mod kernel;

use std::error::Error;
use std::sync::Arc;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{ compile_ptx_with_opts, CompileOptions };
use rand::RngExt;
use crate::kernel::{add_bias_batched, rmsnorm};

pub struct LinearLayer {
    pub in_features: usize,
    pub out_features: usize,

    pub weights: CudaSlice<f32>,
    pub bias: CudaSlice<f32>,
    pub gamma: CudaSlice<f32>,

    pub grad_weights: Option<CudaSlice<f32>>,
    pub grad_bias: Option<CudaSlice<f32>>,
    pub grad_gamma: Option<CudaSlice<f32>>,

    pub saved_input: Option<CudaSlice<f32>>,
    pub saved_no_normalized_input: Option<CudaSlice<f32>>,
    pub saved_linear: Option<CudaSlice<f32>>, // z до ReLU
}

impl LinearLayer {
    fn new(stream: &Arc<CudaStream>, in_features: usize, out_features: usize, batches: i32, mode: bool) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            in_features,
            out_features,

            weights: stream.alloc_zeros(in_features * (out_features * 2))?,
            bias: stream.alloc_zeros(out_features * 2)?,
            gamma: stream.alloc_zeros(in_features)?,

            grad_weights: alloc_learning_slice(stream, (in_features * (out_features * 2)) as i32, mode), // x2 для SwiGLU весов gate и нового up
            grad_bias: alloc_learning_slice(stream, (out_features * 2) as i32, mode),
            grad_gamma: alloc_learning_slice(stream, in_features as i32, mode), // in_features потому что pre нормализация

            saved_input: alloc_learning_slice(stream, batches * in_features as i32, mode),
            saved_no_normalized_input: alloc_learning_slice(stream, batches * in_features as i32, mode),
            saved_linear: alloc_learning_slice(stream, batches * (out_features * 2) as i32, mode)
        })
    }
}

pub struct NeuralNetwork {
    pub layers: Vec<LinearLayer>,
}

pub struct KernelFunctions {
    swiglu: CudaFunction,
    add_bias_batched: CudaFunction,
    rmsnorm: CudaFunction,
}

fn main() -> Result<(), Box<dyn Error>> {
    let mode = true; // true - training; false - inference

    // Device
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let blas = CudaBlas::new(stream.clone())?;

    // Module ('s)
    let module = compile_module(ctx)?;

    // Functions
    let kernel_functions = KernelFunctions {
        swiglu: module.load_function("swiglu_forward")?,
        add_bias_batched: module.load_function("add_bias_batched")?,
        rmsnorm: module.load_function("rmsnorm")?,
    };

    // Структуры тестовой нейронки. 5 входных - 3 - 3 - 2 выходных
    let batches = 1;

    let mut network = create_network(&stream, batches, mode)?;

    fill_network_with_noise(&stream, &mut network)?;

    let mut buf_a = stream.alloc_zeros::<f32>(5 * 2 * batches as usize)?;
    let mut buf_b = stream.alloc_zeros::<f32>(5 * 2 * batches as usize)?;


    let input = [1f32, 2f32, 3f32, 4f32, 5f32, 1f32, 2f32, 3f32, 4f32, 5f32];

    stream.memcpy_htod(&input, &mut buf_a)?;

    let ping_pong = process_network(&stream, &blas, &mut network, batches, &mut buf_a, &mut buf_b, &kernel_functions, mode)?;

    let out_buf = if ping_pong % 2 == 0 { &buf_a } else { &buf_b };

    let test_out = stream.clone_dtoh(out_buf)?;

    println!("Активация второго слоя {test_out:?}");

    Ok(())
}

fn compile_module(ctx: Arc<CudaContext>) -> Result<Arc<CudaModule>, Box<dyn Error>> {
    let opts = CompileOptions {
        arch: Some("compute_120"),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(include_str!("../kernel.cu"), opts)?;

    let module = ctx.load_module(ptx)?;

    Ok(module)
}

fn alloc_learning_slice(stream: &Arc<CudaStream>, n: i32, mode: bool) -> Option<CudaSlice<f32>> {
     if !mode { return None }

    Some(stream.alloc_zeros(n as usize).unwrap())
}

fn create_network(stream: &Arc<CudaStream>, batches: i32, mode: bool) -> Result<NeuralNetwork, Box<dyn Error>> {
    let mut network = NeuralNetwork {
        layers: Vec::new()
    };

    let l1 = LinearLayer::new(&stream, 5, 3, batches, mode)?;
    let l2 = LinearLayer::new(&stream, 3, 3, batches, mode)?;
    let l3 = LinearLayer::new(&stream, 3, 2, batches, mode)?;

    network.layers.push(l1);
    network.layers.push(l2);
    network.layers.push(l3);

    Ok(network)
}

fn fill_network_with_noise(stream: &Arc<CudaStream>, neural_network: &mut NeuralNetwork) -> Result<(), Box<dyn Error>> {
    let mut rng = rand::rng();

    for layer in neural_network.layers.iter_mut() {
        let limit = (6.0f32 / layer.in_features as f32).sqrt();

        let total_weights = layer.in_features * layer.out_features * 2;

        let weights: Vec<f32> = (0..total_weights)
            .map(|_| rng.random_range(-limit..limit))
            .collect();

        let bias = vec![0f32; layer.out_features * 2];
        let gamma = vec![1f32; layer.in_features];

        let layer_weight = &mut layer.weights;
        let layer_bias = &mut layer.bias;
        let layer_gamma = &mut layer.gamma;

        stream.memcpy_htod(weights.as_slice(), layer_weight)?;
        stream.memcpy_htod(bias.as_slice(), layer_bias)?;
        stream.memcpy_htod(gamma.as_slice(), layer_gamma)?;
    }

    Ok(())
}

fn process_network(
    stream: &Arc<CudaStream>,
    blas: &CudaBlas,
    network: &mut NeuralNetwork,
    batches: i32,
    buf_a: &mut CudaSlice<f32>,
    buf_b: &mut CudaSlice<f32>,
    kernel_functions: &KernelFunctions,
    mode: bool
) -> Result<i32, Box<dyn Error>> {
    let layers = network.layers.len();

    let mut ping_pong_index = 0;
    let mut layer_index = 0usize;

    for _ in 0..layers {
        step(&stream, &blas, network, batches, buf_a, buf_b, ping_pong_index, layer_index, kernel_functions, mode)?;
        ping_pong_index += 1;
        layer_index += 1;

        let test_out;

        if ping_pong_index % 2 == 0 { // TODO Удалить. для дебага
            test_out = stream.clone_dtoh(buf_a)?;
        } else {
            test_out = stream.clone_dtoh(buf_b)?;
        }

        println!("Активация {layer_index:?} слоя {test_out:?}");
    }

    Ok(ping_pong_index)
}

fn step(
    stream: &Arc<CudaStream>,
    blas: &CudaBlas,
    network: &mut NeuralNetwork,
    batches: i32,
    buf_a: &mut CudaSlice<f32>,
    buf_b: &mut CudaSlice<f32>,
    ping_pong_index: i32,
    layer_index: usize,
    kernel_functions: &KernelFunctions,
    mode: bool
) -> Result<(), Box<dyn Error>> {
    let layer = network.layers.get_mut(layer_index).unwrap();

    let out_buf;
    let in_buf;

    if ping_pong_index % 2 == 0 {
        in_buf = buf_a;
        out_buf = buf_b;
    } else {
        out_buf = buf_a;
        in_buf = buf_b;
    }

    let in_features = layer.in_features as i32;
    let out_features = layer.out_features as i32;
    let swiglu_features = out_features * 2;

    if mode { // Training
        let u_batches = batches as usize;

        stream.memcpy_dtod(&in_buf.slice(0..u_batches * layer.in_features), layer.saved_no_normalized_input.as_mut().unwrap())?;
    }

    rmsnorm(stream, kernel_functions, in_buf, in_buf, &layer.gamma, in_features, batches)?;

    let gemm_cfg = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: swiglu_features,
        n: batches,
        k: in_features,
        alpha: 1f32,
        beta: 0f32,
        lda: swiglu_features,
        ldb: in_features,
        ldc: swiglu_features
    };

    unsafe {
        blas.gemm(gemm_cfg, &layer.weights, in_buf, out_buf)?
    }

    add_bias_batched(stream, kernel_functions, out_buf, &layer.bias, &swiglu_features, batches)?;

    if mode { // Training
        let u_batches = batches as usize;

        stream.memcpy_dtod(&out_buf.slice(0..u_batches * swiglu_features as usize), layer.saved_linear.as_mut().unwrap())?;
        stream.memcpy_dtod(&in_buf.slice(0..u_batches * layer.in_features), layer.saved_input.as_mut().unwrap())?;
    }

    let launch_cfg = LaunchConfig::for_num_elems(batches as u32 * out_features as u32);

    let mut launch_args = stream.launch_builder(&kernel_functions.swiglu);
    launch_args.arg(&*out_buf);
    launch_args.arg(&out_features);
    launch_args.arg(&batches);

    unsafe { launch_args.launch(launch_cfg)? };

    Ok(())
}

fn backprop() -> Result<(), Box<dyn Error>> {

    Ok(())
}