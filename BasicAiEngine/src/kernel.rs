use std::error::Error;
use std::sync::Arc;
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use crate::KernelFunctions;

pub fn add_bias_batched(
    stream: &Arc<CudaStream>,
    kernel_functions: &KernelFunctions,
    out_buf: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    swiglu_features: &i32,
    batches: i32
) -> Result<(), Box<dyn Error>> {
    let launch_cfg = LaunchConfig::for_num_elems(batches as u32 * *swiglu_features as u32);

    let mut launch_args = stream.launch_builder(&kernel_functions.add_bias_batched);
    launch_args.arg(&*out_buf);
    launch_args.arg(bias);
    launch_args.arg(&batches);
    launch_args.arg(swiglu_features);

    unsafe { launch_args.launch(launch_cfg)? };

    Ok(())
}

const EPS: f32 = 0.00001;

pub fn rmsnorm(
    stream: &Arc<CudaStream>,
    kernel_functions: &KernelFunctions,
    in_buf: &CudaSlice<f32>,
    out_buf: &CudaSlice<f32>,
    gamma: &CudaSlice<f32>,
    features: i32,
    batches: i32
) -> Result<(), Box<dyn Error>> {
    let launch_cfg = LaunchConfig {
        grid_dim: (batches as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch_args = stream.launch_builder(&kernel_functions.rmsnorm);
    launch_args.arg(&*in_buf);
    launch_args.arg(&*out_buf);
    launch_args.arg(&*gamma);
    launch_args.arg(&features);
    launch_args.arg(&EPS);
    launch_args.arg(&batches);

    unsafe { launch_args.launch(launch_cfg)? };

    Ok(())
}

pub fn rmsnorm_backward(
    stream: &Arc<CudaStream>,
    kernel_functions: &KernelFunctions,
    grad_output: &CudaSlice<f32>, // dy: пришедший градиент (после GEMM)
    input: &CudaSlice<f32>,       // x: сырой вход (layer.saved_no_normalized_input)
    gamma: &CudaSlice<f32>,       // layer.gamma
    grad_input: &CudaSlice<f32>,  // dx: градиент входа (полетит в предыдущий слой)
    grad_gamma: &CudaSlice<f32>,  // d_gamma: куда накопить градиент гаммы (layer.grad_gamma)
    features: i32,
    batches: i32
) -> Result<(), Box<dyn Error>> {
    let launch_cfg = LaunchConfig {
        grid_dim: (batches as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch_args = stream.launch_builder(&kernel_functions.rmsnorm_backward);
    launch_args.arg(&*grad_output);
    launch_args.arg(&*input);
    launch_args.arg(&*gamma);
    launch_args.arg(&*grad_input);
    launch_args.arg(&*grad_gamma);
    launch_args.arg(&features);
    launch_args.arg(&EPS);
    launch_args.arg(&batches);

    unsafe { launch_args.launch(launch_cfg)? };

    Ok(())
}

pub fn mse_loss_backward(
    stream: &Arc<CudaStream>,
    kernel_functions: &KernelFunctions,
    pred: &CudaSlice<f32>,
    target: &CudaSlice<f32>,
    grad_out: &CudaSlice<f32>,
    total: i32
) -> Result<(), Box<dyn Error>> {
    let launch_cfg = LaunchConfig::for_num_elems(total as u32);

    let mut launch_args = stream.launch_builder(&kernel_functions.mse_loss_backward);
    launch_args.arg(&*pred);
    launch_args.arg(&*target);
    launch_args.arg(&*grad_out);
    launch_args.arg(&total);

    unsafe { launch_args.launch(launch_cfg)? };

    Ok(())
}

pub fn swiglu_backward(
    stream: &Arc<CudaStream>,
    kernel_functions: &KernelFunctions,
    gradient: &CudaSlice<f32>,     // dy: прилетевший градиент (размер batches * out_features)
    grad_linear: &CudaSlice<f32>,  // выход: сдвоенный градиент для gate и up (batches * out_features * 2)
    saved_linear: &CudaSlice<f32>, // layer.saved_linear (gate и up до активации)
    out_features: i32,
    batches: i32
) -> Result<(), Box<dyn Error>> {
    let launch_cfg = LaunchConfig::for_num_elems(batches as u32 * out_features as u32);

    let mut launch_args = stream.launch_builder(&kernel_functions.swiglu_backward);
    launch_args.arg(&*gradient);
    launch_args.arg(&*grad_linear);
    launch_args.arg(&*saved_linear);
    launch_args.arg(&out_features);
    launch_args.arg(&batches);

    unsafe { launch_args.launch(launch_cfg)? };

    Ok(())
}

pub fn bias_backward(
    stream: &Arc<CudaStream>,
    kernel_functions: &KernelFunctions,
    grad_linear: &CudaSlice<f32>, // сдвоенный градиент из swiglu_backward
    grad_bias: &CudaSlice<f32>,   // куда сложить сумму (layer.grad_bias)
    swiglu_features: i32,         // out_features * 2
    batches: i32
) -> Result<(), Box<dyn Error>> {
    let launch_cfg = LaunchConfig::for_num_elems(swiglu_features as u32);
    let mut launch_args = stream.launch_builder(&kernel_functions.bias_backward);
    launch_args.arg(&*grad_linear);
    launch_args.arg(&*grad_bias);
    launch_args.arg(&swiglu_features);
    launch_args.arg(&batches);
    unsafe { launch_args.launch(launch_cfg)? };
    Ok(())
}

pub fn sgd_step(
    stream: &Arc<CudaStream>,
    kernel_functions: &KernelFunctions,
    param: &mut CudaSlice<f32>,
    grad: &CudaSlice<f32>,
    lr: f32,
) -> Result<(), Box<dyn Error>> {
    let total = param.len() as i32;
    let launch_cfg = LaunchConfig::for_num_elems(total as u32);

    let mut launch_args = stream.launch_builder(&kernel_functions.sgd_step);
    launch_args.arg(param);
    launch_args.arg(&*grad);
    launch_args.arg(&lr);
    launch_args.arg(&total);

    unsafe { launch_args.launch(launch_cfg)? };
    Ok(())
}