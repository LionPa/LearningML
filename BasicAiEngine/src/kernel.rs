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