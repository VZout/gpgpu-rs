// Required preamble
#![feature(asm, register_attr)]
#![cfg_attr(feature = "gpu", no_std)]
#![register_attr(spirv)]

#[macro_use]
extern crate gpgpu;

// Normal functions are usable on GPU and CPU.
fn do_stuff() -> f32 {
    0.0
}

#[cpu_only] // `#[cpu_only]` Excludes code for the GPU.
use futures::executor::block_on;

// `#[gpu]` marks a function that is executed on the GPU.
#[gpu]
pub fn calculate(value: f32) -> f32 {
    return do_stuff();
}

// `async` is allowed (even recommended).
#[gpu]
pub async fn calculate_async(value: f32) {
    do_stuff();
}

// `#[entry]` creates compute context and implies its cpu only.
#[entry]
fn main() {
    // Executes function on the GPU.
    calculate(0.0);
    block_on(calculate_async(0.0));
    // GPU functions can still be ran on the CPU.
    let _ = calculate__cpu(0.0);
    let _ = block_on(calculate_async__cpu(0.0));
}
