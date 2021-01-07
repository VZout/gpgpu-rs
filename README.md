<p align="center"><strong>gpgpu-rs</strong></p>
<p align="center">
  <i><q>Exploration of inline GPU code within rust</q></i>
</p>

### Introduction

This is a proof of concept showcasing inline shaders/kernels intertwined with CPU code. Currently only the graphical compute shading stage is supported.

### Demo

```rust
// Normal functions are usable on GPU and CPU.
fn do_stuff() -> f32 { 0.0 }

// `#[gpu]` marks a function that is executed on the GPU.
#[gpu] pub fn calculate(value: f32) -> f32 {
    return do_stuff();
}

// `async` is allowed (even recommended).
#[gpu] pub async fn calculate_async() -> f32 {
    return do_stuff();
}

#[entry] fn main() {
    // Executes function on the GPU.
    let _ = calculate(0.0);
    let _ = block_on(calculate_async());
    // GPU functions can still be ran on the CPU.
    let _ = calculate__cpu(0.0);
    let _ = block_on(calculate_async__cpu());
}
```

See the complete example in [`./example/`](#)

**Glossary**

* [**`#[cpu_only]`**](#) - *Not compiled for GPU*
* [**`#[entry]`**](#) - *Impies `#[cpu_only]` and creates a compute context*
* [**`#[gpu]`**](#) - *A function that is capable of starting GPU work*

### To-Do

- Upload arguments to GPU.
- Expand function parameters. (Removing the need to call Load on args)
- Write return value.
- GPU readback.

* Any type of return expression.
* Multiple input variables.
* Multiple return experssions.
* Allow calling `#[gpu]` functions from GPU.

- `no_std` version of `spirv-builder`.

* Speed up GPU execution.
