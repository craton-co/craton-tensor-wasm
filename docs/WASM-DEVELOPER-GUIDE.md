# Wasm Developer Guide

This guide walks you through writing Wasm functions for **TensorWasm**, from a trivial `add(a, b)` through hand-tuned GPU kernels and the auto-offload fast path. If you haven't already deployed a hello-world, start with [GETTING-STARTED.md](./GETTING-STARTED.md).

## 1. The Wasm target

Craton TensorWasm targets **`wasm32-wasip1`** — the WebAssembly System Interface, Preview 1. This is the modern, stable WASI target and the one you almost always want:

```sh
rustup target add wasm32-wasip1
```

`wasm32-wasip1` gives your guest access to a curated set of host imports: clocks, random, filesystem (sandboxed), and TensorWasm's own `wasi:cuda/host@0.1.0` for GPU work.

If you're writing a **pure compute kernel** with no I/O, you can also use `wasm32-unknown-unknown`. The output is smaller and the link is faster, but you lose all WASI imports — no clocks, no random, no GPU. Use it only when you genuinely need nothing from the host.

| Target | WASI imports | `wasi:cuda` available | Typical use |
|---|---|---|---|
| `wasm32-wasip1` | Yes | Yes | Almost everything |
| `wasm32-unknown-unknown` | No | No | Tiny pure-compute kernels |

## 2. Project layout

A minimal `Cargo.toml` for a TensorWasm function:

```toml
[package]
name = "my_fn"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]
```

`crate-type = ["cdylib"]` is what produces a `.wasm` file with C-ABI exports rather than a Rust `rlib`. Use this layout for compute libraries that the host calls into by export name.

If instead you want a **WASI binary** with a `_start` entry point — useful for one-shot batch jobs — use:

```toml
[[bin]]
name = "my_fn"
path = "src/main.rs"
```

TensorWasm handles both layouts; the choice is yours.

## 3. A minimal compute function

Here's a complete `src/lib.rs` exposing both a scalar `add` and a vectorized `run`:

```rust
#[no_mangle]
pub extern "C" fn add(a: f32, b: f32) -> f32 {
    a + b
}

/// Read `len` floats from `input_ptr`, write `len` doubled values to `out`.
#[no_mangle]
pub extern "C" fn run(input_ptr: *const f32, len: usize, out: *mut f32) {
    // SAFETY: the host guarantees these point to len*4 bytes of guest memory.
    let input = unsafe { core::slice::from_raw_parts(input_ptr, len) };
    let output = unsafe { core::slice::from_raw_parts_mut(out, len) };
    for i in 0..len {
        output[i] = input[i] * 2.0;
    }
}
```

Build it:

```sh
cargo build --target wasm32-wasip1 --release
```

The resulting `.wasm` exports `add` and `run`. The host invokes them by name with the arguments you pass through the API.

Pointers (`*const f32`, `*mut f32`) are **guest-relative offsets** into the Wasm linear memory — they're plain `i32` values at the ABI level. TensorWasm's host translates them transparently.

## 4. Using wasi-cuda for explicit GPU kernels

When you've written a CUDA kernel by hand and compiled it to PTX, you can launch it from your Wasm guest using the `wasi:cuda/host@0.1.0` import surface.

Declare the host imports your guest will call:

```rust
#[link(wasm_import_module = "wasi:cuda/host@0.1.0")]
extern "C" {
    fn wasi_cuda_load_ptx(
        ptx_ptr: i32, ptx_len: i32,
        entry_ptr: i32, entry_len: i32,
    ) -> i64;
    fn wasi_cuda_launch(
        kernel_id: i64,
        grid_x: i32, grid_y: i32, grid_z: i32,
        block_x: i32, block_y: i32, block_z: i32,
        shared_mem: i32,
        args_ptr: i32, args_len: i32,
    ) -> i32;
    fn wasi_cuda_sync() -> i32;
    fn wasi_cuda_last_error_len() -> i32;
}
```

A complete vector-add kernel that runs on the GPU:

```rust
static PTX: &[u8] = include_bytes!("../kernels/vector_add.ptx");

#[no_mangle]
pub extern "C" fn vector_add_gpu(
    a: *const f32, b: *const f32, out: *mut f32, len: usize,
) -> i32 {
    let entry = b"vector_add\0";
    let kernel_id = unsafe {
        wasi_cuda_load_ptx(
            PTX.as_ptr() as i32, PTX.len() as i32,
            entry.as_ptr() as i32, (entry.len() - 1) as i32,
        )
    };
    if kernel_id < 0 { return -1; }

    // Pack arguments as a flat array of pointer/scalar slots.
    let args: [i64; 4] = [
        a as i64, b as i64, out as i64, len as i64,
    ];

    let block = 256i32;
    let grid = ((len as i32) + block - 1) / block;

    let rc = unsafe {
        wasi_cuda_launch(
            kernel_id,
            grid, 1, 1,
            block, 1, 1,
            0,
            args.as_ptr() as i32, (args.len() * 8) as i32,
        )
    };
    if rc != 0 { return -2; }

    unsafe { wasi_cuda_sync() }
}
```

Three things to notice:

1. **PTX is embedded** via `include_bytes!`. TensorWasm caches loaded PTX per-instance keyed by hash, so repeat loads are free.
2. **`kernel_id` is opaque** — treat it as a handle. It's only valid within the lifetime of the current instance.
3. **`wasi_cuda_sync()` is explicit** — kernel launches are asynchronous on the device. Always sync before reading results from host-shared memory.

If anything goes wrong, `wasi_cuda_last_error_len()` returns the length of a host-side error string; pair it with a `wasi_cuda_last_error_copy` to retrieve the message.

## 5. Auto-offload (opt-in)

Many compute loops don't need a hand-written PTX kernel — TensorWasm can detect SIMD-shaped Rust loops and promote them to GPU kernels at instantiation time.

The promotion criteria, at a high level:

- The loop must use `core::arch::wasm32` **v128** SIMD intrinsics, or be auto-vectorized by `rustc` into v128 ops.
- The estimated **trip count** must exceed the `trip_count` threshold (default 4096).
- The **v128-ratio** — fraction of body ops that are SIMD — must exceed the configured threshold (default 0.5).

Both thresholds are configurable per deployment. Full details — including the IR pattern matcher and how to inspect promotion decisions — live in [AUTO-OFFLOAD.md](./AUTO-OFFLOAD.md).

Auto-offload always has a **CPU fallback path** wired up; you'll never silently fail to run because the GPU rejected your kernel.

## 6. Memory model

Your Wasm guest sees its own **linear memory** — a flat `u8` buffer addressed by `i32` offsets. From the guest's point of view, that's the entire universe.

Under the hood, TensorWasm backs that linear memory with a **`UnifiedBuffer`** that's also mapped into the CUDA device's address space. When you call `wasi_cuda_launch` with a pointer, you're passing the host a **guest-relative offset**; the host translates that into the equivalent device pointer for the kernel. The kernel reads and writes the same bytes your guest sees — no explicit `cudaMemcpy` needed.

The practical implications:

- Allocate buffers with `Vec<T>` or `Box<[T]>` and pass `.as_ptr()` / `.as_mut_ptr()` directly to host calls. They're already in unified memory.
- Don't assume pointers are stable across `sync` if you've grown a `Vec` — linear memory may have been re-paged.
- Cross-instance buffer sharing is not supported; each instance owns its memory.

## 7. Limits per instance

| Limit | Default | Notes |
|---|---|---|
| `MAX_PTX_BYTES` | 8 MiB | Per `wasi_cuda_load_ptx` call. |
| `MAX_KERNELS_PER_INSTANCE` | 256 | Across the instance's lifetime. |
| Epoch deadline | 30 s | Wired from `SpawnConfig`; configurable per deployment. |
| Linear memory | 256 MiB | `EngineConfig::max_memory_bytes`. |

Hitting any of these terminates the instance with a clear diagnostic; they're guardrails, not silent truncations.

## 8. Debugging

The single most useful knob is `TENSOR_WASM_LOG`:

```sh
TENSOR_WASM_LOG=debug cargo run --bin tensor-wasm -- run my_fn.wasm
```

At `debug`, the host logs every WASI-CUDA call with its arguments — PTX hash on load, grid/block dims on launch, return codes on every call. At `trace`, you also get per-arg byte dumps.

Tracing spans are grouped per instance, so when you're running under the HTTP server you can pivot in Jaeger from a single invocation down to every kernel it dispatched.

For deeper performance work, see [PERFORMANCE.md](./PERFORMANCE.md) and [COLD-START.md](./COLD-START.md).
