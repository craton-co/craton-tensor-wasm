<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Craton Software Company
-->

# Writing CUDA Kernels for Craton TensorWasm

A practical guide for developers who already know CUDA and want to
ship a kernel that loads and dispatches under TensorWasm's
`wasi:cuda` surface. Covers **explicit dispatch** (PTX you write and
the guest loads at runtime), the **auto-offload** path (Cranelift
pattern detection emits PTX for you), and the shared machinery they
ride on: the W1.1 typed-argv wire format, host-side bounds checking,
and the launch contract from [`wit/wasi-cuda.wit`](../wit/wasi-cuda.wit).

Status: **v0.3 workstream**. The kernel-args lane in Section 6
shipped in W1.1 (March 2026). The auto-offload runtime-swap hook
(Section 11) is still gated behind the per-block Cranelift hook
tracked in [`docs/WASMTIME-FORK.md`](WASMTIME-FORK.md). Path C
(Section 5) is forward-looking scaffold per
[RFC 0001](../rfcs/0001-cuda-oxide-integration.md).

## Contents

1. [Audience and prerequisites](#1-audience-and-prerequisites)
2. [Two dispatch surfaces](#2-two-dispatch-surfaces)
3. [Writing a PTX kernel by hand](#3-writing-a-ptx-kernel-by-hand)
4. [Compiling .cu to PTX](#4-compiling-cu-to-ptx)
5. [Path C: Rust kernels via cuda-oxide](#5-path-c-rust-kernels-via-cuda-oxide)
6. [Loading and launching from a wasm guest](#6-loading-and-launching-from-a-wasm-guest)
7. [Bounds checking and pointer args](#7-bounds-checking-and-pointer-args)
8. [Performance tips](#8-performance-tips)
9. [Common pitfalls](#9-common-pitfalls)
10. [Testing](#10-testing)
11. [Auto-offload coverage](#11-auto-offload-coverage)
12. [Related](#12-related)

---

## 1. Audience and prerequisites

This guide is for a developer who has written at least one CUDA
kernel in `.cu` or hand-rolled PTX, understands the SIMT execution
model (warp = 32 threads, blocks on SMs), knows what `cuLaunchKernel`'s
`gridDim` / `blockDim` mean, and wants the kernel to run inside a
Wasm module dispatched by TensorWasm. A working CUDA toolchain pinned
per [`docs/CUDA-SETUP.md`](CUDA-SETUP.md) (12.4 on the S22 runner;
12.0–13.2 on contributor dev boxes) is assumed. If you don't have a
CUDA-capable host, the launch path is still exercisable against the
no-CUDA stub (see [Section 10](#10-testing)).

You do **not** need Wasm internals beyond: a module imports
functions, exports a linear memory, and writes bytes into it before
calling imports. Host functions are declared in
[`wit/wasi-cuda.wit`](../wit/wasi-cuda.wit) and implemented in
[`crates/tensor-wasm-wasi-gpu`](../crates/tensor-wasm-wasi-gpu).

---

## 2. Two dispatch surfaces

TensorWasm exposes two ways for a Wasm guest's GPU work to reach the
driver. They share machinery (the kernel registry, the bounds checker,
the back-pressure semaphore) but the developer-facing experience is
different.

### 2.1 Explicit dispatch (PTX written by hand)

The guest treats PTX as data: (1) compile `.cu` → PTX ahead of time;
(2) bundle or fetch the PTX; (3) call
`wasi_cuda_load_ptx(ptx_bytes, entry_name)` for a `kernel_id`; (4)
pack parameters using the W1.1 typed-argv wire format
([Section 3.3](#33-calling-convention)); (5) call
`wasi_cuda_launch(kernel_id, grid, block, shared_mem, args)`; (6)
call `wasi_cuda_sync()` before reading results.

This is what most of this guide covers — the only path that ships
with v0.2 and the only path where you control the PTX.

### 2.2 Auto-offload (Cranelift to PTX)

The guest writes nothing CUDA-aware. Plain Wasm SIMD (`v128.*`) loops
are inspected by [`tensor_wasm_jit::detector::classify`](../crates/tensor-wasm-jit/src/detector.rs)
at JIT-pipeline time; recognized patterns (element-wise f32, FMA
GEMV / dot-product, tiled f16 → f32 matmul, 3x3 conv2d stencil) are
lowered to PTX blueprints and cached. See [Section 11](#11-auto-offload-coverage)
and [`docs/AUTO-OFFLOAD.md`](AUTO-OFFLOAD.md). The auto-offload runtime
swap is **not yet wired** in v0.2 — the pipeline pre-emits PTX but
Wasmtime still dispatches the Cranelift body at runtime.

---

## 3. Writing a PTX kernel by hand

### 3.1 PTX version target

TensorWasm pins the toolchain to **CUDA 12.4** on the S22 runner (see
[`docs/CUDA-SETUP.md` §Required versions](CUDA-SETUP.md#required-versions)),
so **PTX ISA 8.0** is the lowest common denominator your kernel must
declare. Anything above 8.0 will fail to JIT on the runner; the
fixtures under `kernels/` all pin `.version 8.0`.

```ptx
.version 8.0
.target  sm_80
.address_size 64
```

Bumping `.version` is a build-host concern; bumping `.target` is a
runtime concern — see the SM matrix next.

### 3.2 SM target matrix

Mirrors [`CUDA-SETUP.md` §SM-level compatibility matrix](CUDA-SETUP.md#sm-level-compatibility-matrix):

| Kernel kind | Minimum `.target` | Hardware floor |
|---|---|---|
| Scalar / vector arithmetic | `sm_70` | V100 / Titan V |
| `cp.async` / async-copy intrinsics | `sm_80` | A100 |
| `wmma.mma.sync` tensor-core matmul (f16 → f32) | `sm_80` | A100 |
| `cp.async.bulk` / tensor-memory accelerator | `sm_90` | H100 |

Set `.target sm_70` for broadest coverage. A `.target` exceeding the
running GPU's compute capability surfaces as `MalformedPtx` (`-4`) —
`cust::module::Module::from_ptx` fails the JIT compile inside ptxas.

### 3.3 Calling convention

In a C host you pass arguments via a `void**` that `cuLaunchKernel`
reads. A Wasm guest can't author a `void**` directly — it lives in
linear memory and only sees i32 offsets — so the W1.1 wire format
flattens the argument list into one tagged byte buffer that the host
parses, bounds-checks, and lowers into the `void**` for you.

The wire format is documented in full in
[`crates/tensor-wasm-wasi-gpu/src/kernel_args.rs`](../crates/tensor-wasm-wasi-gpu/src/kernel_args.rs):

| Tag byte | PTX type | Value bytes (LE) | Wire size |
|---|---|---|---|
| `0x01` | `.s32` | 4 | 5 |
| `0x02` | `.s64` | 8 | 9 |
| `0x03` | `.f32` | 4 | 5 |
| `0x04` | `.f64` | 8 | 9 |
| `0x05` | `.u32` | 4 | 5 |
| `0x06` | `.u64` | 8 | 9 |
| `0x07` | `.ptr` (pointer arg) | 4 (guest offset) + 4 (byte length) | 9 |

No inter-argument padding. The buffer is a flat concatenation of
`(tag, value)` records; the host reads exactly `tag.value_bytes()`
after each tag byte. The packing maps one-to-one onto a PTX `.param`
block — the PTX-side type drives the tag:

```ptx
.visible .entry my_kernel(
    .param .u64 a_ptr,     // tag 0x07: ptr + len
    .param .u64 b_ptr,     // tag 0x07: ptr + len
    .param .u32 n,         // tag 0x05: u32
    .param .f32 scale      // tag 0x03: f32
)
```

Pointer args carry **both** the guest-memory offset **and** the byte
length. The host bounds-checks `[ptr, ptr+len)` against the guest's
linear memory; the checked window becomes a raw host pointer that —
under CUDA Unified Memory — doubles as a device address. The original
guest offset is preserved for log lines and tests
([`LoweredArg::Ptr`](../crates/tensor-wasm-wasi-gpu/src/kernel_args.rs)).

Sanity caps enforced by `parse_argv`:

- `MAX_KERNEL_ARGS = 128`
- `MAX_KERNEL_ARGS_BYTES = 4 * 1024` (4 KiB)

Either violation surfaces as `KernelArgsUnsupported` (`-10`). The
caps are deliberately low so a malicious guest can't pin host memory
inside the parser ([`docs/RISKS.md`](RISKS.md)).

### 3.4 Worked example: vector_add.ptx

The reference fixture
[`kernels/vector_add.ptx`](../kernels/vector_add.ptx) round-trips the
calling convention with three pointer args and one scalar:

```ptx
.version 8.0
.target sm_80
.address_size 64

.visible .entry vector_add(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 n
)
{
    .reg .pred  %p<2>;
    .reg .s32   %r<6>;
    .reg .s64   %rd<11>;
    .reg .f32   %f<4>;

    ld.param.u64    %rd1, [a_ptr];
    ld.param.u64    %rd2, [b_ptr];
    ld.param.u64    %rd3, [c_ptr];
    ld.param.u32    %r1,  [n];

    mov.u32         %r2, %ntid.x;
    mov.u32         %r3, %ctaid.x;
    mov.u32         %r4, %tid.x;
    mad.lo.s32      %r5, %r2, %r3, %r4; // i = bd.x*bi.x + ti.x

    setp.ge.s32     %p1, %r5, %r1;
    @%p1 bra        L_done;

    cvt.s64.s32     %rd4, %r5;
    shl.b64         %rd5, %rd4, 2;      // i * sizeof(f32)
    add.s64         %rd6, %rd1, %rd5;
    add.s64         %rd7, %rd2, %rd5;
    add.s64         %rd8, %rd3, %rd5;

    ld.global.f32   %f1, [%rd6];
    ld.global.f32   %f2, [%rd7];
    add.f32         %f3, %f1, %f2;
    st.global.f32   [%rd8], %f3;

L_done:
    ret;
}
```

The four `.param` declarations correspond, in order, to four argv
records the guest packs:

```text
[tag=0x07, a_offset:u32, a_len:u32,   // a_ptr
 tag=0x07, b_offset:u32, b_len:u32,   // b_ptr
 tag=0x07, c_offset:u32, c_len:u32,   // c_ptr
 tag=0x05, n:u32]                     // n
```

Total wire size: 9 + 9 + 9 + 5 = 32 bytes, well under the 4 KiB cap.

---

## 4. Compiling .cu to PTX

Most contributors write the source in `.cu` and let `nvcc` emit PTX,
rather than hand-rolling PTX. Both reach the same `wasi_cuda_load_ptx`
call site.

### 4.1 Basic compile

```bash
nvcc --ptx -arch=sm_75 vector_add.cu -o vector_add.ptx
```

`--ptx` stops at PTX; the output is UTF-8 text the guest can bundle
as-is. `-arch=sm_75` is a **concrete** target: ship to a higher SM
and the driver JIT-recompiles; ship to a lower SM and the launch
fails.

### 4.2 Virtual vs concrete targets

For kernels that ship once and run on a range of GPUs, use the
**virtual** form:

```bash
nvcc --ptx -arch=compute_70 vector_add.cu -o vector_add.ptx
```

`compute_70` PTX is valid input for any concrete SM at or above 70;
the driver JIT picks the final code at load time. Use `compute_70`
for scalar / vector, `compute_80` for wmma. To support both Turing
(no wmma) and Ampere+ (with wmma), compile two PTX modules and pick
at runtime — there is no `wasi_cuda_get_device_capability` function
yet (tracked in [`docs/PATH-TO-V1.md`](PATH-TO-V1.md)).

### 4.3 Validation and register usage

`ptxas` validates the PTX and reports register pressure:

```bash
ptxas --gpu-name=sm_75 -O3 -v vector_add.ptx
```

Sample output (vector_add on SM_75):

```text
ptxas info    : Compiling entry function 'vector_add' for 'sm_75'
ptxas info    : Function properties for vector_add
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 10 registers, 376 bytes cmem[0]
```

Watch: **registers** above ~64 limit occupancy on SM_75 / SM_80,
above 255 the kernel can't launch with a 1024-thread block; non-zero
**stack frame** means nvcc spilled to local memory; **cmem[0]** above
~32 KiB eats into the 64 KiB per-launch cap. The host won't reject
high register pressure but contended launches silently lose throughput.

---

## 5. Path C: Rust kernels via cuda-oxide

A third authoring path: write the kernel as a `#[no_std]` Rust
function annotated with cuda-oxide's `#[cuda_module]` macro, build it
with `cargo oxide build`, and load the emitted PTX through the same
`wasi_cuda_load_ptx` host fn that consumes Path A and Path B output.
The pitch is single-language ergonomics — no `.ptx` or `.cu`
sidecars, no nvcc-out-of-band step — paired with Rust's safety on
the host-facing pieces of the kernel (lifetimes for pointer args,
trait-checked numeric types) that Path A simply does not check.
The PTX bytes Path C emits are interchangeable with the bytes Paths
A and B emit; nothing downstream of `load_ptx` knows or cares which
path produced them.

### 5.1 Status (v0.3.1)

cuda-oxide is **v0.1.0 alpha** (NVIDIA Labs, released 2026-05-09).
The TensorWasm `cuda-oxide-backend` feature flag in `tensor-wasm-mem`
is a scaffold — the host-runtime surface is stubbed, and the
kernel-authoring side (`cuda-device`, `cuda-macros`) is not yet
wired into any in-tree example. The real implementation lands in
the v0.4 port tracked by
[RFC 0001](../rfcs/0001-cuda-oxide-integration.md). The guidance
below is forward-looking; anyone exercising this path against the
v0.1.x crates today should expect API breakage at v0.2 and budget
for rework. See
[`crates/tensor-wasm-mem/README-cuda-oxide.md`](../crates/tensor-wasm-mem/README-cuda-oxide.md)
for the host-runtime scaffold's current shape and feature-flag
matrix.

### 5.2 Toolchain prereqs

cuda-oxide pins `nightly-2026-04-03` in its own
`rust-toolchain.toml`. TensorWasm's workspace pins
`nightly-2026-03-15`, so Path C requires a **local toolchain
override** — building this path on the workspace default fails with
a cryptic missing-feature error. Per RFC 0001 "Toolchain plan" step
3, the workspace default bumps to a cuda-oxide-compatible nightly
at v0.4; until then the override is the contract.

```bash
# One-shot override via env var:
RUSTUP_TOOLCHAIN=nightly-2026-04-03 cargo oxide build --release

# Or branch-local rust-toolchain.toml:
#   [toolchain]
#   channel = "nightly-2026-04-03"
#   components = ["rust-src", "rustc-dev", "llvm-tools-preview"]
```

Components required on the override toolchain:

- `rust-src` — cuda-oxide's codegen backend re-compiles `core` for
  the `nvptx64-nvidia-cuda` target.
- `rustc-dev` — cuda-oxide is a `rustc` codegen backend and links
  against `rustc_public` internals.
- `llvm-tools-preview` — supplies the `llvm-*` binaries the
  `dialect-llvm` → LLVM IR → PTX stage shells out to.

The `cargo oxide` subcommand is installed via cargo:

```bash
cargo install cargo-oxide  # TODO(v0.4): verify exact crate name once cuda-oxide v0.2 ships
```

### 5.3 Write a kernel in Rust

Skeleton, illustrative. The exact APIs in `cuda_device::prelude` may
shift between v0.1.x patch releases:

```rust
// src/kernels/vector_add.rs
#![no_std]
use cuda_device::prelude::*;

#[cuda_module]
pub mod vector_add {
    pub unsafe fn vector_add(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        n: usize,
    ) {
        let i = thread::index_x() as usize;
        if i < n {
            *out.add(i) = *a.add(i) + *b.add(i);
        }
    }
}
// TODO(v0.4): verify against cuda-oxide v0.2 — the prelude
// re-exports (thread::index_x, etc.) are still in flux.
```

The body mirrors the `vector_add` PTX fixture from
[Section 3.4](#34-worked-example-vector_addptx): one thread per
element, bounds check against `n`, single `add.f32`. The
`#[cuda_module]` macro is what flips the inner items onto the
`nvptx64-nvidia-cuda` codegen path; everything outside the macro
remains a normal host-side Rust crate.

### 5.4 Build to PTX

```bash
cargo oxide build --release --target nvptx64-nvidia-cuda
# outputs target/nvptx64-nvidia-cuda/release/vector_add.ptx
```

The compiler pipeline under the hood is
`Rust source → rustc_public Stable MIR → dialect-mir → mem2reg →
dialect-llvm → LLVM IR → PTX`. The emitted `.ptx` is the same UTF-8
text format Path A authors by hand and Path B emits from nvcc; the
`.entry` symbol matches the inner-fn name (`vector_add`). Validate
register pressure with `ptxas -v` exactly as for Path B — see
[Section 4.3](#43-validation-and-register-usage).

### 5.5 Load and dispatch from TensorWasm

The PTX bytes are interchangeable with Paths A and B, so the guest
side is the existing `load_ptx` + `launch` sequence verbatim — see
[Section 6.2](#62-worked-example-rust-guest) for the Rust guest and
[Section 6.3](#63-worked-example-c-guest) for the C guest. The
five-line shape:

```rust
const PTX: &[u8] = include_bytes!(
    "../target/nvptx64-nvidia-cuda/release/vector_add.ptx");
let kid = unsafe { wasi_cuda_load_ptx(
    PTX.as_ptr() as i32, PTX.len() as i32,
    b"vector_add".as_ptr() as i32, 10) };
// argv packing + wasi_cuda_launch unchanged from Section 6.2.
```

Argv packing (the W1.1 wire format from
[Section 3.3](#33-calling-convention)) is unchanged: the host parses
the same tagged byte stream regardless of how the PTX was authored.

### 5.6 Limitations vs Paths A/B today

- **Toolchain split** is the headline cost. Until the v0.4 workspace
  bump, Path C lives on a different nightly than the rest of the
  codebase; cross-cutting refactors that touch a Path C kernel and
  any other crate require two `cargo` invocations.
- **v0.1.0 alpha churn.** cuda-oxide explicitly warns about API
  breakage between v0.1 patch releases. The `cuda_device::prelude`
  surface, the `#[cuda_module]` macro inputs, and the `cargo oxide`
  CLI flags are all candidates for renaming before v0.2.
- **Kernel stdlib coverage.** `cuda-device` is materially smaller
  than the CUDA C++ runtime. Atomics, warp-shuffle intrinsics,
  cooperative-groups, and the bulk of `<cuda/std/*>` are not yet
  exposed; kernels that need them stay on Path A or Path B.
- **No SM_80 wmma yet.** cuda-oxide has not lowered tensor-core
  intrinsics (`wmma.mma.sync`, `cp.async.bulk`) through its dialect
  pipeline. Per the SM matrix in
  [`docs/CUDA-SETUP.md`](CUDA-SETUP.md#sm-level-compatibility-matrix),
  this rules out the wmma-blueprint matmul path; tensor-core kernels
  stay on Path A.
- **Pliron supply-chain footprint.** cuda-oxide pulls Pliron via a
  `git` revision pin; `cargo-deny` and the W3.6 reproducible-builds
  pipeline both need allowlist entries that the cust and nvcc paths
  do not require.

### 5.7 When to pick Path C vs A vs B

| Pick | When |
|---|---|
| A (hand-PTX) | Maximum control; warp-level tuning; tensor-core intrinsics (`wmma`, `cp.async.bulk`); SM-specific micro-optimisations the higher paths cannot express. |
| B (.cu via nvcc) | Existing CUDA C++ codebase or team CUDA expertise; full CUDA runtime headers (`<cuda/std/*>`, atomics, cooperative groups); kernels shared with non-TensorWasm consumers. |
| C (Rust via cuda-oxide) | Rust-native team that values single-language ergonomics; safety on pointer/lifetime bookkeeping; willing to track v0.1.x alpha churn and the toolchain split until v0.4. |

The choice is per kernel, not per project — the same crate can ship
Path A wmma kernels alongside Path C element-wise kernels, loaded
through the same `wasi_cuda_load_ptx` call site.

### 5.8 Future (v0.4+)

Path C exists today because v0.4 needs a documented author-side
surface, but the longer-term direction is that **many users will
not write CUDA kernels at all**. RFC 0001's Pliron lever (see the
module docs in
[`crates/tensor-wasm-jit/src/pliron_dialect.rs`](../crates/tensor-wasm-jit/src/pliron_dialect.rs))
adds a fourth front-end to cuda-oxide's pipeline — `Wasm →
Cranelift IR → dialect-mir` — so the auto-offload detector
(Section 11) can lower arbitrary pure-compute Wasm loops to PTX
through the same dialect machinery cuda-oxide uses for Rust source.
That expands auto-offload coverage from the three hand-written
blueprints today to anything the detector can prove safe. Path C
remains for the kernels that benefit from being written explicitly;
the Pliron lowering pass covers everything else.

---

## 6. Loading and launching from a wasm guest

The three host functions declared in
[`wit/wasi-cuda.wit`](../wit/wasi-cuda.wit):

- `load-ptx(ptx: list<u8>, entry: string) -> result<kernel-id, abi-error>`
- `launch(kernel, grid, block, shared-mem, args: list<u8>) -> result<_, abi-error>`
- `sync() -> result<_, abi-error>`

In the raw function-import ABI (pre-Component-Model, what most guests
use today):

- `wasi_cuda_load_ptx(ptx_ptr, ptx_len, entry_ptr, entry_len: i32) -> i64`
- `wasi_cuda_launch(kernel_id: i64, grid_x..block_z, shared_mem, args_ptr, args_len: i32) -> i32`
- `wasi_cuda_sync() -> i32`

All names live under the import module `wasi:cuda/host@0.2.0`. See
[`crates/tensor-wasm-wasi-gpu/src/abi.rs`](../crates/tensor-wasm-wasi-gpu/src/abi.rs)
for the full constant set, including the `wasi_cuda_last_error_len` /
`wasi_cuda_last_error_copy` pair used to retrieve the host's last
recorded error string.

Note: the inline `launch` docs in `wit/wasi-cuda.wit` still describe
v0.1.0 behavior (rejects non-empty `args` with
`kernel-args-unsupported`). That comment is stale — W1.1 now accepts
typed argv; `kernel-args-unsupported` is reserved for sanity-cap
busts (Section 3.3).

### 6.1 Encoding the argv buffer

The argv buffer is a flat sequence of tagged records. Both example
guests below pack three pointer args plus a scalar count for the
`vector_add` kernel from [Section 3.4](#34-worked-example-vector_addptx).
The host-side helper
[`encode_argv`](../crates/tensor-wasm-wasi-gpu/src/kernel_args.rs)
is the canonical byte-layout reference.

### 6.2 Worked example: Rust guest

`wasm32-wasip1` guest. Build with `cargo build --target wasm32-wasip1 --release`.

```rust
// SPDX-License-Identifier: Apache-2.0
const PTX: &[u8] = include_bytes!("vector_add.ptx");
const ENTRY: &[u8] = b"vector_add";
const N: u32 = 1024;
const TAG_U32: u8 = 0x05;
const TAG_PTR: u8 = 0x07;

#[link(wasm_import_module = "wasi:cuda/host@0.2.0")]
extern "C" {
    fn wasi_cuda_load_ptx(
        ptx_ptr: i32, ptx_len: i32,
        entry_ptr: i32, entry_len: i32,
    ) -> i64;
    fn wasi_cuda_launch(
        kernel_id: i64,
        gx: i32, gy: i32, gz: i32, bx: i32, by: i32, bz: i32,
        shared_mem: i32, args_ptr: i32, args_len: i32,
    ) -> i32;
    fn wasi_cuda_sync() -> i32;
}

fn push_ptr(buf: &mut Vec<u8>, offset: u32, byte_len: u32) {
    buf.push(TAG_PTR);
    buf.extend_from_slice(&offset.to_le_bytes());
    buf.extend_from_slice(&byte_len.to_le_bytes());
}
fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.push(TAG_U32);
    buf.extend_from_slice(&v.to_le_bytes());
}

#[no_mangle]
pub extern "C" fn _start() {
    // Allocate input/output buffers in linear memory.
    let bytes_per_buf = (N as usize) * std::mem::size_of::<f32>();
    let mut a = vec![0.0f32; N as usize];
    let mut b = vec![0.0f32; N as usize];
    let mut c = vec![0.0f32; N as usize];
    for i in 0..(N as usize) { a[i] = i as f32; b[i] = (2*i) as f32; }

    // Register the PTX. Entry must match the .entry symbol exactly.
    let kid = unsafe {
        wasi_cuda_load_ptx(
            PTX.as_ptr() as i32, PTX.len() as i32,
            ENTRY.as_ptr() as i32, ENTRY.len() as i32,
        )
    };
    assert!(kid >= 0, "load_ptx failed: {kid}");

    // Pack the argv buffer for (a, b, c, n).
    let mut argv = Vec::with_capacity(9*3 + 5);
    push_ptr(&mut argv, a.as_ptr() as u32, bytes_per_buf as u32);
    push_ptr(&mut argv, b.as_ptr() as u32, bytes_per_buf as u32);
    push_ptr(&mut argv, c.as_mut_ptr() as u32, bytes_per_buf as u32);
    push_u32(&mut argv, N);

    // Block dim must be a warp multiple (32). 256 threads/block is the
    // modal sweet spot; grid = ceil(N / 256).
    let block_x: i32 = 256;
    let grid_x: i32 = ((N as i32) + block_x - 1) / block_x;

    let rc = unsafe {
        wasi_cuda_launch(kid, grid_x, 1, 1, block_x, 1, 1,
            0, argv.as_ptr() as i32, argv.len() as i32)
    };
    assert_eq!(rc, 0, "launch failed: {rc}");
    assert_eq!(unsafe { wasi_cuda_sync() }, 0);

    // c[i] == a[i] + b[i] under CUDA UVM; under no-CUDA, launch
    // returned NotAvailable (-1) and the assert above tripped.
    assert!((c[7] - 21.0).abs() < 1e-5);
}
```

Two non-obvious things: `a.as_ptr() as u32` assumes 32-bit Wasm
linear memory (`wasm32-wasip1` default) — never `as usize`, on a
future 64-bit memory build it would silently widen. The argv buffer
has **no length prefix** or padding; the host walks one tag at a
time until `args_len` bytes are consumed.

### 6.3 Worked example: C guest

`wasi-sdk` C guest. Build with `clang --target=wasm32-wasi -O2 guest.c -o guest.wasm`.

```c
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include <string.h>
#include <assert.h>

#define IMPORT __attribute__((import_module("wasi:cuda/host@0.2.0")))
IMPORT int64_t wasi_cuda_load_ptx(int32_t pp, int32_t pl, int32_t ep, int32_t el);
IMPORT int32_t wasi_cuda_launch(int64_t kid,
    int32_t gx, int32_t gy, int32_t gz, int32_t bx, int32_t by, int32_t bz,
    int32_t shmem, int32_t ap, int32_t al);
IMPORT int32_t wasi_cuda_sync(void);

#define TAG_U32 0x05
#define TAG_PTR 0x07
#define N 1024

extern const uint8_t vector_add_ptx[];
extern const uint32_t vector_add_ptx_len;

static float a[N], b[N], c[N];
static uint8_t argv_buf[32]; /* 3 * 9 (ptr) + 5 (u32) */

static size_t push_ptr(uint8_t *p, uint32_t off, uint32_t len) {
    *p++ = TAG_PTR; memcpy(p, &off, 4); p += 4; memcpy(p, &len, 4); return 9;
}
static size_t push_u32(uint8_t *p, uint32_t v) {
    *p++ = TAG_U32; memcpy(p, &v, 4); return 5;
}

void _start(void) {
    for (uint32_t i = 0; i < N; ++i) { a[i] = (float)i; b[i] = (float)(2*i); }

    int64_t kid = wasi_cuda_load_ptx(
        (int32_t)(uintptr_t)vector_add_ptx, (int32_t)vector_add_ptx_len,
        (int32_t)(uintptr_t)"vector_add", 10);
    assert(kid >= 0);

    uint8_t *p = argv_buf;
    p += push_ptr(p, (uint32_t)(uintptr_t)a, sizeof(a));
    p += push_ptr(p, (uint32_t)(uintptr_t)b, sizeof(b));
    p += push_ptr(p, (uint32_t)(uintptr_t)c, sizeof(c));
    p += push_u32(p, N);

    int32_t rc = wasi_cuda_launch(kid,
        (N + 255) / 256, 1, 1, 256, 1, 1, 0,
        (int32_t)(uintptr_t)argv_buf, (int32_t)(p - argv_buf));
    assert(rc == 0);
    assert(wasi_cuda_sync() == 0);
    /* c[7] == 21.0f under CUDA. */
}
```

Identical packing: same tag bytes, same little-endian value bytes,
no padding. Any guest language that can write a `&[u8]` and call an
`extern` can drive the host.

---

## 7. Bounds checking and pointer args

Every pointer arg is bounds-checked before any CUDA call. Inside
[`parse_argv`](../crates/tensor-wasm-wasi-gpu/src/kernel_args.rs),
for each `Ptr` record the host computes `end = guest_offset + len`
and rejects with `InvalidPointer` (`-2`) if the add overflows `u32`
or `end > caller_memory_length`. Once the check passes, the kernel
dereferences the resolved host pointer — under CUDA Unified Memory it
doubles as a device address.

**Cost**: ~50 ns per pointer arg (overflow + compare + slice index).
Negligible against the ~5 µs single-block launch overhead, but the
128-arg cap from [Section 3.3](#33-calling-convention) caps the
worst-case overhead at a few microseconds.

Invariants:

- A zero-length pointer at the exact end of memory (`offset =
  memory_len`, `len = 0`) is allowed — safe sentinel for "no buffer
  to pass" without conditional argv packing.
- A non-zero `len` with `offset == memory_len` is rejected.
- The pointer's region may overlap the argv buffer; the host treats
  them as independent reads.

**HAZARD**: the resolved host pointer is captured into
`cuLaunchKernel` synchronously, before the wasmtime fiber suspends.
A subsequent `memory.grow` inside the same guest could relocate
linear memory, invalidating host pointers stored elsewhere. The
launch path is correct because CUDA has already consumed the pointer;
do not hold a `LoweredArg::Ptr` across a guest-callable boundary
([`host.rs` HAZARD note](../crates/tensor-wasm-wasi-gpu/src/host.rs)).

---

## 8. Performance tips

The launch path is thin (parse argv → bounds-check → `cuLaunchKernel`
→ `spawn_blocking(stream.synchronize)` → return), so the usual
CUDA-host performance disciplines apply.

**Match the wave size.** Launch dims so `block_x * block_y * block_z`
is a multiple of 32 (warp size). A 33-thread block spins up two warps
but wastes one full lane group. 256 threads/block is the modal sweet
spot.

**Avoid divergent branches in tight loops.** A divergent `if`
serializes lanes. The `vector_add` PTX from
[Section 3.4](#34-worked-example-vector_addptx) gets this right with
one `setp.ge.s32` + early `bra` — all 32 warp lanes branch together.

**Use shared memory for matmul tiles.** Declare a `.shared` tile and
pass `shared_mem` (bytes) to `wasi_cuda_launch`; the host forwards
unchanged to `cuLaunchKernel`. 16x16 f32 (1 KiB) or 32x32 f32 (4 KiB)
per block are common; hardware cap is 48 KiB/block on SM_70+ (96 KiB
opt-in on SM_80+).

**Profile with nsys**:

```bash
nsys profile --stats=true --trace=cuda,osrt ./tensor-wasm run my_module.wasm
```

`--stats=true` summarizes per-kernel duration, launch overhead, and
sync wait. Compare against
[`bench-results/baseline.json`](../bench-results/baseline.json) for
regressions. For TensorWasm's own counters, use the W1.5
`tensor-wasm-cli observe` subcommand ([`docs/CLI.md`](CLI.md)).

**Sizing.** Each in-flight launch holds a back-pressure permit until
`stream.synchronize` returns; a full queue stalls new launches. The
formulas in
[`docs/CAPACITY-PLANNING.md` §4](CAPACITY-PLANNING.md#4-sizing-formulas)
relate launches/sec to permit count to GPU memory budget.

---

## 9. Common pitfalls

**`KernelArgsUnsupported` (`-10`)** — Reserved for sanity-cap busts
only: argv buffer above 4 KiB or more than 128 records. The v0.1.0
contract (returned for **any** non-empty argv) is gone — W1.1 lowers
typed argv directly into `cuLaunchKernel`
([`host.rs`](../crates/tensor-wasm-wasi-gpu/src/host.rs)).

**`InvalidPointer` (`-2`)** — A pointer arg's `[guest_offset,
guest_offset + len)` window is outside the guest's linear memory.
Common causes: (1) the length in the argv record is **bytes**, not
elements — a `Vec<f32>` of length 1024 is 4096 bytes; (2) growing a
`Vec` after packing argv, which reallocs and invalidates the recorded
offset. Pin buffers (`Vec::with_capacity` + no more pushes, or a
fixed-size array) before packing.

**`unsupported gpu architecture 'compute_80'` on SM_75** — PTX
targets an architecture the GPU doesn't support. On RTX 2060 (SM_75),
`.target sm_80`+ fails at JIT time inside `ptxas`; the host surfaces
this as `MalformedPtx` (`-4`). Fix: re-emit with `-arch=compute_70`
or `-arch=sm_75`. See
[`CUDA-SETUP.md` §SM-level compatibility](CUDA-SETUP.md#sm-level-compatibility-matrix)
for the wmma-on-Turing caveat.

**Forgetting to call `sync`** — In v0.2 `wasi_cuda_launch` is
synchronous from the guest's POV (returns 0 after `stream.synchronize`
completes). But future extensions (multi-launch batching, async launch
— see [`PATH-TO-V1.md` §v0.5](PATH-TO-V1.md#v050-beta--external-validation))
will require explicit `wasi_cuda_sync`. Forward-compatible guests call
`sync` defensively before reading kernel-written buffers.

**Entry name mismatch** — The `entry` string passed to
`wasi_cuda_load_ptx` must match the `.visible .entry foo(...)` symbol
byte-for-byte (no nul terminator, no leading underscore, no namespace
mangling). Passing `"_vector_add"` when the PTX declares `vector_add`
returns `MalformedPtx`.

---

## 10. Testing

### 10.1 Host-side mock dispatch (no GPU required)

W1.1 added `WasiCudaContext::last_lowered_args` so argv lowering can
be tested end-to-end without a GPU. On a no-CUDA host,
`wasi_cuda_launch` returns `NotAvailable` (`-1`) after parsing the
argv, and the parsed `Vec<LoweredArg>` is recorded for inspection.

Representative test
([`tests/kernel_args_e2e.rs`](../crates/tensor-wasm-wasi-gpu/tests/kernel_args_e2e.rs)):

```rust
let expected = vec![
    LoweredArg::I32(-13),
    LoweredArg::U32(0x1234_5678),
    LoweredArg::Ptr { host_ptr: std::ptr::null(), len: 64, guest_offset: 256 },
];
let argv = encode_argv(&expected);
let wat = build_launch_wat(&argv, /*offset=*/1024, kid);
// Build a wasmtime Store with WasiCudaContext, instantiate, call.
let rc = launch_fn.call_async(&mut store, ()).await.unwrap();
#[cfg(not(feature = "cuda"))]
assert_eq!(rc, AbiError::NotAvailable.code());
let recorded = store.data().wasi_cuda().last_lowered_args();
assert_eq!(recorded.len(), 3); // host_ptr re-resolves against live memory.
```

Use this pattern to verify a guest's argv packer, pin the contract on
a kernel before shipping, and regression-test parser / bounds-checker
changes. It does not exercise the kernel itself.

### 10.2 Real-GPU integration tests

Tests needing a real CUDA device are marked
`#[ignore = "requires CUDA hardware"]`:

```rust
#[tokio::test]
#[ignore = "requires CUDA hardware"]
async fn scalar_argv_real_cuda_launch() {
    // Same shape as the mock-dispatch test, but launch returns 0 and
    // the test reads back the output buffer.
}
```

The S22 self-hosted CUDA runner runs these via `cargo test --features
tensor-wasm-wasi-gpu/cuda -- --include-ignored`. On a no-CUDA dev
box the bodies still compile, catching ABI drift early. Guideline:
keep the kernel-correctness assertion behind
`#[cfg(feature = "cuda")]`; keep the argv round-trip assertion
unconditional so it runs on every PR.

---

## 11. Auto-offload coverage

Pointer to [`docs/AUTO-OFFLOAD.md`](AUTO-OFFLOAD.md). The detector
recognizes element-wise f32 vector arithmetic (vector_add), fused
multiply-add (GEMV / dot-product), and tiled f16 → f32 matmul
(tensor-core path on SM_80+); a 3x3 conv2d stencil blueprint is
queued but not yet wired. The pipeline produces validated PTX and
caches it; the runtime swap that would replace the Cranelift body
is gated on a per-block hook
([`docs/WASMTIME-FORK.md`](WASMTIME-FORK.md)) — until that lands,
auto-offload pre-emits but does not replace. Blocks with control
flow, dynamic trip counts, or cross-block dependencies are skipped
([`AUTO-OFFLOAD.md` §Known limitations](AUTO-OFFLOAD.md#known-limitations)).

**Write PTX** when the kernel needs control flow, integer SIMD, f64,
custom shared-memory tiling, or intrinsics above the blueprint set.
**Use auto-offload** for element-wise loops or textbook matmul when
you'd rather write idiomatic Rust + `v128` intrinsics. **Use both**
when the hot inner loop is a blueprint and surrounding logic is custom.

**v0.4 expansion via Pliron.** The blueprint set above is the
v0.3.x coverage ceiling because the detector emits PTX from three
hand-written templates. The v0.4 Pliron lowering pass tracked in
[`crates/tensor-wasm-jit/src/pliron_dialect.rs`](../crates/tensor-wasm-jit/src/pliron_dialect.rs)
(scaffold today; real lowering per RFC 0001 step 4) replaces the
template-emitter with a `Cranelift IR → Pliron dialect-mir → ... →
PTX` pipeline that shares cuda-oxide's backend. Once it lands,
coverage expands from the three blueprints to arbitrary pure-compute
loops the detector can prove safe — see
[RFC 0001 "Pliron lever and the auto-offload pipeline"](../rfcs/0001-cuda-oxide-integration.md#pliron-lever-and-the-auto-offload-pipeline).
That is what shrinks Path C's audience over time: many users will
not write CUDA kernels at all, the lowering pass auto-offloads them.

---

## 12. Related

- [`docs/CUDA-SETUP.md`](CUDA-SETUP.md) — toolchain, driver matrix,
  SM compatibility, troubleshooting.
- [`docs/AUTO-OFFLOAD.md`](AUTO-OFFLOAD.md) — pattern list, detector
  verdicts, deopt behaviour.
- [`docs/RISKS.md`](RISKS.md) — v0.1.0 known limitations including
  the superseded kernel-args contract.
- [`docs/CAPACITY-PLANNING.md`](CAPACITY-PLANNING.md) — sizing
  formulas for launch rate, permits, GPU memory per tenant.
- [`docs/PERFORMANCE.md`](PERFORMANCE.md) — measured dispatch medians.
- [`docs/BENCHMARKING.md`](BENCHMARKING.md) — external-comparison methodology.
- [`docs/PATH-TO-V1.md`](PATH-TO-V1.md) — v0.2 / v0.3 milestone exit criteria.
- [`wit/wasi-cuda.wit`](../wit/wasi-cuda.wit) — authoritative
  Component-Model interface. The inline `launch` docstring is stale
  (describes v0.1.0); the shipped behavior is what `host.rs` and
  `kernel_args.rs` implement.
- [`crates/tensor-wasm-wasi-gpu/src/kernel_args.rs`](../crates/tensor-wasm-wasi-gpu/src/kernel_args.rs) — wire format, sanity caps, encoder / decoder.
- [`crates/tensor-wasm-wasi-gpu/src/host.rs`](../crates/tensor-wasm-wasi-gpu/src/host.rs) — launch impl: validation, parsing, `cuLaunchKernel`, sync.
- [`crates/tensor-wasm-wasi-gpu/tests/kernel_args_e2e.rs`](../crates/tensor-wasm-wasi-gpu/tests/kernel_args_e2e.rs) — e2e tests for scalar / pointer argv and the `#[ignore]` pattern.
- [`rfcs/0001-cuda-oxide-integration.md`](../rfcs/0001-cuda-oxide-integration.md) — Path C provenance: the v0.5 cust-successor RFC that motivates the `cuda-oxide-backend` feature and the v0.4 toolchain bump.
- [`crates/tensor-wasm-mem/README-cuda-oxide.md`](../crates/tensor-wasm-mem/README-cuda-oxide.md) — user-facing reference for the `cuda-oxide-backend` feature flag, current stub behaviour, and the `RUSTUP_TOOLCHAIN` override.
- [`crates/tensor-wasm-jit/src/pliron_dialect.rs`](../crates/tensor-wasm-jit/src/pliron_dialect.rs) — module-doc scaffold for the v0.4 Wasm-to-`dialect-mir` lowering pass that expands auto-offload coverage beyond the three blueprints.

---

_Updated for tensor-wasm v0.3.1 (PATH-TO-V1 W4.5 + RFC 0001 step 5).
Worked examples are pinned to the W1.1 wire format; re-render if the
tag table changes. Path C content is forward-looking against
cuda-oxide v0.1.0 alpha and will need a refresh once cuda-oxide v0.2
ships — see the inline `TODO(v0.4)` markers in Section 5._
