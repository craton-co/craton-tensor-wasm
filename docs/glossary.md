<!-- SPDX-License-Identifier: Apache-2.0 -->

# Craton TensorWasm — Glossary

Short definitions of recurring terms in TensorWasm's documentation,
RFCs, and crate-internal comments. Each entry is a two-to-three-line
paragraph; deeper material lives in the linked specs and design docs.

## UVM (Unified Virtual Memory)

CUDA's managed-memory facility, accessed via `cudaMallocManaged`, that
lets host code and device kernels share a single address space. The
driver page-migrates the backing pages between host RAM and GPU memory
on demand. TensorWasm's `UnifiedBuffer` is allocated through UVM so a
Wasm guest's linear memory is directly addressable by a launched
kernel.

## MPS (Multi-Process Service)

NVIDIA's CUDA context-sharing daemon (`nvidia-cuda-mps-control`) that
multiplexes several client processes onto one device-side context, so
co-located tenants share SMs without serialising at the driver. See
[`docs/MPS-SETUP.md`](MPS-SETUP.md) for the startup, capabilities, and
the runtime probe TensorWasm uses to decide between MPS-shared and
per-tenant CUDA contexts.

## MIG (Multi-Instance GPU)

Hardware-level GPU partitioning on A100, H100, and later datacenter
parts. The GPU is split into up to seven isolated instances, each with
its own SMs, L2 slice, and memory partition. MIG is the long-term
mitigation for the L2 cache timing side channel called out in
[`SECURITY.md`](../SECURITY.md#known-gaps); it provides hardware
isolation that MPS alone does not.

## PTX

NVIDIA's intermediate-representation assembly for CUDA kernels —
human-readable, architecture-agnostic, and consumed by `ptxas` to
produce the SASS that runs on a specific compute capability. Wasm
guests upload PTX via `wasi_cuda_load_ptx`; TensorWasm validates it
with `ptxas --gpu-name <arch>` before caching the compiled module.

## WMMA

Warp Matrix Multiply-Accumulate — the tensor-core instruction class
(`mma.sync` family in PTX) that performs a small fixed-shape
matrix-multiply-and-accumulate in one warp-wide operation. WMMA is the
path through which the auto-offload pipeline lights up the tensor
cores on Volta-and-newer hardware.

## BLAKE3 fingerprint

A 32-byte BLAKE3 digest used as the cache key in
`tensor-wasm-exec` for compiled Wasm modules. The full digest is
stored (post-H11 fix); earlier code truncated to 16 bytes which the
H11 hardening item ruled out as too narrow a key for an
adversarially-chosen-collision threat model.

## Deopt guard

A JIT-mechanism check inserted around an auto-offloaded kernel that
verifies the preconditions used to admit the offload still hold at
each invocation. On a miss — alignment, shape, dtype, or tenant-quota
violation — the guard falls back to CPU execution rather than
launching the kernel.

## Dispatch future

The F3-backend abstraction in `tensor-wasm-exec` over kernel-launch
completion. The same `DispatchFuture` type covers the busy-poll,
sleep-loop, event-driven, and CUDA-async backends, so the call site
does not need to know which polling strategy the operator selected.

## Back-pressure semaphore

The bounded counting semaphore that caps the number of in-flight
asynchronous kernel dispatches per tenant context. The bound prevents
the CUDA driver from saturating on a single tenant's burst and is the
mechanism through which `tensor-wasm-tenant` quota counters take
effect on the async path.

## Tenant capability

A newtype guard token (added by the H17 hardening item) required to
mutate any `TenantContext` quota counter. The token is unforgeable
within the crate's public surface, so any code path that updates a
counter has had to acquire a `TenantCapability` first — the type
system enforces the invariant.

## Snapshot wire format

The v2 binary format for captured-instance snapshots: zstd-compressed
bincode 2.x legacy encoding, wrapped in a header carrying a magic
constant, a format version byte, and a CRC32 over the payload. The
authoritative byte-layout spec is
[`crates/tensor-wasm-snapshot/FORMAT.md`](../crates/tensor-wasm-snapshot/FORMAT.md);
the cross-version compatibility promise is in
[`SNAPSHOT-COMPATIBILITY.md`](SNAPSHOT-COMPATIBILITY.md).

## Auto-offload

The runtime decision to JIT a Wasm function down to a GPU kernel,
made by the auto-offload detector based on a vector-op-density
heuristic on the function's body. See
[`AUTO-OFFLOAD.md`](AUTO-OFFLOAD.md) for the precise admission rules
and the patterns the detector recognises versus rejects.

## Cold start

The interval from the first request for a function instance to that
instance being fully warm and serving steady-state latency. The five
additive components — image fetch, Wasm compile, snapshot restore,
CUDA-context warmup, first-request overhead — are decomposed in
[`COLD-START.md`](COLD-START.md); snapshot capture/restore latency is
the dominant lever on this number.

## WASI-cuda

The host-interface package, currently versioned as
`wasi:cuda/host@0.2.0`, that gives Wasm guests typed access to PTX
load and kernel-launch host functions. The WIT lives under `wit/` and
is the surface against which both the explicit kernel-dispatch path
and the auto-offload fast path are bound.

## pliron / cuda-oxide

The two Rust scaffold backends prototyped in RFC 0001 (next-generation
lowering) as candidates for the post-v1 kernel-IR pipeline. Neither is
wired into the v0.3.x release — they are scaffold-only — and the RFC
records the trade-offs that will inform the eventual lowering-backend
choice.
