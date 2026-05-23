# Auto-Offload — Supported Patterns and Known Limitations

_Status: introduced in S14. The Cranelift-fork question is settled in
`docs/WASMTIME-FORK.md`; this document is the user-facing companion that
explains what code patterns Bali will (and won't) auto-offload to the GPU._

## Triggering auto-offload

Auto-offload is **off by default**. Enable it with the cargo feature flag:

```
cargo build --workspace --features bali-jit/auto-offload
```

or via the HTTP API by submitting a function with `"jit_auto_offload": true`
(see `nova-api/API.md`, S17 — note: the field name preserves API stability
across the Nova→Bali rename).

When enabled, candidate basic blocks are inspected at *pipeline-driver* time
by [`bali_jit::detector::classify`](../crates/bali-jit/src/detector.rs) —
that is, by programs that call the `bali-jit` pipeline directly, not by
Wasmtime at dispatch time. Blocks marked [`DetectorVerdict::Offload`] are
lowered to PTX via the pipeline described in this document and the result is
inserted into the kernel cache. The rest run on the standard Wasmtime CPU
path. In v0.1.0, an `Offload` verdict produces a cached PTX entry but does
**not** yet replace the Cranelift-compiled function body at runtime — see
the next section for why.

## Integration status in v0.1.0

Bali v0.1.0 ships the auto-offload **pipeline as a library** — the detector,
clif-lower, PTX emitter, kernel cache, and deopt guard are all production-
quality and exhaustively tested. What v0.1.0 does NOT yet ship is the
runtime swap: at execution time, Wasmtime continues to dispatch the
Cranelift-compiled function body. Replacing that body with a dispatch stub
into the cached PTX kernel requires per-block compile hooks in
`wasmtime-cranelift` that upstream does not currently expose — see
[`WASMTIME-FORK.md`](./WASMTIME-FORK.md) for the rationale of not forking.

Concretely:

- The pipeline runs ahead-of-time inside `bali-jit` and produces validated
  PTX kernels. Programs can drive it directly via the public API
  (`bali_jit::detector::classify` → `clif_lower::lower_block` →
  `ptx_emit::emit` → `cache::KernelCache::put`).
- The runtime-swap hook is intentionally deferred. When Wasmtime upstreams
  the per-block hook (tracked in `WASMTIME-FORK.md`), or when we explicitly
  decide to fork, the executor can begin consulting the cache at dispatch
  time. The pipeline output is then a drop-in.
- Until then, the `bali-jit/auto-offload` feature flag toggles the *pipeline*
  on/off (controlling whether kernels are pre-emitted at all) but does not
  change runtime dispatch.

## Supported patterns

### 1. Element-wise vector arithmetic on f32

Code that maps a single SIMD op across a statically-bounded loop is the
sweet spot:

```rust
// In Wasm (wasm32-wasi), after rustc + wasm-opt:
for i in 0..N {  // N must be a constant
    c[i] = a[i] + b[i];
}
```

The detector flags this as `Offload` when N ≥ 64 and the body is ≥ 80% `v128.*`.

### 2. Fused multiply-add (vector dot product, GEMV)

```rust
for i in 0..N {
    sum[i] += a[i] * b[i];  // lowers to v128.fma
}
```

Lowered to `fma.rn.f32` PTX instructions for sm_80+.

### 3. Tiled matrix multiply (f16 → f32 accumulator)

A naive `(M, K) × (K, N) → (M, N)` GEMM written with explicit blocks is
lowered to `wmma.mma.sync.aligned.row.col.m16n16k16.f32.f16.f16.f32`
(tensor-core path). M, N, K must each be a multiple of 16.

## Known limitations

### 1. Control flow inside the candidate block

If a basic block contains `Op::Branch` or `Op::Call`, the lowering returns
`LowerError::UnsupportedOp`. The block falls back to the CPU path. Loops
with early exits will not auto-offload.

### 2. f64, integer SIMD, mixed precision

The PTX emitter currently emits only f32 (and f16 for tensor-core MatMul).
v128 ops over i32/i64 lanes are detected but trigger `UnsupportedOp` in the
lowering pass. Tracked as a follow-up.

### 3. Dynamic-trip-count loops

`loop_trip_count = None` (a Wasm `loop` with a runtime-determined count) is
never offloaded — the launch overhead beats any wins. This is intentional.

### 4. Cross-block dependencies

Each candidate is one basic block. Patterns that span multiple blocks
(prologue + loop + epilogue) are split; only the loop body is considered.
Most useful kernels are already inside a single basic block after
Cranelift's pre-passes, so in practice this is rarely a problem.

## Deopt-on-error behaviour

[`DeoptGuard`](../crates/bali-jit/src/deopt.rs) tracks per-fingerprint deopt
state. A kernel that has been deopted in this process is *not* retried on
the GPU until the cache is invalidated. Numerical correctness is checked by
the executor on the first invocation of every newly-emitted kernel
(comparing against the Cranelift reference within a configurable tolerance —
default 1e-4 absolute for f32).

## Metrics

The following Prometheus counters surface auto-offload behaviour:

- `bali_offload_success_total` — kernels that ran on GPU and passed
  correctness checks.
- `bali_offload_fallback_total` — kernels that deopted, plus their reason
  label (`cuda_error`, `numerical_divergence`, `assembly_failed`).

Compute the success rate as
`bali_offload_success_total / (bali_offload_success_total + bali_offload_fallback_total)`.

---

_Status: S14 of the plan. Expand as we add patterns in S19+._
