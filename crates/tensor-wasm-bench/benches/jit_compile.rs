// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! S19 JIT-compile bench: PTX emit latency for representative kernels.
//!
//! On a CUDA host the full pipeline also runs `ptxas` to produce SASS;
//! that step dominates wall-clock (10-50 ms per kernel). The non-CUDA
//! path measured here is the text-emit cost only — which is what S13's
//! kernel cache aims to amortise. A 10x regression on emit latency is
//! still worth catching.
//!
//! The `jit_compile/cache` group directly exercises S13's "cache hit
//! latency <1ms for pre-warmed MatMul kernel" done-when: `warm_hit`
//! measures only the cache lookup on a pre-populated entry (expected
//! sub-microsecond), while `cold_miss_then_insert` pays the full
//! emit+insert cost as a baseline for the hit/miss ratio.

use std::time::Duration;

use tensor_wasm_jit::ir::{TensorWasmKernelBlueprint, TensorWasmOp, GridHint};
use tensor_wasm_jit::ptx_emit::emit;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

fn vector_add_blueprint(lanes: u32) -> TensorWasmKernelBlueprint {
    TensorWasmKernelBlueprint::new("vector_add")
        .push(TensorWasmOp::LoadUnified { lanes })
        .push(TensorWasmOp::LoadUnified { lanes })
        .push(TensorWasmOp::VecAdd { lanes })
        .push(TensorWasmOp::StoreUnified { lanes })
        .with_grid(GridHint {
            total_threads: 1024,
            preferred_block_size: 128,
        })
}

fn matmul_blueprint() -> TensorWasmKernelBlueprint {
    // Approximates a 16x16 inner-loop matmul body as a stack of FMAs
    // without invoking the wmma `MatMul` op — PTX emission for that op
    // is deferred to v0.4 and the emitter explicitly refuses it (see
    // `tensor_wasm_jit::ptx_emit::EmitError::NotYetImplemented`). The
    // shape (loads → many FMAs → store) keeps the cache-key fingerprint
    // and the emit cost in the same order of magnitude as a real matmul
    // kernel so the bench remains representative for S13's cache hit/
    // miss measurements.
    let mut bp = TensorWasmKernelBlueprint::new("matmul_16x16x16")
        .push(TensorWasmOp::LoadUnified { lanes: 4 })
        .push(TensorWasmOp::LoadUnified { lanes: 4 });
    for _ in 0..16 {
        bp = bp.push(TensorWasmOp::VecFma { lanes: 4 });
    }
    bp.push(TensorWasmOp::StoreUnified { lanes: 4 })
        .with_grid(GridHint {
            total_threads: 4096,
            preferred_block_size: 128,
        })
        .with_shared_mem(8 * 1024)
}

fn conv2d_blueprint() -> TensorWasmKernelBlueprint {
    // Conv2D modelled as a stack of FMA+barrier ops representing a tile.
    let mut bp = TensorWasmKernelBlueprint::new("conv2d_3x3")
        .with_grid(GridHint {
            total_threads: 16 * 1024,
            preferred_block_size: 256,
        })
        .with_shared_mem(16 * 1024);
    for _ in 0..9 {
        bp = bp.push(TensorWasmOp::VecFma { lanes: 4 });
    }
    bp.push(TensorWasmOp::Barrier)
        .push(TensorWasmOp::StoreUnified { lanes: 4 })
}

fn bench_emit_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_compile/emit_text");
    group.measurement_time(Duration::from_secs(3));

    let blueprints: &[(&str, TensorWasmKernelBlueprint)] = &[
        ("vector_add[4]", vector_add_blueprint(4)),
        ("vector_add[16]", vector_add_blueprint(16)),
        ("matmul[16x16x16]", matmul_blueprint()),
        ("conv2d[3x3]", conv2d_blueprint()),
    ];

    for (name, bp) in blueprints {
        group.bench_with_input(BenchmarkId::from_parameter(name), bp, |b, bp| {
            b.iter(|| {
                let ptx = emit(bp).expect("emit");
                criterion::black_box(ptx);
            });
        });
    }
    group.finish();
}

fn bench_blueprint_fingerprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_compile/fingerprint");
    group.measurement_time(Duration::from_secs(3));
    let bp = matmul_blueprint();
    group.bench_function("matmul_16x16x16", |b| {
        b.iter(|| {
            let fp = bp.fingerprint();
            criterion::black_box(fp);
        });
    });
    group.finish();
}

fn bench_cache_hit_vs_miss(c: &mut Criterion) {
    use tensor_wasm_core::types::TenantId;
    use tensor_wasm_jit::cache::{CacheKey, CachedKernel, CompiledHandle, KernelCache};
    use std::sync::Arc;

    let mut group = c.benchmark_group("jit_compile/cache");
    group.measurement_time(Duration::from_secs(3));

    let bp = matmul_blueprint();
    // Bench-only: no real tenant context here, so use the placeholder
    // `TenantId(0)`. Real dispatch sites must pass the calling tenant —
    // see `CacheKey::for_tenant` docs.
    let key = CacheKey::for_tenant(TenantId(0), bp.fingerprint(), 80);

    // Cold: pretend the cache is empty — pay the emit cost AND insert.
    // Cache construction is hoisted via `iter_batched_ref` so each sample
    // measures emit+put+get only, not the one-time DashMap allocation.
    group.bench_function("cold_miss_then_insert", |b| {
        b.iter_batched_ref(
            KernelCache::new,
            |cache| {
                let ptx = emit(&bp).expect("emit");
                let entry = CachedKernel::new(
                    bp.fingerprint(),
                    Arc::new(ptx),
                    CompiledHandle::default(),
                );
                cache.put(key, entry);
                criterion::black_box(cache.get(&key));
            },
            BatchSize::SmallInput,
        );
    });

    // Warm: pre-populate; iter measures only the lookup.
    let warm_cache = KernelCache::with_capacity(16);
    let warm_entry = CachedKernel::new(
        bp.fingerprint(),
        Arc::new(emit(&bp).expect("emit")),
        CompiledHandle::default(),
    );
    warm_cache.put(key, warm_entry);
    group.bench_function("warm_hit", |b| {
        b.iter(|| {
            criterion::black_box(warm_cache.get(&key));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_emit_text,
    bench_blueprint_fingerprint,
    bench_cache_hit_vs_miss
);
criterion_main!(benches);
