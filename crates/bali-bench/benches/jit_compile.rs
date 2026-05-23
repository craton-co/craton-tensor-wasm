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

use bali_jit::ir::{BaliKernelBlueprint, BaliOp, GridHint};
use bali_jit::ptx_emit::emit;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn vector_add_blueprint(lanes: u32) -> BaliKernelBlueprint {
    BaliKernelBlueprint::new("vector_add")
        .push(BaliOp::LoadUnified { lanes })
        .push(BaliOp::LoadUnified { lanes })
        .push(BaliOp::VecAdd { lanes })
        .push(BaliOp::StoreUnified { lanes })
        .with_grid(GridHint {
            total_threads: 1024,
            preferred_block_size: 128,
        })
}

fn matmul_blueprint() -> BaliKernelBlueprint {
    BaliKernelBlueprint::new("matmul_16x16x16")
        .push(BaliOp::LoadUnified { lanes: 4 })
        .push(BaliOp::LoadUnified { lanes: 4 })
        .push(BaliOp::MatMul {
            m: 16,
            n: 16,
            k: 16,
        })
        .push(BaliOp::StoreUnified { lanes: 4 })
        .with_grid(GridHint {
            total_threads: 4096,
            preferred_block_size: 128,
        })
        .with_shared_mem(8 * 1024)
}

fn conv2d_blueprint() -> BaliKernelBlueprint {
    // Conv2D modelled as a stack of FMA+barrier ops representing a tile.
    let mut bp = BaliKernelBlueprint::new("conv2d_3x3")
        .with_grid(GridHint {
            total_threads: 16 * 1024,
            preferred_block_size: 256,
        })
        .with_shared_mem(16 * 1024);
    for _ in 0..9 {
        bp = bp.push(BaliOp::VecFma { lanes: 4 });
    }
    bp.push(BaliOp::Barrier)
        .push(BaliOp::StoreUnified { lanes: 4 })
}

fn bench_emit_text(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_compile/emit_text");
    group.measurement_time(Duration::from_secs(3));

    let blueprints: &[(&str, BaliKernelBlueprint)] = &[
        ("vector_add[4]", vector_add_blueprint(4)),
        ("vector_add[16]", vector_add_blueprint(16)),
        ("matmul[16x16x16]", matmul_blueprint()),
        ("conv2d[3x3]", conv2d_blueprint()),
    ];

    for (name, bp) in blueprints {
        group.bench_with_input(BenchmarkId::from_parameter(name), bp, |b, bp| {
            b.iter(|| {
                let ptx = emit(bp);
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
    use bali_jit::cache::{CacheKey, CachedKernel, CompiledHandle, KernelCache};
    use std::sync::Arc;

    let mut group = c.benchmark_group("jit_compile/cache");
    group.measurement_time(Duration::from_secs(3));

    let bp = matmul_blueprint();
    let key = CacheKey {
        blueprint: bp.fingerprint(),
        sm_version: 80,
    };

    // Cold: pretend the cache is empty — pay the emit cost AND insert.
    group.bench_function("cold_miss_then_insert", |b| {
        b.iter(|| {
            let cache = KernelCache::with_capacity(16);
            let ptx = emit(&bp);
            let entry = CachedKernel {
                fingerprint: bp.fingerprint(),
                ptx: Arc::new(ptx),
                compiled: CompiledHandle::default(),
            };
            cache.put(key, entry);
            criterion::black_box(cache.get(&key));
        });
    });

    // Warm: pre-populate; iter measures only the lookup.
    let warm_cache = KernelCache::with_capacity(16);
    let warm_entry = CachedKernel {
        fingerprint: bp.fingerprint(),
        ptx: Arc::new(emit(&bp)),
        compiled: CompiledHandle::default(),
    };
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
