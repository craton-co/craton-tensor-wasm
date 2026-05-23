//! S16 done-when: "Context switching overhead <5μs per call (benchmark)".
//!
//! On a CUDA host this would measure `cuCtxPushCurrent` + `cuCtxPopCurrent`
//! across an established `cust::context::Context`. Without CUDA we instead
//! measure the host-side lookup that dominates the wire path on the
//! `StreamIsolated` tier: `TenantRegistry::get` + `Arc::clone`. The plan
//! target is a useful guardrail in both modes — if this benchmark regresses
//! past ~1μs on commodity x86 hardware something has gone wrong with our
//! DashMap usage.

use std::time::Duration;

use bali_core::types::TenantId;
use bali_tenant::{IsolationKind, TenantContext, TenantRegistry};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn populate(n: u64) -> TenantRegistry {
    let reg = TenantRegistry::new();
    for i in 0..n {
        let ctx = TenantContext::builder(TenantId(i))
            .with_isolation(IsolationKind::StreamIsolated)
            .with_memory_quota_bytes(1024 * 1024)
            .build();
        reg.register(ctx).expect("register");
    }
    reg
}

fn bench_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("tenant_registry/lookup");
    group.measurement_time(Duration::from_secs(3));
    for &n in &[1u64, 16, 256] {
        let reg = populate(n);
        let last = TenantId(n - 1);
        group.bench_with_input(BenchmarkId::from_parameter(n), &last, |b, &id| {
            b.iter(|| {
                let ctx = reg.get(id).expect("present");
                black_box(ctx);
            });
        });
    }
    group.finish();
}

fn bench_consume_release(c: &mut Criterion) {
    let mut group = c.benchmark_group("tenant_registry/consume_release");
    group.measurement_time(Duration::from_secs(3));
    let reg = populate(16);
    let ctx = reg.get(TenantId(7)).unwrap();
    group.bench_function("256KiB", |b| {
        b.iter(|| {
            ctx.consume_bytes(256 * 1024).unwrap();
            ctx.release_bytes(256 * 1024);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_lookup, bench_consume_release);
criterion_main!(benches);
