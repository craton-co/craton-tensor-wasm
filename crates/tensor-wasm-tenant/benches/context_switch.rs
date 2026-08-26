// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! S16 done-when: "Context switching overhead <5μs per call (benchmark)".
//!
//! On a CUDA host this would measure `cuCtxPushCurrent` + `cuCtxPopCurrent`
//! across an established `cust::context::Context`. Without CUDA we instead
//! measure the host-side lookup that dominates the wire path on the
//! `StreamIsolated` tier: `TenantRegistry::get` + `Arc::clone`. The plan
//! target is a useful guardrail in both modes — if this benchmark regresses
//! past ~1μs on commodity x86 hardware something has gone wrong with our
//! DashMap usage.
//!
//! TESTS (Finding 8): the consume/release microbench now exercises the
//! cap-CHECKED `consume_bytes_with_capability` / `release_bytes_with_capability`
//! path (the supported API) rather than the deprecated unchecked variants —
//! the cap check is a single integer compare on top of the same inner CAS
//! loop, so this measures what production actually runs. A contended-`get`
//! bench is added to catch DashMap shard-lock regressions under
//! multi-threaded lookups, the scenario the single-threaded lookup bench
//! cannot see.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::{
    IsolationKind, RegistryAdminCapability, TenantCapability, TenantContext, TenantRegistry,
};

fn populate(n: u64) -> (TenantRegistry, RegistryAdminCapability) {
    let (reg, cap) = TenantRegistry::new();
    for i in 0..n {
        let ctx = TenantContext::builder(TenantId(i))
            .with_isolation(IsolationKind::StreamIsolated)
            .with_memory_quota_bytes(1024 * 1024)
            .build();
        reg.register(ctx).expect("register");
    }
    (reg, cap)
}

fn bench_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("tenant_registry/lookup");
    group.measurement_time(Duration::from_secs(3));
    for &n in &[1u64, 16, 256] {
        let (reg, cap) = populate(n);
        let last = TenantId(n - 1);
        group.bench_with_input(BenchmarkId::from_parameter(n), &last, |b, &id| {
            b.iter(|| {
                let ctx = reg.get(id, &cap).expect("present");
                black_box(ctx);
            });
        });
    }
    group.finish();
}

/// Contended `get`: a fixed pool of background threads hammers `get` on the
/// same registry while the measured thread also issues `get`s. This surfaces
/// DashMap shard-lock contention that the single-threaded `bench_lookup`
/// cannot — a regression that serialised lookups (e.g. a coarser lock) would
/// show up here but not there.
fn bench_lookup_contended(c: &mut Criterion) {
    const CONTENDERS: usize = 4;
    const TENANTS: u64 = 256;

    let mut group = c.benchmark_group("tenant_registry/lookup_contended");
    group.measurement_time(Duration::from_secs(3));

    let (reg, cap) = populate(TENANTS);
    let reg = Arc::new(reg);
    let cap = Arc::new(cap);
    let stop = Arc::new(AtomicBool::new(false));
    let start = Arc::new(Barrier::new(CONTENDERS + 1));

    let mut contenders = Vec::with_capacity(CONTENDERS);
    for t in 0..CONTENDERS {
        let reg = Arc::clone(&reg);
        let cap = Arc::clone(&cap);
        let stop = Arc::clone(&stop);
        let start = Arc::clone(&start);
        // Each contender targets a different id so they spread across shards.
        let id = TenantId((t as u64 * 37) % TENANTS);
        contenders.push(thread::spawn(move || {
            start.wait();
            while !stop.load(Ordering::Relaxed) {
                let ctx = reg.get(id, &cap);
                black_box(ctx);
            }
        }));
    }

    start.wait();
    let measured_id = TenantId(TENANTS - 1);
    group.bench_function("4_contenders", |b| {
        b.iter(|| {
            let ctx = reg.get(measured_id, &cap).expect("present");
            black_box(ctx);
        });
    });
    group.finish();

    stop.store(true, Ordering::Relaxed);
    for h in contenders {
        h.join().expect("contender thread panicked");
    }
}

fn bench_consume_release(c: &mut Criterion) {
    let mut group = c.benchmark_group("tenant_registry/consume_release");
    group.measurement_time(Duration::from_secs(3));

    // Register through the capability API so we bench the supported,
    // cap-CHECKED path (the deprecated unchecked variants are gone from this
    // bench — see the module doc).
    let (reg, _admin) = TenantRegistry::new();
    let (ctx, tenant_cap): (Arc<TenantContext>, TenantCapability) = reg
        .register_with_capability(
            TenantContext::builder(TenantId(7))
                .with_isolation(IsolationKind::StreamIsolated)
                .with_memory_quota_bytes(1024 * 1024)
                .build(),
        )
        .expect("register_with_capability");

    group.bench_function("256KiB_checked", |b| {
        b.iter(|| {
            ctx.consume_bytes_with_capability(&tenant_cap, 256 * 1024)
                .unwrap();
            ctx.release_bytes_with_capability(&tenant_cap, 256 * 1024)
                .unwrap();
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_lookup,
    bench_lookup_contended,
    bench_consume_release
);
criterion_main!(benches);
