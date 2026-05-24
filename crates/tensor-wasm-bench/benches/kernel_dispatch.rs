// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! S9 benchmark: serial vs concurrent kernel dispatch throughput.
//!
//! Without CUDA the [`DispatchFuture`] resolves immediately, so this
//! benchmark really measures TensorWasm's scheduling overhead (back-pressure
//! semaphore acquire/release, future polling). On CUDA hosts the same
//! benchmark measures real launch-to-completion latency.
//!
//! `BackPressure::with_cap` is hoisted out of the timed region via
//! `iter_batched_ref` so each sample measures dispatch only (permit
//! acquire+release + future poll), not the one-time semaphore setup.

use std::time::Duration;

use tensor_wasm_wasi_gpu::async_dispatch::{BackPressure, DispatchFuture};
use criterion::{
    criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput,
};

fn bench_serial(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("dispatch/serial");
    group.measurement_time(Duration::from_secs(3));
    for &n in &[1u64, 10, 100, 1000] {
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched_ref(
                || BackPressure::with_cap(1),
                |bp| {
                    rt.block_on(async {
                        for _ in 0..n {
                            let permit = bp.acquire().await;
                            DispatchFuture::ready(permit).await;
                        }
                    });
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

fn bench_concurrent(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("dispatch/concurrent_cap64");
    group.measurement_time(Duration::from_secs(3));
    for &n in &[1u64, 10, 100, 1000] {
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter_batched_ref(
                || BackPressure::with_cap(64),
                |bp| {
                    rt.block_on(async {
                        let mut handles = Vec::with_capacity(n as usize);
                        for _ in 0..n {
                            let bp = bp.clone();
                            handles.push(tokio::spawn(async move {
                                let permit = bp.acquire().await;
                                DispatchFuture::ready(permit).await;
                            }));
                        }
                        for h in handles {
                            h.await.unwrap();
                        }
                    });
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_serial, bench_concurrent);
criterion_main!(benches);
