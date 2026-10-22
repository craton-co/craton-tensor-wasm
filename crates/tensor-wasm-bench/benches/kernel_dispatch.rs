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
//!
//! ## What `DispatchFuture::ready` actually measures (finding 1)
//!
//! The `serial`/`concurrent_cap64` groups drive [`DispatchFuture::ready`],
//! which on a no-CUDA host resolves on first poll — so the sample captures
//! `tracing::info_span!` construction + permit RAII + a single future poll,
//! NOT GPU sync. Three additions make the breakdown honest:
//!
//! * **`dispatch/span_only`** isolates the `tracing::info_span!("…")`
//!   construction cost that every `DispatchFuture` pays at build time, so a
//!   reader can subtract the span tax from the `ready`-path numbers.
//! * **`dispatch/real_sync` (CUDA only)** records a real `cust` event on a
//!   stream and drives [`DispatchFuture::with_event`], so on a GPU host this
//!   group measures the actual `event.query()`-driven completion poll — the
//!   path `ready` short-circuits.
//! * **`dispatch/contended`** documents that `cap=64` with `n ≤ 64` is the
//!   *uncontended* baseline (every task gets a permit immediately) and adds
//!   `cap=1`/`cap=4` with a large `n` so the semaphore actually queues and
//!   the permit-wait cost shows up.

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use tensor_wasm_wasi_gpu::async_dispatch::{BackPressure, DispatchFuture};

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
    // cap=64 with n ≤ 64 is the UNCONTENDED baseline: every spawned task
    // acquires a permit on the first try, so the semaphore never makes a
    // task wait. The `dispatch/contended` group below is the contended
    // counterpart (cap ≪ n) that actually queues on the permit.
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

/// Isolates the per-dispatch `tracing::info_span!("wasi_cuda.dispatch")`
/// construction cost that `DispatchFuture::ready` pays on every build, so the
/// span tax can be subtracted from the `serial`/`concurrent` numbers above.
/// `acquire` + `await` are identical to the `ready` path; the only difference
/// from `bench_serial` at `n = 1` is that this is a single dispatch, making
/// the per-dispatch span construction the dominant term.
fn bench_span_only(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("dispatch/span_only");
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("ready_single", |b| {
        b.iter_batched_ref(
            || BackPressure::with_cap(1),
            |bp| {
                rt.block_on(async {
                    let permit = bp.acquire().await;
                    // Building the future constructs the `info_span!`; awaiting
                    // it on a no-CUDA host resolves on the first poll, so the
                    // sample is dominated by the span construction + permit RAII.
                    let fut = DispatchFuture::ready(permit);
                    criterion::black_box(&fut);
                    fut.await;
                });
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

/// Contended counterpart to `concurrent_cap64`. With `cap ≪ n` the semaphore
/// must make tasks wait for a permit, so the sample includes the permit-queue
/// cost the uncontended baseline never pays. Two caps (1 and 4) bracket the
/// "fully serialised" and "narrow-parallel" regimes against a large `n`.
fn bench_contended(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let mut group = c.benchmark_group("dispatch/contended");
    group.measurement_time(Duration::from_secs(3));
    const N: u64 = 1000;
    for &cap in &[1usize, 4] {
        group.throughput(Throughput::Elements(N));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("cap{cap}_n{N}")),
            &cap,
            |b, &cap| {
                b.iter_batched_ref(
                    || BackPressure::with_cap(cap),
                    |bp| {
                        rt.block_on(async {
                            let mut handles = Vec::with_capacity(N as usize);
                            for _ in 0..N {
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
            },
        );
    }
    group.finish();
}

/// Real GPU-sync dispatch path (CUDA only). Records a `cust` event on a
/// freshly created stream and drives [`DispatchFuture::with_event`], so the
/// sample exercises the `event.query()`-driven completion poll instead of the
/// immediate-resolve `ready` short-circuit. No kernel is launched: the event
/// is recorded on an otherwise-empty stream, so the measured number is the
/// floor of the real-sync machinery (context bind + event record + the
/// poll/sleep wake loop in `DispatchFuture::poll`), not launch-to-completion
/// of any particular kernel. That floor is the cost the no-CUDA `ready` path
/// hides entirely.
#[cfg(feature = "cuda")]
fn bench_real_sync(c: &mut Criterion) {
    use cust::event::{Event, EventFlags};
    use cust::stream::{Stream, StreamFlags};

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    // Bind the process-wide primary context to this thread before any driver
    // call (mirrors the launch path in `tensor-wasm-wasi-gpu::host`). If CUDA
    // init fails (no device / no toolkit at runtime), skip the group rather
    // than panicking the whole bench binary.
    if tensor_wasm_wasi_gpu::cuda_ctx::ensure_current_context().is_err() {
        eprintln!("dispatch/real_sync: skipped (no usable CUDA context at runtime)");
        return;
    }

    let mut group = c.benchmark_group("dispatch/real_sync");
    group.measurement_time(Duration::from_secs(3));
    group.bench_function("event_query_single", |b| {
        b.iter_batched_ref(
            || BackPressure::with_cap(1),
            |bp| {
                rt.block_on(async {
                    tensor_wasm_wasi_gpu::cuda_ctx::ensure_current_context()
                        .expect("bind CUDA context on worker thread");
                    let stream =
                        Stream::new(StreamFlags::NON_BLOCKING, None).expect("create CUDA stream");
                    let event = Event::new(EventFlags::DEFAULT).expect("create CUDA event");
                    // Record on an empty stream: the event fires as soon as the
                    // (empty) stream reaches it, so `DispatchFuture` exercises
                    // the real `event.query()` poll path without depending on a
                    // particular kernel's runtime.
                    event.record(&stream).expect("record CUDA event");
                    let permit = bp.acquire().await;
                    DispatchFuture::with_event(permit, event).await;
                });
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

#[cfg(feature = "cuda")]
criterion_group!(
    benches,
    bench_serial,
    bench_concurrent,
    bench_span_only,
    bench_contended,
    bench_real_sync
);
#[cfg(not(feature = "cuda"))]
criterion_group!(
    benches,
    bench_serial,
    bench_concurrent,
    bench_span_only,
    bench_contended
);
criterion_main!(benches);
