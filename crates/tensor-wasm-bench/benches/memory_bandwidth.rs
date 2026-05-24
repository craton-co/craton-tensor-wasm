// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! S19 memory-bandwidth bench: sequential vs strided host-copy patterns.
//!
//! On a CUDA host this would measure `cudaMemcpyAsync` device-to-device
//! and unified-memory page migration. On a non-CUDA host the same code
//! exercises host-side memcpy and serves as a regression backstop.
//!
//! Note: the `strided` group used to be named `random_stride`, but the
//! access pattern is actually *fixed-stride sequential*, not random — we
//! step by a constant `STRIDE` rather than indexing through a shuffled
//! permutation. The rename keeps the metric name honest. The bench id
//! change is recorded in `bench-results/baseline-notes.md` so the CI
//! regression-gate parser can be patched in a follow-up.

use std::time::Duration;

use tensor_wasm_mem::pinned_host::GuardedHostBuffer;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_sequential_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_bandwidth/sequential");
    group.measurement_time(Duration::from_secs(3));
    for &size in &[4096usize, 64 * 1024, 1024 * 1024, 16 * 1024 * 1024] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut src = GuardedHostBuffer::new(size).unwrap();
            let mut dst = GuardedHostBuffer::new(size).unwrap();
            // Pre-fill source with a deterministic pattern.
            for (i, byte) in src.as_mut_slice().iter_mut().enumerate() {
                *byte = (i % 251) as u8;
            }
            b.iter(|| {
                dst.as_mut_slice().copy_from_slice(src.as_slice());
                criterion::black_box(&dst);
            });
        });
    }
    group.finish();
}

fn bench_strided_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_bandwidth/strided");
    group.measurement_time(Duration::from_secs(3));
    const STRIDE: usize = 4096;
    for &size in &[64 * 1024usize, 1024 * 1024, 16 * 1024 * 1024] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let mut src = GuardedHostBuffer::new(size).unwrap();
            let mut dst = GuardedHostBuffer::new(size).unwrap();
            for (i, byte) in src.as_mut_slice().iter_mut().enumerate() {
                *byte = (i % 251) as u8;
            }
            // Copy in 64-byte chunks spaced by STRIDE to exercise non-sequential
            // access patterns. `src` is read-only inside the loop, so we can
            // hold the shared slice for the iteration; `dst` is taken mut anew
            // each pass.
            b.iter(|| {
                let src_slice = src.as_slice();
                let dst_slice = dst.as_mut_slice();
                let src_len = src_slice.len();
                let mut off = 0usize;
                while off < src_len {
                    let chunk_end = (off + 64).min(src_len);
                    dst_slice[off..chunk_end].copy_from_slice(&src_slice[off..chunk_end]);
                    let step = STRIDE.min(src_len - off).max(1);
                    off += step;
                }
                criterion::black_box(&dst_slice);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_sequential_copy, bench_strided_copy);
criterion_main!(benches);
