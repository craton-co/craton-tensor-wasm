//! S19 cold-start bench: snapshot save -> reload round-trip latency at
//! 1 MiB, 16 MiB, 128 MiB, 512 MiB payload sizes.
//!
//! On a CUDA host this would additionally measure UVM warmup; here we
//! capture the host-side `bincode -> zstd -> fs::write -> fs::read -> zstd ->
//! bincode` path. PERFORMANCE.md (S19) documents the expected gap to a
//! full CUDA-restore path.
//!
//! Note: the `disk_round_trip` group intentionally stops at 16 MiB. Above
//! that, disk IO dominates and blows the bench wall-clock with little
//! signal — capture/restore alone are exercised at the larger sizes.

use std::time::Duration;

use bali_core::types::{InstanceId, TenantId};
use bali_snapshot::reader::SnapshotReader;
use bali_snapshot::writer::{InstanceState, SnapshotWriter};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Build a `(wasm, gpu, regs)` triple totalling roughly `size_bytes`.
fn fixture_bytes(size_bytes: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let half = size_bytes / 2;
    let wasm = vec![0xAAu8; half];
    let gpu = vec![0xBBu8; half];
    let regs = vec![0u8; 256];
    (wasm, gpu, regs)
}

fn bench_snapshot_capture(c: &mut Criterion) {
    let mut group = c.benchmark_group("cold_start/capture");
    group.measurement_time(Duration::from_secs(3));
    for &size in &[
        1024usize * 1024,
        16 * 1024 * 1024,
        128 * 1024 * 1024,
        512 * 1024 * 1024,
    ] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let (wasm, gpu, regs) = fixture_bytes(size);
            let writer = SnapshotWriter::new();
            b.iter(|| {
                let bytes = writer
                    .capture(InstanceState {
                        tenant_id: TenantId(1),
                        instance_id: InstanceId(1),
                        wasm_memory: &wasm,
                        gpu_memory: &gpu,
                        registers: &regs,
                    })
                    .expect("capture");
                criterion::black_box(bytes);
            });
        });
    }
    group.finish();
}

fn bench_snapshot_restore(c: &mut Criterion) {
    let mut group = c.benchmark_group("cold_start/restore");
    group.measurement_time(Duration::from_secs(3));
    for &size in &[
        1024usize * 1024,
        16 * 1024 * 1024,
        128 * 1024 * 1024,
        512 * 1024 * 1024,
    ] {
        group.throughput(Throughput::Bytes(size as u64));
        let (wasm, gpu, regs) = fixture_bytes(size);
        let captured = SnapshotWriter::new()
            .capture(InstanceState {
                tenant_id: TenantId(1),
                instance_id: InstanceId(1),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            })
            .expect("pre-capture");
        let reader = SnapshotReader::new();
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &captured,
            |b, captured| {
                b.iter(|| {
                    let snap = reader.restore(captured).expect("restore");
                    criterion::black_box(snap);
                });
            },
        );
    }
    group.finish();
}

fn bench_round_trip_via_disk(c: &mut Criterion) {
    let mut group = c.benchmark_group("cold_start/disk_round_trip");
    group.measurement_time(Duration::from_secs(3));
    // Only measure smaller sizes for the disk path -- disk dominates above 16 MiB.
    for &size in &[1024usize * 1024, 16 * 1024 * 1024] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let (wasm, gpu, regs) = fixture_bytes(size);
            let writer = SnapshotWriter::new();
            let reader = SnapshotReader::new();
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("snap.bali");
            b.iter(|| {
                let bytes = writer
                    .capture(InstanceState {
                        tenant_id: TenantId(1),
                        instance_id: InstanceId(1),
                        wasm_memory: &wasm,
                        gpu_memory: &gpu,
                        registers: &regs,
                    })
                    .unwrap();
                std::fs::write(&path, &bytes).unwrap();
                let read = std::fs::read(&path).unwrap();
                let restored = reader.restore(&read).unwrap();
                criterion::black_box(restored);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_snapshot_capture,
    bench_snapshot_restore,
    bench_round_trip_via_disk
);
criterion_main!(benches);
