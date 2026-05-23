//! End-to-end round-trip test for `bali-snapshot`.
//!
//! Captures a synthesised [`InstanceState`], writes the blob to a temp file,
//! reads it back through [`SnapshotReader`], and checks every field is
//! byte-for-byte identical.

use std::fs;

use bali_core::types::{InstanceId, TenantId};
use bali_snapshot::reader::SnapshotReader;
use bali_snapshot::writer::{InstanceState, SnapshotWriter, SNAPSHOT_MAGIC, SNAPSHOT_VERSION};
use tempfile::NamedTempFile;

/// Build a deterministic-but-not-trivial set of memory bodies so the test
/// catches off-by-one / aliasing bugs in the writer.
fn synth_state() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let wasm: Vec<u8> = (0u32..8192).map(|i| (i % 251) as u8).collect();
    let gpu: Vec<u8> = (0u32..4096)
        .map(|i| ((i.wrapping_mul(17)) % 253) as u8)
        .collect();
    let regs: Vec<u8> = (0u32..512).map(|i| ((i ^ 0xA5) & 0xFF) as u8).collect();
    (wasm, gpu, regs)
}

#[test]
fn full_round_trip_through_temp_file() {
    let (wasm, gpu, regs) = synth_state();

    let writer = SnapshotWriter::new();
    let blob = writer
        .capture(InstanceState {
            tenant_id: TenantId(0xABCD),
            instance_id: InstanceId(0x1234_5678_9ABC_DEF0),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture");

    // Persist to a temp file and read back via the filesystem to mimic a real
    // cold-start path.
    let tmp = NamedTempFile::new().expect("temp file");
    fs::write(tmp.path(), &blob).expect("write blob");
    let on_disk = fs::read(tmp.path()).expect("read blob");
    assert_eq!(on_disk, blob);

    let restored = SnapshotReader::new().restore(&on_disk).expect("restore");
    assert_eq!(restored.magic, SNAPSHOT_MAGIC);
    assert_eq!(restored.version, SNAPSHOT_VERSION);
    assert_eq!(restored.wasm_memory, wasm);
    assert_eq!(restored.gpu_memory, gpu);
    assert_eq!(restored.registers, regs);
    assert_eq!(restored.metadata.tenant_id, TenantId(0xABCD));
    assert_eq!(
        restored.metadata.instance_id,
        InstanceId(0x1234_5678_9ABC_DEF0)
    );
    assert_eq!(
        restored.metadata.total_uncompressed_bytes,
        (wasm.len() + gpu.len() + regs.len()) as u64,
    );
    // Metadata timestamp is filled in by the writer; just sanity-check it's
    // non-zero on a host with a working clock.
    assert!(restored.metadata.created_unix_ms > 0);
}

#[test]
fn re_export_at_crate_root_works() {
    // The crate root re-exports `Snapshot` and `SnapshotMetadata` for callers
    // that don't want to reach into submodules. This compiles iff the re-export
    // is in place.
    let _: Option<bali_snapshot::Snapshot> = None;
    let _: Option<bali_snapshot::SnapshotMetadata> = None;
}
