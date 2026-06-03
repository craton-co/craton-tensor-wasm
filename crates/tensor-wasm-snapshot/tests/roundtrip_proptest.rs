// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Property-based roundtrip: arbitrary memory + register state must survive
//! `capture` + `restore` byte-for-byte.
//!
//! Complements the deterministic round-trip tests (`round_trip.rs`,
//! `hmac_round_trip.rs`) by exercising the bincode/zstd pipeline against
//! adversarially-shaped inputs — empty bodies, single-byte bodies, exact
//! power-of-two lengths, runs of `0x00` and `0xFF` that may compress to
//! degenerate frames, and `Vec<u64>` register payloads whose byte
//! representation may put kind-byte sentinels at trailer offsets.
//!
//! Sizes are deliberately bounded (≤16 KiB per memory blob, ≤32 register
//! globals) to keep the test under a few seconds in CI — the goal is shape
//! coverage, not throughput.

use proptest::prelude::*;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{
    InstanceState, SnapshotWriter, SNAPSHOT_MAGIC, SNAPSHOT_VERSION,
};

proptest! {
    // Keep cases bounded so this test completes quickly in CI. The goal is
    // shape coverage (zero-length blobs, all-zero / all-0xFF runs, register
    // byte patterns) rather than throughput.
    #![proptest_config(ProptestConfig {
        cases: 64,
        .. ProptestConfig::default()
    })]

    /// Capture an arbitrary `InstanceState` with no HMAC, then restore it,
    /// and assert byte-equality on every blob and metadata field.
    #[test]
    fn arbitrary_state_round_trips(
        cpu_mem in proptest::collection::vec(any::<u8>(), 0..16 * 1024),
        gpu_mem in proptest::collection::vec(any::<u8>(), 0..16 * 1024),
        globals in proptest::collection::vec(any::<u64>(), 0..32),
    ) {
        // Flatten the `globals: Vec<u64>` into the snapshot's `registers`
        // byte-blob using little-endian encoding. This is the same on-wire
        // layout the rest of the codebase uses for register-file dumps and
        // gives proptest direct coverage of multi-byte alignment shapes
        // inside the registers blob.
        let mut registers: Vec<u8> = Vec::with_capacity(globals.len() * 8);
        for g in &globals {
            registers.extend_from_slice(&g.to_le_bytes());
        }

        // Build the state and capture with no HMAC (default writer emits v2).
        let state = InstanceState {
            tenant_id: TenantId(0xABCD),
            instance_id: InstanceId(0x1234_5678_9ABC_DEF0),
            wasm_memory: &cpu_mem,
            gpu_memory: &gpu_mem,
            registers: &registers,
        };

        let blob = SnapshotWriter::new()
            .capture(state)
            .expect("capture must succeed for in-cap arbitrary state");

        let restored = SnapshotReader::new()
            .restore(&blob)
            .expect("restore must succeed for blob produced by capture");

        // Byte-equality on every section — the load-bearing property.
        prop_assert_eq!(restored.magic, SNAPSHOT_MAGIC);
        prop_assert_eq!(restored.version, SNAPSHOT_VERSION);
        prop_assert_eq!(&restored.wasm_memory, &cpu_mem);
        prop_assert_eq!(&restored.gpu_memory, &gpu_mem);
        prop_assert_eq!(&restored.registers, &registers);
        prop_assert_eq!(restored.metadata.tenant_id, TenantId(0xABCD));
        prop_assert_eq!(
            restored.metadata.instance_id,
            InstanceId(0x1234_5678_9ABC_DEF0),
        );
        prop_assert_eq!(
            restored.metadata.total_uncompressed_bytes,
            (cpu_mem.len() + gpu_mem.len() + registers.len()) as u64,
        );
    }
}
