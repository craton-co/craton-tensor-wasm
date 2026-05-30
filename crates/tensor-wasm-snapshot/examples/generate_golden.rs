// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Golden-fixture generator for the cross-version snapshot compatibility tests.
//!
//! Produces the two deterministic snapshot blobs consumed by
//! `tests/compat.rs`:
//!
//! - `golden_v0_1_0_minimal.snap` — empty body, fixed tenant/instance IDs.
//! - `golden_v0_1_0_with_wasm_memory.snap` — small but non-trivial wasm linear
//!   memory plus a handful of GPU bytes and registers.
//!
//! The blobs are written to `<out_dir>/golden_v0_1_0_minimal.snap` and
//! `<out_dir>/golden_v0_1_0_with_wasm_memory.snap` where `out_dir` is the first
//! positional argument (defaults to the current working directory if omitted).
//!
//! ## Determinism
//!
//! `SnapshotWriter::capture` stamps `metadata.created_unix_ms` with the host's
//! wall clock, which would make the resulting bytes change run-to-run. To
//! produce a stable golden fixture this generator bypasses the writer's
//! timestamp path: it constructs the [`Snapshot`] struct directly with a fixed
//! `created_unix_ms`, runs bincode + zstd with the same defaults the writer
//! uses ([`DEFAULT_ZSTD_LEVEL`]), and writes the resulting bytes verbatim.
//!
//! The bytes-on-disk are therefore a function of: input bodies, fixed
//! metadata, [`SNAPSHOT_MAGIC`], [`SNAPSHOT_VERSION`], the bincode 1.x default
//! config, and the zstd level. Any change to those inputs is a deliberate
//! fixture refresh.
//!
//! ## Usage
//!
//! ```text
//! cargo run -p tensor-wasm-snapshot --example generate_golden -- \
//!     crates/tensor-wasm-snapshot/tests/fixtures
//! ```
//!
//! Re-run whenever the snapshot format bumps. See
//! `docs/SNAPSHOT-COMPATIBILITY.md` for the procedure to add a new golden
//! fixture for a new format version (the existing fixtures must continue to
//! load — they encode the previous version's wire bytes verbatim).

use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;

use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::payload_crc32;
use tensor_wasm_snapshot::writer::{
    Snapshot, SnapshotMetadata, DEFAULT_ZSTD_LEVEL, SNAPSHOT_MAGIC, SNAPSHOT_VERSION,
};

/// Fixed timestamp embedded in every golden fixture so the bytes are stable
/// across machines and runs. Chosen as `2026-01-01T00:00:00Z` in milliseconds.
const GOLDEN_CREATED_UNIX_MS: u64 = 1_767_225_600_000;

/// Tenant ID used in the minimal fixture.
const MINIMAL_TENANT_ID: u64 = 0xA;
/// Instance ID used in the minimal fixture.
const MINIMAL_INSTANCE_ID: u128 = 0xB;

/// Tenant ID used in the richer fixture.
const RICH_TENANT_ID: u64 = 0xC0FFEE;
/// Instance ID used in the richer fixture.
const RICH_INSTANCE_ID: u128 = 0xDEAD_BEEF_CAFE_F00D;

/// Build a deterministic, length-stable wasm memory body for the richer
/// fixture. Pattern: `i % 251` over 4096 bytes — coprime with 256 so every
/// byte value is visited, no run-length compresses to nothing, and the total
/// length is predictable.
fn synth_wasm_memory() -> Vec<u8> {
    (0u32..4096).map(|i| (i % 251) as u8).collect()
}

/// GPU body for the richer fixture: 1 KiB with a different stride so the two
/// blobs are byte-distinguishable in the resulting blob.
fn synth_gpu_memory() -> Vec<u8> {
    (0u32..1024)
        .map(|i| ((i.wrapping_mul(17)) % 253) as u8)
        .collect()
}

/// Register-file body for the richer fixture: 256 bytes of a fixed XOR mask.
fn synth_registers() -> Vec<u8> {
    (0u32..256).map(|i| ((i ^ 0x5A) & 0xFF) as u8).collect()
}

/// Build a [`Snapshot`] with a fixed timestamp and the current magic + version.
///
/// Returns the assembled struct (callers run bincode + zstd to land bytes on
/// disk). Keeping the assembly step separate from the encoding step lets the
/// generator emit two fixtures from the same code path with different bodies
/// while sharing every framing detail.
fn build_snapshot(
    tenant_id: TenantId,
    instance_id: InstanceId,
    wasm_memory: Vec<u8>,
    gpu_memory: Vec<u8>,
    registers: Vec<u8>,
) -> Snapshot {
    let total_uncompressed_bytes = (wasm_memory.len() + gpu_memory.len() + registers.len()) as u64;
    let crc32 = payload_crc32(&wasm_memory, &gpu_memory, &registers);
    Snapshot {
        magic: SNAPSHOT_MAGIC,
        version: SNAPSHOT_VERSION,
        wasm_memory,
        gpu_memory,
        registers,
        metadata: SnapshotMetadata {
            tenant_id,
            instance_id,
            created_unix_ms: GOLDEN_CREATED_UNIX_MS,
            total_uncompressed_bytes,
            // Mirror `SnapshotWriter::capture`'s v0.3.x defaults so the
            // golden bytes stay wire-identical to a real capture: no
            // sequence number, no replay nonce (both land in v0.4).
            sequence_no: 0,
            nonce: None,
        },
        crc32,
    }
}

/// Encode `snapshot` using the same bincode + zstd settings the production
/// writer uses, so the resulting bytes are wire-identical to what
/// `SnapshotWriter::capture` would emit for the same logical inputs.
fn encode(snapshot: &Snapshot) -> io::Result<Vec<u8>> {
    let encoded = bincode::serde::encode_to_vec(snapshot, bincode::config::legacy())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("bincode encode: {e}")))?;
    zstd::encode_all(encoded.as_slice(), DEFAULT_ZSTD_LEVEL)
}

fn main() -> ExitCode {
    let out_dir: PathBuf = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));

    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("could not create {}: {e}", out_dir.display());
        return ExitCode::from(1);
    }

    // Fixture 1: minimal — empty bodies, just the envelope and metadata.
    let minimal = build_snapshot(
        TenantId(MINIMAL_TENANT_ID),
        InstanceId(MINIMAL_INSTANCE_ID),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    let minimal_bytes = match encode(&minimal) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("encode minimal fixture: {e}");
            return ExitCode::from(2);
        }
    };
    let minimal_path = out_dir.join("golden_v0_1_0_minimal.snap");
    if let Err(e) = fs::write(&minimal_path, &minimal_bytes) {
        eprintln!("write {}: {e}", minimal_path.display());
        return ExitCode::from(3);
    }
    println!(
        "wrote {} ({} bytes)",
        minimal_path.display(),
        minimal_bytes.len()
    );

    // Fixture 2: slightly richer — real wasm/gpu/registers bodies.
    let rich = build_snapshot(
        TenantId(RICH_TENANT_ID),
        InstanceId(RICH_INSTANCE_ID),
        synth_wasm_memory(),
        synth_gpu_memory(),
        synth_registers(),
    );
    let rich_bytes = match encode(&rich) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("encode rich fixture: {e}");
            return ExitCode::from(4);
        }
    };
    let rich_path = out_dir.join("golden_v0_1_0_with_wasm_memory.snap");
    if let Err(e) = fs::write(&rich_path, &rich_bytes) {
        eprintln!("write {}: {e}", rich_path.display());
        return ExitCode::from(5);
    }
    println!("wrote {} ({} bytes)", rich_path.display(), rich_bytes.len());

    ExitCode::SUCCESS
}
