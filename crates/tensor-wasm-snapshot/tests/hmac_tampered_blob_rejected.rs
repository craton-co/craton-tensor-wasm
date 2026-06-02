// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Tampering inside the signed payload region must be caught by the HMAC.
//!
//! The CRC32 alone is not an integrity check against a motivated attacker —
//! anyone who can mutate the compressed blob can also recompute CRC32 over
//! the tampered prefix. This test simulates exactly that: it captures a
//! signed v3 snapshot, decompresses the inner payload, flips one byte of
//! `wasm_memory`, *re-fixes* the CRC32 so the IEEE checksum still validates,
//! re-encodes and re-compresses, then reattaches the *original* signature
//! trailer. The reader must reject the result because the HMAC computed over
//! the tampered ciphertext no longer matches the (unchanged) trailer — the
//! key is unknown to the attacker, so they cannot re-sign.
//!
//! Mirrors the handcrafted-blob style of `restore_validation.rs`.

#![cfg(feature = "signed-snapshots")]

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tensor_wasm_snapshot::format::V3_TRAILER_MAGIC_LEN;
use tensor_wasm_snapshot::payload_crc32;
use tensor_wasm_snapshot::reader::SnapshotReader;
use tensor_wasm_snapshot::writer::{InstanceState, Snapshot, SnapshotWriter, DEFAULT_ZSTD_LEVEL};

/// Fixed HMAC key for the capture step. The restore step intentionally uses
/// the *same* key — the rejection must come from the HMAC-over-tampered-bytes
/// mismatch, not from a key swap. That isolates the test to the integrity
/// property, distinct from `hmac_wrong_key_rejected.rs` which exercises the
/// confidentiality-of-the-key property.
const KEY: [u8; 32] = [0xAB; 32];

/// Size of the v3 signature trailer. The T8 format change grew this from the
/// original `[kind][sig]` (33 bytes) to `[V3_TRAILER_MAGIC][kind][sig]`:
/// 4-byte magic + 1-byte `SignatureKind` + 32-byte HMAC-SHA256 = 37 bytes.
const SIG_TRAILER_LEN: usize = V3_TRAILER_MAGIC_LEN + 1 + 32;

#[test]
fn tampered_wasm_memory_with_refixed_crc_is_rejected() {
    let wasm = vec![0x10u8, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80];
    let gpu = vec![0xAAu8, 0xBB, 0xCC];
    let regs = vec![0x77u8; 8];

    // 1. Capture a real signed v3 archive. T40: this test splices the v3
    //    trailer, so pin the writer to the legacy envelope — the v0.4
    //    artifact-envelope default produces no v3 trailer.
    let archive = SnapshotWriter::new()
        .with_hmac_sha256_key(KEY)
        .with_legacy_envelope()
        .capture(InstanceState {
            tenant_id: TenantId(0xC0FFEE),
            instance_id: InstanceId(0xDEAD_BEEF),
            wasm_memory: &wasm,
            gpu_memory: &gpu,
            registers: &regs,
        })
        .expect("capture signed v3");

    assert!(
        archive.len() > SIG_TRAILER_LEN,
        "signed archive must contain payload + 33-byte trailer (got {} bytes)",
        archive.len(),
    );

    // 2. Split off the signature trailer (last 33 bytes) so we can mutate the
    //    inner compressed payload, then reattach the trailer verbatim.
    let split = archive.len() - SIG_TRAILER_LEN;
    let (compressed_payload, sig_trailer) = archive.split_at(split);
    let sig_trailer = sig_trailer.to_vec();

    // 3. Decompress to recover the bincode-encoded Snapshot.
    let decompressed = zstd::decode_all(compressed_payload).expect("decompress signed payload");

    // 4. Decode, mutate one byte of wasm_memory, and re-fix CRC32 so the
    //    inner CRC check still passes for the tampered bytes. This proves
    //    that the HMAC — not CRC — is what catches the attack.
    let cfg = bincode::config::legacy();
    let (mut snap, _read): (Snapshot, usize) =
        bincode::serde::decode_from_slice(&decompressed, cfg).expect("decode snapshot");
    assert!(
        !snap.wasm_memory.is_empty(),
        "tamper test needs a non-empty wasm_memory blob",
    );
    snap.wasm_memory[2] ^= 0xFF;
    snap.crc32 = payload_crc32(&snap.wasm_memory, &snap.gpu_memory, &snap.registers);

    // 5. Re-encode and re-compress with the same config the writer uses.
    let re_encoded = bincode::serde::encode_to_vec(&snap, cfg).expect("re-encode");
    let re_compressed =
        zstd::encode_all(re_encoded.as_slice(), DEFAULT_ZSTD_LEVEL).expect("re-compress");

    // 6. Reattach the original signature trailer. The HMAC inside this trailer
    //    is valid for the ORIGINAL compressed payload, not for the tampered
    //    one — the attacker has no key to re-sign.
    let mut tampered = re_compressed;
    tampered.extend_from_slice(&sig_trailer);

    // 7. Restore with the legitimate key. The reader must reject with a
    //    signature-class error, not a CRC or bincode error.
    let err = SnapshotReader::new()
        .with_hmac_sha256_key(KEY)
        .restore(&tampered)
        .expect_err("tampered signed blob must be rejected");

    match err {
        TensorWasmError::Serialization(msg) => {
            let lower = msg.to_ascii_lowercase();
            assert!(
                lower.contains("hmac")
                    || lower.contains("signature mismatch")
                    || lower.contains("signature is invalid")
                    || lower.contains("invalid signature"),
                "expected HMAC/signature rejection (CRC was deliberately re-fixed), got: {msg}",
            );
        }
        other => panic!("expected Serialization error, got {other:?}"),
    }
}
