// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Robustness property: `SnapshotReader::restore` must never panic on
//! arbitrary input.
//!
//! The reader's whole contract (see `reader.rs`) is that *every* malformed
//! input — random bytes, truncated zstd frames, broken bincode, oversized
//! declared lengths, stray trailer-magic coincidences — is surfaced as a
//! `Result::Err`, never an unwind. This proptest throws random byte vectors of
//! varied length at `restore` (across several reader configurations) and
//! asserts only that the call *returns* (`is_err() || is_ok()`), so a panic in
//! any decode path — zstd init, bincode length-prefix handling, the v3 trailer
//! classifier, the v4 artifact-envelope detector — fails the test instead of
//! aborting the process.
//!
//! A random blob will essentially always be rejected; the rare `Ok` (e.g. a
//! random sequence that happens to be a valid empty zstd frame around a valid
//! bincode payload) is equally acceptable — the property is "no panic", not
//! "always error".

use proptest::prelude::*;
use tensor_wasm_snapshot::reader::SnapshotReader;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 512,
        .. ProptestConfig::default()
    })]

    /// Default reader (no HMAC key, signature not required): random bytes in,
    /// no panic out.
    #[test]
    fn restore_default_reader_never_panics(
        data in proptest::collection::vec(any::<u8>(), 0..4096),
    ) {
        let reader = SnapshotReader::new();
        let result = reader.restore(&data);
        prop_assert!(result.is_err() || result.is_ok());
    }

    /// Same, but with a reader that has an HMAC key configured and requires a
    /// signature — exercises the v3 trailer classifier and (under the default
    /// `artifact-backing` feature) the v4 artifact-envelope detector on random
    /// input, which is where a length-prefix or offset-arithmetic panic would
    /// most plausibly hide.
    #[test]
    fn restore_signing_reader_never_panics(
        data in proptest::collection::vec(any::<u8>(), 0..4096),
    ) {
        let reader = build_signing_reader();
        let result = reader.restore(&data);
        prop_assert!(result.is_err() || result.is_ok());
    }

    /// Bias the generator toward inputs that *start* like a real snapshot: a
    /// zstd frame magic prefix followed by random bytes. These drive the
    /// decoder deeper than uniformly-random noise (which usually fails at
    /// "zstd init"), exercising the streaming decode + bincode path.
    #[test]
    fn restore_zstd_prefixed_garbage_never_panics(
        tail in proptest::collection::vec(any::<u8>(), 0..4096),
    ) {
        // zstd frame magic: 0x28 0xB5 0x2F 0xFD (little-endian 0xFD2FB528).
        let mut data = vec![0x28u8, 0xB5, 0x2F, 0xFD];
        data.extend_from_slice(&tail);
        let reader = SnapshotReader::new();
        let result = reader.restore(&data);
        prop_assert!(result.is_err() || result.is_ok());
    }

    /// Append random "trailer-shaped" tails to a real capture so the v3 trailer
    /// classifier / verifier sees adversarial trailer bytes. The base capture
    /// guarantees a valid zstd+bincode prefix; the random tail probes the
    /// trailer-offset arithmetic for panics on out-of-range / inconsistent
    /// lengths.
    #[test]
    fn restore_real_capture_with_random_tail_never_panics(
        tail in proptest::collection::vec(any::<u8>(), 0..128),
    ) {
        let mut data = base_capture();
        data.extend_from_slice(&tail);
        let reader = build_signing_reader();
        let result = reader.restore(&data);
        prop_assert!(result.is_err() || result.is_ok());
    }
}

/// A reader configured with an HMAC key and `require_signature`, so random and
/// trailer-shaped inputs exercise the strictest classification/verification
/// path. Falls back to a default reader on builds without `signed-snapshots`.
fn build_signing_reader() -> SnapshotReader {
    #[cfg(feature = "signed-snapshots")]
    {
        SnapshotReader::new()
            .with_hmac_sha256_key([0x5Au8; 32])
            .require_signature()
    }
    #[cfg(not(feature = "signed-snapshots"))]
    {
        SnapshotReader::new()
    }
}

/// A real, minimal v2 capture used as a valid prefix for the random-tail case.
fn base_capture() -> Vec<u8> {
    use tensor_wasm_core::types::{InstanceId, TenantId};
    use tensor_wasm_snapshot::writer::{InstanceState, SnapshotWriter};
    SnapshotWriter::new()
        .capture(InstanceState {
            tenant_id: TenantId(1),
            instance_id: InstanceId(1),
            wasm_memory: &[1, 2, 3, 4, 5, 6, 7, 8],
            gpu_memory: &[],
            registers: &[],
        })
        .expect("base capture")
}
