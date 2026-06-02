// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Fuzz target for `tensor_wasm_artifacts::decode_envelope_from_bytes` and its
//! cap-parameterised sibling `decode_envelope_from_bytes_with_cap`.
//!
//! The envelope decoder is the artifact store's untrusted-input entry point:
//! it parses an attacker-controllable byte blob (magic + version + BLAKE3
//! content-hash + zstd body + HMAC tag) under a caller-supplied 32-byte key.
//! It is the verifier half of the disk-cache and v4-snapshot trust boundary, so
//! it must refuse every malformed shape — bad magic, bad version, HMAC
//! mismatch, zstd garbage, zip-bomb, content-hash mismatch — as a structured
//! [`ArtifactError`] rather than panicking. Any panic is a finding.
//!
//! We split the first 32 bytes of fuzzer input into a synthetic HMAC key and
//! feed the remainder to both decode entry points. The overwhelmingly common
//! outcome is `Err(BadMagic)` / `Err(BadHmac)` — exactly the property under
//! test. To also exercise the zip-bomb `TooLarge` ceiling and the cap-threading
//! that the snapshot reader's `with_max_decompressed` knob relies on, we derive
//! a small `max_decompressed` cap from the input and run the `_with_cap`
//! variant alongside the default-cap wrapper.

#![no_main]

use libfuzzer_sys::fuzz_target;

use tensor_wasm_artifacts::{decode_envelope_from_bytes, decode_envelope_from_bytes_with_cap};

fuzz_target!(|data: &[u8]| {
    // Need 32 bytes for the synthetic key prefix. Shorter inputs can't drive
    // the keyed decode path, so skip them to keep the mutator productive.
    if data.len() < 32 {
        return;
    }
    // Cap the envelope length (after consuming the 32-byte key prefix) to a
    // 64 MiB ceiling so the fuzzer doesn't burn cycles on oversized inputs the
    // decoder would reject in its first cheap length/magic checks anyway.
    if data.len() - 32 > 64 * 1024 * 1024 {
        return;
    }
    let (key_bytes, envelope) = data.split_at(32);
    let key: [u8; 32] = key_bytes
        .try_into()
        .expect("split_at(32) yields exactly 32 bytes");

    // Default-cap wrapper. Contract: never panics — every adversarial input is
    // either decoded or surfaced as a structured `ArtifactError`. We ignore the
    // Ok/Err discriminant; libFuzzer catches panics and UB.
    let _ = decode_envelope_from_bytes(envelope, &key);

    // Cap-parameterised variant with a tight, input-derived ceiling so the
    // zip-bomb `TooLarge` path and the `Take`-probe boundary get exercised too.
    // The cap is bounded to a small range to keep mutated zstd frames bumping
    // against it rather than always fitting under the 1 GiB default.
    let cap = 1 + (envelope.len() % 65_536);
    let _ = decode_envelope_from_bytes_with_cap(envelope, &key, cap);
});
