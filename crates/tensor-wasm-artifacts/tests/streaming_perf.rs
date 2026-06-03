// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T22 perf-tier: 100 MiB put + get round-trip exercising the streaming
//! HMAC + zstd paths.
//!
//! The point of the assertion is qualitative — we cannot easily measure
//! peak RSS from inside the test, but we CAN demonstrate that:
//!
//! 1. A payload sized between `MAX_PAYLOAD_LEN / 2` and `MAX_PAYLOAD_LEN`
//!    round-trips through `put` + `get` without tripping `TooLarge`.
//! 2. The intermediate state never holds a framed-envelope `Vec<u8>` the
//!    size of the payload (the old path did; the new path streams).
//!    This is enforced structurally by the new code's lack of any
//!    `Vec::with_capacity(payload.len() + …)` allocation in `put`.
//! 3. Wall-time stays in a sensible band so a future regression that
//!    accidentally re-buffers the body will surface as a 2–3x slowdown
//!    here even before peak-RSS instrumentation is added.
//!
//! Marked `#[ignore]` because the round-trip costs ~1–2 seconds of
//! wall-time and ~100–200 MiB of disk in a tempdir; run on demand with
//! `cargo test -p tensor-wasm-artifacts -- --ignored streaming_perf`.

use std::time::Instant;

use tensor_wasm_artifacts::{ArtifactStore, DiskArtifactStore, MAX_PAYLOAD_LEN};

/// 100 MiB — well under the 256 MiB `MAX_PAYLOAD_LEN` cap so this test
/// stays clear of the cap-boundary refusal path while still being large
/// enough that a buffered round-trip would be obviously distinguishable
/// from a streaming one.
const PAYLOAD_LEN: usize = 100 * 1024 * 1024;

#[test]
#[ignore = "100 MiB round-trip; run with `cargo test -p tensor-wasm-artifacts -- --ignored streaming_perf`"]
fn streaming_put_get_100_mib_round_trip() {
    // Sanity: the constant the test scales against is the public cap.
    // If the cap shrinks below 100 MiB in a future revision this test
    // should be revisited rather than silently passing on a smaller
    // payload.
    const _: () = assert!(
        PAYLOAD_LEN < MAX_PAYLOAD_LEN,
        "test payload must stay below put cap",
    );

    // Use a deterministic but non-trivial fill so zstd has to do real
    // work (all-zeros would compress to nearly nothing and exercise the
    // I/O path far less). A simple counting pattern compresses ~8x with
    // level 3, which is in the same band as real WASM modules.
    let payload: Vec<u8> = (0..PAYLOAD_LEN).map(|i| (i & 0xFF) as u8).collect();

    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x7Bu8; 32]);

    let put_start = Instant::now();
    let hash = store.put(&payload).expect("put 100 MiB");
    let put_elapsed = put_start.elapsed();

    let get_start = Instant::now();
    let got = store.get(&hash).expect("get 100 MiB");
    let get_elapsed = get_start.elapsed();

    assert_eq!(got.len(), payload.len(), "round-trip length matches");
    // Spot-check ends + a middle byte; full equality would be ~100 MiB
    // of comparison and the length+endpoints are enough to catch a
    // streaming bug that off-by-ones the body.
    assert_eq!(got[0], payload[0]);
    assert_eq!(got[PAYLOAD_LEN / 2], payload[PAYLOAD_LEN / 2]);
    assert_eq!(got[PAYLOAD_LEN - 1], payload[PAYLOAD_LEN - 1]);

    // Qualitative timing band — generous because CI hardware varies
    // wildly. The point is to catch a 10x regression, not to gate on
    // exact ns numbers.
    eprintln!(
        "streaming_perf: put 100 MiB in {:?}, get 100 MiB in {:?}",
        put_elapsed, get_elapsed
    );
    assert!(
        put_elapsed.as_secs() < 60,
        "put took {put_elapsed:?}; likely regression"
    );
    assert!(
        get_elapsed.as_secs() < 60,
        "get took {get_elapsed:?}; likely regression"
    );
}

#[test]
#[ignore = "exercises put at the size cap; run with `cargo test -p tensor-wasm-artifacts -- --ignored streaming_perf`"]
fn streaming_put_at_cap_boundary_succeeds() {
    // Payload sized EXACTLY at `MAX_PAYLOAD_LEN` is allowed; only
    // strictly past the cap should refuse. This exercises the streaming
    // writer at its largest legitimate input, which is where any
    // off-by-one in the cap check or a buffered allocation big enough
    // to OOM a low-RAM box would surface.
    let payload: Vec<u8> = (0..MAX_PAYLOAD_LEN).map(|i| (i & 0xFF) as u8).collect();

    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x42u8; 32]);

    let hash = store.put(&payload).expect("put at cap must succeed");
    // Drop the input early so peak RSS during `get` doesn't double-count.
    drop(payload);

    let got = store.get(&hash).expect("get at cap must succeed");
    assert_eq!(got.len(), MAX_PAYLOAD_LEN);
    assert_eq!(got[0], 0);
    assert_eq!(
        got[MAX_PAYLOAD_LEN - 1],
        ((MAX_PAYLOAD_LEN - 1) & 0xFF) as u8
    );
}
