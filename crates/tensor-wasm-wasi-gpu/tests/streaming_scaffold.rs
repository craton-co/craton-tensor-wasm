// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Unit tests for the `wasi:tensor/host` streaming scaffold
//! ([`tensor_wasm_wasi_gpu::streaming::StreamingContext`]).
//!
//! Covers the four error codes documented on `emit-chunk`:
//!   * disabled context returns `-1`,
//!   * total-bytes cap returns `-2`,
//!   * receiver-dropped returns `-3`,
//!   * happy path increments `bytes_emitted`.
//!
//! See `wit/wasi-tensor.wit` for the WIT contract these tests pin.

use tensor_wasm_wasi_gpu::streaming::{StreamingContext, MAX_TOTAL_STREAM_BYTES};
use tokio::sync::mpsc;

#[tokio::test]
async fn disabled_emit_returns_minus_one() {
    let ctx = StreamingContext::disabled();
    assert!(!ctx.is_enabled());
    let rc = ctx.emit_chunk(vec![0u8; 16]).await;
    assert_eq!(rc, -1, "disabled context must signal -1 to the guest");
    // Counter must NOT advance on a rejected emit.
    assert_eq!(ctx.bytes_emitted(), 0);
    // Flush on a disabled context also returns -1 so the guest can
    // probe streaming-enabled state without an emit.
    assert_eq!(ctx.flush(), -1);
}

#[tokio::test]
async fn happy_path_advances_bytes_emitted() {
    let (tx, mut rx) = mpsc::channel::<Vec<u8>>(4);
    let ctx = StreamingContext::with_channel(tx);
    assert!(ctx.is_enabled());

    let chunk = vec![0xDE, 0xAD, 0xBE, 0xEF];
    let rc = ctx.emit_chunk(chunk.clone()).await;
    assert_eq!(rc, 0, "happy path must return 0 (bytes accepted)");
    assert_eq!(ctx.bytes_emitted(), chunk.len() as u64);

    // Bytes must arrive on the receiver verbatim.
    let received = rx.recv().await.expect("chunk delivered");
    assert_eq!(received, chunk);

    // Flush on an enabled context returns 0.
    assert_eq!(ctx.flush(), 0);

    // Second chunk accumulates.
    let chunk2 = vec![0xCA, 0xFE];
    assert_eq!(ctx.emit_chunk(chunk2.clone()).await, 0);
    assert_eq!(
        ctx.bytes_emitted(),
        (chunk.len() + chunk2.len()) as u64,
        "bytes_emitted must accumulate across emits"
    );
    let received2 = rx.recv().await.expect("second chunk delivered");
    assert_eq!(received2, chunk2);
}

#[tokio::test]
async fn cap_exceeded_returns_minus_two() {
    // Use the explicit-cap constructor so this test doesn't have to
    // emit MAX_TOTAL_STREAM_BYTES of data.
    let (tx, _rx) = mpsc::channel::<Vec<u8>>(4);
    let ctx = StreamingContext::with_channel_and_cap(tx, 8);

    // First emit (4 bytes) fits.
    assert_eq!(ctx.emit_chunk(vec![0u8; 4]).await, 0);
    assert_eq!(ctx.bytes_emitted(), 4);

    // Second emit (5 bytes) trips the cap (4 + 5 = 9 > 8).
    assert_eq!(
        ctx.emit_chunk(vec![0u8; 5]).await,
        -2,
        "cap-exceeded must signal -2"
    );
    // Counter must NOT advance on a cap-rejected emit.
    assert_eq!(
        ctx.bytes_emitted(),
        4,
        "bytes_emitted rolls back on cap rejection"
    );

    // A subsequent in-bounds emit must still succeed (3 bytes fits
    // exactly into the 8-byte budget).
    assert_eq!(ctx.emit_chunk(vec![0u8; 4]).await, 0);
    assert_eq!(ctx.bytes_emitted(), 8);
}

#[tokio::test]
async fn receiver_dropped_returns_minus_three() {
    let (tx, rx) = mpsc::channel::<Vec<u8>>(4);
    let ctx = StreamingContext::with_channel(tx);
    // Simulate "downstream HTTP client disconnected" by dropping the
    // gateway-side receiver before the guest emits.
    drop(rx);

    let rc = ctx.emit_chunk(vec![0u8; 4]).await;
    assert_eq!(rc, -3, "receiver-dropped must signal -3");
    // Counter rolls back on send failure so the per-invocation total
    // reflects only successfully forwarded bytes.
    assert_eq!(ctx.bytes_emitted(), 0);
}

#[test]
fn default_cap_constant_is_sane() {
    // Sanity-check: 64 MiB is large enough for LLM-style token streams
    // but small enough that a runaway guest cannot exhaust gateway
    // memory. If this assertion ever changes, update `docs/STREAMING.md`
    // and the `wasi:tensor/host.emit-chunk` doc comment.
    assert_eq!(MAX_TOTAL_STREAM_BYTES, 64 * 1024 * 1024);
}
