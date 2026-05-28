// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `wasi:tensor/host` streaming surface (roadmap feature #2).
//!
//! Lets guests emit chunks of output that the host gateway flushes to the
//! HTTP client as SSE or chunked-transfer bytes. v0.3.7 lands the WIT
//! contract + a buffered in-memory channel; v0.4 lands actual streaming
//! through the axum response path.
//!
//! ## Surface
//!
//! [`StreamingContext`] owns an optional Tokio `mpsc::Sender<Vec<u8>>`
//! channel. The host functions wrapped by [`add_streaming_to_linker`]
//! call into [`StreamingContext::emit_chunk`] and
//! [`StreamingContext::flush`]; the API gateway holds the matching
//! `mpsc::Receiver` and forwards the chunks into the axum SSE / chunked
//! response body.
//!
//! ## Caps
//!
//! Two hard caps are enforced before any chunk is forwarded:
//!
//! * [`MAX_CHUNK_BYTES`] (64 KiB) — single-chunk size cap. Currently the
//!   parser-side check is left for the v0.4 host-fn implementation (the
//!   wasi-tensor WIT signature does not yet carry a per-call cap distinct
//!   from the total cap); the constant is exported so the host-fn wrapper
//!   and tests share a single source of truth.
//! * [`MAX_TOTAL_STREAM_BYTES`] (64 MiB) — total bytes a single
//!   invocation may emit before the host returns the documented
//!   `-2 = cap exceeded` code from `emit-chunk`.
//!
//! Caps are intentionally conservative for the scaffold so a runaway
//! guest cannot exhaust gateway memory while v0.4 is in flight.
//!
//! ## Error codes
//!
//! Mirrors the WIT contract in `wit/wasi-tensor.wit`:
//! * `>= 0` — bytes accepted.
//! * `-1`   — streaming not enabled for this invocation.
//! * `-2`   — guest tried to emit past the documented size cap.
//! * `-3`   — downstream client disconnected (receiver dropped).
//!
//! Tests in `tests/streaming_scaffold.rs` exercise every branch.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use wasmtime::{Caller, Linker};

/// Maximum size, in bytes, of a single `emit-chunk` call. 64 KiB matches
/// typical HTTP chunk-encoder buffer sizes; guests producing larger
/// payloads should call `emit-chunk` repeatedly. Enforced by the v0.4
/// host-fn wrapper; exported here so wrapper and tests share a constant.
pub const MAX_CHUNK_BYTES: usize = 64 * 1024;

/// Maximum total bytes a single invocation may emit across all
/// `emit-chunk` calls before the host starts returning the documented
/// `-2 = cap exceeded` code. 64 MiB is generous for LLM-style token
/// streams while keeping the worst-case per-invocation memory footprint
/// bounded.
pub const MAX_TOTAL_STREAM_BYTES: u64 = 64 * 1024 * 1024;

/// Per-invocation streaming context. Owns the producer end of the
/// `mpsc::Sender<Vec<u8>>` channel into which guest-emitted chunks are
/// pushed; the API gateway holds the matching receiver and forwards the
/// bytes onto the wire.
///
/// Cloning is cheap (a refcount bump on each of the two `Arc` fields).
/// The atomic byte-counter is shared across clones so the
/// per-invocation cap remains a single source of truth even if the
/// context is duplicated across host-fn closures.
#[derive(Debug, Clone)]
pub struct StreamingContext {
    /// `None` means streaming is disabled for this invocation — every
    /// `emit_chunk` returns the `-1` error code immediately. `Some(_)`
    /// means a downstream receiver is attached.
    sender: Option<Arc<mpsc::Sender<Vec<u8>>>>,
    /// Running total of bytes successfully accepted (i.e. forwarded
    /// onto the channel). Compared against `max_total` on every emit.
    bytes_emitted: Arc<AtomicU64>,
    /// Cap on `bytes_emitted`. Zero when streaming is disabled — emit
    /// returns `-1` before this is consulted, but a zero cap is the
    /// safe scaffold default for the disabled path.
    max_total: u64,
}

impl StreamingContext {
    /// Construct a context with streaming disabled. Every `emit_chunk`
    /// returns `-1`; useful as the default for invocations the gateway
    /// did not opt into the streaming response path.
    pub fn disabled() -> Self {
        Self {
            sender: None,
            bytes_emitted: Arc::new(AtomicU64::new(0)),
            max_total: 0,
        }
    }

    /// Construct a context wired to `sender`. Bytes pushed via
    /// `emit_chunk` are forwarded onto the channel; the gateway-side
    /// receiver drives the SSE / chunked response body.
    ///
    /// The total-bytes cap defaults to [`MAX_TOTAL_STREAM_BYTES`] —
    /// embedders that need a different cap should construct the context
    /// directly via [`Self::with_channel_and_cap`].
    pub fn with_channel(sender: mpsc::Sender<Vec<u8>>) -> Self {
        Self::with_channel_and_cap(sender, MAX_TOTAL_STREAM_BYTES)
    }

    /// Construct a context wired to `sender` with an explicit total-bytes
    /// cap. Intended primarily for tests that exercise the cap branch
    /// without having to emit 64 MiB of data.
    pub fn with_channel_and_cap(sender: mpsc::Sender<Vec<u8>>, max_total: u64) -> Self {
        Self {
            sender: Some(Arc::new(sender)),
            bytes_emitted: Arc::new(AtomicU64::new(0)),
            max_total,
        }
    }

    /// Forward `bytes` to the downstream receiver if streaming is
    /// enabled.
    ///
    /// Returns:
    /// * `0` on success (bytes accepted). The chunk's contribution is
    ///   added to [`Self::bytes_emitted`].
    /// * `-1` if streaming is disabled (no channel attached).
    /// * `-2` if accepting this chunk would push the total past
    ///   [`Self::max_total`]. The counter is rolled back so the
    ///   per-invocation total reflects only successfully forwarded
    ///   bytes.
    /// * `-3` if the receiver has been dropped — the downstream HTTP
    ///   client disconnected. The counter is rolled back symmetrically.
    pub async fn emit_chunk(&self, bytes: Vec<u8>) -> i32 {
        let Some(s) = &self.sender else {
            return -1;
        };
        let added = bytes.len() as u64;
        // Optimistic-add + rollback-on-failure keeps the success path a
        // single atomic op while preserving an accurate running total
        // across the cap and receiver-dropped failure branches.
        let new_total = self.bytes_emitted.fetch_add(added, Ordering::SeqCst) + added;
        if new_total > self.max_total {
            self.bytes_emitted.fetch_sub(added, Ordering::SeqCst);
            return -2;
        }
        match s.send(bytes).await {
            Ok(_) => 0,
            Err(_) => {
                // Receiver dropped — roll back the accounting so a
                // subsequent attempt (if the gateway swaps in a new
                // receiver) sees an accurate total.
                self.bytes_emitted.fetch_sub(added, Ordering::SeqCst);
                -3
            }
        }
    }

    /// Flush any buffered chunks. The scaffold uses an unbuffered
    /// `mpsc::Sender` whose `send` already delivers per-call, so this
    /// is a no-op. v0.4 may introduce a coalescing buffer in front of
    /// the channel; the flush hook is reserved for that.
    ///
    /// Returns `0` on success, `-1` when streaming is disabled.
    pub fn flush(&self) -> i32 {
        if self.sender.is_none() {
            return -1;
        }
        0
    }

    /// Snapshot of bytes successfully emitted so far. Reads through
    /// [`Ordering::SeqCst`] for symmetry with [`Self::emit_chunk`].
    pub fn bytes_emitted(&self) -> u64 {
        self.bytes_emitted.load(Ordering::SeqCst)
    }

    /// `true` if a receiver channel is attached (i.e. the invocation
    /// was dispatched through the streaming response path).
    pub fn is_enabled(&self) -> bool {
        self.sender.is_some()
    }
}

/// Trait implemented by store data types that can hand out a
/// [`StreamingContext`]. Parallels [`crate::host::HasWasiCuda`]; the
/// executor's `InstanceState` will implement this once the v0.4 wiring
/// lands.
pub trait HasStreaming {
    /// Borrow the streaming context.
    fn streaming(&self) -> &StreamingContext;
}

/// Module name used to register the wasi-tensor host functions.
///
/// The on-the-wire string matches the WIT package name in
/// `wit/wasi-tensor.wit` so wit-bindgen-generated guest bindings resolve
/// the imports against this module.
pub const STREAMING_MODULE: &str = "wasi:tensor/host";

/// Host-function name for `emit-chunk`.
pub const FN_EMIT_CHUNK: &str = "emit-chunk";

/// Host-function name for `flush`.
pub const FN_FLUSH: &str = "flush";

/// Register the wasi-tensor host functions on a wasmtime `Linker`.
///
/// `T` is the store data type and must implement [`HasStreaming`].
///
/// The host-fn wrappers registered here read the `(buf_ptr, buf_len)`
/// argument pair from the guest, bounds-check the region against the
/// guest's linear memory exported as `"memory"`, then forward the bytes
/// into [`StreamingContext::emit_chunk`].
///
/// Single-chunk size cap: emits whose `buf_len` exceeds
/// [`MAX_CHUNK_BYTES`] return `-2` (cap exceeded) without touching the
/// channel — chunks above the cap would interleave badly with the
/// downstream SSE / chunked-transfer framing, so we refuse them at the
/// boundary. The total-bytes cap is still enforced inside
/// [`StreamingContext::emit_chunk`] on top of this per-call check.
///
/// Returns the same `wasmtime::Result` shape as [`crate::host::add_to_linker`]
/// for parity. Idempotency: registering the same `(module, name)`
/// twice on the same `Linker` is an error — callers should construct a
/// fresh `Linker` per build, just as the existing wasi-cuda registration
/// expects.
pub fn add_streaming_to_linker<T: HasStreaming + Send + 'static>(
    linker: &mut Linker<T>,
) -> wasmtime::Result<()> {
    // `func_wrap_async` mirrors the wasi-cuda launch path: emit_chunk
    // performs an `.await` on the `mpsc::Sender`, so it must run on a
    // wasmtime async fiber. Even a synchronous wrapper would have to
    // poll the future to completion; using `func_wrap_async` keeps the
    // backpressure semantics honest.
    linker.func_wrap_async(
        STREAMING_MODULE,
        FN_EMIT_CHUNK,
        |mut caller: Caller<'_, T>,
         (buf_ptr, buf_len): (i32, i32)|
         -> Box<dyn std::future::Future<Output = i32> + Send + '_> {
            // Synchronously: validate the (ptr, len) pair, copy the
            // bytes out of guest linear memory, and clone the streaming
            // context out of store data. The `Caller`'s memory borrow
            // cannot survive an `.await` (wasmtime's `Memory::data`
            // borrow may be invalidated by any await on a different
            // host fn or by guest re-entry), so we materialise the
            // `Vec<u8>` up front and emit on the cloned context.
            //
            // Cloning `StreamingContext` is two refcount bumps — see
            // the type's `Clone` impl — so the cost is negligible
            // compared to the byte copy.
            let prep = prepare_emit_chunk(&mut caller, buf_ptr, buf_len);
            Box::new(async move {
                match prep {
                    Ok((bytes, ctx)) => ctx.emit_chunk(bytes).await,
                    Err(code) => code,
                }
            })
        },
    )?;

    linker.func_wrap(STREAMING_MODULE, FN_FLUSH, |caller: Caller<'_, T>| -> i32 {
        caller.data().streaming().flush()
    })?;

    Ok(())
}

/// Synchronous preamble for [`add_streaming_to_linker`]'s `emit-chunk`
/// host function. Validates the `(buf_ptr, buf_len)` pair against the
/// guest's exported `"memory"`, copies the bytes out of linear memory,
/// and clones the [`StreamingContext`] out of store data.
///
/// Returns `Err(-2)` (cap-exceeded / invalid pointer) on any failure
/// path — both `MAX_CHUNK_BYTES` overflow and out-of-bounds region
/// surface the same documented `-2` code so the guest cannot
/// fingerprint the host's memory layout from the return value. The
/// disabled / receiver-dropped branches (`-1` / `-3`) are reached only
/// once the bytes have been forwarded onto the channel.
fn prepare_emit_chunk<T: HasStreaming>(
    caller: &mut Caller<'_, T>,
    buf_ptr: i32,
    buf_len: i32,
) -> Result<(Vec<u8>, StreamingContext), i32> {
    if buf_len < 0 || buf_ptr < 0 {
        return Err(-2);
    }
    if (buf_len as usize) > MAX_CHUNK_BYTES {
        return Err(-2);
    }
    let memory = caller
        .get_export("memory")
        .and_then(|e| e.into_memory())
        .ok_or(-2_i32)?;
    let start = buf_ptr as usize;
    let end = start.checked_add(buf_len as usize).ok_or(-2_i32)?;
    // `Memory::data` returns a slice tied to the borrow of the
    // store, so we scope the borrow tightly and copy out before the
    // subsequent `caller.data()` call re-borrows for the
    // streaming-context clone. Mirrors the pattern used by
    // `wasi-cuda`'s `read_bytes` helper in `src/host.rs`.
    let bytes: Vec<u8> = {
        let data = memory.data(&caller);
        if end > data.len() {
            return Err(-2);
        }
        data[start..end].to_vec()
    };
    let ctx = caller.data().streaming().clone();
    Ok((bytes, ctx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn disabled_context_returns_minus_one() {
        let ctx = StreamingContext::disabled();
        assert!(!ctx.is_enabled());
        assert_eq!(ctx.emit_chunk(vec![1, 2, 3]).await, -1);
        assert_eq!(ctx.flush(), -1);
        assert_eq!(ctx.bytes_emitted(), 0);
    }

    #[tokio::test]
    async fn enabled_context_forwards_bytes() {
        let (tx, mut rx) = mpsc::channel::<Vec<u8>>(4);
        let ctx = StreamingContext::with_channel(tx);
        assert!(ctx.is_enabled());
        assert_eq!(ctx.emit_chunk(vec![0xAA, 0xBB, 0xCC]).await, 0);
        assert_eq!(ctx.bytes_emitted(), 3);
        let chunk = rx.recv().await.expect("chunk delivered");
        assert_eq!(chunk, vec![0xAA, 0xBB, 0xCC]);
        assert_eq!(ctx.flush(), 0);
    }
}
