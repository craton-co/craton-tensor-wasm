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
//! Mirrors the WIT contract in `wit/wasi-tensor.wit`. Acceptance is
//! all-or-nothing — the host forwards the whole chunk or none of it, so
//! success is a flat `0` rather than a byte count:
//! * `0`    — chunk fully accepted (the entire payload was forwarded).
//! * `-1`   — streaming not enabled for this invocation.
//! * `-2`   — guest tried to emit past the documented size cap.
//! * `-3`   — downstream client disconnected (receiver dropped).
//!
//! Tests in `tests/streaming_scaffold.rs` exercise every branch.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use wasmtime::{Caller, Linker};

use crate::abi::AbiError;

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
    ///   `max_total`. The counter is rolled back afterward.
    /// * `-3` if the receiver has been dropped — the downstream HTTP
    ///   client disconnected. The counter is rolled back symmetrically.
    ///
    /// Concurrency note: the running total is maintained with an
    /// optimistic `fetch_add` followed by a rollback on the failure
    /// branches, so it reflects only successfully forwarded bytes *once
    /// quiescent*. While concurrent emits are in flight, a reader of
    /// [`Self::bytes_emitted`] may briefly observe a value above the bytes
    /// actually forwarded — each in-flight emit can overshoot by its own
    /// chunk size before its rollback lands, so with the per-call cap the
    /// transient overshoot is bounded by roughly `2 * MAX_CHUNK_BYTES` per
    /// racing emit. The overshoot is reconciled (rolled back) before each
    /// failing call returns; it never causes the cap to under-count
    /// accepted bytes.
    pub async fn emit_chunk(&self, bytes: Vec<u8>) -> i32 {
        let Some(s) = &self.sender else {
            return -1;
        };
        let added = bytes.len() as u64;
        // Optimistic-add + rollback-on-failure keeps the success path a
        // single atomic op while keeping the running total accurate once
        // quiescent. Under concurrency the total can transiently overshoot
        // (each racing emit adds its chunk size before any rollback lands),
        // bounded by ~2*MAX_CHUNK_BYTES per racing emit — see the
        // `emit_chunk` doc comment.
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

/// Module name used to register the wasi-tensor host functions:
/// `wasi:tensor/host@0.1.0`.
///
/// The on-the-wire string carries the `@x.y.z` version segment so it
/// matches the `package wasi:tensor@0.1.0;` declaration in
/// `wit/wasi-tensor.wit`. wit-bindgen-generated guests for
/// `wasi:tensor@0.1.0` emit imports named `wasi:tensor/host@0.1.0`, so the
/// version segment is required for those bindings to resolve against this
/// host module. The version is kept in lockstep with the WIT package
/// declaration — bumping one without the other will cause guests generated
/// from the WIT to fail to link, mirroring the `wasi:cuda/host@0.2.0`
/// ([`crate::abi::MODULE`]) and `wasi:scheduler/host@0.1.0`
/// ([`crate::scheduler::SCHEDULER_MODULE`]) conventions.
pub const STREAMING_MODULE: &str = "wasi:tensor/host@0.1.0";

/// Host-function name for `emit-chunk`.
pub const FN_EMIT_CHUNK: &str = "emit-chunk";

/// Host-function name for `flush`.
pub const FN_FLUSH: &str = "flush";

/// Host-function name for `input-len`.
pub const FN_INPUT_LEN: &str = "input-len";

/// Host-function name for `read-input`.
pub const FN_READ_INPUT: &str = "read-input";

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

// ---------------------------------------------------------------------------
// Guest input channel (pull model): `input-len` / `read-input`
// ---------------------------------------------------------------------------

/// Per-invocation input context. Owns the bytes the host staged for the
/// guest to pull via the `wasi:tensor/host` `input-len` / `read-input`
/// host functions.
///
/// This is the input counterpart to [`StreamingContext`] (which is
/// output-only, guest → host). The gateway stages the request payload —
/// e.g. the OpenAI completions shim stages the assembled prompt bytes —
/// before the guest runs, and the guest copies them into its own linear
/// memory.
///
/// Cloning is cheap: the bytes live behind an `Arc<[u8]>`, so a clone is
/// a single refcount bump. An empty context (`InputContext::empty`) is
/// the default for invocations the gateway did not stage input for —
/// `len()` is `0` and `read-input` copies nothing.
#[derive(Debug, Clone, Default)]
pub struct InputContext {
    /// Bytes staged for the guest. Empty by default.
    bytes: Arc<[u8]>,
}

impl InputContext {
    /// Construct an empty context. `len()` returns `0`; `read-input`
    /// copies nothing. This is the default for invocations the gateway
    /// did not stage any input for.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Construct a context wrapping the given staged input bytes.
    pub fn new(bytes: impl Into<Arc<[u8]>>) -> Self {
        Self {
            bytes: bytes.into(),
        }
    }

    /// Borrow the staged input bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Number of staged input bytes. Clamped to `u32::MAX` so the value
    /// fits the WIT `input-len() -> u32` return type — a staged buffer
    /// above 4 GiB is implausible (the API body-size cap is far below
    /// that) but we saturate rather than wrap on the cast.
    pub fn len_u32(&self) -> u32 {
        u32::try_from(self.bytes.len()).unwrap_or(u32::MAX)
    }

    /// `true` when no input bytes are staged.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

/// Trait implemented by store data types that can hand out an
/// [`InputContext`]. Parallels [`HasStreaming`]; the executor's
/// `InstanceState` implements it so the `input-len` / `read-input` host
/// functions reach the per-instance staged buffer.
pub trait HasInput {
    /// Borrow the input context.
    fn input(&self) -> &InputContext;
}

/// Register the `wasi:tensor/host` guest-input host functions
/// (`input-len`, `read-input`) on a wasmtime `Linker`.
///
/// `T` is the store data type and must implement [`HasInput`]. Mirrors
/// [`add_streaming_to_linker`]: both register host functions on the same
/// [`STREAMING_MODULE`] (`wasi:tensor/host@0.1.0`) so a guest importing
/// either surface links against one host module. They are split into two
/// registration functions because the input surface needs only
/// [`HasInput`] while the streaming surface needs [`HasStreaming`]; the
/// executor's `InstanceState` implements both and calls both.
///
/// `read-input` bounds-checks the `(ptr, len)` destination region against
/// the guest's exported `"memory"` with the same `checked_add` pattern as
/// `prepare_emit_chunk` and the wasi-cuda `read_bytes` path, returning
/// [`AbiError::InvalidPointer`] (`-2`) on any out-of-bounds / overflow /
/// missing-memory failure. A successful copy returns the number of bytes
/// written (`min(len, input-len())`); `0` means "nothing staged" or
/// `len == 0`, never an error.
pub fn add_input_to_linker<T: HasInput + Send + 'static>(
    linker: &mut Linker<T>,
) -> wasmtime::Result<()> {
    linker.func_wrap(
        STREAMING_MODULE,
        FN_INPUT_LEN,
        |caller: Caller<'_, T>| -> u32 { caller.data().input().len_u32() },
    )?;

    linker.func_wrap(
        STREAMING_MODULE,
        FN_READ_INPUT,
        |mut caller: Caller<'_, T>, ptr: u32, len: u32| -> i32 {
            // Clone the staged bytes out of store data before borrowing
            // memory mutably for the write. The `Arc<[u8]>` clone is a
            // refcount bump.
            let input = caller.data().input().clone();
            let staged = input.bytes();
            if staged.is_empty() || len == 0 {
                return 0;
            }
            let to_copy = std::cmp::min(staged.len(), len as usize);
            let memory = match caller.get_export("memory").and_then(|e| e.into_memory()) {
                Some(m) => m,
                None => return AbiError::InvalidPointer.code(),
            };
            // Bounds-check the destination region against the current
            // linear-memory size with the same `checked_add` guard the
            // emit-chunk / read_bytes paths use: without it a guest could
            // pass `(ptr = u32::MAX - 1, len = 4)` and wrap to a small
            // `end` that looks in-bounds.
            let start = ptr as usize;
            let end = match start.checked_add(to_copy) {
                Some(e) => e,
                None => return AbiError::InvalidPointer.code(),
            };
            let mem_len = memory.data(&caller).len();
            if end > mem_len {
                return AbiError::InvalidPointer.code();
            }
            // Copy out of the Arc-backed buffer into an owned Vec so the
            // subsequent mutable `memory.write` borrow does not alias the
            // `input` borrow (mirrors `last_error_copy` in `host.rs`).
            let buf = staged[..to_copy].to_vec();
            if memory.write(&mut caller, start, &buf).is_err() {
                return AbiError::InvalidPointer.code();
            }
            // `to_copy <= len <= u32::MAX` and `to_copy <= staged.len()`,
            // and the WIT contract caps a single read at `input-len()`
            // which is itself clamped to `u32::MAX`, so the `as i32` cast
            // is non-negative for any realistic buffer. Guard the
            // (implausible) >2 GiB case so we never hand back a negative
            // count that a guest would read as an error.
            i32::try_from(to_copy).unwrap_or(i32::MAX)
        },
    )?;

    Ok(())
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

    #[test]
    fn input_context_empty_by_default() {
        let ctx = InputContext::empty();
        assert!(ctx.is_empty());
        assert_eq!(ctx.len_u32(), 0);
        assert_eq!(ctx.bytes(), b"");
        // Default impl agrees with `empty()`.
        assert!(InputContext::default().is_empty());
    }

    #[test]
    fn input_context_carries_staged_bytes() {
        let ctx = InputContext::new(b"hello prompt".to_vec());
        assert!(!ctx.is_empty());
        assert_eq!(ctx.len_u32(), 12);
        assert_eq!(ctx.bytes(), b"hello prompt");
        // Cheap clone shares the same backing buffer.
        let cloned = ctx.clone();
        assert_eq!(cloned.bytes(), ctx.bytes());
    }

    #[test]
    fn streaming_module_is_versioned() {
        assert!(STREAMING_MODULE.contains("wasi:tensor"));
        assert!(STREAMING_MODULE.ends_with("@0.1.0"));
    }

    /// Pin the host [`STREAMING_MODULE`] string against drift from
    /// `wit/wasi-tensor.wit`.
    ///
    /// Mirrors `abi.rs::module_version_matches_wit_package_decl`. The WIT
    /// file is the authoritative spec; the host's import-module name has to
    /// carry the same `@x.y.z` segment so wit-bindgen-generated guests for
    /// `wasi:tensor@x.y.z` (which emit imports named
    /// `wasi:tensor/host@x.y.z`) can link against this host. Parse the
    /// version out of the WIT `package wasi:tensor@x.y.z;` line and compare
    /// it to the suffix of [`STREAMING_MODULE`]. If somebody bumps one
    /// without the other, this trips before the linker error reaches
    /// downstream users.
    #[test]
    fn streaming_module_version_matches_wit_package_decl() {
        // Path is relative to this source file
        // (`crates/tensor-wasm-wasi-gpu/src/streaming.rs`):
        //   ../ -> crates/tensor-wasm-wasi-gpu/, where the in-crate
        //          `wit/wasi-tensor.wit` copy lives. Kept in-crate so the
        //          published tarball is self-contained (`cargo publish`
        //          rejects paths that escape the crate root).
        const WIT: &str = include_str!("../wit/wasi-tensor.wit");
        let pkg_line = WIT
            .lines()
            .find(|l| l.trim_start().starts_with("package wasi:tensor@"))
            .expect("wit/wasi-tensor.wit must declare `package wasi:tensor@x.y.z;`");
        let version = pkg_line
            .trim()
            .trim_start_matches("package wasi:tensor@")
            .trim_end_matches(';')
            .trim();
        assert!(
            !version.is_empty(),
            "could not parse a version out of: {pkg_line:?}"
        );
        let expected_suffix = format!("@{version}");
        assert!(
            STREAMING_MODULE.ends_with(&expected_suffix),
            "STREAMING_MODULE ({STREAMING_MODULE:?}) drifted from \
             wit/wasi-tensor.wit package version ({version:?}); update \
             src/streaming.rs::STREAMING_MODULE or the WIT file so they agree."
        );
    }
}
