// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Behavioural tests for [`TensorWasmMemoryCreator`] / [`TensorWasmLinearMemory`]:
//!
//! - pool exhaustion falls back to a fresh `UnifiedBuffer` (no failure);
//! - `grow_to(current_size)` is a no-op success;
//! - a creator whose `device_id` disagrees with its pool's logs a `WARN`
//!   when the pool is exhausted and the fallback path runs on the
//!   creator's (different) device;
//! - the `From<UnifiedError> for TensorWasmError` conversion turns a
//!   structured `UnifiedError::TooLarge` into `TensorWasmError::MemoryExhausted`
//!   carrying the real `requested` / `limit` figures (no substring matching).

use std::sync::{Arc, Mutex};

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_mem::pool::UnifiedMemoryPool;
use tensor_wasm_mem::unified::{DeviceId, UnifiedError};
use tensor_wasm_mem::wasm_memory::{TensorWasmLinearMemory, TensorWasmMemoryCreator};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;
use wasmtime::{LinearMemory, MemoryCreator, MemoryType};

/// Capture every `WARN`-level event seen while a `tracing::subscriber::with_default`
/// guard is active. We only record the formatted message body — that is
/// enough to assert on the device-mismatch warning.
#[derive(Default, Clone)]
struct WarnCapture {
    events: Arc<Mutex<Vec<String>>>,
}

impl<S> Layer<S> for WarnCapture
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        if *event.metadata().level() != Level::WARN {
            return;
        }
        // Render the event's fields into a single string so the test can
        // grep for the warning text without dragging in a full fmt layer.
        struct Visitor<'a> {
            out: &'a mut String,
        }
        impl<'a> tracing::field::Visit for Visitor<'a> {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                use std::fmt::Write;
                let _ = write!(self.out, " {}={:?}", field.name(), value);
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                use std::fmt::Write;
                let _ = write!(self.out, " {}={}", field.name(), value);
            }
        }
        let mut rendered = String::new();
        rendered.push_str(event.metadata().target());
        let mut v = Visitor { out: &mut rendered };
        event.record(&mut v);
        self.events.lock().unwrap().push(rendered);
    }
}

#[test]
fn pool_exhaustion_fallback() {
    // Pool sized for exactly one 64 KiB Wasm page; ask for two — second one
    // must fall back to a fresh UnifiedBuffer rather than fail.
    const PAGE: usize = 65_536;
    let pool = Arc::new(UnifiedMemoryPool::new(PAGE).expect("pool"));
    let creator = TensorWasmMemoryCreator::with_pool(DeviceId::default(), pool.clone());

    // First allocation fills the slab exactly.
    let first = creator
        .new_memory(MemoryType::new(1, Some(1)), PAGE, Some(PAGE), None, 0)
        .expect("first new_memory must succeed from the pool");
    assert_eq!(first.byte_size(), PAGE);
    assert_eq!(pool.live_allocations(), 1, "first carved from pool");

    // Second allocation cannot fit — must transparently fall back.
    let second = creator
        .new_memory(MemoryType::new(1, Some(1)), PAGE, Some(PAGE), None, 0)
        .expect("second new_memory must succeed via fallback");
    assert_eq!(second.byte_size(), PAGE);
    // The pool's live count must NOT have incremented — the fallback path
    // does not touch the pool counter.
    assert_eq!(
        pool.live_allocations(),
        1,
        "fallback allocation must not bump the pool's live count"
    );
}

#[test]
fn grow_to_equal_current_is_noop_success() {
    let mut mem = TensorWasmLinearMemory::new(128 * 1024, Some(1024 * 1024)).expect("alloc");
    let before = mem.byte_size();
    mem.grow_to(before).expect("grow_to(current) must succeed");
    assert_eq!(mem.byte_size(), before, "size must be unchanged");
}

#[test]
fn creator_with_pool_device_mismatch_warns_on_fallback() {
    let capture = WarnCapture::default();
    let events_handle = capture.events.clone();

    // Pool on device 0; creator targets device 7 — and the pool is too
    // small to carve a single Wasm page so the very first allocation
    // exhausts it and trips the fallback path that emits the warning.
    let pool = Arc::new(UnifiedMemoryPool::new_on(1024, DeviceId(0)).expect("pool on device 0"));
    let creator = TensorWasmMemoryCreator::with_pool(DeviceId(7), pool);

    let subscriber = tracing_subscriber::registry().with(capture);
    tracing::subscriber::with_default(subscriber, || {
        let mt = MemoryType::new(1, Some(1));
        // 64 KiB exceeds the 1 KiB slab — pool.allocate fails, fallback runs.
        let _mem = creator
            .new_memory(mt, 65_536, Some(65_536), None, 0)
            .expect("fallback must succeed");
    });

    let events = events_handle.lock().unwrap();
    let mismatch_seen = events.iter().any(|e| {
        e.contains("differs from the exhausted pool")
            && e.contains("creator_device_id")
            && e.contains("pool_device_id")
    });
    assert!(
        mismatch_seen,
        "expected device-mismatch warning; saw events: {events:?}"
    );
}

#[test]
fn unified_error_too_large_maps_to_memory_exhausted_with_figures() {
    // Exhaustion is reported via the structured `TooLarge` variant; the
    // `From` impl plumbs `requested` / `limit` straight through to
    // `MemoryExhausted` (no string parsing, no zeroed placeholders).
    let e = UnifiedError::TooLarge {
        requested: 4096,
        limit: 1024,
    };
    let b: TensorWasmError = e.into();
    match b {
        TensorWasmError::MemoryExhausted { requested, limit } => {
            assert_eq!(requested, 4096);
            assert_eq!(limit, 1024);
        }
        other => panic!("expected MemoryExhausted, got {other:?}"),
    }
}

#[test]
fn unified_error_allocation_falls_through_to_serialization() {
    // Any `Allocation` payload reaching the conversion is a caller bug
    // (bad alignment, `minimum > maximum`, etc.); exhaustion goes through
    // `TooLarge`. The detail string is forwarded into `Serialization`.
    let e = UnifiedError::Allocation("minimum 8 > maximum 4".into());
    let b: TensorWasmError = e.into();
    assert!(matches!(b, TensorWasmError::Serialization(_)));
    // `TensorWasmError`'s Display is sanitised and omits the inner detail; the
    // forwarded context is reachable via `inner()`.
    assert!(b.inner().unwrap_or("").contains("minimum 8"));
}

#[test]
fn new_memory_rejects_non_zero_guard_size() {
    let creator = TensorWasmMemoryCreator::default();
    let mt = MemoryType::new(1, Some(1));
    // `Box<dyn LinearMemory>` is not `Debug`, so `expect_err` won't compile —
    // unwrap the error manually instead.
    let err = match creator.new_memory(mt, 65_536, Some(65_536), None, 4096) {
        Ok(_) => panic!("non-zero guard_size_in_bytes must be rejected"),
        Err(e) => e,
    };
    assert!(
        err.contains("guard_size_in_bytes"),
        "error must mention the offending knob, got: {err}"
    );
}

#[test]
fn new_memory_rejects_oversized_reservation() {
    let creator = TensorWasmMemoryCreator::default();
    let mt = MemoryType::new(1, Some(1));
    // Ask for 64 KiB max but reserve 1 MiB — impossible to satisfy.
    let err = match creator.new_memory(mt, 65_536, Some(65_536), Some(1024 * 1024), 0) {
        Ok(_) => panic!("reservation > capacity must be rejected"),
        Err(e) => e,
    };
    assert!(
        err.contains("reserve"),
        "error must mention reservation, got: {err}"
    );
}
