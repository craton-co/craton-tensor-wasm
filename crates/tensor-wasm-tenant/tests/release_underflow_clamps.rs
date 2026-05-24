// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! `release_bytes` must saturate on underflow and emit a warning.
//!
//! Releasing more bytes than were ever consumed is a bookkeeping bug,
//! but it must not panic, wrap the counter into the high u64 range, or
//! silently corrupt subsequent quota accounting. The fix replaces the
//! historical CAS loop with `fetch_sub` + post-hoc clamp; this test
//! pins both halves of that behaviour:
//!
//! 1. Counter ends at exactly 0 after the overshoot.
//! 2. A `tracing::warn!` event is emitted on the
//!    `tensor_wasm_tenant::context` target with the expected `before` and
//!    `bytes` fields, so operators can chase the bug upstream.

use std::sync::{Arc, Mutex};

use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::TenantContext;
use tracing::field::{Field, Visit};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

/// Captured copy of a single tracing event we care about.
#[derive(Debug, Default, Clone)]
struct CapturedWarn {
    target: String,
    message: Option<String>,
    before: Option<u64>,
    bytes: Option<u64>,
}

#[derive(Default)]
struct WarnCapture {
    events: Arc<Mutex<Vec<CapturedWarn>>>,
}

impl WarnCapture {
    fn handle(&self) -> Arc<Mutex<Vec<CapturedWarn>>> {
        Arc::clone(&self.events)
    }
}

impl<S> Layer<S> for WarnCapture
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        if *event.metadata().level() != tracing::Level::WARN {
            return;
        }
        let mut cap = CapturedWarn {
            target: event.metadata().target().to_string(),
            ..Default::default()
        };
        struct V<'a>(&'a mut CapturedWarn);
        impl Visit for V<'_> {
            fn record_u64(&mut self, field: &Field, value: u64) {
                match field.name() {
                    "before" => self.0.before = Some(value),
                    "bytes" => self.0.bytes = Some(value),
                    _ => {}
                }
            }
            fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0.message = Some(format!("{value:?}"));
                }
            }
        }
        event.record(&mut V(&mut cap));
        self.events.lock().unwrap().push(cap);
    }
}

#[test]
fn release_more_than_consumed_clamps_to_zero_and_warns() {
    let capture = WarnCapture::default();
    let handle = capture.handle();

    let subscriber = tracing_subscriber::registry().with(capture);
    tracing::subscriber::with_default(subscriber, || {
        let ctx = TenantContext::builder(TenantId(1))
            .with_memory_quota_bytes(1024)
            .build();
        ctx.consume_bytes(100).expect("consume");
        assert_eq!(ctx.bytes_in_use(), 100);

        // Over-release by 100 bytes (200 released vs 100 consumed).
        ctx.release_bytes(200);

        // Counter is clamped to zero, NOT wrapped to u64::MAX - 100.
        assert_eq!(
            ctx.bytes_in_use(),
            0,
            "underflow must clamp to 0 (saturating)",
        );

        // Subsequent quota accounting still works after the underflow.
        ctx.consume_bytes(50).expect("post-underflow consume");
        assert_eq!(ctx.bytes_in_use(), 50);
    });

    let events = handle.lock().unwrap();
    let underflow = events
        .iter()
        .find(|e| {
            e.target == "tensor_wasm_tenant::context"
                && e.message
                    .as_deref()
                    .map(|m| m.contains("release_bytes underflow clamped"))
                    .unwrap_or(false)
        })
        .expect("expected a warn event from release_bytes on underflow");

    assert_eq!(
        underflow.before,
        Some(100),
        "warn must report the observed pre-update value",
    );
    assert_eq!(
        underflow.bytes,
        Some(200),
        "warn must report the requested release size",
    );
}
