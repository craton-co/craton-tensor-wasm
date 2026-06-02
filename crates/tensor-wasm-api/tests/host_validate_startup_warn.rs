// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration test for the T16 startup-safety warning:
//! `tensor_wasm_api::server::maybe_warn_host_validation_disabled`.
//!
//! ## Background (T16 finding)
//!
//! The `host_validate` middleware (`crate::middleware::host_validate`) is
//! a runtime no-op when `TENSOR_WASM_API_TRUSTED_HOSTS` is unset — every
//! `Host` header is admitted. That is the legacy behaviour and the safe
//! default for local-dev deployments where the operator usually has no
//! ingress in front of the gateway. Behind a misconfigured production
//! ingress that forwards arbitrary `Host` headers, however, the
//! pass-through default is silently exposed to virtual-host confusion
//! and cache-poisoning attacks. Operators rolling out the gateway with
//! `TENSOR_WASM_API_TOKENS` set (production-mode bearer auth) might
//! reasonably assume the rest of the safety surface is also active.
//!
//! T16 closes that gap by emitting a single `tracing::warn!` from the
//! router builder when production mode is on but the Host allowlist is
//! off. The runtime middleware behaviour stays unchanged — only the
//! operator-facing visibility is new.
//!
//! ## What this file covers
//!
//! * **`warns_when_tokens_set_but_trusted_hosts_unset`** — the asymmetric
//!   case the audit flagged. Capture a tracing layer, build the router
//!   under `temp_env::with_vars` with `TENSOR_WASM_API_TOKENS=test` and
//!   `TENSOR_WASM_API_TRUSTED_HOSTS` unset, and assert the warning event
//!   was recorded against the documented target.
//! * **`no_warn_when_both_vars_set`** — symmetric "all good" case: both
//!   env vars set, no warning is emitted. Pins the rule that the warn is
//!   conditional on the asymmetric configuration, not just on production
//!   mode.
//! * **`no_warn_when_neither_var_set`** — pure dev-mode: neither env
//!   var configured, the gateway is wide-open by design and the warn
//!   stays quiet. The dev-mode `TENSOR_WASM_API_TOKENS` warning is a
//!   different log line owned by `AuthConfig::from_env`.
//!
//! ## Why a custom capture layer instead of `tracing_test`
//!
//! `tracing-test` is not in the workspace's dev-deps and the crate
//! already depends on `tracing-subscriber` for the concurrent-load test
//! (`tests/trace_concurrent_load_test.rs`). Re-using `tracing-subscriber`
//! keeps the dep graph lean and matches the established pattern in this
//! repo for "capture events into a Vec for assertion" tests.
//!
//! ## Why each test resets the latch
//!
//! `tensor_wasm_api::server` keeps a process-wide `AtomicBool` so the
//! warning fires at most once per process. Without resetting it the
//! second test in this binary would observe zero warnings even if the
//! production-mode branch ran, because the first test would have
//! consumed the CAS. `__reset_host_validation_warn_for_test` is the
//! deliberately-ugly public-but-doc-hidden hook the server module
//! exposes for exactly this purpose.

#![allow(clippy::expect_used)]

use std::sync::{Arc, Mutex, Once};

use tensor_wasm_api::server::__reset_host_validation_warn_for_test;
use tensor_wasm_api::{build_router, AppState, ENV_API_TOKENS, ENV_TRUSTED_HOSTS};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

/// Target string the T16 warning is emitted against — matches the
/// `target: "tensor_wasm_api::server"` argument in `maybe_warn_host_validation_disabled`.
/// Pinned as a constant so a future refactor that changes the target
/// surfaces here as a compile-time-stable failure rather than a string
/// drift the test author has to chase down.
const T16_WARN_TARGET: &str = "tensor_wasm_api::server";

/// Substring that must appear in the formatted message body of the T16
/// warning. Picking a unique, low-churn fragment of the documented
/// message keeps the assertion resilient to wording tweaks while still
/// catching a regression that would drop the env-var name from the
/// operator-facing message.
const T16_MESSAGE_FRAGMENT: &str = "TENSOR_WASM_API_TRUSTED_HOSTS";

/// One captured warn-level tracing event. Stored in a shared
/// `Mutex<Vec<...>>` by [`CapturingLayer`] so the assertions can iterate
/// the recorded events without holding the subscriber lock.
#[derive(Debug, Clone)]
struct CapturedEvent {
    target: String,
    message: String,
}

/// `tracing_subscriber::Layer` that records every WARN-level event into
/// a shared `Vec<CapturedEvent>`. We do NOT filter on the target in the
/// layer itself — the per-test assertion filters — so that an
/// unexpected warn on a different target surfaces as visible debug
/// context rather than getting silently dropped.
#[derive(Clone)]
struct CapturingLayer {
    events: Arc<Mutex<Vec<CapturedEvent>>>,
}

impl<S> Layer<S> for CapturingLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        // Only care about WARN — INFO/DEBUG from the router builder
        // (e.g. the `install_w3c_propagator` path) would otherwise fill
        // the buffer and make message-substring assertions brittle.
        if *event.metadata().level() != tracing::Level::WARN {
            return;
        }
        let target = event.metadata().target().to_string();

        // Render the event's fields into a single string we can substring-
        // match on. `tracing::field::display` writes the message field
        // into the `Visit` impl; collecting into a `String` is the
        // standard pattern for log capture in tests.
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        let captured = CapturedEvent {
            target,
            message: visitor.message,
        };
        if let Ok(mut guard) = self.events.lock() {
            guard.push(captured);
        }
    }
}

/// `tracing::field::Visit` impl that concatenates every field's
/// `Debug` rendering into a single string. The standard `Visit` API
/// hands each field separately; we don't care about field names here,
/// only that the message body (and any structured fields) shows up in
/// the haystack our `contains(...)` assertion runs against.
#[derive(Default)]
struct MessageVisitor {
    message: String,
}

impl tracing::field::Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write;
        // The default `message` field is the format-args body — keep it
        // unprefixed for readable assertions. Other structured fields
        // get a `name=value` prefix so they still appear in the haystack.
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        } else {
            let _ = write!(self.message, " {}={value:?}", field.name());
        }
    }
}

/// Install the capturing subscriber globally, once per test binary.
///
/// `tracing::subscriber::set_global_default` (and the `try_init` helper
/// underneath the subscriber builder) panics if called twice from the
/// same process, so we serialise behind `Once`. The returned handle is
/// the shared `Vec` the layer pushes into — each test snapshots the
/// length before the action under test, runs the action, and asserts on
/// the suffix.
fn install_capturing_subscriber() -> Arc<Mutex<Vec<CapturedEvent>>> {
    static INIT: Once = Once::new();
    static EVENTS: std::sync::OnceLock<Arc<Mutex<Vec<CapturedEvent>>>> = std::sync::OnceLock::new();
    let events = EVENTS
        .get_or_init(|| Arc::new(Mutex::new(Vec::new())))
        .clone();
    INIT.call_once(|| {
        let layer = CapturingLayer {
            events: Arc::clone(&events),
        };
        // `try_init` rather than `init` so a sibling test that already
        // installed a subscriber (none today in this binary, but cheap
        // insurance against future cross-file globals) doesn't panic.
        let _ = tracing_subscriber::registry().with(layer).try_init();
    });
    events
}

/// Drain every event captured since the supplied baseline length. The
/// per-test pattern is: snapshot `events.lock().unwrap().len()` before
/// invoking `build_router`, then call this with that snapshot to get
/// only the events emitted by the call under test.
fn drain_new_events(
    events: &Arc<Mutex<Vec<CapturedEvent>>>,
    baseline: usize,
) -> Vec<CapturedEvent> {
    let guard = events.lock().expect("capture mutex poisoned");
    guard[baseline..].to_vec()
}

/// `true` when `events` contains at least one entry whose target is the
/// T16 target and whose message body carries the T16 fragment. Used by
/// the positive-case assertion below.
fn t16_warn_present(events: &[CapturedEvent]) -> bool {
    events
        .iter()
        .any(|e| e.target == T16_WARN_TARGET && e.message.contains(T16_MESSAGE_FRAGMENT))
}

#[test]
fn warns_when_tokens_set_but_trusted_hosts_unset() {
    let events = install_capturing_subscriber();

    // `temp_env::with_vars` mutates the process env synchronously and
    // restores it on closure exit. Pass `None` to scope the variable as
    // "unset for the duration", which is what we need for
    // `TENSOR_WASM_API_TRUSTED_HOSTS`. The router build path reads both
    // env vars via `std::env::var(...)`, so the closure body must hold
    // the env mutation across the `build_router` call.
    //
    // We do the latch reset, the baseline snapshot, the `build_router`
    // call, and the assertion ALL inside the closure. `temp_env` holds
    // its internal mutex across that closure, so the three tests in
    // this binary serialise on it and a parallel sibling cannot
    // (a) flip the latch between our reset and our build, or
    // (b) push unrelated capture events into the shared buffer between
    //     our baseline snapshot and our drain. Doing this outside the
    //     closure would race on both axes under Cargo's default
    //     parallel-test runner.
    temp_env::with_vars(
        [
            (ENV_API_TOKENS, Some("test")),
            (ENV_TRUSTED_HOSTS, None::<&str>),
        ],
        || {
            __reset_host_validation_warn_for_test();
            let baseline = events.lock().expect("baseline lock").len();
            let state = Arc::new(AppState::default());
            let _router = build_router(state);
            let new_events = drain_new_events(&events, baseline);
            assert!(
                t16_warn_present(&new_events),
                "expected T16 warn against target {T16_WARN_TARGET:?} containing \
                 {T16_MESSAGE_FRAGMENT:?}, got: {new_events:#?}",
            );
        },
    );
}

#[test]
fn no_warn_when_both_vars_set() {
    let events = install_capturing_subscriber();

    temp_env::with_vars(
        [
            (ENV_API_TOKENS, Some("test")),
            (ENV_TRUSTED_HOSTS, Some("api.example.com")),
        ],
        || {
            __reset_host_validation_warn_for_test();
            let baseline = events.lock().expect("baseline lock").len();
            let state = Arc::new(AppState::default());
            let _router = build_router(state);
            let new_events = drain_new_events(&events, baseline);
            assert!(
                !t16_warn_present(&new_events),
                "T16 warn must NOT fire when TENSOR_WASM_API_TRUSTED_HOSTS \
                 is set; got: {new_events:#?}",
            );
        },
    );
}

#[test]
fn no_warn_when_neither_var_set() {
    // Pure dev mode: neither TENSOR_WASM_API_TOKENS nor
    // TENSOR_WASM_API_TRUSTED_HOSTS is configured. The gateway is wide-
    // open by design and the T16 warning is conditional on production-
    // mode tokens being set, so it must stay quiet. The dev-mode
    // `AuthConfig::from_env` warning fires under a different target
    // (`tensor_wasm_api::middleware`) and is not the warning under
    // test here.
    let events = install_capturing_subscriber();

    temp_env::with_vars(
        [
            (ENV_API_TOKENS, None::<&str>),
            (ENV_TRUSTED_HOSTS, None::<&str>),
        ],
        || {
            __reset_host_validation_warn_for_test();
            let baseline = events.lock().expect("baseline lock").len();
            let state = Arc::new(AppState::default());
            let _router = build_router(state);
            let new_events = drain_new_events(&events, baseline);
            assert!(
                !t16_warn_present(&new_events),
                "T16 warn must NOT fire in pure dev mode (neither env var \
                 set); got: {new_events:#?}",
            );
        },
    );
}
