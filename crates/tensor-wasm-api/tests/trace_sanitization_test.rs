// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Unit-level coverage for the path and method sanitisers used by the
//! `http.request` tracing span in [`tensor_wasm_api::middleware`].
//!
//! These helpers run on every inbound request, so each one is on the
//! hot path for log injection / dashboard-pollution attacks:
//!
//! * `sanitize_path` is fed `req.uri().path()` — attacker-controlled
//!   ASCII (and worse, percent-decoded bytes once axum's router has
//!   matched a route). Path-traversal probes
//!   (`/functions/../etc/passwd`) and CRLF-injection payloads
//!   (`/functions/foo%0d%0aevil-header:%20yes`) end up here verbatim
//!   unless the helper sanitises them.
//! * `normalize_method` is fed `req.method().as_str()`. hyper rejects
//!   most malformed values before this point, but a method-label
//!   cardinality blow-up (`Get`, `gEt`, `GET\r\n`, …) on the
//!   `tensor_wasm_http_requests_total` series would still degrade
//!   dashboards. We normalise so only `[A-Z]{1,16}` reaches the span.
//!
//! Integration-level proof that the helpers actually take effect in
//! the live `tracing::info_span!` macro lives in
//! `trace_span_does_not_leak_query.rs`; the cases here cover the
//! pure-function contract documented in `middleware.rs`.

#![allow(clippy::expect_used)]

use std::borrow::Cow;

use tensor_wasm_api::{normalize_method, sanitize_path, MAX_PATH_LEN};

#[test]
fn sanitize_path_strips_crlf() {
    // The classic log-injection vector: a request URI that contains
    // raw CR/LF. If `sanitize_path` ever forgets to filter these, a
    // hostile client can forge a second JSON log line in any sink
    // that escapes per-line rather than per-field. The assertion is
    // intentionally byte-level (`\r`, `\n`) rather than substring of
    // the original input so the test catches partial-filter
    // regressions (e.g. dropping only one of the two).
    let raw = "/api/v1\r\nevil:";
    let out = sanitize_path(raw);
    assert!(
        !out.contains('\r'),
        "sanitize_path must strip CR, got {out:?}"
    );
    assert!(
        !out.contains('\n'),
        "sanitize_path must strip LF, got {out:?}"
    );
    // The output should also be `Cow::Owned` because we had to
    // rewrite the bytes; a `Borrowed` here would mean the helper
    // claimed the input was clean and short-circuited.
    assert!(
        matches!(out, Cow::Owned(_)),
        "sanitize_path on CRLF-bearing input must return Cow::Owned, \
         got Borrowed (helper claimed input was clean): {out:?}"
    );
}

#[test]
fn sanitize_path_truncates_long_paths() {
    // 1000-byte all-`a` path. Pre-fix the helper would have shipped
    // every byte into the span attribute, ballooning every log line
    // that touched the request. Post-fix the helper truncates to
    // `MAX_PATH_LEN` (with room for the ellipsis suffix).
    //
    // The bound `MAX_PATH_LEN + 4` is the spec the agent prompt
    // pinned: 256 byte budget + up to 4 bytes of ellipsis padding.
    // The actual ellipsis is `…` (3 UTF-8 bytes); the extra byte
    // gives the helper room to evolve without breaking this test.
    let raw: String = format!("/{}", "a".repeat(1000));
    let out = sanitize_path(&raw);
    assert!(
        out.len() <= MAX_PATH_LEN + 4,
        "sanitize_path must truncate long paths, got {} bytes (budget {} + 4 ellipsis)",
        out.len(),
        MAX_PATH_LEN,
    );
    // And of course the value must actually have changed (Owned).
    assert!(
        matches!(out, Cow::Owned(_)),
        "sanitize_path on a 1000-byte input must return Cow::Owned"
    );
}

#[test]
fn sanitize_path_replaces_non_ascii() {
    // `é` is `0xC3 0xA9` in UTF-8. The pre-fix helper recorded these
    // bytes verbatim, which is fine for a JSON sink but unsafe for a
    // terminal viewer: a hostile path could contain ANSI-escape
    // bytes that move the cursor / repaint the screen. Post-fix the
    // helper replaces any byte outside `0x20..=0x7E` with `?`.
    let raw = "/café";
    let out = sanitize_path(raw);

    // Neither byte of the `é` UTF-8 encoding may survive into the
    // span attribute.
    let bytes = out.as_bytes();
    assert!(
        !bytes.contains(&0xC3),
        "sanitize_path leaked 0xC3 byte from UTF-8 `é`: {out:?}"
    );
    assert!(
        !bytes.contains(&0xA9),
        "sanitize_path leaked 0xA9 byte from UTF-8 `é`: {out:?}"
    );

    // The replacement character must actually appear so operators
    // see that *something* was stripped — silent truncation would
    // hide the attack.
    assert!(
        out.contains('?'),
        "sanitize_path on `/café` must surface a `?` placeholder, got {out:?}"
    );

    // Defensive: the printable ASCII prefix `/caf` should round-trip
    // unchanged.
    assert!(
        out.starts_with("/caf"),
        "sanitize_path mangled the printable prefix `/caf` of `/café`: {out:?}"
    );
}

#[test]
fn sanitize_path_passes_through_clean() {
    // The hot-path case: a clean, short, printable-ASCII path must
    // return `Cow::Borrowed` so we don't allocate per request. A
    // regression that always returned `Cow::Owned` here would land
    // an allocation in the critical span-construction path for
    // every single inbound request — measurable in p99 under load.
    let raw = "/healthz";
    let out = sanitize_path(raw);
    assert!(
        matches!(out, Cow::Borrowed(_)),
        "sanitize_path on clean input must return Cow::Borrowed, got Owned: {out:?}"
    );
    assert_eq!(&*out, raw, "sanitize_path mutated a clean input");
}

#[test]
fn method_normalizer_rejects_lowercase() {
    // A lowercase method name is the canonical "not standard"
    // signal: hyper accepts it on the wire but the W3C grammar and
    // every observability dashboard expects upper-case. We map any
    // non-`[A-Z]{1,16}` input to the literal `"OTHER"` sentinel so
    // the metrics label cardinality stays bounded.
    let out = normalize_method("get");
    assert_eq!(
        &*out, "OTHER",
        "normalize_method must collapse lowercase `get` to `OTHER`, got {out:?}"
    );
}
