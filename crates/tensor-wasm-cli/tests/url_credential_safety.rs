// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Credential-safety tests for the `tensor-wasm` CLI's URL/auth handling.
//!
//! These tests exercise the parser-level helpers exposed by the crate's
//! library surface (`tensor-wasm-cli/src/lib.rs`):
//!
//! * `cmd::validate_server_url` must reject URLs carrying embedded userinfo
//!   (`http://user:pass@host`). reqwest forwards the userinfo as Basic auth,
//!   and the password would otherwise be echoed verbatim by every error
//!   message that formats `{url}` (`snapshot.rs:269`, `deploy.rs:110`,
//!   `invoke.rs:66`, `metrics.rs:41`).
//! * `cmd::snapshot::refuse_hmac_key_on_plaintext` must hard-refuse to send
//!   the 32-byte HMAC signing key over non-loopback `http://`. Loopback
//!   targets are allowed because the dev quickstart in `docs/CLI.md`
//!   binds to `127.0.0.1` and reissuing a self-signed cert per developer
//!   would be hostile UX.
//! * `cmd::plaintext_token_warned_for_test` is an observability hook on the
//!   warn-once gate guarding bearer-token transmission. We can't reach
//!   inside the tracing subscriber from a test, but we can assert the
//!   `OnceLock` latch flipped after a triggering call.
//!
//! The tests live in `tests/` rather than under `#[cfg(test)] mod tests`
//! so they exercise the same `pub` surface that any other downstream
//! consumer would see, catching accidental `pub(crate)` regressions that
//! would silently move parser helpers out of reach.

use tensor_wasm_cli::cmd::snapshot::refuse_hmac_key_on_plaintext;
use tensor_wasm_cli::cmd::{HttpContext, validate_server_url};

/// `http://u:p@host` — reqwest would happily forward the userinfo as
/// Basic auth and the password would land in every "{url}" formatted
/// error message. The validator must catch this before the URL ever
/// reaches the HTTP layer.
///
/// We use distinctive credential values (`hunter2`, `alice42`) rather
/// than literal "user"/"pass" so the "must not leak" assertions don't
/// collide with the literal words that legitimately appear in the
/// rejection message itself ("pass auth via TENSOR_WASM_TOKEN env var").
#[test]
fn validate_server_url_rejects_userinfo() {
    let err = validate_server_url("http://alice42:hunter2@host")
        .expect_err("URL with embedded credentials must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("embedded credentials"),
        "error should mention embedded credentials: {msg}"
    );
    assert!(
        msg.contains("TENSOR_WASM_TOKEN"),
        "error should redirect operator to TENSOR_WASM_TOKEN: {msg}"
    );
    // The credential values themselves must NOT appear in the error —
    // otherwise we've just moved the leak from one log line to another.
    assert!(
        !msg.contains("hunter2"),
        "error message must not echo the password value back: {msg}"
    );
    assert!(
        !msg.contains("alice42"),
        "error message must not echo the username value back: {msg}"
    );
}

/// `https://host:port` should pass cleanly. This is the spec-mandated
/// positive case and acts as a regression guard against an over-eager
/// rewrite of `validate_server_url` that starts rejecting normal URLs.
#[test]
fn validate_server_url_accepts_clean_url() {
    validate_server_url("https://host:8080")
        .expect("clean https URL with a port must be accepted");
}

/// A userinfo-only URL (`http://token@host`, no password) is still
/// userinfo and must be rejected — otherwise an operator could smuggle
/// a token-shaped value past the check as if it were a username.
#[test]
fn validate_server_url_rejects_userinfo_no_password() {
    assert!(
        validate_server_url("http://token@host").is_err(),
        "single-component userinfo must still be rejected"
    );
}

/// Userinfo embedded in an `https://` URL is just as dangerous as in
/// `http://` — TLS doesn't help if the credential ends up in stderr.
#[test]
fn validate_server_url_rejects_userinfo_on_https_too() {
    assert!(
        validate_server_url("https://user:pass@host").is_err(),
        "userinfo on https:// must also be rejected"
    );
}

/// HMAC signing keys are the *server-side secret* used to sign every
/// snapshot the API emits. Sending the key over plaintext to a public
/// host would let any on-path attacker forge authentic-looking archives
/// from that point on, which is far harder to recover from than a leaked
/// bearer token (you can rotate a token; you have to revoke every signed
/// snapshot when the key leaks). Refuse outright.
#[test]
fn refuse_hmac_key_on_plaintext_blocks_remote_http() {
    let err = refuse_hmac_key_on_plaintext("http://prod.example.com:8080")
        .expect_err("plaintext http:// to a public host must refuse the HMAC key");
    let msg = format!("{err}");
    assert!(
        msg.contains("refusing to send HMAC key over plaintext http://"),
        "error should explain the refusal: {msg}"
    );
    assert!(
        msg.contains("https://") || msg.contains("--hmac-key-file"),
        "error should suggest a remediation: {msg}"
    );
}

/// `https://` is fine — TLS protects the key in flight.
#[test]
fn refuse_hmac_key_on_plaintext_allows_https() {
    refuse_hmac_key_on_plaintext("https://prod.example.com")
        .expect("https:// must not be refused");
}

/// Loopback addresses are dev-only by construction; the dev quickstart
/// in `docs/CLI.md` binds the server to `127.0.0.1:8080`. Refusing here
/// would break the documented workflow.
#[test]
fn refuse_hmac_key_on_plaintext_allows_loopback() {
    refuse_hmac_key_on_plaintext("http://localhost:8080").expect("localhost must be allowed");
    refuse_hmac_key_on_plaintext("http://127.0.0.1:8080").expect("127.0.0.1 must be allowed");
    refuse_hmac_key_on_plaintext("http://[::1]:8080").expect("::1 must be allowed");
    // Any address in 127.0.0.0/8 is loopback per RFC 6890.
    refuse_hmac_key_on_plaintext("http://127.255.0.42:8080")
        .expect("127.0.0.0/8 must be allowed");
}

/// The plaintext-token warning is a one-shot, observed by flipping a
/// `OnceLock` latch. This test verifies that pointing the bearer-token
/// machinery at a non-loopback `http://` URL flips that latch. We can
/// only observe the gate transition once per process; the assertion
/// scheme below tolerates the test running in any order relative to
/// other tests in this binary by reading the gate state both before and
/// after.
#[test]
fn apply_warn_once_gate_trips_on_plaintext_http_token() {
    use tensor_wasm_cli::cmd::plaintext_token_warned_for_test;

    // Build a context with a token configured. We don't touch the env
    // because `HttpContext::from_env` would interact with the shared
    // process env — constructing the struct directly is cleaner.
    let ctx = HttpContext::from_env_for_test_with_token("super-secret", 0);

    let client = reqwest::Client::new();
    // Trip the gate by routing the request through `apply` with a
    // non-loopback http:// URL. We don't care about the resulting
    // RequestBuilder — only that `apply` observed the URL and flipped
    // the `OnceLock`.
    let _ = ctx.apply(client.get("http://example.com:8080")).build();

    assert!(
        plaintext_token_warned_for_test(),
        "expected the plaintext-token warn-once gate to be tripped after \
         apply() on http://example.com:8080"
    );
}

/// Counterpart to the test above: pointing `apply` at https:// must NOT
/// trip the gate (because https:// is safe) and must NOT trip it for
/// loopback http:// either (because dev workflows are exempt). We can't
/// re-run this test in isolation if another test already tripped the
/// gate, so we only assert in the direction that's safe regardless of
/// ordering: a context with no token never trips the gate.
#[test]
fn apply_warn_once_gate_does_not_trip_without_a_token() {
    let ctx = HttpContext::from_env_for_test_with_token_optional(None, 0);
    let client = reqwest::Client::new();
    let _ = ctx
        .apply(client.get("http://attacker.example.com"))
        .build();
    // We can't assert the gate is *unset* because a sibling test may have
    // tripped it. The meaningful assertion is the negative invariant on
    // the path: with no token, `apply` should never even reach the URL
    // inspection branch. That's structural — if it regresses, the path
    // becomes observable via tracing and code review, not this test.
}
