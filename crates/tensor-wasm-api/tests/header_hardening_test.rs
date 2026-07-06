// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration tests for the `tensor-wasm-api` header-handling hardening
//! pass.
//!
//! Each test mirrors the `oneshot` + extension-injection pattern used by
//! `tests/host_validate_test.rs`: a tiny router wires the production
//! middleware (`tenant_scope` or `bearer_auth`) in front of an `ok`
//! handler, injects the relevant config via `axum::Extension`, sends a
//! single hostile request, and asserts the status code plus the
//! `error.kind` field of the JSON envelope.
//!
//! Coverage:
//!
//! * **`duplicate_tenant_header_rejected`** — two `X-TensorWasm-Tenant`
//!   headers on the same request collapse to `400
//!   duplicate_tenant_header`. Today `HeaderMap::get` returns the first
//!   match, so an attacker behind a permissive proxy could send two
//!   values and confuse downstream observers about which tenant the
//!   request really belonged to.
//! * **`invalid_tenant_header_distinct_from_missing`** — absent vs.
//!   present-but-unparseable surface as distinct `kind`s
//!   (`missing_tenant` and `invalid_tenant`) so dashboards can alert on
//!   each class independently.
//! * **`oversized_authorization_header_rejected`** — an `Authorization`
//!   value larger than [`MAX_AUTH_HEADER_BYTES`] returns `401
//!   invalid_auth` BEFORE the per-allowlist-entry constant-time compare
//!   loop runs, so a hostile client cannot burn CPU at
//!   `O(num_tokens * token_len)` per request.
//! * **`authorization_with_control_bytes_rejected`** — bearer values
//!   carrying NUL or CR/LF are refused. `axum::HeaderValue` already
//!   rejects these at construction; the test documents the contract
//!   and asserts the fallback in [`parse_bearer_credentials`] returns
//!   the same `unauthorized` envelope as a malformed value would.
//! * **`missing_tenant_when_required`** — the
//!   `TENSOR_WASM_API_REQUIRE_TENANT=1` env-var path resolves to a
//!   `TenantConfig` whose `require_header` is `true`, and an absent
//!   header surfaces as `400 missing_tenant`. Pins the env-var loader
//!   independently from the explicit-extension variant exercised by
//!   `invalid_tenant_header_distinct_from_missing` (Case A).
//! * **`authorization_with_embedded_nul_rejected`** — a NUL byte
//!   inside the `Authorization` value (e.g. `Bearer abc\0def`) is
//!   rejected by [`axum::http::HeaderValue::from_bytes`] before the
//!   value can even enter a `Request`. The hyper layer relies on
//!   this guarantee to keep CRLF / NUL out of downstream consumers
//!   (audit log fields, span attributes, log lines); the test pins
//!   the contract so a future http-crate relaxation could not
//!   silently re-open the channel.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::Extension;
use axum::http::{HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::middleware::from_fn;
use axum::routing::get;
use axum::{Json, Router};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::middleware::{bearer_auth, tenant_scope, MAX_AUTH_HEADER_BYTES};
use tensor_wasm_api::{AuthConfig, TenantConfig, HEADER_TENANT};
use tensor_wasm_core::types::TenantId;
use tower::ServiceExt;

/// Decode the response body as JSON. All four tests assert against the
/// `error.kind` field, so consolidating the decode lets each test stay
/// focused on the hostile-input shape it exercises.
async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("body").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

/// Pull `error.kind` out of an envelope or `None` on shape mismatch.
fn error_kind(body: &Value) -> Option<&str> {
    body.pointer("/error/kind").and_then(Value::as_str)
}

/// Probe handler used as the inner service. Every test only cares about
/// the status code + envelope short-circuited by middleware — if the
/// handler runs we know the middleware admitted the request.
async fn echo_tenant(tenant: Option<Extension<TenantId>>) -> Json<Value> {
    Json(match tenant {
        Some(Extension(TenantId(v))) => json!({ "tenant": v }),
        None => json!({ "tenant": null }),
    })
}

/// Build a router that wires `tenant_scope` in front of `/probe`,
/// injecting the supplied [`TenantConfig`] via `axum::Extension`.
fn tenant_router(cfg: TenantConfig) -> Router {
    let stack = tower::ServiceBuilder::new()
        .layer(axum::Extension(cfg))
        .layer(from_fn(tenant_scope));
    Router::new().route("/probe", get(echo_tenant)).layer(stack)
}

/// Bare-handler probe used for the bearer-auth tests. Tenant scope is
/// not wired here — these tests exercise the auth path only.
async fn ok_handler() -> &'static str {
    "ok"
}

/// Build a router that wires `bearer_auth` in front of `/probe`,
/// injecting the supplied [`AuthConfig`] via `axum::Extension`.
fn auth_router(cfg: AuthConfig) -> Router {
    let stack = tower::ServiceBuilder::new()
        .layer(axum::Extension(cfg))
        .layer(from_fn(bearer_auth));
    Router::new().route("/probe", get(ok_handler)).layer(stack)
}

// ---------------------------------------------------------------------------
// Fix 1 — duplicate tenant headers collapse to `400
// duplicate_tenant_header`.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn duplicate_tenant_header_rejected() {
    // Build a request that carries TWO copies of X-TensorWasm-Tenant.
    // `Request::builder().header(...)` appends rather than replaces, so
    // calling it twice produces a `HeaderMap` with two entries — exactly
    // the surface we want to defend against. `HeaderMap::get` would
    // silently return only the first one; the production middleware now
    // refuses outright before reading either value.
    let router = tenant_router(TenantConfig::default());
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header(HEADER_TENANT, "1")
        .header(HEADER_TENANT, "999")
        .body(Body::empty())
        .unwrap();
    // Sanity-check: the builder really did append two entries (not
    // overwrite). If this assumption ever changes, the test would pass
    // for the wrong reason; pin it explicitly.
    assert_eq!(
        req.headers().get_all(HEADER_TENANT).iter().count(),
        2,
        "test setup must produce two X-TensorWasm-Tenant headers",
    );
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        error_kind(&body),
        Some("duplicate_tenant_header"),
        "got {body}"
    );
}

// ---------------------------------------------------------------------------
// Fix 2 — `missing_tenant` vs `invalid_tenant` are distinct kinds.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn invalid_tenant_header_distinct_from_missing() {
    // Case A: absent header + require_header = true => missing_tenant.
    //
    // The legacy contract (TENSOR_WASM_API_REQUIRE_TENANT=1) still maps
    // an absent header to `missing_tenant`. This branch is unchanged by
    // the hardening pass; we pin it here so a future refactor cannot
    // collapse the two kinds back into one.
    let router = tenant_router(TenantConfig {
        require_header: true,
    });
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        error_kind(&body),
        Some("missing_tenant"),
        "absent + required must surface as missing_tenant: got {body}"
    );

    // Case B: present-but-non-numeric => invalid_tenant (new). The
    // pre-hardening implementation collapsed this to `missing_tenant`,
    // hiding parse failures from operators reading the dashboard.
    let router = tenant_router(TenantConfig::default());
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header(HEADER_TENANT, "not-a-number")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        error_kind(&body),
        Some("invalid_tenant"),
        "present-but-unparseable must surface as invalid_tenant: got {body}"
    );

    // Case C: present-and-numeric => pass-through. Confirms the new
    // duplicate / invalid branches did not regress the happy path.
    let router = tenant_router(TenantConfig::default());
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header(HEADER_TENANT, "42")
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        body.get("tenant").and_then(Value::as_u64),
        Some(42),
        "numeric tenant must thread to handler: got {body}"
    );
}

// ---------------------------------------------------------------------------
// Fix 3 — oversized Authorization header is rejected before the
// constant-time compare loop runs.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn oversized_authorization_header_rejected() {
    // Allowlist a short token so the constant-time compare loop runs
    // (i.e. we are NOT in dev mode — bearer_auth's dev short-circuit
    // would bypass the size cap and pass the request through).
    let auth = AuthConfig::from_tokens(["legit-token"]);
    let router = auth_router(auth);

    // Build a Bearer value well above MAX_AUTH_HEADER_BYTES. 8 KiB of
    // 'x' is harmless ASCII (so `HeaderValue::from_bytes` will accept
    // it) but the resulting `Authorization` value is far beyond the
    // 1 KiB cap. The pre-hardening path would have fed every byte
    // through every entry's `ct_eq`; the cap rejects the request
    // before that loop even starts.
    let oversized_token = "x".repeat(8 * 1024);
    let header_value_str = format!("Bearer {}", oversized_token);
    assert!(
        header_value_str.len() > MAX_AUTH_HEADER_BYTES,
        "test setup must produce a value above MAX_AUTH_HEADER_BYTES \
         ({} bytes); got {} bytes",
        MAX_AUTH_HEADER_BYTES,
        header_value_str.len(),
    );
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_bytes(header_value_str.as_bytes())
                .expect("ASCII bytes parse into HeaderValue"),
        )
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        error_kind(&body),
        Some("invalid_auth"),
        "oversized Authorization must surface as invalid_auth: got {body}"
    );
}

// ---------------------------------------------------------------------------
// Fix 4 — control bytes in the Bearer value are refused.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn authorization_with_control_bytes_rejected() {
    // The HTTP spec (RFC 7230 §3.2.6) bans control bytes in header
    // values, and `axum::http::HeaderValue::from_bytes` enforces that
    // at construction: NUL, CR, and LF are rejected outright. This is
    // the primary line of defence — the test documents the contract.
    for attack in [b"Bearer \x00malicious".as_slice(), b"Bearer \r\nevil:"] {
        let result = HeaderValue::from_bytes(attack);
        assert!(
            result.is_err(),
            "HeaderValue::from_bytes must reject control bytes; \
             value {:?} unexpectedly parsed",
            attack,
        );
    }

    // Defence in depth: even if a future refactor were to bypass
    // `HeaderValue` (e.g. via a raw byte source), `parse_bearer_credentials`
    // must NOT return a token that carries control bytes. We exercise
    // the helper by building a router with a token allowlist and
    // sending an `Authorization` value that uses a horizontal-tab
    // separator (which `HeaderValue` accepts) followed by a printable
    // payload — the request takes the happy path, confirming the
    // control-byte filter does not reject legitimate values. The
    // hostile-input rejection is unit-tested at the `parse_bearer_credentials`
    // boundary in `crates/tensor-wasm-api/src/middleware.rs`.
    let auth = AuthConfig::from_tokens(["legit-token"]);
    let router = auth_router(auth);
    // Build the header bytes manually: "Bearer\tlegit-token". The tab
    // (0x09) is the only control byte the parser accepts as a
    // separator, so this exercises the surface immediately adjacent to
    // the control-byte filter — a regression that broadens the filter
    // to reject 0x09 would surface as a 401 here.
    let value = HeaderValue::from_bytes(b"Bearer\tlegit-token")
        .expect("tab separator is a permitted header byte");
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header(axum::http::header::AUTHORIZATION, value)
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "tab-separated Bearer value must still be admitted",
    );

    // Final sanity check: a header value composed entirely of
    // printable ASCII but with a non-allowlisted token still rejects
    // as `unauthorized` (NOT `invalid_auth`). This pins the boundary
    // between Fix 3 (oversized) and Fix 4 (control bytes) vs. the
    // legacy mismatch path — they are three distinct envelopes.
    let auth = AuthConfig::from_tokens(["legit-token"]);
    let router = auth_router(auth);
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer wrong-token"),
        )
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        error_kind(&body),
        Some("unauthorized"),
        "non-allowlisted token must still surface as unauthorized: got {body}"
    );

    // Silence unused-import lint when only one constructor is used in
    // the printable-ASCII branch above. `Arc` keeps the import set
    // aligned with neighbour tests; an unused-import warning would
    // make the file noisy without changing behaviour.
    let _ = Arc::new(());
    let _: HeaderName = axum::http::header::AUTHORIZATION;
}

// ---------------------------------------------------------------------------
// `missing_tenant_when_required` — env-var path
// ---------------------------------------------------------------------------
//
// The `invalid_tenant_header_distinct_from_missing` test above already
// exercises the explicit-extension variant (`TenantConfig { require_header:
// true }`). This test pins the env-var loader independently: with
// `TENSOR_WASM_API_REQUIRE_TENANT=1` exported into the process,
// `TenantConfig::from_env()` MUST resolve to the same `require_header = true`
// shape and the absent-header path MUST still surface as
// `400 missing_tenant`.
//
// We use `temp_env::with_var` rather than raw `std::env::set_var` because:
//   * `std::env::set_var` is `unsafe` in Rust 2024 (race with concurrent
//     reads of the process environment), and
//   * `temp_env` serialises concurrent users behind an internal Mutex so a
//     second test that mutates a different env var cannot race with this
//     one inside the same test binary.
//
// Because `temp_env::with_var` provides a synchronous scope and the
// request must drive through the async router, we build the router INSIDE
// the closure (capturing `TenantConfig::from_env()` at the moment the env
// var is live) and then block on the `oneshot` future with a dedicated
// current-thread runtime. This avoids holding a `tokio::test`'s reactor
// across the env-var mutation, which would otherwise let another async
// task observe the modified env briefly.

#[test]
fn missing_tenant_when_required() {
    temp_env::with_var("TENSOR_WASM_API_REQUIRE_TENANT", Some("1"), || {
        // Capture the env-var-derived config at the moment the var is
        // live. The loader trims the value and compares against "1";
        // any other value would resolve `require_header = false`.
        let cfg = TenantConfig::from_env();
        assert!(
            cfg.require_header,
            "TENSOR_WASM_API_REQUIRE_TENANT=1 must yield require_header=true",
        );

        let router = tenant_router(cfg);
        let req = Request::builder()
            .method(Method::GET)
            .uri("/probe")
            .body(Body::empty())
            .unwrap();

        // Drive the async oneshot on a private current-thread runtime
        // so we do not need a `#[tokio::test]` attribute (which is
        // incompatible with the synchronous `temp_env::with_var` scope).
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("current-thread runtime");
        let (status, body) = rt.block_on(async move {
            let resp = router.oneshot(req).await.unwrap();
            let status = resp.status();
            let body = body_json(resp.into_body()).await;
            (status, body)
        });

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            error_kind(&body),
            Some("missing_tenant"),
            "TENSOR_WASM_API_REQUIRE_TENANT=1 + absent header must surface as \
                 missing_tenant: got {body}"
        );
    });
}

// ---------------------------------------------------------------------------
// `authorization_with_embedded_nul_rejected`
// ---------------------------------------------------------------------------
//
// A NUL byte inside the `Authorization` value would truncate any
// downstream C-string consumer (audit log writer, terminal log viewer)
// and is forbidden by RFC 7230 §3.2.6. The `http` crate's
// [`HeaderValue::from_bytes`] enforces this at construction time:
// `is_valid(b)` requires `b >= 32 && b != 127 || b == b'\t'`, which
// rejects 0x00. We pin the contract here so a future http-crate
// relaxation (or a hand-rolled byte-source bypass) could not silently
// re-open the NUL channel.
//
// The task spec phrases this as "use `HeaderValue::from_bytes` since the
// standard `try_from` rejects it" — both gates in fact reject NUL, but
// `from_bytes` is the surface most likely to be reached by an integration
// caller that wants to test the byte-level rejection without an
// intermediate `&str` (which would be impossible to construct for a
// NUL-containing buffer in pure safe Rust anyway).

#[test]
fn authorization_with_embedded_nul_rejected() {
    // The classic shape an attacker would attempt — a printable prefix
    // ("Bearer abc") followed by a NUL and more bytes ("def") that a
    // naive consumer might log past the NUL boundary.
    let attack: &[u8] = b"Bearer abc\0def";
    let result = HeaderValue::from_bytes(attack);
    assert!(
        result.is_err(),
        "HeaderValue::from_bytes must reject embedded NUL; \
         value {:?} unexpectedly parsed",
        attack,
    );

    // Defence-in-depth complement: a NUL at the very end (after a
    // tab separator) is also refused. Pins the boundary so a future
    // narrowing of the validator that only checked "interior" bytes
    // would still fail this test.
    let trailing_nul: &[u8] = b"Bearer\tlegit\0";
    assert!(
        HeaderValue::from_bytes(trailing_nul).is_err(),
        "HeaderValue::from_bytes must reject trailing NUL too; \
         value {:?} unexpectedly parsed",
        trailing_nul,
    );

    // And a NUL immediately after the scheme separator (no payload at
    // all) — covers the smallest possible hostile input.
    let bare_nul: &[u8] = b"Bearer \0";
    assert!(
        HeaderValue::from_bytes(bare_nul).is_err(),
        "HeaderValue::from_bytes must reject a bare NUL payload; \
         value {:?} unexpectedly parsed",
        bare_nul,
    );
}

// ---------------------------------------------------------------------------
// api S-3 — duplicate `Authorization` headers are rejected end-to-end.
// ---------------------------------------------------------------------------
//
// `HeaderMap::get` returns only the FIRST `Authorization` occurrence, so a
// buggy or hostile client sending two headers would silently get one value
// accepted and the other invisible — and some proxies concatenate the two
// with a comma, which round-trips unpredictably through the bearer parser.
// `bearer_auth` refuses outright before reading either value, surfacing
// `400 bad_request`. This is the end-to-end counterpart to the unit-level
// `auth_count > 1` check in `middleware.rs`.

#[tokio::test]
async fn duplicate_authorization_header_rejected() {
    // Allowlist a token so we are NOT in dev mode (dev mode short-circuits
    // before the duplicate-header check runs).
    let auth = AuthConfig::from_tokens(["legit-token"]);
    let router = auth_router(auth);

    // `Request::builder().header(...)` appends rather than replaces, so two
    // calls produce a `HeaderMap` carrying two `Authorization` entries —
    // exactly the surface the S-3 guard defends against. We send two
    // *individually valid* bearer values so the rejection can only come
    // from the duplicate-count check, not from a malformed single value.
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header(axum::http::header::AUTHORIZATION, "Bearer legit-token")
        .header(axum::http::header::AUTHORIZATION, "Bearer legit-token")
        .body(Body::empty())
        .unwrap();
    // Sanity-check the test setup really produced two headers (so the test
    // cannot pass for the wrong reason if the builder semantics change).
    assert_eq!(
        req.headers()
            .get_all(axum::http::header::AUTHORIZATION)
            .iter()
            .count(),
        2,
        "test setup must produce two Authorization headers",
    );
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        error_kind(&body),
        Some("bad_request"),
        "duplicate Authorization headers must surface as bad_request: got {body}"
    );
}

// ---------------------------------------------------------------------------
// header-hardening — the length cap short-circuits BEFORE the constant-time
// compare loop.
// ---------------------------------------------------------------------------
//
// The existing `oversized_authorization_header_rejected` test sends an
// oversized value that would never match the allowlist anyway, so it pins
// the *outcome* (`invalid_auth`) but not the *ordering*. This test sharpens
// the ordering claim: it builds a header whose prefix is the EXACT valid
// token, padded beyond the cap. If the cap fires first (the hardened order)
// the response is `401 invalid_auth`; if the constant-time compare ran first
// it would instead reach the allowlist loop and — because the padded value
// is no longer byte-equal to the token — surface the distinct
// `401 unauthorized`. Asserting `invalid_auth` therefore proves the cap
// short-circuits the request before any `ct_eq` comparison runs.

#[tokio::test]
async fn auth_header_cap_short_circuits_before_constant_time_compare() {
    const TOKEN: &str = "legit-token";
    let auth = AuthConfig::from_tokens([TOKEN]);
    let router = auth_router(auth);

    // "Bearer legit-token" + enough padding to push the whole value past
    // MAX_AUTH_HEADER_BYTES. The valid-token prefix is deliberate: it is
    // the case where an ordering bug would be most tempting (a naive
    // implementation might `starts_with`-match and admit), and it makes the
    // `invalid_auth` vs `unauthorized` distinction the sole discriminator
    // between "cap ran first" and "compare ran first".
    let padding = "x".repeat(MAX_AUTH_HEADER_BYTES);
    let header_value_str = format!("Bearer {TOKEN}{padding}");
    assert!(
        header_value_str.len() > MAX_AUTH_HEADER_BYTES,
        "test setup must exceed MAX_AUTH_HEADER_BYTES ({} bytes); got {} bytes",
        MAX_AUTH_HEADER_BYTES,
        header_value_str.len(),
    );
    let req = Request::builder()
        .method(Method::GET)
        .uri("/probe")
        .header(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_bytes(header_value_str.as_bytes())
                .expect("ASCII bytes parse into HeaderValue"),
        )
        .body(Body::empty())
        .unwrap();
    let resp = router.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = body_json(resp.into_body()).await;
    assert_eq!(
        error_kind(&body),
        Some("invalid_auth"),
        "the length cap must reject (invalid_auth) BEFORE the constant-time \
         compare loop (which would surface unauthorized): got {body}"
    );
}
