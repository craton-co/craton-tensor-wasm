// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration tests for the XFCC trusted-proxy gate.
//!
//! Verifies that `X-Forwarded-Client-Cert` is only honoured when the
//! immediate TCP peer is in the operator-curated `TrustedProxies`
//! allowlist. Without this gate any TCP peer could write an arbitrary
//! certificate `Subject` into the audit stream, defeating non-repudiation
//! (the original vulnerability — see the threat-model comment on
//! `tensor_wasm_api::audit::extract_client_cert_subject_gated`).
//!
//! Cases:
//!
//! * `xfcc_from_untrusted_peer_is_dropped` — empty allowlist (the safe
//!   default) → `client_cert_subject` is `None` regardless of header.
//! * `xfcc_from_trusted_peer_is_parsed` — allowlist `127.0.0.1/32`,
//!   request from `127.0.0.1` → audit record carries the parsed Subject.
//! * `xfcc_cidr_range_membership` — allowlist `10.0.0.0/8`, peer
//!   `10.5.3.7` is trusted; peer `192.168.1.1` is not.
//!
//! Tests drive the router via `tower::ServiceExt::oneshot` and manually
//! stamp `axum::extract::ConnectInfo<SocketAddr>` into the request
//! extensions (oneshot does not bind a socket, so axum's
//! `IntoMakeServiceWithConnectInfo` path is not exercised).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{Method, Request, StatusCode};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tensor_wasm_api::{
    build_router_with_trusted_proxies, AppState, AuditConfig, AuditRecord, AuditSink, AuthConfig,
    CorsConfig, RateLimitConfig, RateLimiter, TenantConfig, TokenScope, TrustedProxies,
    HEADER_XFCC,
};
use tower::ServiceExt;

/// Test sink that buffers every emitted record so the test can assert
/// on the structured contents.
#[derive(Debug, Default)]
struct CapturingSink {
    records: Mutex<Vec<AuditRecord>>,
}

impl CapturingSink {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn snapshot(&self) -> Vec<AuditRecord> {
        self.records.lock().expect("records mutex").clone()
    }
}

impl AuditSink for CapturingSink {
    fn emit(&self, record: &AuditRecord) {
        self.records
            .lock()
            .expect("records mutex")
            .push(record.clone());
    }
}

async fn body_json(body: Body) -> Value {
    let bytes = body.collect().await.expect("body").to_bytes().to_vec();
    serde_json::from_slice(&bytes).expect("body is JSON")
}

/// Build a router whose audit sink is a capturing sink and whose
/// `TrustedProxies` allowlist is the explicitly supplied one (bypassing
/// the ambient `TENSOR_WASM_API_TRUSTED_XFCC_PROXIES` env var).
fn router_with_proxies(
    scopes: &[(&str, TokenScope)],
    trusted: TrustedProxies,
) -> (axum::Router, Arc<CapturingSink>) {
    let mut map: HashMap<String, TokenScope> = HashMap::new();
    for (k, v) in scopes {
        map.insert((*k).to_owned(), v.clone());
    }
    let auth = AuthConfig::from_scopes(map);
    let sink: Arc<CapturingSink> = CapturingSink::new();
    let audit_cfg = AuditConfig::from_sink(sink.clone() as Arc<dyn AuditSink>);
    let router = build_router_with_trusted_proxies(
        Arc::new(AppState::default()),
        auth,
        TenantConfig::default(),
        RateLimiter::new(RateLimitConfig::disabled()),
        audit_cfg,
        CorsConfig::default(),
        trusted,
    );
    (router, sink)
}

/// Deploy a trivial function so a subsequent invoke has something to
/// reference. Returns the assigned function id.
async fn deploy_trivial_function(router: &axum::Router, bearer: &str, peer: SocketAddr) -> String {
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).expect("WAT parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("authorization", format!("Bearer {bearer}"))
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "t", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    // Stamp the ConnectInfo so the audit middleware sees this synthetic
    // peer — `oneshot` does not run axum's listener path that normally
    // populates the extension.
    req.extensions_mut().insert(ConnectInfo(peer));
    let resp = router.clone().oneshot(req).await.expect("deploy");
    assert_eq!(resp.status(), StatusCode::OK, "deploy must succeed");
    body_json(resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .expect("id")
}

/// Empty allowlist (the safe default): an inbound `X-Forwarded-Client-Cert`
/// header must NOT reach the audit record.
#[tokio::test]
async fn xfcc_from_untrusted_peer_is_dropped() {
    let (router, sink) =
        router_with_proxies(&[("scoped", TokenScope::all())], TrustedProxies::empty());

    let peer: SocketAddr = "127.0.0.1:54321".parse().unwrap();
    // Use POST /functions so the audit middleware emits a record we can
    // inspect (read-only routes are filtered at the route-shape stage).
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).unwrap();
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("authorization", "Bearer scoped")
        .header("content-type", "application/json")
        .header(HEADER_XFCC, "Subject=\"CN=evil-attacker,O=Spoofed\"")
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "t", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(peer));
    let resp = router.oneshot(req).await.expect("deploy");
    assert_eq!(resp.status(), StatusCode::OK);

    let records = sink.snapshot();
    assert_eq!(records.len(), 1, "exactly one audit record emitted");
    let v: Value = serde_json::to_value(&records[0]).expect("serialises");
    // The spoofed Subject must NOT be present — either the field is
    // absent (when None) or it is explicitly null. Both shapes are
    // acceptable; the load-bearing property is that the spoofed string
    // never reaches the audit stream.
    let subj = v.get("client_cert_subject");
    assert!(
        subj.is_none_or(Value::is_null),
        "untrusted peer's XFCC must be dropped; got {subj:?}",
    );
}

/// Allowlist `127.0.0.1/32` and a request from `127.0.0.1` → the audit
/// record must contain the parsed Subject.
#[tokio::test]
async fn xfcc_from_trusted_peer_is_parsed() {
    let trusted = TrustedProxies::parse("127.0.0.1/32");
    let (router, sink) = router_with_proxies(&[("scoped", TokenScope::all())], trusted);

    let peer: SocketAddr = "127.0.0.1:54321".parse().unwrap();
    let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).unwrap();
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let mut req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("authorization", "Bearer scoped")
        .header("content-type", "application/json")
        .header(
            HEADER_XFCC,
            "Hash=abc123;Subject=\"CN=client-prod,O=Acme\";URI=spiffe://acme.io/client",
        )
        .body(Body::from(
            serde_json::to_vec(&json!({ "name": "t", "wasm_b64": wasm_b64 })).unwrap(),
        ))
        .unwrap();
    req.extensions_mut().insert(ConnectInfo(peer));
    let resp = router.oneshot(req).await.expect("deploy");
    assert_eq!(resp.status(), StatusCode::OK);

    let records = sink.snapshot();
    assert_eq!(records.len(), 1, "exactly one audit record emitted");
    let rec = &records[0];
    assert_eq!(
        rec.client_cert_subject.as_deref(),
        Some("CN=client-prod,O=Acme"),
        "trusted peer's XFCC Subject must reach the audit record",
    );
    // Serialised form carries the value too.
    let v: Value = serde_json::to_value(rec).expect("serialises");
    assert_eq!(
        v["client_cert_subject"], "CN=client-prod,O=Acme",
        "wire shape carries the parsed Subject: {v}",
    );
}

/// Allowlist `10.0.0.0/8` — `10.5.3.7` is inside the range and is
/// trusted; `192.168.1.1` is outside and is not.
#[tokio::test]
async fn xfcc_cidr_range_membership() {
    let trusted = TrustedProxies::parse("10.0.0.0/8");
    let (router, sink) = router_with_proxies(&[("scoped", TokenScope::all())], trusted);

    // First request: peer inside the CIDR → XFCC should pass through.
    let inside: SocketAddr = "10.5.3.7:12345".parse().unwrap();
    let _id = deploy_trivial_function(&router, "scoped", inside).await;
    // Buffer holds the deploy record but it had no XFCC header. Reset
    // the buffer so we focus on the next two requests, which DO carry
    // XFCC headers but differ on peer trust.
    sink.records.lock().unwrap().clear();

    let inside_req = {
        let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).unwrap();
        let wasm_b64 = BASE64.encode(&wasm_bytes);
        let mut req = Request::builder()
            .method(Method::POST)
            .uri("/functions")
            .header("authorization", "Bearer scoped")
            .header("content-type", "application/json")
            .header(HEADER_XFCC, "Subject=\"CN=trusted-from-10-net\"")
            .body(Body::from(
                serde_json::to_vec(&json!({ "name": "t2", "wasm_b64": wasm_b64 })).unwrap(),
            ))
            .unwrap();
        req.extensions_mut().insert(ConnectInfo(inside));
        req
    };
    let resp = router.clone().oneshot(inside_req).await.expect("inside");
    assert_eq!(resp.status(), StatusCode::OK);

    // Second request: peer outside the CIDR → XFCC must be dropped.
    let outside: SocketAddr = "192.168.1.1:12345".parse().unwrap();
    let outside_req = {
        let wasm_bytes = wat::parse_str(r#"(module (func (export "_start")))"#).unwrap();
        let wasm_b64 = BASE64.encode(&wasm_bytes);
        let mut req = Request::builder()
            .method(Method::POST)
            .uri("/functions")
            .header("authorization", "Bearer scoped")
            .header("content-type", "application/json")
            .header(HEADER_XFCC, "Subject=\"CN=spoofed-from-192-net\"")
            .body(Body::from(
                serde_json::to_vec(&json!({ "name": "t3", "wasm_b64": wasm_b64 })).unwrap(),
            ))
            .unwrap();
        req.extensions_mut().insert(ConnectInfo(outside));
        req
    };
    let resp = router.clone().oneshot(outside_req).await.expect("outside");
    assert_eq!(resp.status(), StatusCode::OK);

    let records = sink.snapshot();
    assert_eq!(records.len(), 2, "one record per request");

    // First record: trusted peer, Subject present.
    assert_eq!(
        records[0].client_cert_subject.as_deref(),
        Some("CN=trusted-from-10-net"),
        "10.5.3.7 is inside 10.0.0.0/8; its XFCC must be honoured",
    );
    // Second record: untrusted peer, Subject dropped.
    assert!(
        records[1].client_cert_subject.is_none(),
        "192.168.1.1 is outside 10.0.0.0/8; its XFCC must be dropped, \
         got {:?}",
        records[1].client_cert_subject,
    );
}
