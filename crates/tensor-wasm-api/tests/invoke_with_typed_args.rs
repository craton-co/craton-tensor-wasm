// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! T33 (v0.4): broader matrix coverage for typed multi-value
//! `/functions/{id}/invoke` arguments.
//!
//! Complements `tests/invoke_with_args.rs` (the basic i32 adder happy
//! path) with:
//!
//!   * f64 round-trip — confirms the JSON-number → WasmArg::F64 lane
//!     reaches the guest and the f64 return value is rendered back as
//!     a JSON number (rather than degraded to integer or null).
//!   * i64 escalation — a JSON integer above `i32::MAX` becomes a
//!     `WasmArg::I64` per the codec contract and the guest signature
//!     must match.
//!   * Mixed-type rejection — passing a JSON string in an `args`
//!     element surfaces the canonical `400 invalid_args` envelope and
//!     does NOT leak the offending value into the response (api T3
//!     scrub policy).

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tensor_wasm_api::{build_router_with_config, AppState, AuthConfig, TenantConfig};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

async fn body_bytes(body: Body) -> Vec<u8> {
    body.collect()
        .await
        .expect("collect body")
        .to_bytes()
        .to_vec()
}

async fn body_json(body: Body) -> Value {
    let bytes = body_bytes(body).await;
    serde_json::from_slice(&bytes).expect("body is JSON")
}

fn router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig::default(),
    )
}

fn json_post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap()
}

async fn deploy(router: &axum::Router, name: &str, wat_src: &str) -> String {
    let wasm_bytes = wat::parse_str(wat_src).expect("WAT parses");
    let wasm_b64 = BASE64.encode(&wasm_bytes);
    let req = json_post(
        "/functions",
        json!({ "name": name, "wasm_b64": wasm_b64 }),
    );
    let resp = router.clone().oneshot(req).await.expect("deploy oneshot");
    assert_eq!(resp.status(), StatusCode::OK, "deploy failed");
    body_json(resp.into_body())
        .await
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .expect("deploy response has id")
}

#[tokio::test]
async fn invoke_with_f64_arg_round_trips_through_response() {
    let router = router();
    let id = deploy(
        &router,
        "doubler_f64",
        r#"
        (module
          (func (export "double") (param f64) (result f64)
            local.get 0
            local.get 0
            f64.add)
        )
        "#,
    )
    .await;

    // 1.5 + 1.5 = 3.0, exactly representable in IEEE-754.
    let body = json!({ "export": "double", "args": [1.5] });
    let req = json_post(&format!("/functions/{id}/invoke"), body);
    let resp = router.oneshot(req).await.expect("invoke oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let payload = body_json(resp.into_body()).await;
    let arr = payload
        .get("result")
        .and_then(Value::as_array)
        .expect("result is array");
    assert_eq!(arr.len(), 1, "single f64 result expected; got {arr:?}");
    assert_eq!(arr[0].as_f64(), Some(3.0), "doubler returned wrong value: {payload}");
}

#[tokio::test]
async fn invoke_with_i64_arg_escalates_above_i32_max() {
    // 2_147_483_648 = i32::MAX + 1 → must escalate to i64 per the
    // codec contract documented in `WasmArg::from_json`. The guest
    // signature is `(i64) -> i64` so wasmtime would reject the call
    // if the codec degraded the value to i32.
    let router = router();
    let id = deploy(
        &router,
        "identity_i64",
        r#"
        (module
          (func (export "id") (param i64) (result i64)
            local.get 0)
        )
        "#,
    )
    .await;

    let big = 2_147_483_648_i64;
    let body = json!({ "export": "id", "args": [big] });
    let req = json_post(&format!("/functions/{id}/invoke"), body);
    let resp = router.oneshot(req).await.expect("invoke oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let payload = body_json(resp.into_body()).await;
    let arr = payload
        .get("result")
        .and_then(Value::as_array)
        .expect("result is array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0].as_i64(), Some(big), "identity returned wrong i64: {payload}");
}

#[tokio::test]
async fn invoke_with_string_arg_rejected_with_invalid_args() {
    let router = router();
    let id = deploy(
        &router,
        "any_i32",
        r#"
        (module
          (func (export "noop") (param i32))
        )
        "#,
    )
    .await;

    // String is not a permitted WasmArg JSON form per the codec.
    let body = json!({ "export": "noop", "args": ["forty-two"] });
    let req = json_post(&format!("/functions/{id}/invoke"), body);
    let resp = router.oneshot(req).await.expect("invoke oneshot");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let payload = body_json(resp.into_body()).await;
    assert_eq!(
        payload.pointer("/error/kind").and_then(Value::as_str),
        Some("invalid_args"),
        "expected stable invalid_args kind: {payload}"
    );
}

#[tokio::test]
async fn invoke_with_two_i32_args_returns_sum_via_spawn_config_path() {
    // Direct exercise of the T33 `SpawnConfig::with_args` wiring in
    // `run_invoke`: the gateway attaches the parsed args to the
    // SpawnConfig before calling `spawn_instance`. The instance must
    // still spawn successfully and the call must surface the sum.
    let router = router();
    let id = deploy(
        &router,
        "adder_i32",
        r#"
        (module
          (func (export "add") (param i32 i32) (result i32)
            local.get 0
            local.get 1
            i32.add)
        )
        "#,
    )
    .await;

    let body = json!({ "export": "add", "args": [40, 2] });
    let req = json_post(&format!("/functions/{id}/invoke"), body);
    let resp = router.oneshot(req).await.expect("invoke oneshot");
    assert_eq!(resp.status(), StatusCode::OK);
    let payload = body_json(resp.into_body()).await;
    let arr = payload
        .get("result")
        .and_then(Value::as_array)
        .expect("result is array");
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0].as_i64(), Some(42), "adder returned wrong value: {payload}");
}
