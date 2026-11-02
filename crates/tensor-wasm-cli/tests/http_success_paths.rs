// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! End-to-end HTTP success-path + hardening tests for the `tensor-wasm` CLI
//! (cli tests F7).
//!
//! The existing `cli_smoke.rs` tests only exercise the HTTP-shaped commands
//! against `DEAD_SERVER` (nothing bound) so they prove the request *fails*
//! cleanly. These tests stand up a real in-process `axum` stub and run the
//! compiled binary against it, so they prove the *success* path works
//! end-to-end and that the security hardening (T17 in-memory body cap, T18
//! terminal-output sanitisation) actually fires on real responses:
//!
//!   * `invoke` / `metrics` / `kernel list` / `kernel publish` / `observe`
//!     parse a real 2xx response and print the expected output.
//!   * A malicious server response containing ANSI escapes / a bare `\r` in
//!     an `invoke` body or a `kernel list` manifest is scrubbed before it
//!     reaches stdout.
//!   * `metrics` / `invoke` reject an over-cap response body (the
//!     `bounded_text` T17 path) instead of buffering it.
//!   * `--output json` emits a valid machine-readable document for
//!     `invoke` / `deploy` / `metrics`.
//!
//! Harness shape mirrors `snapshot_save_streams_to_disk.rs`: a `spawn_blocking`
//! task runs the (synchronous) `assert_cmd` binary so it doesn't park the
//! tokio runtime driving the axum server.

use std::net::SocketAddr;

use assert_cmd::Command as AssertCmd;
use axum::body::{Body, Bytes};
use axum::extract::Path;
use axum::http::{header, HeaderValue, Response, StatusCode};
use axum::routing::{get, post};
use axum::{Json, Router};
use tensor_wasm_cli::cmd::MAX_RESPONSE_BODY_BYTES;
use tensor_wasm_jit::registry::KernelManifest;
use tokio::net::TcpListener;

/// Bind a fresh axum server on `127.0.0.1:0`; return its address + the task
/// handle (kept alive by the caller so `axum::serve` is not cancelled).
async fn spawn_test_server(router: Router) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind 127.0.0.1:0");
    let addr = listener.local_addr().expect("local_addr");
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    (addr, handle)
}

/// Fresh `tensor-wasm` command with developer env scrubbed. Matches the helper
/// in `cli_smoke.rs` / `snapshot_save_streams_to_disk.rs`.
fn tensor_wasm() -> AssertCmd {
    let mut cmd = AssertCmd::cargo_bin("tensor-wasm").expect("tensor-wasm binary built");
    cmd.env_remove("TENSOR_WASM_TOKEN")
        .env_remove("TENSOR_WASM_LOG");
    cmd
}

/// Run a CLI invocation in a blocking task (assert_cmd is synchronous) so it
/// doesn't park the runtime driving the axum stub. Returns (stdout, stderr,
/// success).
async fn run_cli(args: Vec<String>) -> (String, String, bool) {
    tokio::task::spawn_blocking(move || {
        let assertion = tensor_wasm()
            .args(&args)
            .timeout(std::time::Duration::from_secs(30))
            .assert();
        let out = assertion.get_output();
        (
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
            out.status.success(),
        )
    })
    .await
    .expect("blocking CLI task did not panic")
}

// ---------------------------------------------------------------------------
// Success-path handlers
// ---------------------------------------------------------------------------

async fn invoke_ok(Path(_id): Path<String>) -> Json<serde_json::Value> {
    Json(serde_json::json!({ "ok": true, "result": [42] }))
}

async fn metrics_ok() -> Response<Body> {
    let body = "# HELP foo a counter\n# TYPE foo counter\nfoo 7\nbar{a=\"1\"} 2.5\n";
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/plain")
        .body(Body::from(body))
        .unwrap()
}

fn sample_manifest(name: &str, publisher: &str) -> KernelManifest {
    KernelManifest::new(
        name.to_string(),
        "1.0.0".to_string(),
        80,
        [0u8; 32],
        [0u8; 32],
        0,
        publisher.to_string(),
    )
}

async fn kernels_list_ok() -> Json<serde_json::Value> {
    let m = sample_manifest("matmul.f32", "acme");
    Json(serde_json::json!({ "manifests": [m] }))
}

async fn kernels_publish_ok() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "id": "k-1" }))
}

async fn healthz_ok() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok", "uptime_seconds": 5 }))
}

// ---------------------------------------------------------------------------
// Success-path tests
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invoke_success_path_prints_response() {
    let app = Router::new().route("/functions/:id/invoke", post(invoke_ok));
    let (addr, _h) = spawn_test_server(app).await;
    let (stdout, stderr, ok) = run_cli(vec![
        "invoke".into(),
        "fn-1".into(),
        "--server".into(),
        format!("http://{addr}"),
    ])
    .await;
    assert!(ok, "invoke should succeed; stderr: {stderr}");
    assert!(stdout.contains("\"ok\""), "missing response body: {stdout}");
    assert!(stdout.contains("42"), "missing result value: {stdout}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metrics_success_path_prints_exposition() {
    let app = Router::new().route("/metrics", get(metrics_ok));
    let (addr, _h) = spawn_test_server(app).await;
    let (stdout, stderr, ok) = run_cli(vec![
        "metrics".into(),
        "--server".into(),
        format!("http://{addr}"),
    ])
    .await;
    assert!(ok, "metrics should succeed; stderr: {stderr}");
    assert!(stdout.contains("foo 7"), "missing metric line: {stdout}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kernel_list_success_path_renders_table() {
    let app = Router::new().route("/kernels", get(kernels_list_ok));
    let (addr, _h) = spawn_test_server(app).await;
    let (stdout, stderr, ok) = run_cli(vec![
        "kernel".into(),
        "list".into(),
        "--server".into(),
        format!("http://{addr}"),
    ])
    .await;
    assert!(ok, "kernel list should succeed; stderr: {stderr}");
    assert!(
        stdout.contains("matmul.f32@1.0.0"),
        "missing manifest row: {stdout}"
    );
    assert!(
        stdout.contains("publisher=acme"),
        "missing publisher: {stdout}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kernel_publish_success_path_reports_published() {
    let app = Router::new().route("/kernels", post(kernels_publish_ok));
    let (addr, _h) = spawn_test_server(app).await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let ptx = tmp.path().join("kernel.ptx");
    std::fs::write(&ptx, b"// ptx text\n").expect("write ptx");
    let key = tmp.path().join("hmac.key");
    std::fs::write(&key, "42".repeat(32)).expect("write key");

    let (stdout, stderr, ok) = run_cli(vec![
        "kernel".into(),
        "publish".into(),
        "matmul.f32".into(),
        "1.0.0".into(),
        "--ptx-file".into(),
        ptx.to_str().unwrap().into(),
        "--sm".into(),
        "80".into(),
        "--key-file".into(),
        key.to_str().unwrap().into(),
        "--server".into(),
        format!("http://{addr}"),
    ])
    .await;
    assert!(ok, "kernel publish should succeed; stderr: {stderr}");
    assert!(
        stdout.contains("published matmul.f32@1.0.0"),
        "missing publish ack: {stdout}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn observe_one_tick_json_renders_health_and_metrics() {
    // `observe` loops until Ctrl-C, so we use `--output json` (NDJSON, no
    // screen clear) and kill the process after a moment via the assert_cmd
    // timeout, asserting on whatever it printed before being terminated.
    let app = Router::new()
        .route("/healthz", get(healthz_ok))
        .route("/metrics", get(metrics_ok));
    let (addr, _h) = spawn_test_server(app).await;

    let args = vec![
        "observe".into(),
        "--addr".into(),
        format!("http://{addr}"),
        "--interval".into(),
        "1".into(),
        "--output".into(),
        "json".into(),
    ];
    // Run with a short timeout; observe never exits on its own. We capture
    // stdout up to the kill. The first NDJSON line is enough to assert on.
    let stdout = tokio::task::spawn_blocking(move || {
        let output = tensor_wasm()
            .args(&args)
            .timeout(std::time::Duration::from_secs(3))
            .output()
            .expect("spawn observe");
        String::from_utf8_lossy(&output.stdout).into_owned()
    })
    .await
    .expect("blocking observe task did not panic");

    let line = stdout
        .lines()
        .find(|l| !l.trim().is_empty())
        .expect("observe should emit at least one NDJSON tick");
    let v: serde_json::Value = serde_json::from_str(line).expect("observe tick is valid JSON");
    assert_eq!(v["liveness"], "ok", "tick: {v}");
    assert_eq!(v["health_status"], "ok", "tick: {v}");
}

// ---------------------------------------------------------------------------
// T18 terminal-output sanitisation (malicious response bytes scrubbed)
// ---------------------------------------------------------------------------

async fn invoke_malicious() -> Response<Body> {
    // A NON-JSON body so the CLI's text path echoes it verbatim through
    // `sanitise_terminal_output` (the JSON pretty-print path would JSON-escape
    // control bytes itself, which wouldn't exercise the sanitiser). The raw
    // ESC + bare CR must be scrubbed before reaching stdout.
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/plain")
        .body(Body::from("evil\x1b[31mred\rOVERWRITE"))
        .unwrap()
}

async fn kernels_list_malicious() -> Json<serde_json::Value> {
    // ANSI escape + CR in the manifest `name` / `publisher` fields.
    let m = sample_manifest("ker\x1b[2Jnel", "pub\rwriter");
    Json(serde_json::json!({ "manifests": [m] }))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invoke_scrubs_malicious_response_bytes() {
    let app = Router::new().route("/functions/:id/invoke", post(invoke_malicious));
    let (addr, _h) = spawn_test_server(app).await;
    let (stdout, _stderr, ok) = run_cli(vec![
        "invoke".into(),
        "fn-1".into(),
        "--server".into(),
        format!("http://{addr}"),
    ])
    .await;
    assert!(ok, "invoke should succeed");
    assert!(
        !stdout.contains('\x1b'),
        "ESC byte must be scrubbed from invoke stdout: {stdout:?}"
    );
    assert!(
        !stdout.contains('\r'),
        "bare CR must be scrubbed from invoke stdout: {stdout:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kernel_list_scrubs_malicious_manifest_bytes() {
    let app = Router::new().route("/kernels", get(kernels_list_malicious));
    let (addr, _h) = spawn_test_server(app).await;
    let (stdout, _stderr, ok) = run_cli(vec![
        "kernel".into(),
        "list".into(),
        "--server".into(),
        format!("http://{addr}"),
    ])
    .await;
    assert!(ok, "kernel list should succeed");
    assert!(
        !stdout.contains('\x1b'),
        "ESC byte must be scrubbed from kernel list stdout: {stdout:?}"
    );
    assert!(
        !stdout.contains('\r'),
        "bare CR must be scrubbed from kernel list stdout: {stdout:?}"
    );
}

// ---------------------------------------------------------------------------
// T17 bounded_text (over-cap response body rejected, not buffered)
// ---------------------------------------------------------------------------

/// Handler that declares a `Content-Length` larger than the CLI's in-memory
/// cap. `bounded_text` must fast-fail on the declared length. The body is
/// streamed lazily so the well-formed HTTP message never materialises in RAM
/// (the CLI drops the connection after rejecting on the header).
async fn oversize_response() -> Response<Body> {
    use futures::stream;
    let oversize = (MAX_RESPONSE_BODY_BYTES as u64) + 1;
    let mut remaining = oversize;
    let chunks = std::iter::from_fn(move || {
        if remaining == 0 {
            return None;
        }
        let n = remaining.min(64 * 1024) as usize;
        remaining -= n as u64;
        Some(Ok::<Bytes, std::io::Error>(Bytes::from(vec![0u8; n])))
    });
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_LENGTH,
            HeaderValue::from_str(&oversize.to_string()).unwrap(),
        )
        .body(Body::from_stream(stream::iter(chunks)))
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metrics_rejects_oversize_body() {
    let app = Router::new().route("/metrics", get(oversize_response));
    let (addr, _h) = spawn_test_server(app).await;
    let (_stdout, stderr, ok) = run_cli(vec![
        "metrics".into(),
        "--server".into(),
        format!("http://{addr}"),
    ])
    .await;
    assert!(!ok, "metrics must fail on an over-cap body");
    assert!(
        stderr.to_lowercase().contains("cap") || stderr.to_lowercase().contains("exceeds"),
        "error should mention the response-body cap: {stderr}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invoke_rejects_oversize_body() {
    let app = Router::new().route("/functions/:id/invoke", post(oversize_response));
    let (addr, _h) = spawn_test_server(app).await;
    let (_stdout, stderr, ok) = run_cli(vec![
        "invoke".into(),
        "fn-1".into(),
        "--server".into(),
        format!("http://{addr}"),
    ])
    .await;
    assert!(!ok, "invoke must fail on an over-cap body");
    assert!(
        stderr.to_lowercase().contains("cap") || stderr.to_lowercase().contains("exceeds"),
        "error should mention the response-body cap: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// --output json end-to-end
// ---------------------------------------------------------------------------

async fn deploy_ok() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "id": "fn-deployed-1" }))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invoke_output_json_emits_valid_envelope() {
    let app = Router::new().route("/functions/:id/invoke", post(invoke_ok));
    let (addr, _h) = spawn_test_server(app).await;
    let (stdout, stderr, ok) = run_cli(vec![
        "invoke".into(),
        "fn-1".into(),
        "--server".into(),
        format!("http://{addr}"),
        "--output".into(),
        "json".into(),
    ])
    .await;
    assert!(ok, "invoke --output json should succeed; stderr: {stderr}");
    let line = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .expect("json line");
    let v: serde_json::Value = serde_json::from_str(line).expect("invoke json valid");
    assert_eq!(v["id"], "fn-1");
    assert_eq!(v["export"], "_start");
    assert_eq!(v["response"]["ok"], true);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deploy_output_json_emits_valid_envelope() {
    let app = Router::new().route("/functions", post(deploy_ok));
    let (addr, _h) = spawn_test_server(app).await;

    let tmp = tempfile::tempdir().expect("tempdir");
    let wasm = wat::parse_str("(module)").expect("compile empty module");
    let p = tmp.path().join("m.wasm");
    std::fs::write(&p, &wasm).expect("write wasm");

    let (stdout, stderr, ok) = run_cli(vec![
        "deploy".into(),
        p.to_str().unwrap().into(),
        "--server".into(),
        format!("http://{addr}"),
        "--output".into(),
        "json".into(),
    ])
    .await;
    assert!(ok, "deploy --output json should succeed; stderr: {stderr}");
    let line = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .expect("json line");
    let v: serde_json::Value = serde_json::from_str(line).expect("deploy json valid");
    assert_eq!(v["id"], "fn-deployed-1");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metrics_output_json_emits_valid_samples() {
    let app = Router::new().route("/metrics", get(metrics_ok));
    let (addr, _h) = spawn_test_server(app).await;
    let (stdout, stderr, ok) = run_cli(vec![
        "metrics".into(),
        "--server".into(),
        format!("http://{addr}"),
        "--output".into(),
        "json".into(),
    ])
    .await;
    assert!(ok, "metrics --output json should succeed; stderr: {stderr}");
    let line = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .expect("json line");
    let v: serde_json::Value = serde_json::from_str(line).expect("metrics json valid");
    let samples = v["samples"].as_array().expect("samples array");
    assert!(
        samples.iter().any(|s| s["name"] == "foo"),
        "missing foo sample: {v}"
    );
}
