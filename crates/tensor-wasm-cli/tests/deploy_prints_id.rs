// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! End-to-end smoke test for the `tensor-wasm deploy` happy path.
//!
//! Stands up a tiny axum server that responds to `POST /functions` with the
//! server-assigned-id ack envelope (`{"id":"i-abc"}`), then runs
//! `tensor-wasm deploy` against it as a real subprocess and asserts the CLI
//! prints the id on stdout.
//!
//! Conceptually a sibling of `snapshot_save_streams_to_disk.rs`: same axum
//! harness, same `spawn_test_server` shape, but exercising the deploy
//! id-extraction path rather than the streaming-write path.

use std::net::SocketAddr;
use std::time::Duration;

use assert_cmd::Command as AssertCmd;
use axum::routing::post;
use axum::Json;
use axum::Router;
use predicates::prelude::*;
use tokio::net::TcpListener;

/// axum handler for `POST /functions`. Mirrors the `tensor-wasm-api` ack
/// shape: a short `{"id": "<id>"}` JSON body. The CLI extracts `id` and
/// prints it verbatim (after terminal sanitisation).
async fn functions_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "id": "i-abc" }))
}

/// Spin a fresh axum server bound to `127.0.0.1:0` and return its address.
/// The join handle is returned so the caller keeps the runtime task alive
/// for the duration of the test; dropping it would cancel `axum::serve`.
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

/// Returns an `assert_cmd::Command` for the `tensor-wasm` binary with any
/// developer env vars stripped so the test isn't sensitive to the host
/// shell config. Matches `cli_smoke.rs`'s helper.
fn tensor_wasm() -> AssertCmd {
    let mut cmd = AssertCmd::cargo_bin("tensor-wasm").expect("tensor-wasm binary built");
    cmd.env_remove("TENSOR_WASM_TOKEN")
        .env_remove("TENSOR_WASM_LOG");
    cmd
}

/// Drive the real CLI through the deploy happy path and confirm it prints
/// the server-assigned id to stdout.
///
/// We pin `flavor = "multi_thread"` because the test uses
/// [`tokio::task::spawn_blocking`] to call `assert_cmd` (a synchronous API)
/// without parking the runtime that's driving the axum server in the
/// background — the single-threaded flavour can deadlock when the blocking
/// task and the server's `accept` loop want to make progress at the same
/// time on the same thread.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deploy_prints_server_assigned_id() {
    let app = Router::new().route("/functions", post(functions_handler));
    let (addr, _handle) = spawn_test_server(app).await;
    let server_url = format!("http://{addr}");

    // `deploy` reads the file's bytes and base64-encodes them; the contents
    // are opaque to the CLI on the happy path (the stub accepts any body),
    // so a tiny non-empty file is enough.
    let tmp = tempfile::tempdir().expect("tempdir");
    let wasm = tmp.path().join("demo.wasm");
    std::fs::write(&wasm, b"\0asm\x01\0\0\0").expect("write stub wasm");

    let wasm_for_cli = wasm.clone();
    let server_for_cli = server_url.clone();
    let cli_result = tokio::task::spawn_blocking(move || {
        tensor_wasm()
            .args([
                "deploy",
                wasm_for_cli.to_str().unwrap(),
                "--server",
                &server_for_cli,
            ])
            .timeout(Duration::from_secs(30))
            .assert()
            .success()
            .stdout(predicate::str::contains("i-abc"));
    })
    .await;
    cli_result.expect("blocking CLI task did not panic");
}
