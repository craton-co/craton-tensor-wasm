// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! End-to-end smoke test for the `tensor-wasm snapshot save` streaming path
//! (T24 BufWriter perf optimisation).
//!
//! Stands up a tiny axum server that responds to `POST
//! /instances/{id}/snapshot` with a synthetic body delivered in multiple
//! ~16 KiB chunks (the size `reqwest::Response::bytes_stream` typically
//! yields), then runs `tensor-wasm snapshot save` against it as a real
//! subprocess. After the CLI exits success, the test reads the persisted
//! `.tensor-wasm` file and verifies the bytes match what the server sent.
//!
//! Why this is the right shape for T24:
//!
//! * The optimisation wraps the `NamedTempFile` in a
//!   [`BufWriter`](std::io::BufWriter) before the chunk loop. A regression
//!   that drops the `flush()` / mis-handles `into_inner()` would leave the
//!   tail of the body buffered in RAM at the moment `persist` runs, so the
//!   persisted file would be SHORTER than the response. This test would
//!   fail loudly on a byte-for-byte comparison.
//! * The cap-enforcement / chunk-loop error handling is exercised by the
//!   sibling `bounded_response.rs` and `cli_smoke.rs` tests — we explicitly
//!   keep payloads well under the default 256 MiB cap so we are
//!   exercising the success path of the streaming write.
//!
//! Conceptually a sibling of `bounded_response.rs`: same axum harness, same
//! `spawn_test_server` shape, but exercising the snapshot-save path rather
//! than the bounded-body helper.

use std::net::SocketAddr;
use std::time::Duration;

use assert_cmd::Command as AssertCmd;
use axum::body::Body;
use axum::http::{Response, StatusCode};
use axum::routing::post;
use axum::Router;
use tokio::net::TcpListener;

/// Synthetic snapshot body delivered as a stream of small chunks. The chunk
/// size mirrors what `reqwest::bytes_stream()` typically hands the CLI off
/// a chunked-transfer wire (~16 KiB), so the test exercises the same
/// "many small writes" pattern T24 was written to optimise.
const CHUNK_SIZE: usize = 16 * 1024;
const N_CHUNKS: usize = 8; // 128 KiB total — well under the 256 MiB cap
const TOTAL_BYTES: usize = CHUNK_SIZE * N_CHUNKS;

/// Deterministic payload: chunk `i` is filled with the byte value `(i &
/// 0xFF)`. The CLI just streams bytes to disk verbatim, so we don't need a
/// real snapshot envelope — we only need to verify that *every byte the
/// server sent shows up on disk in the same order*. A pattern that varies
/// per chunk catches a bug where (say) the BufWriter dropped the last
/// chunk silently.
fn build_payload() -> Vec<u8> {
    let mut out = Vec::with_capacity(TOTAL_BYTES);
    for i in 0..N_CHUNKS {
        let byte = (i & 0xFF) as u8;
        out.extend(std::iter::repeat(byte).take(CHUNK_SIZE));
    }
    out
}

/// axum handler for `POST /instances/{id}/snapshot`. Streams the payload
/// as `N_CHUNKS` separate chunks so the CLI's `bytes_stream` loop iterates
/// multiple times (rather than reading a single buffered Bytes).
async fn snapshot_handler() -> Response<Body> {
    use axum::body::Bytes;
    use futures::stream;
    let payload = build_payload();
    let items: Vec<Result<Bytes, std::io::Error>> = payload
        .chunks(CHUNK_SIZE)
        .map(|c| Ok(Bytes::copy_from_slice(c)))
        .collect();
    let s = stream::iter(items);
    Response::builder()
        .status(StatusCode::OK)
        .body(Body::from_stream(s))
        .unwrap()
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

/// Drive the real CLI through the streaming-write path and confirm the
/// persisted file matches the bytes the axum server emitted.
///
/// We pin `flavor = "multi_thread"` because the test uses
/// [`tokio::task::spawn_blocking`] to call `assert_cmd` (a synchronous API)
/// without parking the runtime that's driving the axum server in the
/// background — the single-threaded flavour can deadlock when the blocking
/// task and the server's `accept` loop want to make progress at the same
/// time on the same thread.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_save_streams_to_disk_byte_for_byte() {
    // axum 0.7 path parameter syntax is `:id`, not `{id}` (the latter is
    // axum 0.8). The CLI's request URL is
    // `<server>/instances/<id>/snapshot`, so this matches every `i-*` id
    // the test passes via `--instance`.
    let app = Router::new().route("/instances/:id/snapshot", post(snapshot_handler));
    let (addr, _handle) = spawn_test_server(app).await;
    let server_url = format!("http://{addr}");

    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("snap.tensor-wasm");

    // Spawn the CLI in a blocking task so we don't park the tokio runtime
    // that's driving the axum server. `assert_cmd` is sync.
    let out_for_cli = out.clone();
    let server_for_cli = server_url.clone();
    let cli_result = tokio::task::spawn_blocking(move || {
        tensor_wasm()
            .args([
                "snapshot",
                "save",
                "--instance",
                "i-test",
                "--output",
                out_for_cli.to_str().unwrap(),
                "--server",
                &server_for_cli,
            ])
            .timeout(Duration::from_secs(30))
            .assert()
            .success();
    })
    .await;
    cli_result.expect("blocking CLI task did not panic");

    // The whole point of T24: the BufWriter MUST be flushed before
    // `persist`. If a regression drops the flush we'd see a file that is
    // shorter than TOTAL_BYTES (the last buffer remains in the BufWriter
    // and is dropped on the way out of scope).
    let bytes = std::fs::read(&out).expect("persisted snapshot exists");
    assert_eq!(
        bytes.len(),
        TOTAL_BYTES,
        "persisted snapshot size mismatch — likely a missing BufWriter flush"
    );
    assert_eq!(
        bytes,
        build_payload(),
        "persisted snapshot bytes do not match the wire payload"
    );
}
