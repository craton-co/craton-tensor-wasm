// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration tests for the CLI's bounded response-body helper (T17 DoS
//! hardening).
//!
//! Stand up a tiny axum server that either declares an oversized
//! `Content-Length` (fast-fail path) or streams more bytes than the cap
//! advertises no length (mid-stream tripwire path), then assert that
//! `bounded_text` / `bounded_bytes` reject the response with the
//! [`ApiClientError::ResponseTooLarge`] variant rather than silently
//! buffering everything into the CLI's RAM.
//!
//! The helper is `pub` + `#[doc(hidden)]` purely so this integration
//! test can reach it through the lib surface — see `src/cmd/mod.rs`.

use std::net::SocketAddr;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, HeaderValue, Response, StatusCode};
use axum::routing::get;
use axum::Router;
use tokio::net::TcpListener;

use tensor_wasm_cli::cmd::{bounded_bytes, bounded_text, ApiClientError, MAX_RESPONSE_BODY_BYTES};

/// Bind a fresh axum server on `127.0.0.1:0`, return its bound address and
/// the join handle of the background task that drives it. The handle is
/// kept alive by the caller so the runtime drops it at end-of-test (axum's
/// `serve` future runs forever otherwise).
async fn spawn_test_server(router: Router) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind 127.0.0.1:0");
    let addr = listener.local_addr().expect("local_addr");
    let handle = tokio::spawn(async move {
        // We deliberately ignore the result: when the test runtime tears
        // down, axum's serve future is cancelled and that surfaces as an
        // Err we don't care about.
        let _ = axum::serve(listener, router).await;
    });
    (addr, handle)
}

/// Handler that advertises `Content-Length: MAX_RESPONSE_BODY_BYTES + 1`
/// in the response headers. The actual body is built from a streaming
/// channel that yields a single tiny chunk so axum / hyper does NOT
/// auto-compute and override our explicit `Content-Length` header — they
/// only auto-fill it when the body has a statically-known size. The test
/// asserts that `bounded_bytes` FAST-FAILS on the declared length without
/// reading a byte off the socket; the body contents do not matter.
async fn oversized_content_length() -> Response<Body> {
    use axum::body::Bytes;
    use futures::stream;
    let oversize = (MAX_RESPONSE_BODY_BYTES as u64) + 1;
    let s = stream::iter(vec![Ok::<Bytes, std::io::Error>(Bytes::from_static(
        b"tiny",
    ))]);
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_LENGTH,
            HeaderValue::from_str(&oversize.to_string()).unwrap(),
        )
        .body(Body::from_stream(s))
        .unwrap()
}

#[tokio::test]
async fn bounded_text_rejects_oversize_content_length() {
    let app = Router::new().route("/metrics", get(oversized_content_length));
    let (addr, _handle) = spawn_test_server(app).await;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let resp = client
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .expect("GET succeeds");

    let err = bounded_text(resp)
        .await
        .expect_err("bounded_text must reject Content-Length > cap");

    match err {
        ApiClientError::ResponseTooLarge { actual, limit } => {
            assert_eq!(
                limit, MAX_RESPONSE_BODY_BYTES,
                "limit should equal the cap constant"
            );
            assert_eq!(
                actual,
                (MAX_RESPONSE_BODY_BYTES as u64) + 1,
                "actual should equal the declared Content-Length"
            );
        }
        other => panic!("expected ResponseTooLarge, got {other:?}"),
    }
}

/// Handler that streams MORE than [`MAX_RESPONSE_BODY_BYTES`] bytes back
/// WITHOUT declaring a `Content-Length`. This exercises the per-chunk
/// tripwire in `bounded_bytes` rather than the fast-fail branch.
///
/// We chunk a single byte at a time to keep the test memory cheap — axum
/// 0.7's `Body::from_stream` lets us hand it a `futures::stream::Stream`
/// of `Result<Bytes, _>` items.
async fn oversize_streamed() -> Response<Body> {
    use axum::body::Bytes;
    use futures::stream;

    // Cap + a small overflow. We deliberately yield 64 KiB chunks instead
    // of one byte per chunk; the cap check fires the moment cumulative
    // bytes exceed the limit so we hit the tripwire after ~256 chunks
    // rather than after 16 million. Streaming exactly `MAX + chunk_size`
    // is enough to provoke the trip without producing 16+ MiB of
    // throwaway data.
    // `Bytes::from_static` needs a `&'static [u8]`. Promoting the zeroed
    // array to a `static` gives every chunk a cheap shared pointer; cloning
    // a `Bytes` over a static slice is a refcount bump, NOT a copy. 64 KiB
    // chunk size is large enough that the trip wire fires after ~256
    // iterations rather than 16 million one-byte yields.
    const CHUNK: usize = 64 * 1024;
    static ZERO_CHUNK: [u8; 64 * 1024] = [0u8; 64 * 1024];
    let payload = Bytes::from_static(&ZERO_CHUNK);
    let n_chunks = (MAX_RESPONSE_BODY_BYTES / CHUNK) + 2;
    let items: Vec<Result<Bytes, std::io::Error>> =
        (0..n_chunks).map(|_| Ok(payload.clone())).collect();
    let s = stream::iter(items);

    Response::builder()
        .status(StatusCode::OK)
        // Deliberately omit Content-Length so the fast-fail branch is
        // bypassed; the body uses chunked transfer encoding.
        .body(Body::from_stream(s))
        .unwrap()
}

#[tokio::test]
async fn bounded_bytes_trips_mid_stream_when_no_content_length() {
    let app = Router::new().route("/metrics", get(oversize_streamed));
    let (addr, _handle) = spawn_test_server(app).await;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();
    let resp = client
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .expect("GET succeeds");

    let err = bounded_bytes(resp)
        .await
        .expect_err("bounded_bytes must trip the mid-stream cap");

    match err {
        ApiClientError::ResponseTooLarge { actual, limit } => {
            assert_eq!(limit, MAX_RESPONSE_BODY_BYTES);
            // The running total at trip time is at least cap+1; the exact
            // value depends on chunk alignment with the cap boundary.
            assert!(
                actual > MAX_RESPONSE_BODY_BYTES as u64,
                "actual={actual} must exceed cap={MAX_RESPONSE_BODY_BYTES}"
            );
        }
        other => panic!("expected ResponseTooLarge, got {other:?}"),
    }
}

/// A small in-cap response should pass through `bounded_text` cleanly so
/// we know the helper isn't false-positiving on legitimate payloads.
async fn small_ok() -> &'static str {
    "ok"
}

#[tokio::test]
async fn bounded_text_passes_small_response() {
    let app = Router::new().route("/healthz", get(small_ok));
    let (addr, _handle) = spawn_test_server(app).await;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let resp = client
        .get(format!("http://{addr}/healthz"))
        .send()
        .await
        .expect("GET succeeds");
    let text = bounded_text(resp)
        .await
        .expect("small response should pass");
    assert_eq!(text, "ok");
}
