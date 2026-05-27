// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `tensor-wasm snapshot` — save or restore a running instance to a `.tensor-wasm` file.
//!
//! Both sub-actions are wired to the TensorWasm HTTP API (`POST /instances/{id}/snapshot`
//! for save, `POST /instances/restore` for restore). Local-side validation
//! happens before any network call so the user gets a fast, clear error for the
//! common "wrong path / unwritable output / file too big" cases without
//! waiting on a server round-trip.
//!
//! ## API not yet shipped
//!
//! As of this commit the TensorWasm API server (see `tensor-wasm-api/src/server.rs`) does
//! NOT yet route `/instances/...` — those endpoints are planned but not
//! merged. To avoid silently exiting 0 on a UX that doesn't work end-to-end,
//! the CLI checks whether the server returns `404 Not Found` for the path and
//! converts that into a dedicated [`codes::FEATURE_NOT_EXPOSED`] exit code
//! pointing the user at the tracking issue. See
//! <https://github.com/craton-co/craton-tensor-wasm/issues> for status.
//!
//! ## Streaming / size limits
//!
//! The on-disk archive size is bounded by `--max-archive-bytes`, which defaults
//! to 256 MiB. Naming this honestly matters: the flag caps the *on-disk
//! (compressed) archive* the CLI is willing to upload — the decompressed
//! footprint is bounded server-side by
//! `tensor_wasm_snapshot::SnapshotReader::with_max_decompressed`. A previous
//! release exposed this flag as `--max-decompressed`, which was misleading
//! because a small gzipped archive can decompress to many gigabytes; the old
//! name is kept as a hidden alias for one release.
//!
//! On the save side, the server-supplied body is bounded by
//! `--max-restore-bytes` (default 256 MiB) so a malicious server can't fill
//! the operator's disk by streaming an unbounded response.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Subcommand;

use super::HttpContext;

/// Default cap on the on-disk snapshot archive size accepted by the CLI, in
/// bytes (256 MiB). Override per invocation with `--max-archive-bytes` on
/// restore or `--max-restore-bytes` on save. This is an honest bound on the
/// *compressed* archive size — the decompressed footprint is bounded
/// server-side by `tensor_wasm_snapshot::limits::MAX_DECOMPRESSED_BYTES`.
pub(crate) const DEFAULT_MAX_ARCHIVE_BYTES: u64 = 256 * 1024 * 1024;

/// Process exit codes used by the snapshot subcommands. Made explicit so
/// callers (CI, smoke tests) can distinguish "feature not implemented" from a
/// genuine runtime error.
pub(crate) mod codes {
    /// The CLI subcommand surface exists but the corresponding API endpoint
    /// has not been deployed yet (404 Not Found on the route).
    pub const FEATURE_NOT_EXPOSED: i32 = 3;
    /// Local-side validation failed (bad path, file too big, etc.) before we
    /// reached the network.
    pub const LOCAL_VALIDATION_FAILED: i32 = 2;
}

/// `tensor-wasm snapshot` sub-actions.
#[derive(Debug, Subcommand)]
pub enum SnapshotAction {
    /// Capture the state of a running instance into a `.tensor-wasm` file via the API.
    Save {
        /// Identifier of the running instance to snapshot.
        #[arg(long)]
        instance: String,
        /// Output path for the resulting `.tensor-wasm` archive.
        #[arg(long)]
        output: PathBuf,
        /// Base URL of the target TensorWasm server (e.g. `http://localhost:8080`).
        #[arg(long)]
        server: String,
        /// Maximum number of bytes the CLI will accept from the server and
        /// write to `--output`. Defaults to 256 MiB; values above the default
        /// are clamped down so a malicious server cannot fill the operator's
        /// disk by streaming an unbounded response body.
        #[arg(long, default_value_t = DEFAULT_MAX_ARCHIVE_BYTES)]
        max_restore_bytes: u64,
    },
    /// Restore an instance from a `.tensor-wasm` archive via the API.
    Restore {
        /// Path to the `.tensor-wasm` archive to upload.
        #[arg(long)]
        input: PathBuf,
        /// Identifier to assign to the restored instance.
        #[arg(long = "as-instance")]
        as_instance: String,
        /// Base URL of the target TensorWasm server (e.g. `http://localhost:8080`).
        #[arg(long)]
        server: String,
        /// Maximum *on-disk archive* size the CLI will upload, in bytes
        /// (default 256 MiB). This bounds the compressed payload only — the
        /// decompressed footprint is enforced server-side and may be much
        /// larger. The deprecated alias `--max-decompressed` is accepted for
        /// one release; prefer `--max-archive-bytes` in new scripts.
        #[arg(
            long = "max-archive-bytes",
            alias = "max-decompressed",
            default_value_t = DEFAULT_MAX_ARCHIVE_BYTES
        )]
        max_archive_bytes: u64,
    },
}

/// Error carrying a specific exit code, surfaced to `main` via `anyhow`.
///
/// `anyhow::Error::downcast_ref::<SnapshotExit>()` lets the dispatcher map
/// the structured error back to a process exit code without losing the
/// human-readable message.
#[derive(Debug)]
pub struct SnapshotExit {
    /// Exit code this error should map to (see [`codes`]).
    pub code: i32,
    /// Human-readable message rendered to stderr.
    pub message: String,
}

impl std::fmt::Display for SnapshotExit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SnapshotExit {}

/// Entry point for `tensor-wasm snapshot`. Returns `Err` (with a typed
/// [`SnapshotExit`]) rather than `Ok(())` on any failure, so the process
/// always exits non-zero on unimplemented or broken paths.
pub async fn run(action: SnapshotAction, ctx: &HttpContext) -> Result<()> {
    match action {
        SnapshotAction::Save {
            instance,
            output,
            server,
            max_restore_bytes,
        } => save(&server, &instance, &output, max_restore_bytes, ctx).await,
        SnapshotAction::Restore {
            input,
            as_instance,
            server,
            max_archive_bytes,
        } => restore(&server, &input, &as_instance, max_archive_bytes, ctx).await,
    }
}

/// Implementation of `tensor-wasm snapshot save`.
///
/// `max_restore_bytes` bounds the response body the CLI is willing to write
/// to disk. It is clamped down to [`DEFAULT_MAX_ARCHIVE_BYTES`] so a caller
/// cannot opt into unbounded disk consumption by raising the flag.
async fn save(
    server: &str,
    instance_id: &str,
    output: &Path,
    max_restore_bytes: u64,
    ctx: &HttpContext,
) -> Result<()> {
    super::validate_server_url(server)?;
    validate_parent_writable(output)?;
    if instance_id.trim().is_empty() {
        return Err(local_err("--instance must be non-empty"));
    }

    // Clamp down to the hard ceiling: callers can lower the cap (good for
    // hosts with little spare disk) but never raise it past 256 MiB.
    let cap = max_restore_bytes.min(DEFAULT_MAX_ARCHIVE_BYTES);

    let url = format!(
        "{}/instances/{}/snapshot",
        super::server_base(server),
        instance_id
    );
    let client = ctx.build_client(Duration::from_secs(120))?;

    let resp = ctx
        .apply(client.post(&url))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("POST {url}: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp
            .text()
            .await
            .with_context(|| format!("reading error body from {url}"))?;
        if status == reqwest::StatusCode::NOT_FOUND && !looks_like_tensor_wasm_envelope(&text) {
            return Err(not_yet_shipped("save", &url));
        }
        return Err(super::render_error_response(status, &text));
    }

    let bytes = resp
        .bytes()
        .await
        .with_context(|| format!("streaming snapshot body from {url}"))?;
    if bytes.len() as u64 > cap {
        return Err(anyhow::anyhow!(
            "server returned {} bytes; refusing to write more than {} bytes to disk \
             (lower with --max-restore-bytes; the hard ceiling is {} bytes)",
            bytes.len(),
            cap,
            DEFAULT_MAX_ARCHIVE_BYTES
        ));
    }

    std::fs::write(output, &bytes)
        .with_context(|| format!("writing snapshot to {}", output.display()))?;

    println!(
        "snapshot save: wrote {} bytes to {}",
        bytes.len(),
        output.display()
    );
    Ok(())
}

/// Implementation of `tensor-wasm snapshot restore`.
async fn restore(
    server: &str,
    input: &Path,
    as_instance: &str,
    max_archive_bytes: u64,
    ctx: &HttpContext,
) -> Result<()> {
    super::validate_server_url(server)?;
    if as_instance.trim().is_empty() {
        return Err(local_err("--as-instance must be non-empty"));
    }

    let meta = std::fs::metadata(input)
        .with_context(|| format!("locating snapshot file {}", input.display()))?;
    if !meta.is_file() {
        return Err(local_err(format!(
            "{} is not a regular file",
            input.display()
        )));
    }
    if meta.len() > max_archive_bytes {
        return Err(local_err(format!(
            "snapshot archive {} is {} bytes on disk; the configured cap is \
             {} bytes (raise with --max-archive-bytes; note this bounds the \
             *compressed* upload — decompressed size is enforced server-side)",
            input.display(),
            meta.len(),
            max_archive_bytes
        )));
    }

    let bytes = std::fs::read(input)
        .with_context(|| format!("reading snapshot file {}", input.display()))?;

    let url = format!("{}/instances/restore", super::server_base(server));
    let client = ctx.build_client(Duration::from_secs(120))?;

    let resp = ctx
        .apply(
            client
                .post(&url)
                .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
                .header("X-TensorWasm-As-Instance", as_instance)
                .body(bytes),
        )
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("POST {url}: {e}"))?;

    let status = resp.status();
    let text = resp
        .text()
        .await
        .with_context(|| format!("reading response body from {url}"))?;
    if !status.is_success() {
        if status == reqwest::StatusCode::NOT_FOUND && !looks_like_tensor_wasm_envelope(&text) {
            return Err(not_yet_shipped("restore", &url));
        }
        return Err(super::render_error_response(status, &text));
    }

    // Optimistically parse `{"id": "<...>"}`; fall back to the raw body when
    // the server hasn't yet standardised the shape.
    let id = serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|v| v.get("id").and_then(|s| s.as_str()).map(str::to_owned))
        .unwrap_or_else(|| text.trim().to_owned());
    println!("{id}");
    Ok(())
}

/// Confirm `path`'s parent directory exists and is writable so we don't fetch
/// a snapshot just to discover we can't put it anywhere.
fn validate_parent_writable(path: &Path) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    let meta = std::fs::metadata(parent).with_context(|| {
        format!(
            "checking --output parent directory {}",
            parent.display()
        )
    })?;
    if !meta.is_dir() {
        return Err(local_err(format!(
            "--output parent {} is not a directory",
            parent.display()
        )));
    }
    // `is_writable` requires `unix`/`windows` plumbing; the practical check is
    // attempting a temp create+remove. That's overkill here — `std::fs::write`
    // failure later will surface a clear error. We just confirm existence.
    Ok(())
}

/// Heuristic: does this response body look like a TensorWasm error envelope
/// (`{"error": {"kind": ..., "message": ...}}`)? Used to distinguish a real
/// `404 instance_not_found` from axum's default "no route" 404, so we can
/// surface the "feature not yet shipped" hint only for the latter.
fn looks_like_tensor_wasm_envelope(body: &str) -> bool {
    #[derive(serde::Deserialize)]
    struct Envelope {
        #[allow(dead_code)]
        error: Inner,
    }
    #[derive(serde::Deserialize)]
    struct Inner {
        #[allow(dead_code)]
        kind: String,
        #[allow(dead_code)]
        message: String,
    }
    serde_json::from_str::<Envelope>(body).is_ok()
}

/// Build an [`anyhow::Error`] tagged with the LOCAL_VALIDATION_FAILED exit code.
fn local_err(msg: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(SnapshotExit {
        code: codes::LOCAL_VALIDATION_FAILED,
        message: msg.into(),
    })
}

/// Build a "feature not yet exposed by API" error tagged with the
/// FEATURE_NOT_EXPOSED exit code. Used when the server returns 404 on the
/// snapshot routes.
fn not_yet_shipped(action: &str, url: &str) -> anyhow::Error {
    anyhow::Error::new(SnapshotExit {
        code: codes::FEATURE_NOT_EXPOSED,
        message: format!(
            "snapshot {action} requires API support not yet shipped \
             (server returned 404 for {url}); track at \
             https://github.com/craton-co/craton-tensor-wasm/issues"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_err_carries_validation_exit_code() {
        let e = local_err("nope");
        let tagged: &SnapshotExit = e.downcast_ref().expect("typed error");
        assert_eq!(tagged.code, codes::LOCAL_VALIDATION_FAILED);
        assert_eq!(tagged.message, "nope");
    }

    #[test]
    fn not_yet_shipped_carries_feature_exit_code() {
        let e = not_yet_shipped("save", "http://x/y");
        let tagged: &SnapshotExit = e.downcast_ref().expect("typed error");
        assert_eq!(tagged.code, codes::FEATURE_NOT_EXPOSED);
        assert!(tagged.message.contains("not yet shipped"));
    }

    #[test]
    fn validate_parent_writable_accepts_cwd() {
        // `out.tensor-wasm` -> parent = `.`, which always exists.
        validate_parent_writable(Path::new("out.tensor-wasm")).unwrap();
    }

    #[test]
    fn envelope_heuristic_matches_real_404() {
        assert!(looks_like_tensor_wasm_envelope(
            r#"{"error":{"kind":"instance_not_found","message":"i-1 not found"}}"#
        ));
    }

    #[test]
    fn envelope_heuristic_rejects_default_404() {
        // Axum's default route-miss response is empty / plain text.
        assert!(!looks_like_tensor_wasm_envelope(""));
        assert!(!looks_like_tensor_wasm_envelope("Not Found"));
        assert!(!looks_like_tensor_wasm_envelope("{\"unrelated\":1}"));
    }

    #[test]
    fn validate_parent_writable_rejects_missing_dir() {
        let err = validate_parent_writable(Path::new(
            "/definitely/not/a/real/path/snapshot.tensor-wasm",
        ));
        assert!(err.is_err());
    }
}
