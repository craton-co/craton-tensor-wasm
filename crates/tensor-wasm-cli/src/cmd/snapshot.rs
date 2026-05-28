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
//!
//! ## Snapshot signing
//!
//! `--hmac-key-file PATH` on `snapshot save` causes the CLI to forward the
//! 32-byte HMAC-SHA256 key (hex-encoded, in the
//! `X-TensorWasm-Snapshot-HMAC-Key` header) to the server, which signs the
//! emitted archive with it. On `snapshot restore`, passing `--hmac-key-file`
//! supplies the verifying key, and `--require-signature` instructs the
//! server to refuse unsigned (v2) archives outright. See
//! `docs/SNAPSHOT-FORMAT.md` for the on-disk layout.

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

/// Hard ceiling on `--max-restore-bytes`. A caller may lower the cap (good for
/// hosts with little spare disk) but never raise it past this constant; this
/// stops a malicious or buggy server from convincing the CLI to fill the
/// operator's disk by way of a large `--max-restore-bytes` flag.
pub(crate) const MAX_RESTORE_BYTES_CEILING: u64 = DEFAULT_MAX_ARCHIVE_BYTES;

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

/// HTTP header on `snapshot save` carrying the hex-encoded 32-byte
/// HMAC-SHA256 key the server should use to sign the emitted archive.
/// Server-side wiring lives in `tensor-wasm-api` (task M8.4).
pub(crate) const HMAC_KEY_HEADER: &str = "X-TensorWasm-Snapshot-HMAC-Key";

/// HTTP header on `snapshot restore` instructing the server to refuse
/// unsigned (v2) snapshots. The header value is always the literal `true`
/// when present; absence is treated as `false` server-side.
pub(crate) const REQUIRE_SIGNATURE_HEADER: &str = "X-TensorWasm-Snapshot-Require-Signature";

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
        /// Path to a 32-byte HMAC-SHA256 key. The file is interpreted as
        /// 64 hex characters if it's that length (whitespace trimmed),
        /// otherwise as 32 raw bytes. Mismatched length → error. When set,
        /// the key is forwarded to the server (hex-encoded, in the
        /// `X-TensorWasm-Snapshot-HMAC-Key` header) and the server uses it
        /// to sign the emitted archive.
        #[arg(long, value_name = "PATH")]
        hmac_key_file: Option<PathBuf>,
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
        /// Path to a 32-byte HMAC-SHA256 key. The file is interpreted as
        /// 64 hex characters if it's that length (whitespace trimmed),
        /// otherwise as 32 raw bytes. Mismatched length → error. When set,
        /// the key is forwarded to the server (hex-encoded, in the
        /// `X-TensorWasm-Snapshot-HMAC-Key` header) and the server uses it
        /// to verify the archive's signature.
        #[arg(long, value_name = "PATH")]
        hmac_key_file: Option<PathBuf>,
        /// Refuse to restore an unsigned (v2) snapshot. Sends the
        /// `X-TensorWasm-Snapshot-Require-Signature: true` header so the
        /// server fails closed on archives produced before signing was
        /// enabled.
        #[arg(long)]
        require_signature: bool,
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
            hmac_key_file,
        } => {
            save(
                &server,
                &instance,
                &output,
                max_restore_bytes,
                hmac_key_file.as_deref(),
                ctx,
            )
            .await
        }
        SnapshotAction::Restore {
            input,
            as_instance,
            server,
            max_archive_bytes,
            hmac_key_file,
            require_signature,
        } => {
            restore(
                &server,
                &input,
                &as_instance,
                max_archive_bytes,
                hmac_key_file.as_deref(),
                require_signature,
                ctx,
            )
            .await
        }
    }
}

/// Implementation of `tensor-wasm snapshot save`.
///
/// `max_restore_bytes` bounds the response body the CLI is willing to write
/// to disk. It is clamped down to [`MAX_RESTORE_BYTES_CEILING`] so a caller
/// cannot opt into unbounded disk consumption by raising the flag.
#[allow(clippy::too_many_arguments)]
async fn save(
    server: &str,
    instance_id: &str,
    output: &Path,
    max_restore_bytes: u64,
    hmac_key_file: Option<&Path>,
    ctx: &HttpContext,
) -> Result<()> {
    super::validate_server_url(server)?;
    validate_parent_writable(output)?;
    if instance_id.trim().is_empty() {
        return Err(local_err("--instance must be non-empty"));
    }

    // Clamp down to the hard ceiling: callers can lower the cap (good for
    // hosts with little spare disk) but never raise it past 256 MiB.
    let cap = max_restore_bytes.min(MAX_RESTORE_BYTES_CEILING);

    // Load and validate the HMAC key before any network I/O so a malformed
    // key file fails fast with a LOCAL_VALIDATION_FAILED exit code.
    let hmac_key_hex = match hmac_key_file {
        Some(path) => Some(load_hmac_key(path).map(hex::encode)?),
        None => None,
    };

    // Refuse to send the 32-byte HMAC signing key over plaintext http:// to a
    // non-loopback host. The key is the *secret* used by the server to sign
    // every snapshot it emits; leaking it lets an attacker forge archives
    // that look authentic. Bearer tokens are also sensitive but cheaper to
    // rotate (see `HttpContext::apply`, which only warns for the token
    // case); the HMAC key warrants a hard refuse.
    if hmac_key_hex.is_some() {
        refuse_hmac_key_on_plaintext(server)?;
    }

    let url = format!(
        "{}/instances/{}/snapshot",
        super::server_base(server),
        instance_id
    );
    let client = ctx.build_client(Duration::from_secs(120))?;

    let mut req = client.post(&url);
    if let Some(hex_key) = &hmac_key_hex {
        req = req.header(HMAC_KEY_HEADER, hex_key);
    }
    let resp = ctx
        .apply(req)
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

    // cli 1.2: stream the body instead of buffering into a single Bytes.
    // Three benefits over the previous resp.bytes() approach:
    //   1. A malicious / buggy server cannot OOM the CLI by streaming
    //      arbitrarily many gigabytes — we abort as soon as `cap` bytes
    //      have been accumulated.
    //   2. An early Content-Length check rejects oversized payloads before
    //      a single chunk is read.
    //   3. The write goes to a temp file in the same directory as `output`
    //      and is atomically renamed on success, so a failed download
    //      never leaves a half-written snapshot at the target path.
    if let Some(declared) = resp.content_length() {
        if declared > cap {
            return Err(anyhow::anyhow!(
                "server declared {} bytes; refusing to write more than {} bytes to disk \
                 (lower with --max-restore-bytes; the hard ceiling is {} bytes)",
                declared,
                cap,
                MAX_RESTORE_BYTES_CEILING
            ));
        }
    }
    let parent = output.parent().unwrap_or_else(|| std::path::Path::new("."));
    let tmp = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("creating tempfile in {}", parent.display()))?;
    // perf T24: wrap the tempfile in a BufWriter so each ~16-64 KiB chunk
    // delivered by `bytes_stream()` doesn't translate into a syscall. At a
    // 64 KiB buffer the syscall count drops ~4-16x for a 256 MiB cap. The
    // cap accounting below still uses `chunk.len()` (the bytes we *accepted*
    // for writing), which is unchanged — the BufWriter only batches the
    // actual `write(2)` calls; it does not alter how many bytes flow
    // through it.
    let mut received: u64 = 0;
    use futures::StreamExt;
    use std::io::{BufWriter, Write};
    let mut writer = BufWriter::with_capacity(64 * 1024, tmp);
    let mut stream = resp.bytes_stream();
    while let Some(chunk_res) = stream.next().await {
        let chunk = chunk_res.with_context(|| format!("streaming snapshot body from {url}"))?;
        received = received.saturating_add(chunk.len() as u64);
        if received > cap {
            drop(writer); // discard the temp file (BufWriter drops the inner NamedTempFile)
            return Err(anyhow::anyhow!(
                "server streamed > {} bytes; refusing to write more than {} bytes to disk \
                 (lower with --max-restore-bytes; the hard ceiling is {} bytes)",
                cap,
                cap,
                MAX_RESTORE_BYTES_CEILING
            ));
        }
        writer
            .write_all(&chunk)
            .with_context(|| format!("writing snapshot chunk to tempfile"))?;
    }
    // Critical: flush the BufWriter before unwrapping so any buffered tail
    // bytes hit the tempfile. We then call `into_inner()` to recover the
    // NamedTempFile for `persist`. `into_inner` itself also flushes, so we
    // surface either flush error via the same anyhow path — never unwrap.
    writer
        .flush()
        .with_context(|| "flushing snapshot tempfile buffer".to_string())?;
    let tmp = writer.into_inner().map_err(|e| {
        // `IntoInnerError::into_error()` consumes the wrapper and yields the
        // underlying `io::Error` from the implicit flush — surface it to the
        // caller so a disk-full or perms problem doesn't silently corrupt
        // the persisted snapshot.
        anyhow::Error::new(e.into_error()).context("flushing snapshot tempfile buffer on close")
    })?;
    tmp.persist(output)
        .map_err(|e| anyhow::anyhow!("renaming tempfile to {}: {}", output.display(), e))?;

    println!(
        "snapshot save: wrote {} bytes to {}",
        received,
        output.display()
    );
    Ok(())
}

/// Implementation of `tensor-wasm snapshot restore`.
#[allow(clippy::too_many_arguments)]
async fn restore(
    server: &str,
    input: &Path,
    as_instance: &str,
    max_archive_bytes: u64,
    hmac_key_file: Option<&Path>,
    require_signature: bool,
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

    // Load and validate the HMAC key before any network I/O so a malformed
    // key file fails fast with a LOCAL_VALIDATION_FAILED exit code.
    let hmac_key_hex = match hmac_key_file {
        Some(path) => Some(load_hmac_key(path).map(hex::encode)?),
        None => None,
    };

    // See the matching check in `save`: refuse to expose the HMAC key over
    // plaintext to anything other than a loopback / dev address.
    if hmac_key_hex.is_some() {
        refuse_hmac_key_on_plaintext(server)?;
    }

    // cli fix 1: stream the archive off disk rather than slurping the entire
    // file (capped at 256 MiB) into a `Vec<u8>` before handing it to reqwest.
    // The previous `std::fs::read(input)` path peaked at ~256 MiB resident
    // for the buffer plus whatever reqwest's body machinery allocated on top;
    // the streaming `ReaderStream + Body::wrap_stream` shape keeps the
    // working set bounded to ~one tokio file-read chunk regardless of
    // archive size. The matching `Content-Length` header is set explicitly
    // from the stat we already did above so the server sees a length-prefix
    // and can pre-size its receive buffer (otherwise reqwest falls back to
    // chunked transfer encoding when the body has no known length).
    let file = tokio::fs::File::open(input)
        .await
        .with_context(|| format!("opening snapshot file {}", input.display()))?;
    let stream = tokio_util::io::ReaderStream::new(file);
    let body = reqwest::Body::wrap_stream(stream);

    let url = format!("{}/instances/restore", super::server_base(server));
    let client = ctx.build_client(Duration::from_secs(120))?;

    let mut req = client
        .post(&url)
        .header(reqwest::header::CONTENT_TYPE, "application/octet-stream")
        // `meta.len()` is u64; `HeaderValue` impls `TryFrom<u64>` but
        // formatting to a decimal string here avoids any ambiguity over
        // whichever reqwest version is resolved and is what every other
        // HTTP client does on the wire anyway.
        .header(reqwest::header::CONTENT_LENGTH, meta.len().to_string())
        .header("X-TensorWasm-As-Instance", as_instance);
    if let Some(hex_key) = &hmac_key_hex {
        req = req.header(HMAC_KEY_HEADER, hex_key);
    }
    if require_signature {
        req = req.header(REQUIRE_SIGNATURE_HEADER, "true");
    }
    let resp = ctx
        .apply(req.body(body))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("POST {url}: {e}"))?;

    let status = resp.status();
    // T17: the restore endpoint returns a short `{"id": "<uuid>"}` ack
    // (or an error envelope on failure). Route through `bounded_text`
    // so a malicious server cannot fill the CLI's RAM by sending a
    // multi-gigabyte success body. The streamed save path elsewhere in
    // this file is the legitimate large-body channel — it writes to
    // disk under its own `--max-restore-bytes` cap.
    let text = super::bounded_text(resp)
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
    // T18: the id came off the wire from the server (or, on the fallback
    // path, is the raw response body). Strip ASCII control bytes before
    // displaying so a malicious server cannot inject ANSI escapes that
    // rewrite the operator's terminal title bar, smuggle in a CR, or
    // hide subsequent output.
    println!("{}", super::sanitise_terminal_output(&id));
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
    let meta = std::fs::metadata(parent)
        .with_context(|| format!("checking --output parent directory {}", parent.display()))?;
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

/// Refuse to forward the HMAC signing key when the server URL is a
/// non-loopback `http://`. Returns `Ok(())` for `https://` and loopback
/// (`localhost`, `127.0.0.0/8`, `::1`) targets so dev workflows aren't
/// blocked. The error is tagged with `LOCAL_VALIDATION_FAILED` because
/// the failure is caught entirely on the client side before any bytes hit
/// the wire.
///
/// Exposed as `pub` (gated by `#[doc(hidden)]`) so the integration test in
/// `tests/url_credential_safety.rs` can exercise the refuse branch
/// directly via the lib surface, without spawning a real server.
#[doc(hidden)]
pub fn refuse_hmac_key_on_plaintext(server: &str) -> Result<()> {
    let Some((scheme, host)) = super::extract_scheme_host(server) else {
        // `validate_server_url` is supposed to run first; if it didn't, fall
        // back to silently allowing the call so the existing scheme-shape
        // error elsewhere takes precedence over this defence-in-depth check.
        return Ok(());
    };
    if scheme == "http" && !super::is_loopback_host(host) {
        return Err(local_err(
            "refusing to send HMAC key over plaintext http://; use https:// or omit --hmac-key-file",
        ));
    }
    Ok(())
}

/// Load a 32-byte HMAC-SHA256 key from disk.
///
/// The file content is interpreted in one of two ways:
///
/// * If the file (with leading/trailing whitespace trimmed) is exactly
///   64 characters of hex (`0-9`, `a-f`, `A-F`), it's decoded as hex into
///   32 raw bytes. This is the recommended human-editable format.
/// * Otherwise, if the file is exactly 32 bytes long, those bytes are used
///   verbatim.
///
/// Any other length is rejected with a [`codes::LOCAL_VALIDATION_FAILED`]
/// error so an operator who accidentally points the flag at, say, a
/// passphrase or PEM file gets a clear message instead of a silently
/// truncated key.
pub(crate) fn load_hmac_key(path: &Path) -> Result<[u8; 32]> {
    // cli 1.1.a: warn if the keyfile is readable by group/other on Unix.
    // The file holds a 32-byte signing secret; world-readable means anyone
    // on the host can forge snapshots that look authentic. We warn rather
    // than refuse because (a) `umask 0` developer setups exist and (b) the
    // operator may genuinely want the file group-readable for a service
    // account. Loud warning, not hard failure.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = std::fs::metadata(path) {
            let mode = meta.permissions().mode();
            if mode & 0o077 != 0 {
                tracing::warn!(
                    target: "tensor_wasm_cli::snapshot",
                    file = %path.display(),
                    mode = format!("{:o}", mode & 0o777),
                    "HMAC key file is readable by group/other; tighten to 0600 \
                     (chmod 600 <file>) — the file holds a 32-byte signing \
                     secret and any reader can forge snapshots"
                );
            }
        }
    }
    // cli 1.1.c: wrap the file read in a Zeroized RAII so the heap-resident
    // copy of the key material is scrubbed on every return path, success
    // or error. The returned `[u8; 32]` is the caller's responsibility.
    let raw = Zeroized(
        std::fs::read(path).with_context(|| format!("reading HMAC key file {}", path.display()))?,
    );

    // Try the hex path first: a file that's pure ASCII hex (after trimming
    // surrounding whitespace) is the documented happy path. Mixed binary
    // files containing 64 ASCII-hex bytes plus a trailing newline still
    // resolve via this branch because `trim` strips the newline.
    if let Ok(text) = std::str::from_utf8(&raw.0) {
        let trimmed = text.trim();
        if trimmed.len() == 64 {
            let bytes = Zeroized(hex::decode(trimmed).map_err(|e| {
                local_err(format!(
                    "HMAC key file {} looks like hex but is not valid: {e}",
                    path.display()
                ))
            })?);
            let mut out = [0u8; 32];
            out.copy_from_slice(&bytes.0);
            return Ok(out);
        }
    }

    if raw.0.len() == 32 {
        let mut out = [0u8; 32];
        out.copy_from_slice(&raw.0);
        return Ok(out);
    }

    Err(local_err(format!(
        "HMAC key file {} must be either 64 hex characters or 32 raw bytes; \
         got {} bytes",
        path.display(),
        raw.0.len()
    )))
}

/// Newtype wrapper that scrubs a `Vec<u8>` on Drop (cli 1.1.c).
///
/// Uses `std::ptr::write_volatile` per byte so a future-optimising compiler
/// cannot elide the store as a dead write. This is the same pattern the
/// `zeroize` crate's `volatile_write_bytes` uses; pulling the dep in just
/// for this one call site is overkill, and the cli crate's heap-key
/// lifetime is short enough (single subcommand) that a per-call
/// implementation suffices. If the cli ever grows a long-running daemon
/// path, switch to `zeroize::Zeroizing`.
struct Zeroized(Vec<u8>);

impl Drop for Zeroized {
    fn drop(&mut self) {
        for byte in self.0.iter_mut() {
            // SAFETY: `byte` is a unique &mut u8 inside the Vec; writing 0
            // through it is sound. `write_volatile` is also sound for any
            // `T: Copy`, which `u8` is.
            unsafe {
                std::ptr::write_volatile(byte, 0);
            }
        }
    }
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

    #[test]
    fn load_hmac_key_accepts_64_hex_chars() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k.hex");
        // 64 hex chars = 32 bytes of 0x42.
        let hex_key = "42".repeat(32);
        std::fs::write(&path, &hex_key).unwrap();
        let k = load_hmac_key(&path).unwrap();
        assert_eq!(k, [0x42u8; 32]);
    }

    #[test]
    fn load_hmac_key_trims_trailing_newline_on_hex_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k.hex");
        let mut content = "ab".repeat(32);
        content.push('\n');
        std::fs::write(&path, &content).unwrap();
        let k = load_hmac_key(&path).unwrap();
        assert_eq!(k, [0xabu8; 32]);
    }

    #[test]
    fn load_hmac_key_accepts_32_raw_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k.bin");
        let raw = [0x99u8; 32];
        std::fs::write(&path, raw).unwrap();
        let k = load_hmac_key(&path).unwrap();
        assert_eq!(k, raw);
    }

    #[test]
    fn load_hmac_key_rejects_wrong_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k.bad");
        std::fs::write(&path, b"too short").unwrap();
        let err = load_hmac_key(&path).unwrap_err();
        let tagged: &SnapshotExit = err.downcast_ref().expect("typed error");
        assert_eq!(tagged.code, codes::LOCAL_VALIDATION_FAILED);
        assert!(tagged.message.contains("64 hex characters or 32 raw bytes"));
    }

    #[test]
    fn refuse_hmac_key_on_plaintext_blocks_remote_http() {
        let err = refuse_hmac_key_on_plaintext("http://example.com:8080").unwrap_err();
        let tagged: &SnapshotExit = err.downcast_ref().expect("typed error");
        assert_eq!(tagged.code, codes::LOCAL_VALIDATION_FAILED);
        assert!(
            tagged.message.contains("plaintext http://"),
            "message should mention plaintext: {}",
            tagged.message
        );
        assert!(
            tagged.message.contains("--hmac-key-file"),
            "message should mention the offending flag: {}",
            tagged.message
        );
    }

    #[test]
    fn refuse_hmac_key_on_plaintext_allows_https() {
        refuse_hmac_key_on_plaintext("https://example.com").unwrap();
    }

    #[test]
    fn refuse_hmac_key_on_plaintext_allows_loopback_http() {
        refuse_hmac_key_on_plaintext("http://localhost:8080").unwrap();
        refuse_hmac_key_on_plaintext("http://127.0.0.1:8080").unwrap();
        refuse_hmac_key_on_plaintext("http://[::1]:8080").unwrap();
    }

    #[test]
    fn load_hmac_key_rejects_invalid_hex() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("k.bad");
        // 64 chars but with a non-hex `z`.
        let mut s = "a".repeat(63);
        s.push('z');
        std::fs::write(&path, &s).unwrap();
        let err = load_hmac_key(&path).unwrap_err();
        let tagged: &SnapshotExit = err.downcast_ref().expect("typed error");
        assert_eq!(tagged.code, codes::LOCAL_VALIDATION_FAILED);
        assert!(tagged.message.contains("not valid"));
    }
}
