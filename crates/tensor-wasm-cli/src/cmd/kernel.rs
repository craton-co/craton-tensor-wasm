// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `tensor-wasm kernel` — publish, list, and verify entries in the signed
//! kernel registry (roadmap feature #3).
//!
//! ## v0.3.8 status: server-side endpoints LANDED
//!
//! The B6.4 milestone wired the `/kernels` HTTP routes into the
//! `tensor-wasm-api` gateway (gated behind the `kernel-registry-api`
//! Cargo feature). This CLI replaces the previous v0.3.7 scaffold that
//! exited with [`FEATURE_NOT_EXPOSED`] (3); the actions now actually
//! talk to the server:
//!
//!   * `publish` reads the PTX text, computes BLAKE3, builds a signed
//!     [`KernelManifest`], and POSTs it as JSON to `/kernels`.
//!   * `list` GETs `/kernels` and renders the manifest table.
//!   * `verify` is local-only — it re-computes the HMAC under the
//!     supplied key and compares against a manifest blob on disk.
//!
//! Servers built without `--features kernel-registry-api` (the default)
//! continue to return `503 kernel_registry_not_configured`; the CLI
//! surfaces that envelope as a normal `render_error_response` error so
//! the operator gets a clear actionable signal.
//!
//! See `docs/KERNEL-REGISTRY.md` for the manifest schema, the signing
//! envelope, and the operator deployment guide.
//!
//! [`FEATURE_NOT_EXPOSED`]: super::snapshot::codes::FEATURE_NOT_EXPOSED
//! [`KernelManifest`]: tensor_wasm_jit::registry::KernelManifest

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Subcommand;
use serde::{Deserialize, Serialize};

use super::snapshot::{codes, load_hmac_key, SnapshotExit};
use super::HttpContext;
use tensor_wasm_jit::registry::{sign_manifest, KernelManifest};

/// Default request timeout for kernel-registry HTTP calls. Publish
/// requires the server to write the manifest + PTX through to the
/// (future) on-disk store, so we pick a value generous enough for a
/// few-MB PTX text on a slow link without being so loose that a stuck
/// server hangs the CLI indefinitely.
const KERNEL_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Maximum size of a PTX file the CLI will read into memory and sign.
/// Mirrors the server's body cap and prevents a misconfigured `--ptx-file`
/// from silently OOM-ing the CLI on a huge file. Operators with kernels
/// larger than this should split them across multiple manifests.
const MAX_PTX_BYTES: u64 = 16 * 1024 * 1024;

/// `tensor-wasm kernel` sub-actions.
#[derive(Debug, Subcommand)]
pub enum KernelAction {
    /// Publish a signed PTX kernel to a TensorWasm server.
    ///
    /// The CLI computes BLAKE3 over `--ptx-file`, builds a
    /// [`KernelManifest`], signs it with the 32-byte HMAC-SHA256 key
    /// from `--key-file`, and POSTs the bundle to `<server>/kernels`.
    /// The server re-verifies the signature and the digest match
    /// before accepting the publish.
    Publish {
        /// Stable kernel name (e.g. `matmul.f32`).
        name: String,
        /// SemVer-style version (e.g. `1.0.0`).
        version: String,
        /// Path to the PTX text the manifest will reference.
        #[arg(long, value_name = "PATH")]
        ptx_file: PathBuf,
        /// Compute capability the PTX was built for (e.g. `80` for sm_80).
        #[arg(long, value_name = "SM")]
        sm: u32,
        /// Path to a 32-byte HMAC-SHA256 signing key. Same on-disk
        /// format as `tensor-wasm snapshot save --hmac-key-file`:
        /// either 64 hex chars or 32 raw bytes.
        #[arg(long, value_name = "PATH")]
        key_file: PathBuf,
        /// Optional publisher identifier baked into the manifest's
        /// advisory metadata (NOT covered by the signature in v0.3.x).
        /// Defaults to `"tensor-wasm-cli"` so smoke tests can omit it.
        #[arg(long, default_value = "tensor-wasm-cli")]
        publisher: String,
        /// Base URL of the target TensorWasm server (e.g.
        /// `http://localhost:8080`).
        #[arg(long)]
        server: String,
    },
    /// List manifests registered on a TensorWasm server.
    List {
        /// Base URL of the target TensorWasm server.
        #[arg(long)]
        server: String,
    },
    /// Locally verify a manifest blob's signature without contacting
    /// the server.
    ///
    /// Reads a manifest JSON from `--manifest-file` and re-computes the
    /// HMAC under `--key-file`. Useful for design-partner build
    /// pipelines that want to gate a release on the manifest verifying
    /// before it gets uploaded.
    Verify {
        /// Kernel selector as `name@version`. Used to confirm the
        /// manifest on disk matches the expected target.
        selector: String,
        /// Path to the manifest JSON to verify.
        #[arg(long, value_name = "PATH")]
        manifest_file: PathBuf,
        /// Path to the 32-byte HMAC-SHA256 verifying key.
        #[arg(long, value_name = "PATH")]
        key_file: PathBuf,
    },
}

/// JSON shape POSTed to `/kernels`. Mirrors the server's
/// `tensor_wasm_api::kernels::PublishKernelRequest`; redefined here so
/// the CLI does not pull in the api crate just to serialise one body.
#[derive(Debug, Serialize)]
struct PublishKernelRequest<'a> {
    manifest: &'a KernelManifest,
    ptx_text: &'a str,
}

/// JSON shape returned by `GET /kernels`.
#[derive(Debug, Deserialize)]
struct ListKernelsResponse {
    manifests: Vec<KernelManifest>,
}

/// Entry point for `tensor-wasm kernel`.
pub async fn run(action: KernelAction, ctx: &HttpContext) -> Result<()> {
    match action {
        KernelAction::Publish {
            name,
            version,
            ptx_file,
            sm,
            key_file,
            publisher,
            server,
        } => {
            publish(
                &server, &name, &version, &ptx_file, sm, &key_file, &publisher, ctx,
            )
            .await
        }
        KernelAction::List { server } => list(&server, ctx).await,
        KernelAction::Verify {
            selector,
            manifest_file,
            key_file,
        } => verify(&selector, &manifest_file, &key_file),
    }
}

/// `tensor-wasm kernel publish` — sign and upload a manifest.
#[allow(clippy::too_many_arguments)]
async fn publish(
    server: &str,
    name: &str,
    version: &str,
    ptx_file: &Path,
    sm: u32,
    key_file: &Path,
    publisher: &str,
    ctx: &HttpContext,
) -> Result<()> {
    super::validate_server_url(server)?;
    if name.trim().is_empty() {
        return Err(local_err("kernel name must be non-empty"));
    }
    if version.trim().is_empty() {
        return Err(local_err("kernel version must be non-empty"));
    }

    // Local-side validation BEFORE any network I/O so a malformed file
    // surfaces as LOCAL_VALIDATION_FAILED (exit 2) rather than as a
    // surprise server error.
    let meta = std::fs::metadata(ptx_file)
        .with_context(|| format!("locating PTX file {}", ptx_file.display()))?;
    if !meta.is_file() {
        return Err(local_err(format!(
            "{} is not a regular file",
            ptx_file.display()
        )));
    }
    if meta.len() > MAX_PTX_BYTES {
        return Err(local_err(format!(
            "PTX file {} is {} bytes; the publish cap is {} bytes ({} MiB)",
            ptx_file.display(),
            meta.len(),
            MAX_PTX_BYTES,
            MAX_PTX_BYTES / (1024 * 1024)
        )));
    }
    let ptx_text = std::fs::read_to_string(ptx_file)
        .with_context(|| format!("reading PTX file {}", ptx_file.display()))?;

    // Load the HMAC key via the snapshot helper so the on-disk format
    // (64 hex chars OR 32 raw bytes) and the Zeroize-on-drop scrub
    // semantics line up across both signing surfaces.
    let key = load_hmac_key(key_file)?;

    // Build and sign the manifest. Wall-clock timestamp is best-effort
    // — a clock skewed before UNIX_EPOCH gets 0, matching the api crate's
    // `now_unix_ms` fallback.
    let digest = *blake3::hash(ptx_text.as_bytes()).as_bytes();
    let published_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let mut manifest = KernelManifest::new(
        name.to_string(),
        version.to_string(),
        sm,
        digest,
        [0u8; 32],
        published_unix_ms,
        publisher.to_string(),
    );
    manifest.signature = sign_manifest(&manifest, &key);

    let url = format!("{}/kernels", super::server_base(server));
    let client = ctx.build_client(KERNEL_REQUEST_TIMEOUT)?;
    let body = PublishKernelRequest {
        manifest: &manifest,
        ptx_text: &ptx_text,
    };
    let resp = ctx
        .apply(client.post(&url).json(&body))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("POST {url}: {e}"))?;

    let status = resp.status();
    // T17: bound the buffered response body — the publish endpoint should
    // only ever respond with a short JSON ack, never a streamed blob.
    let text = super::bounded_text(resp)
        .await
        .with_context(|| format!("reading response body from {url}"))?;
    if !status.is_success() {
        return Err(super::render_error_response(status, &text));
    }

    println!("published {name}@{version}");
    Ok(())
}

/// `tensor-wasm kernel list` — fetch and render the manifest table.
async fn list(server: &str, ctx: &HttpContext) -> Result<()> {
    super::validate_server_url(server)?;
    let url = format!("{}/kernels", super::server_base(server));
    let client = ctx.build_client(KERNEL_REQUEST_TIMEOUT)?;
    let resp = ctx
        .apply(client.get(&url))
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;
    let status = resp.status();
    // T17: bound the buffered response body. A massively-populated kernel
    // registry would still fit comfortably under 16 MiB of JSON; an
    // operator who legitimately exceeds that should paginate, not bloat
    // a single `list` call.
    let text = super::bounded_text(resp)
        .await
        .with_context(|| format!("reading response body from {url}"))?;
    if !status.is_success() {
        return Err(super::render_error_response(status, &text));
    }
    let parsed: ListKernelsResponse = serde_json::from_str(&text)
        .with_context(|| format!("decoding /kernels response from {url}"))?;
    if parsed.manifests.is_empty() {
        println!("(no kernels registered)");
    } else {
        for m in &parsed.manifests {
            // T18: `name`, `version`, and `publisher` are server-controlled
            // strings deserialised from the `/kernels` response. Strip
            // ASCII control bytes per field so a malicious manifest cannot
            // smuggle ANSI escapes into the operator's terminal. The
            // `sm_version` field is an integer and is safe as-is.
            println!(
                "{}@{}  sm={}  publisher={}",
                super::sanitise_terminal_output(&m.name),
                super::sanitise_terminal_output(&m.version),
                m.sm_version,
                super::sanitise_terminal_output(&m.publisher),
            );
        }
    }
    Ok(())
}

/// `tensor-wasm kernel verify` — local re-signing + selector check.
///
/// Reads the manifest JSON from `--manifest-file`, decodes it through
/// `serde_json`, re-computes the HMAC under `--key-file`, and compares
/// against `manifest.signature` in constant time. Also asserts that
/// `name@version` matches the supplied selector — useful when a CI
/// pipeline downloaded the manifest blob from a build cache and wants
/// to gate the release on the exact selector before uploading.
fn verify(selector: &str, manifest_file: &Path, key_file: &Path) -> Result<()> {
    let (sel_name, sel_version) = parse_selector(selector)?;
    let raw = std::fs::read_to_string(manifest_file)
        .with_context(|| format!("reading manifest file {}", manifest_file.display()))?;
    let manifest: KernelManifest = serde_json::from_str(&raw).map_err(|e| {
        local_err(format!(
            "manifest file {} is not a valid KernelManifest JSON: {e}",
            manifest_file.display()
        ))
    })?;
    if manifest.name != sel_name || manifest.version != sel_version {
        return Err(local_err(format!(
            "selector {selector} does not match manifest ({}@{})",
            manifest.name, manifest.version
        )));
    }
    let key = load_hmac_key(key_file)?;
    let recomputed = sign_manifest(&manifest, &key);
    // Constant-time comparison: same primitive the server uses in
    // `InMemoryRegistry::verify_signature`. A short-circuit `==` here
    // would leak timing on the first-byte mismatch which an attacker
    // could exploit to bisect a candidate signature.
    use subtle::ConstantTimeEq;
    if bool::from(recomputed.ct_eq(&manifest.signature)) {
        println!("ok: {selector} verifies under the supplied key");
        Ok(())
    } else {
        Err(local_err(format!(
            "signature mismatch for {selector} — manifest does NOT verify under the supplied key"
        )))
    }
}

/// Split a `name@version` selector into its two halves.
///
/// The grammar mirrors the JIT cache key (`{name}@{version}`); kernel
/// names by convention never contain `@`, so a single split at the
/// first `@` is unambiguous. Whitespace-only inputs and empty halves
/// surface as LOCAL_VALIDATION_FAILED so the operator sees a clear
/// error instead of a silent off-by-one match.
fn parse_selector(s: &str) -> Result<(String, String)> {
    let (name, version) = s
        .split_once('@')
        .ok_or_else(|| local_err(format!("selector {s} must be `name@version`")))?;
    if name.trim().is_empty() || version.trim().is_empty() {
        return Err(local_err(format!(
            "selector {s} has an empty name or version"
        )));
    }
    Ok((name.to_string(), version.to_string()))
}

/// Build an [`anyhow::Error`] tagged with the
/// [`codes::LOCAL_VALIDATION_FAILED`] exit code. Mirrors the helper in
/// `super::snapshot` so the CLI dispatcher can map both feature surfaces
/// through the same exit-code table.
fn local_err(msg: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(SnapshotExit {
        code: codes::LOCAL_VALIDATION_FAILED,
        message: msg.into(),
    })
}

// `subtle` is pulled in transitively through `tensor-wasm-jit`'s
// `kernel-registry` feature, so no separate Cargo entry is needed here.
// (The `ConstantTimeEq` impl for `[u8; 32]` lives behind the same
// `subtle = "2"` line as the snapshot signing path uses.)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_selector_splits_on_first_at() {
        let (n, v) = parse_selector("matmul.f32@1.0.0").unwrap();
        assert_eq!(n, "matmul.f32");
        assert_eq!(v, "1.0.0");
    }

    #[test]
    fn parse_selector_rejects_missing_at() {
        let err = parse_selector("matmul.f32").unwrap_err();
        let tagged: &SnapshotExit = err.downcast_ref().expect("typed error");
        assert_eq!(tagged.code, codes::LOCAL_VALIDATION_FAILED);
    }

    #[test]
    fn parse_selector_rejects_empty_name() {
        let err = parse_selector("@1.0.0").unwrap_err();
        let tagged: &SnapshotExit = err.downcast_ref().expect("typed error");
        assert_eq!(tagged.code, codes::LOCAL_VALIDATION_FAILED);
    }

    #[test]
    fn parse_selector_rejects_empty_version() {
        let err = parse_selector("matmul.f32@").unwrap_err();
        let tagged: &SnapshotExit = err.downcast_ref().expect("typed error");
        assert_eq!(tagged.code, codes::LOCAL_VALIDATION_FAILED);
    }

    #[test]
    fn verify_rejects_selector_mismatch() {
        // Build a manifest for matmul.f32@1.0.0, then call verify with
        // a different selector — must fail with LOCAL_VALIDATION_FAILED.
        let dir = tempfile::tempdir().unwrap();
        let key_path = dir.path().join("k.hex");
        let key_hex = "42".repeat(32);
        std::fs::write(&key_path, &key_hex).unwrap();
        let key = [0x42u8; 32];

        let digest = *blake3::hash(b"// ptx\n").as_bytes();
        let mut m = KernelManifest::new(
            "matmul.f32".to_string(),
            "1.0.0".to_string(),
            80,
            digest,
            [0u8; 32],
            0,
            "test".to_string(),
        );
        m.signature = sign_manifest(&m, &key);
        let manifest_path = dir.path().join("m.json");
        std::fs::write(&manifest_path, serde_json::to_vec(&m).unwrap()).unwrap();

        let err = verify("conv2d.f32@1.0.0", &manifest_path, &key_path).unwrap_err();
        let tagged: &SnapshotExit = err.downcast_ref().expect("typed error");
        assert_eq!(tagged.code, codes::LOCAL_VALIDATION_FAILED);
        assert!(tagged.message.contains("does not match"));
    }

    #[test]
    fn verify_accepts_correct_signature() {
        let dir = tempfile::tempdir().unwrap();
        let key_path = dir.path().join("k.hex");
        let key_hex = "42".repeat(32);
        std::fs::write(&key_path, &key_hex).unwrap();
        let key = [0x42u8; 32];

        let digest = *blake3::hash(b"// ptx\n").as_bytes();
        let mut m = KernelManifest::new(
            "matmul.f32".to_string(),
            "1.0.0".to_string(),
            80,
            digest,
            [0u8; 32],
            0,
            "test".to_string(),
        );
        m.signature = sign_manifest(&m, &key);
        let manifest_path = dir.path().join("m.json");
        std::fs::write(&manifest_path, serde_json::to_vec(&m).unwrap()).unwrap();

        verify("matmul.f32@1.0.0", &manifest_path, &key_path).unwrap();
    }

    #[test]
    fn verify_rejects_bad_signature() {
        let dir = tempfile::tempdir().unwrap();
        let key_path = dir.path().join("k.hex");
        std::fs::write(&key_path, "42".repeat(32)).unwrap();

        // Build a manifest signed under a DIFFERENT key.
        let wrong_key = [0x01u8; 32];
        let digest = *blake3::hash(b"// ptx\n").as_bytes();
        let mut m = KernelManifest::new(
            "matmul.f32".to_string(),
            "1.0.0".to_string(),
            80,
            digest,
            [0u8; 32],
            0,
            "test".to_string(),
        );
        m.signature = sign_manifest(&m, &wrong_key);
        let manifest_path = dir.path().join("m.json");
        std::fs::write(&manifest_path, serde_json::to_vec(&m).unwrap()).unwrap();

        let err = verify("matmul.f32@1.0.0", &manifest_path, &key_path).unwrap_err();
        let tagged: &SnapshotExit = err.downcast_ref().expect("typed error");
        assert_eq!(tagged.code, codes::LOCAL_VALIDATION_FAILED);
        assert!(tagged.message.contains("signature mismatch"));
    }
}
