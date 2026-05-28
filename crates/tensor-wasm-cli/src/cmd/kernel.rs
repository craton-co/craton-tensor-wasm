// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `tensor-wasm kernel` — publish, list, and verify entries in the signed
//! kernel registry (roadmap feature #3).
//!
//! ## v0.3.7 status: scaffold
//!
//! The CLI surface exists so design partners can target the final
//! command shape, but the server-side `/kernels` endpoints (`POST
//! /kernels`, `GET /kernels`, `GET /kernels/{name}/{version}`) are v0.4
//! deliverables. Until they ship, every subcommand exits with the
//! [`FEATURE_NOT_EXPOSED`] code (3) and the documented "feature not yet
//! exposed" message — mirroring the B3.6 snapshot scaffold that landed
//! ahead of its server-side wire.
//!
//! See `docs/KERNEL-REGISTRY.md` for the manifest schema, the signing
//! envelope, and the v0.4 rollout plan.
//!
//! ## Why exit code 3 and not just a clean error?
//!
//! Operators wire `tensor-wasm` into CI and orchestration scripts. A
//! generic exit 1 is indistinguishable from "you typed the wrong
//! arguments" or "the server panicked". Exit code 3 is the project's
//! reserved value for "the feature surface is here but the wire isn't
//! deployed yet" — CI can `[[ $? -eq 3 ]]` and skip the step instead of
//! failing the build. See `tensor-wasm snapshot` for the prior art.
//!
//! [`FEATURE_NOT_EXPOSED`]: super::snapshot::codes::FEATURE_NOT_EXPOSED

use std::path::PathBuf;

use anyhow::Result;
use clap::Subcommand;

use super::snapshot::{codes, SnapshotExit};
use super::HttpContext;

/// `tensor-wasm kernel` sub-actions.
///
/// Every subcommand exits with [`codes::FEATURE_NOT_EXPOSED`] in
/// v0.3.7. The flag surface is stabilised here so a v0.4 rollout that
/// wires the server route can land without breaking design-partner
/// scripts.
#[derive(Debug, Subcommand)]
pub enum KernelAction {
    /// Publish a signed PTX kernel to a TensorWasm server.
    ///
    /// The CLI computes BLAKE3 over `--ptx-file`, builds a
    /// `KernelManifest`, signs it with the 32-byte HMAC-SHA256 key from
    /// `--key-file`, and POSTs the bundle to `<server>/kernels`. v0.3.7
    /// SCAFFOLD: the server-side route isn't wired yet, so the command
    /// exits 3 with the "feature not yet exposed" message documented in
    /// `docs/KERNEL-REGISTRY.md`.
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
        /// Base URL of the target TensorWasm server (e.g.
        /// `http://localhost:8080`).
        #[arg(long)]
        server: String,
    },
    /// List manifests registered on a TensorWasm server.
    ///
    /// v0.3.7 SCAFFOLD: exits 3, as above.
    List {
        /// Base URL of the target TensorWasm server.
        #[arg(long)]
        server: String,
    },
    /// Locally verify a manifest blob's signature without contacting
    /// the server.
    ///
    /// Reads a manifest JSON (the wire format the server produces) and
    /// re-computes the HMAC under `--key-file`. Useful for
    /// design-partner build pipelines that want to gate a release on
    /// the manifest verifying before it gets uploaded. v0.3.7
    /// SCAFFOLD: the manifest wire format isn't pinned yet, so the
    /// command exits 3.
    Verify {
        /// Kernel selector as `name@version`.
        selector: String,
        /// Path to the 32-byte HMAC-SHA256 verifying key.
        #[arg(long, value_name = "PATH")]
        key_file: PathBuf,
    },
}

/// Entry point for `tensor-wasm kernel`. Every action funnels through
/// [`not_yet_exposed`] in v0.3.7; the `_ctx` and field-level destructures
/// exist so the v0.4 implementation can drop in without re-shaping
/// `main.rs`'s dispatch table.
pub async fn run(action: KernelAction, _ctx: &HttpContext) -> Result<()> {
    match action {
        KernelAction::Publish {
            name,
            version,
            ptx_file: _,
            sm: _,
            key_file: _,
            server: _,
        } => Err(not_yet_exposed(&format!(
            "kernel publish {name}@{version}"
        ))),
        KernelAction::List { server: _ } => Err(not_yet_exposed("kernel list")),
        KernelAction::Verify {
            selector,
            key_file: _,
        } => Err(not_yet_exposed(&format!("kernel verify {selector}"))),
    }
}

/// Build the standard "feature not yet exposed" error tagged with
/// [`codes::FEATURE_NOT_EXPOSED`].
///
/// The message text is documented in `docs/KERNEL-REGISTRY.md` so
/// design partners know what string to grep for in CI. Mirror's the
/// snapshot scaffold's wording on purpose: a single error surface for
/// every "v0.3 scaffold, v0.4 wire" feature reduces the support load.
fn not_yet_exposed(what: &str) -> anyhow::Error {
    anyhow::Error::new(SnapshotExit {
        code: codes::FEATURE_NOT_EXPOSED,
        message: format!(
            "{what} is a v0.3.7 scaffold; the signed kernel registry feature is not \
             yet exposed by the server. The CLI surface is stable but the \
             `/kernels` HTTP route lands in v0.4 — see docs/KERNEL-REGISTRY.md \
             and https://github.com/craton-co/craton-tensor-wasm/issues for status"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_yet_exposed_carries_feature_exit_code() {
        let e = not_yet_exposed("kernel publish foo@1.0.0");
        let tagged: &SnapshotExit = e.downcast_ref().expect("typed error");
        assert_eq!(tagged.code, codes::FEATURE_NOT_EXPOSED);
        assert!(
            tagged.message.contains("not yet exposed"),
            "message should say 'not yet exposed': {}",
            tagged.message
        );
        assert!(
            tagged.message.contains("docs/KERNEL-REGISTRY.md"),
            "message should cross-link to the doc: {}",
            tagged.message
        );
    }

    #[tokio::test]
    async fn run_publish_returns_feature_not_exposed() {
        let ctx = HttpContext::from_env_for_test_with_token_optional(None, 0);
        let action = KernelAction::Publish {
            name: "matmul.f32".to_string(),
            version: "1.0.0".to_string(),
            ptx_file: PathBuf::from("dummy.ptx"),
            sm: 80,
            key_file: PathBuf::from("dummy.key"),
            server: "http://localhost:8080".to_string(),
        };
        let err = run(action, &ctx).await.unwrap_err();
        let tagged: &SnapshotExit = err.downcast_ref().expect("typed error");
        assert_eq!(tagged.code, codes::FEATURE_NOT_EXPOSED);
    }

    #[tokio::test]
    async fn run_list_returns_feature_not_exposed() {
        let ctx = HttpContext::from_env_for_test_with_token_optional(None, 0);
        let action = KernelAction::List {
            server: "http://localhost:8080".to_string(),
        };
        let err = run(action, &ctx).await.unwrap_err();
        let tagged: &SnapshotExit = err.downcast_ref().expect("typed error");
        assert_eq!(tagged.code, codes::FEATURE_NOT_EXPOSED);
    }

    #[tokio::test]
    async fn run_verify_returns_feature_not_exposed() {
        let ctx = HttpContext::from_env_for_test_with_token_optional(None, 0);
        let action = KernelAction::Verify {
            selector: "matmul.f32@1.0.0".to_string(),
            key_file: PathBuf::from("dummy.key"),
        };
        let err = run(action, &ctx).await.unwrap_err();
        let tagged: &SnapshotExit = err.downcast_ref().expect("typed error");
        assert_eq!(tagged.code, codes::FEATURE_NOT_EXPOSED);
    }
}
