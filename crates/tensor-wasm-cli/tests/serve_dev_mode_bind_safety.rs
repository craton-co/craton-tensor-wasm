// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Bind-safety integration tests for `tensor-wasm serve` (cli S-35 + api S-35).
//!
//! Background: when `serve` is started in dev mode (no bearer-token allowlist
//! configured) the API gateway accepts every request without authentication.
//! Historically the only signal that this had happened was a single
//! startup `tracing::warn!`, which let several operators inadvertently
//! expose `0.0.0.0`-bound no-auth gateways to the public internet. The fix
//! is a hard bind-time gate: dev mode + non-loopback bind = refuse to
//! start unless the operator explicitly opts in with
//! `--allow-plaintext-public` (or its `TENSOR_WASM_ALLOW_PLAINTEXT_PUBLIC`
//! env-var equivalent).
//!
//! These tests drive the gate end-to-end through the compiled binary so
//! they catch wiring regressions (forgetting to call `validate_bind_safety`
//! from `run`, mis-typing the env var, etc.) that the per-module unit
//! tests in `src/cmd/serve.rs::tests` would miss. Accepted-config cases
//! use the hidden `--check-only` flag to exit before binding so the test
//! does not leave a real HTTP listener behind.

use std::time::Duration;

use assert_cmd::Command as AssertCmd;
use predicates::prelude::*;

/// Build a fresh `tensor-wasm` invocation scrubbed of every environment
/// variable that could perturb the bind-safety gate. The gate keys off:
///
/// * `TENSOR_WASM_API_TOKENS` — present + non-empty flips the gateway out
///   of dev mode entirely, masking the bug we are testing for.
/// * `TENSOR_WASM_ALLOW_PLAINTEXT_PUBLIC` — clap binds this directly to
///   `--allow-plaintext-public`, so a developer's shell export would
///   silently disable the gate.
/// * `TENSOR_WASM_TOKEN` / `TENSOR_WASM_LOG` — irrelevant here but mirrored
///   from the other CLI integration tests for consistency.
fn tensor_wasm() -> AssertCmd {
    let mut cmd = AssertCmd::cargo_bin("tensor-wasm").expect("tensor-wasm binary built");
    cmd.env_remove("TENSOR_WASM_TOKEN")
        .env_remove("TENSOR_WASM_LOG")
        .env_remove("TENSOR_WASM_API_TOKENS")
        .env_remove("TENSOR_WASM_ALLOW_PLAINTEXT_PUBLIC");
    // Defence in depth: cap every subprocess at a few seconds. The
    // bind-safety gate runs synchronously inside `serve::run` before
    // anything async happens, so a healthy run exits in tens of
    // milliseconds; the timeout is here purely to prevent a regression
    // (e.g. someone moves the gate after the listener bind) from hanging
    // the test suite.
    cmd.timeout(Duration::from_secs(10));
    cmd
}

/// `serve --addr 0.0.0.0:8080` with no tokens and no opt-in MUST refuse
/// to start. This is the headline regression guard for cli S-35.
///
/// We deliberately do NOT pass `--check-only` here: the goal is to prove
/// the gate fires in the same code path operators take in production,
/// not just under the test escape hatch.
#[test]
fn dev_mode_wildcard_bind_is_rejected() {
    tensor_wasm()
        .args(["serve", "--addr", "0.0.0.0:8080"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("refusing to bind 0.0.0.0:8080")
                .and(predicate::str::contains("loopback"))
                .and(predicate::str::contains("--allow-plaintext-public")),
        );
}

/// Same shape as the headline test but with the IPv6 wildcard — guards
/// against a future refactor that special-cases `Ipv4Addr::UNSPECIFIED`
/// and forgets the v6 equivalent.
#[test]
fn dev_mode_v6_wildcard_bind_is_rejected() {
    tensor_wasm()
        .args(["serve", "--addr", "[::]:8080"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--allow-plaintext-public"));
}

/// `serve --addr 127.0.0.1:8080` in dev mode is the documented quickstart
/// (`docs/GETTING-STARTED.md`) and MUST continue to be accepted by the
/// parser + bind-safety gate. We exit via the hidden `--check-only` flag
/// so the test never actually binds a socket.
#[test]
fn dev_mode_loopback_bind_is_accepted() {
    tensor_wasm()
        .args(["serve", "--addr", "127.0.0.1:8080", "--check-only"])
        .assert()
        .success()
        .stdout(predicate::str::contains("check_only: configuration accepted"));
}

/// IPv6 loopback parity for the accepted-config path.
#[test]
fn dev_mode_v6_loopback_bind_is_accepted() {
    tensor_wasm()
        .args(["serve", "--addr", "[::1]:8080", "--check-only"])
        .assert()
        .success();
}

/// Explicit opt-in path: `--allow-plaintext-public` silences the gate
/// even with no auth configured. Operators who knowingly run a no-auth
/// gateway behind a separate auth proxy / mesh / private subnet rely on
/// this escape hatch.
#[test]
fn dev_mode_wildcard_with_opt_in_is_accepted() {
    tensor_wasm()
        .args([
            "serve",
            "--addr",
            "0.0.0.0:8080",
            "--allow-plaintext-public",
            "--check-only",
        ])
        .assert()
        .success();
}

/// Env-var form of the opt-in (`TENSOR_WASM_ALLOW_PLAINTEXT_PUBLIC=true`)
/// must work too — container deployments typically configure this via
/// environment rather than CLI flags. Uses `true` (not `1`) because clap
/// routes bool env-var values through `bool::from_str`, which only
/// accepts the case-insensitive strings `true` / `false`.
#[test]
fn dev_mode_wildcard_with_env_opt_in_is_accepted() {
    tensor_wasm()
        .env("TENSOR_WASM_ALLOW_PLAINTEXT_PUBLIC", "true")
        .args(["serve", "--addr", "0.0.0.0:8080", "--check-only"])
        .assert()
        .success();
}

/// Auth-configured deployments (any `--token` flag, or
/// `TENSOR_WASM_API_TOKENS` populated) MUST be accepted on every
/// address. The bearer-token allowlist is the gateway's production
/// access-control surface; gating it on the bind address would break
/// every existing production deployment.
#[test]
fn auth_configured_wildcard_bind_is_accepted() {
    tensor_wasm()
        .args([
            "serve",
            "--addr",
            "0.0.0.0:8080",
            "--token",
            "secret",
            "--check-only",
        ])
        .assert()
        .success();
}

/// Parity check for the env-var form of token configuration. A wildcard
/// scope is used (`token:tenant=*`) so the entry survives the
/// scoped-token parser without emitting a deprecation warning that
/// could be mis-read as a failure.
#[test]
fn auth_configured_via_env_var_wildcard_bind_is_accepted() {
    tensor_wasm()
        .env("TENSOR_WASM_API_TOKENS", "secret:tenant=*")
        .args(["serve", "--addr", "0.0.0.0:8080", "--check-only"])
        .assert()
        .success();
}

/// Reject path must surface a single coherent line that names BOTH
/// remediation paths (`--allow-plaintext-public` AND `--token`) so an
/// operator hit by the gate in production gets a complete fix without
/// having to consult the docs. Regression guard against future
/// truncation of the error message.
#[test]
fn reject_error_names_both_remediations() {
    tensor_wasm()
        .args(["serve", "--addr", "0.0.0.0:8080"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("--allow-plaintext-public")
                .and(predicate::str::contains("--token")),
        );
}
