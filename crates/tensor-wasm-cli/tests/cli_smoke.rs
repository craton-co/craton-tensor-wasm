// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! End-to-end smoke tests for the `tensor-wasm` CLI binary.
//!
//! These tests shell out to the compiled `tensor-wasm` binary via `assert_cmd`
//! (which uses Cargo's `CARGO_BIN_EXE_tensor-wasm` env var under the hood) and use
//! `predicates` for fluent stdout/stderr assertions. Wasm fixtures are
//! assembled in-test from WAT so we don't need to ship a pre-built `.wasm`
//! file alongside the source.

use assert_cmd::Command as AssertCmd;
use predicates::prelude::*;

/// Convenience: a fresh `assert_cmd::Command` for the `tensor-wasm` binary that
/// strips `TENSOR_WASM_TOKEN` from the env so a developer's shell config doesn't
/// silently leak credentials into the test runs.
fn tensor-wasm() -> AssertCmd {
    let mut cmd = AssertCmd::cargo_bin("tensor-wasm").expect("tensor-wasm binary built");
    cmd.env_remove("TENSOR_WASM_TOKEN").env_remove("TENSOR_WASM_LOG");
    cmd
}

#[test]
fn help_exits_zero_and_prints_usage() {
    tensor-wasm()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Usage:"));
}

#[test]
fn completions_bash_emits_script() {
    tensor-wasm()
        .args(["completions", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::contains("tensor-wasm"));
}

/// Exercise every `clap_complete::Shell` variant so future flag additions
/// can't break completion generation silently.
#[test]
fn completions_render_for_every_shell() {
    for shell in ["bash", "zsh", "fish", "powershell", "elvish"] {
        tensor-wasm()
            .args(["completions", shell])
            .assert()
            .success()
            .stdout(predicate::str::is_empty().not());
    }
}

#[test]
fn run_missing_file_errors() {
    tensor-wasm()
        .args(["run", "definitely_does_not_exist_42.wasm"])
        .assert()
        .failure();
}

// Snapshot stubs are GONE — they now exit non-zero with a "feature not yet
// shipped" message when no server is reachable. See `tests/snapshot_*` below.

#[test]
fn deploy_validates_server_url() {
    // Bogus URL → non-zero exit, clear error.
    tensor-wasm()
        .args(["deploy", "Cargo.toml", "--server", "not-a-url"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must start with http://"));
}

#[test]
fn invoke_validates_server_url() {
    tensor-wasm()
        .args([
            "invoke",
            "00000000-0000-0000-0000-000000000000",
            "--server",
            "not-a-url",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must start with http://"));
}

#[test]
fn metrics_validates_server_url() {
    tensor-wasm()
        .args(["metrics", "--server", "not-a-url"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must start with http://"));
}

#[test]
fn deploy_rejects_empty_name() {
    let wasm = wat::parse_str("(module)").expect("compile empty module");
    let tmp = tempfile::tempdir().expect("tempdir");
    let p = tmp.path().join("m.wasm");
    std::fs::write(&p, &wasm).expect("write");
    tensor-wasm()
        .args([
            "deploy",
            p.to_str().unwrap(),
            "--server",
            "http://127.0.0.1:1",
            "--name",
            "",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--name must be non-empty"));
}

#[test]
fn deploy_rejects_whitespace_name() {
    let wasm = wat::parse_str("(module)").expect("compile empty module");
    let tmp = tempfile::tempdir().expect("tempdir");
    let p = tmp.path().join("m.wasm");
    std::fs::write(&p, &wasm).expect("write");
    tensor-wasm()
        .args([
            "deploy",
            p.to_str().unwrap(),
            "--server",
            "http://127.0.0.1:1",
            "--name",
            "   ",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--name must be non-empty"));
}

#[test]
fn bench_rejects_zero_iterations() {
    let wasm = wat::parse_str(r#"(module (func (export "noop")))"#).expect("compile");
    let tmp = tempfile::tempdir().expect("tempdir");
    let p = tmp.path().join("m.wasm");
    std::fs::write(&p, &wasm).expect("write");
    tensor-wasm()
        .args(["bench", p.to_str().unwrap(), "--export", "noop", "--n", "0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--n must be >= 1"));
}

#[test]
fn run_executes_inline_wat_fixture() {
    // Build a trivial Wasm fixture with an `add` export the CLI can call.
    // Note: TensorWasmExecutor::call_export currently runs `() -> ()` signatures —
    // so the `add` function in this fixture must take no params and return
    // nothing. We still pass --args to exercise JSON validation; the
    // executor ignores the values.
    let wat = r#"
        (module
            (func (export "add"))
        )
    "#;
    let wasm = wat::parse_str(wat).expect("compile WAT fixture");
    let tmp = tempfile::tempdir().expect("create tempdir");
    let wasm_path = tmp.path().join("fixture.wasm");
    std::fs::write(&wasm_path, &wasm).expect("write wasm fixture");

    tensor-wasm()
        .args([
            "run",
            wasm_path.to_str().unwrap(),
            "--export",
            "add",
            "--args",
            "[1.0, 2.0]",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("ok"));
}

/// `127.0.0.1:1` is reserved (TCPMUX) and never bound in our CI sandboxes,
/// so every connect attempt fails fast. Used by the three HTTP-shaped
/// subcommand tests below to prove the real `reqwest` code path runs without
/// requiring a live server.
const DEAD_SERVER: &str = "http://127.0.0.1:1";

#[test]
fn deploy_against_dead_server_fails_cleanly() {
    let wasm = wat::parse_str("(module)").expect("compile empty module");
    let tmp = tempfile::tempdir().expect("tempdir");
    let p = tmp.path().join("m.wasm");
    std::fs::write(&p, &wasm).expect("write");

    let assertion = tensor-wasm()
        .args(["deploy", p.to_str().unwrap(), "--server", DEAD_SERVER])
        .assert()
        .failure();
    let out = assertion.get_output();
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
    .to_lowercase();
    assert!(
        !combined.contains("would upload"),
        "deploy still printing the legacy stub:\n{combined}"
    );
    assert!(
        !combined.contains("panicked"),
        "deploy panicked instead of returning an error:\n{combined}"
    );
    assert!(
        combined.contains("connect")
            || combined.contains("connection")
            || combined.contains("refused")
            || combined.contains("reset")
            || combined.contains("post http"),
        "expected a connection-failure-shaped error, got:\n{combined}"
    );
}

#[test]
fn invoke_against_dead_server_fails_cleanly() {
    let assertion = tensor-wasm()
        .args([
            "invoke",
            "00000000-0000-0000-0000-000000000000",
            "--server",
            DEAD_SERVER,
        ])
        .assert()
        .failure();
    let out = assertion.get_output();
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
    .to_lowercase();
    assert!(!combined.contains("would post"));
    assert!(!combined.contains("panicked"));
    assert!(
        combined.contains("connect")
            || combined.contains("connection")
            || combined.contains("refused")
            || combined.contains("reset")
            || combined.contains("post http"),
        "expected a connection-failure-shaped error, got:\n{combined}"
    );
}

#[test]
fn metrics_against_dead_server_fails_cleanly() {
    let assertion = tensor-wasm()
        .args(["metrics", "--server", DEAD_SERVER])
        .assert()
        .failure();
    let out = assertion.get_output();
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
    .to_lowercase();
    assert!(!combined.contains("would get"));
    assert!(!combined.contains("panicked"));
    assert!(
        combined.contains("connect")
            || combined.contains("connection")
            || combined.contains("refused")
            || combined.contains("reset")
            || combined.contains("get http"),
        "expected a connection-failure-shaped error, got:\n{combined}"
    );
}

#[test]
fn invoke_rejects_non_array_args() {
    tensor-wasm()
        .args([
            "invoke",
            "00000000-0000-0000-0000-000000000000",
            "--server",
            DEAD_SERVER,
            "--args",
            "{\"not\":\"array\"}",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must be a JSON array"));
}

#[test]
fn run_rejects_non_array_args() {
    let wasm = wat::parse_str(r#"(module (func (export "noop")))"#).expect("compile");
    let tmp = tempfile::tempdir().expect("tempdir");
    let p = tmp.path().join("m.wasm");
    std::fs::write(&p, &wasm).expect("write");
    tensor-wasm()
        .args([
            "run",
            p.to_str().unwrap(),
            "--export",
            "noop",
            "--args",
            "{\"not\": \"array\"}",
        ])
        .assert()
        .failure();
}

#[test]
fn snapshot_save_fails_when_api_unreachable() {
    // The API is not yet shipped. Against a dead server we get either a
    // connection error (server unreachable) — which is fine — or 404 (which
    // would map to our FEATURE_NOT_EXPOSED exit code). In neither case
    // should the CLI exit 0.
    let tmp = tempfile::tempdir().expect("tempdir");
    let out = tmp.path().join("snap.tensor-wasm");
    tensor-wasm()
        .args([
            "snapshot",
            "save",
            "--instance",
            "i-1",
            "--output",
            out.to_str().unwrap(),
            "--server",
            DEAD_SERVER,
        ])
        .assert()
        .failure();
}

#[test]
fn snapshot_restore_fails_when_api_unreachable() {
    // Need a real on-disk input so we get past local validation.
    let tmp = tempfile::tempdir().expect("tempdir");
    let p = tmp.path().join("snap.tensor-wasm");
    std::fs::write(&p, b"fake snapshot bytes").expect("write");
    tensor-wasm()
        .args([
            "snapshot",
            "restore",
            "--input",
            p.to_str().unwrap(),
            "--as-instance",
            "i-2",
            "--server",
            DEAD_SERVER,
        ])
        .assert()
        .failure();
}

#[test]
fn snapshot_restore_rejects_missing_input() {
    tensor-wasm()
        .args([
            "snapshot",
            "restore",
            "--input",
            "definitely_not_a_file_42.tensor-wasm",
            "--as-instance",
            "i-3",
            "--server",
            DEAD_SERVER,
        ])
        .assert()
        .failure();
}

