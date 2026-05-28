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
fn tensor_wasm() -> AssertCmd {
    let mut cmd = AssertCmd::cargo_bin("tensor-wasm").expect("tensor-wasm binary built");
    cmd.env_remove("TENSOR_WASM_TOKEN").env_remove("TENSOR_WASM_LOG");
    cmd
}

#[test]
fn help_exits_zero_and_prints_usage() {
    tensor_wasm()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Usage:"));
}

#[test]
fn completions_bash_emits_script() {
    tensor_wasm()
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
        tensor_wasm()
            .args(["completions", shell])
            .assert()
            .success()
            .stdout(predicate::str::is_empty().not());
    }
}

#[test]
fn run_missing_file_errors() {
    tensor_wasm()
        .args(["run", "definitely_does_not_exist_42.wasm"])
        .assert()
        .failure();
}

// Snapshot stubs are GONE — they now exit non-zero with a "feature not yet
// shipped" message when no server is reachable. See `tests/snapshot_*` below.

#[test]
fn deploy_validates_server_url() {
    // Bogus URL → non-zero exit, clear error.
    tensor_wasm()
        .args(["deploy", "Cargo.toml", "--server", "not-a-url"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("must start with http://"));
}

#[test]
fn invoke_validates_server_url() {
    tensor_wasm()
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
    tensor_wasm()
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
    tensor_wasm()
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
    tensor_wasm()
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
    tensor_wasm()
        .args(["bench", p.to_str().unwrap(), "--export", "noop", "--n", "0"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--n must be >= 1"));
}

#[test]
fn run_executes_inline_wat_fixture() {
    // Build a no-arg fixture and confirm the legacy `() -> ()` path still
    // prints the literal `ok`. The richer `(i32, i32) -> i32` path is
    // covered by `run_passes_args_to_adder` below.
    let wat = r#"
        (module
            (func (export "noop"))
        )
    "#;
    let wasm = wat::parse_str(wat).expect("compile WAT fixture");
    let tmp = tempfile::tempdir().expect("create tempdir");
    let wasm_path = tmp.path().join("fixture.wasm");
    std::fs::write(&wasm_path, &wasm).expect("write wasm fixture");

    tensor_wasm()
        .args(["run", wasm_path.to_str().unwrap(), "--export", "noop"])
        .assert()
        .success()
        .stdout(predicate::str::contains("ok"));
}

/// Verify `--args` is actually threaded into the call. An `(i32, i32) ->
/// i32` adder built inline as WAT receives `[1, 2]` and the CLI prints
/// `3` (the scalar-collapse rule in `run.rs` unwraps the single-element
/// result list). Pre-feature this test would have hung the runtime: the
/// executor only knew the `() -> ()` signature, so the dynamic call
/// arity-mismatched and trapped.
#[test]
fn run_passes_args_to_adder() {
    let wat = r#"
        (module
            (func (export "add") (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add)
        )
    "#;
    let wasm = wat::parse_str(wat).expect("compile adder WAT");
    let tmp = tempfile::tempdir().expect("create tempdir");
    let wasm_path = tmp.path().join("adder.wasm");
    std::fs::write(&wasm_path, &wasm).expect("write wasm fixture");

    tensor_wasm()
        .args([
            "run",
            wasm_path.to_str().unwrap(),
            "--export",
            "add",
            "--args",
            "[1, 2]",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("3"));
}

/// `--args` is parsed and validated but the values are not forwarded to
/// `call_export` today (executor only supports `() -> ()`). When the
/// caller passes a non-empty array we must emit a loud stderr warning so
/// the user knows their arguments were silently dropped. Regression
/// coverage for cli "Bug 1 — `run --args` silently dropped".
#[test]
fn run_warns_when_args_silently_dropped() {
    let wat = r#"
        (module
            (func (export "add"))
        )
    "#;
    let wasm = wat::parse_str(wat).expect("compile WAT fixture");
    let tmp = tempfile::tempdir().expect("create tempdir");
    let wasm_path = tmp.path().join("fixture.wasm");
    std::fs::write(&wasm_path, &wasm).expect("write wasm fixture");

    let assertion = tensor_wasm()
        .args([
            "run",
            wasm_path.to_str().unwrap(),
            "--export",
            "add",
            "--args",
            "[1,2]",
        ])
        .assert()
        .success();
    let stderr = String::from_utf8_lossy(&assertion.get_output().stderr).to_string();
    // Use insta to pin the user-visible warning copy. The line is
    // wrapped/joined into a single rendered string in the binary; the
    // snapshot makes any future drift visible in code review.
    let warning_line = stderr
        .lines()
        .find(|l| l.starts_with("warning: --args"))
        .unwrap_or("");
    insta::assert_snapshot!("run_args_dropped_warning", warning_line);
    assert!(
        stderr.contains("ignoring 2 argument(s)"),
        "warning must name the dropped count, got: {stderr}"
    );
}

/// Empty `--args '[]'` is a no-op: we still validate the JSON shape but
/// must NOT emit the "silently dropped" warning, since the user is not
/// actually passing arguments. The warning is reserved for the case where
/// the user has a real expectation that the values will reach the guest.
#[test]
fn run_does_not_warn_for_empty_args_array() {
    let wat = r#"
        (module
            (func (export "noop"))
        )
    "#;
    let wasm = wat::parse_str(wat).expect("compile WAT fixture");
    let tmp = tempfile::tempdir().expect("create tempdir");
    let wasm_path = tmp.path().join("fixture.wasm");
    std::fs::write(&wasm_path, &wasm).expect("write wasm fixture");

    let assertion = tensor_wasm()
        .args([
            "run",
            wasm_path.to_str().unwrap(),
            "--export",
            "noop",
            "--args",
            "[]",
        ])
        .assert()
        .success();
    let stderr = String::from_utf8_lossy(&assertion.get_output().stderr);
    assert!(
        !stderr.contains("not forwarded to the guest"),
        "empty array must not trigger the dropped-args warning, got: {stderr}"
    );
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

    let assertion = tensor_wasm()
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
    let assertion = tensor_wasm()
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
    let assertion = tensor_wasm()
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
    tensor_wasm()
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
    tensor_wasm()
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
    tensor_wasm()
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
    tensor_wasm()
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
    tensor_wasm()
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

// -- `tensor-wasm kernel` smoke tests -----------------------------------
//
// v0.3.8 (B6.4): the server-side `/kernels` endpoints landed, so the
// CLI now performs real HTTP calls. These smoke tests exercise the new
// surface against `DEAD_SERVER` (no service bound) so connection
// failures stand in for "the wire is live" — the integration tests in
// `crates/tensor-wasm-api/tests/kernel_registry_routes.rs` cover the
// end-to-end success path. See `crates/tensor-wasm-cli/src/cmd/kernel.rs`.

#[test]
fn kernel_publish_against_dead_server_fails_cleanly() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ptx = tmp.path().join("kernel.ptx");
    std::fs::write(&ptx, b"// fake ptx").expect("write ptx");
    let key = tmp.path().join("hmac.key");
    std::fs::write(&key, "42".repeat(32)).expect("write key");

    let assertion = tensor_wasm()
        .args([
            "kernel",
            "publish",
            "matmul.f32",
            "1.0.0",
            "--ptx-file",
            ptx.to_str().unwrap(),
            "--sm",
            "80",
            "--key-file",
            key.to_str().unwrap(),
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
    assert!(
        !combined.contains("panicked"),
        "publish panicked instead of returning an error:\n{combined}"
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
fn kernel_list_against_dead_server_fails_cleanly() {
    let assertion = tensor_wasm()
        .args(["kernel", "list", "--server", DEAD_SERVER])
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
        combined.contains("connect")
            || combined.contains("connection")
            || combined.contains("refused")
            || combined.contains("reset")
            || combined.contains("get http"),
        "expected a connection-failure-shaped error, got:\n{combined}"
    );
}

#[test]
fn kernel_verify_succeeds_locally_when_manifest_signed_under_key() {
    // `verify` is local-only (no network); pin the happy path so the
    // exit-code contract for CI pipelines that gate releases on
    // signature verification stays stable across releases.
    use tensor_wasm_jit::registry::{sign_manifest, KernelManifest};

    let tmp = tempfile::tempdir().expect("tempdir");
    let key_path = tmp.path().join("hmac.key");
    let key_hex = "42".repeat(32);
    std::fs::write(&key_path, &key_hex).expect("write key");
    let key = [0x42u8; 32];

    let digest = *blake3::hash(b"// fake ptx\n").as_bytes();
    let mut m = KernelManifest::new(
        "matmul.f32".to_string(),
        "1.0.0".to_string(),
        80,
        digest,
        [0u8; 32],
        0,
        "smoke-test".to_string(),
    );
    m.signature = sign_manifest(&m, &key);
    let manifest_path = tmp.path().join("manifest.json");
    std::fs::write(&manifest_path, serde_json::to_vec(&m).unwrap()).unwrap();

    tensor_wasm()
        .args([
            "kernel",
            "verify",
            "matmul.f32@1.0.0",
            "--manifest-file",
            manifest_path.to_str().unwrap(),
            "--key-file",
            key_path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("verifies under the supplied key"));
}

#[test]
fn kernel_help_lists_all_subactions() {
    // Help is the design-partner-facing surface — if `publish`, `list`,
    // or `verify` ever silently disappear from the binary, this test
    // catches it before docs/KERNEL-REGISTRY.md drifts.
    tensor_wasm()
        .args(["kernel", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("publish"))
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("verify"));
}

