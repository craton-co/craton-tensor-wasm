//! End-to-end smoke tests for the `bali` CLI binary.
//!
//! These tests shell out to the compiled `bali` binary (via Cargo's
//! `CARGO_BIN_EXE_bali` env var, set automatically for integration tests of
//! the package owning the bin) and assert on exit status and stdout/stderr.
//! The wasm fixture is assembled in-test from WAT so we don't need to ship a
//! pre-built `.wasm` file alongside the source.

use std::path::PathBuf;
use std::process::Command;

/// Resolve the path to the built `bali` binary that Cargo just produced for
/// this integration test.
fn bali_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_bali"))
}

#[test]
fn help_exits_zero_and_prints_usage() {
    let out = Command::new(bali_bin())
        .arg("--help")
        .output()
        .expect("spawn bali --help");
    assert!(
        out.status.success(),
        "bali --help should exit 0, got {:?}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Usage:"),
        "bali --help stdout missing `Usage:`. Got:\n{stdout}"
    );
}

#[test]
fn completions_bash_emits_script() {
    let out = Command::new(bali_bin())
        .args(["completions", "bash"])
        .output()
        .expect("spawn bali completions bash");
    assert!(
        out.status.success(),
        "bali completions bash should exit 0, got {:?}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("bali"),
        "completion script does not mention `bali`. Got:\n{stdout}"
    );
}

#[test]
fn run_missing_file_errors() {
    let out = Command::new(bali_bin())
        .args(["run", "definitely_does_not_exist_42.wasm"])
        .output()
        .expect("spawn bali run nonexistent");
    assert!(
        !out.status.success(),
        "bali run on a missing file should exit non-zero, got {:?}",
        out.status
    );
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        combined
            .to_lowercase()
            .contains("definitely_does_not_exist_42")
            || combined.to_lowercase().contains("reading wasm file")
            || combined.to_lowercase().contains("no such file")
            || combined.to_lowercase().contains("cannot find"),
        "error message did not mention the missing file. Got:\n{combined}"
    );
}

#[test]
fn snapshot_save_stub_exits_zero() {
    let out = Command::new(bali_bin())
        .args(["snapshot", "save", "dummy", "out.bali"])
        .output()
        .expect("spawn bali snapshot save");
    assert!(
        out.status.success(),
        "bali snapshot save (stub) should exit 0, got {:?}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn snapshot_restore_stub_exits_zero() {
    let out = Command::new(bali_bin())
        .args(["snapshot", "restore", "in.bali"])
        .output()
        .expect("spawn bali snapshot restore");
    assert!(
        out.status.success(),
        "bali snapshot restore (stub) should exit 0, got {:?}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn deploy_validates_server_url() {
    // Bogus URL → non-zero exit, clear error.
    let out = Command::new(bali_bin())
        .args(["deploy", "Cargo.toml", "--server", "not-a-url"])
        .output()
        .expect("spawn bali deploy");
    assert!(
        !out.status.success(),
        "bali deploy with malformed --server should fail"
    );
}

#[test]
fn run_executes_inline_wat_fixture() {
    // Build a trivial Wasm fixture with an `add` export the CLI can call.
    // Note: BaliExecutor::call_export currently runs `() -> ()` signatures —
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

    let out = Command::new(bali_bin())
        .args([
            "run",
            wasm_path.to_str().unwrap(),
            "--export",
            "add",
            "--args",
            "[1.0, 2.0]",
        ])
        .output()
        .expect("spawn bali run");
    assert!(
        out.status.success(),
        "bali run on inline fixture should succeed.\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ok"),
        "expected `ok` in stdout, got:\n{stdout}"
    );
}

/// `127.0.0.1:1` is reserved (TCPMUX) and never bound in our CI sandboxes,
/// so every connect attempt fails fast. Used by the three HTTP-shaped
/// subcommand tests below to prove the real `reqwest` code path runs without
/// requiring a live server.
const DEAD_SERVER: &str = "http://127.0.0.1:1";

/// Combined stdout+stderr from a command, lowercased — convenient for
/// substring assertions on error messages whose exact text changes between
/// reqwest versions.
fn combined_lower(out: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
    .to_lowercase()
}

#[test]
fn deploy_against_dead_server_fails_cleanly() {
    // Write a minimal valid wasm so the file-read step succeeds and we
    // exercise the HTTP path.
    let wasm = wat::parse_str("(module)").expect("compile empty module");
    let tmp = tempfile::tempdir().expect("tempdir");
    let p = tmp.path().join("m.wasm");
    std::fs::write(&p, &wasm).expect("write");

    let out = Command::new(bali_bin())
        .args(["deploy", p.to_str().unwrap(), "--server", DEAD_SERVER])
        .output()
        .expect("spawn bali deploy");
    assert!(
        !out.status.success(),
        "bali deploy against {DEAD_SERVER} should fail, got {:?}",
        out.status
    );
    let msg = combined_lower(&out);
    // We expect a connect-style error, NOT the legacy `would upload` stub
    // string, NOT a panic. The exact phrasing comes from reqwest/hyper, so
    // accept any of the common variants.
    assert!(
        !msg.contains("would upload"),
        "deploy still printing the legacy stub:\n{msg}"
    );
    assert!(
        !msg.contains("panicked"),
        "deploy panicked instead of returning an error:\n{msg}"
    );
    assert!(
        msg.contains("connect")
            || msg.contains("connection")
            || msg.contains("refused")
            || msg.contains("reset")
            || msg.contains("post http"),
        "expected a connection-failure-shaped error, got:\n{msg}"
    );
}

#[test]
fn invoke_against_dead_server_fails_cleanly() {
    let out = Command::new(bali_bin())
        .args([
            "invoke",
            "00000000-0000-0000-0000-000000000000",
            "--server",
            DEAD_SERVER,
        ])
        .output()
        .expect("spawn bali invoke");
    assert!(
        !out.status.success(),
        "bali invoke against {DEAD_SERVER} should fail"
    );
    let msg = combined_lower(&out);
    assert!(
        !msg.contains("would post"),
        "invoke still printing the legacy stub:\n{msg}"
    );
    assert!(
        !msg.contains("panicked"),
        "invoke panicked instead of returning an error:\n{msg}"
    );
    assert!(
        msg.contains("connect")
            || msg.contains("connection")
            || msg.contains("refused")
            || msg.contains("reset")
            || msg.contains("post http"),
        "expected a connection-failure-shaped error, got:\n{msg}"
    );
}

#[test]
fn metrics_against_dead_server_fails_cleanly() {
    let out = Command::new(bali_bin())
        .args(["metrics", "--server", DEAD_SERVER])
        .output()
        .expect("spawn bali metrics");
    assert!(
        !out.status.success(),
        "bali metrics against {DEAD_SERVER} should fail"
    );
    let msg = combined_lower(&out);
    assert!(
        !msg.contains("would get"),
        "metrics still printing the legacy stub:\n{msg}"
    );
    assert!(
        !msg.contains("panicked"),
        "metrics panicked instead of returning an error:\n{msg}"
    );
    assert!(
        msg.contains("connect")
            || msg.contains("connection")
            || msg.contains("refused")
            || msg.contains("reset")
            || msg.contains("get http"),
        "expected a connection-failure-shaped error, got:\n{msg}"
    );
}

#[test]
fn invoke_rejects_non_array_args() {
    let out = Command::new(bali_bin())
        .args([
            "invoke",
            "00000000-0000-0000-0000-000000000000",
            "--server",
            DEAD_SERVER,
            "--args",
            "{\"not\":\"array\"}",
        ])
        .output()
        .expect("spawn bali invoke");
    assert!(
        !out.status.success(),
        "bali invoke with object --args should fail before any HTTP call"
    );
    let msg = combined_lower(&out);
    assert!(
        msg.contains("--args must be a json array") || msg.contains("must be a json array"),
        "expected validation error about JSON array, got:\n{msg}"
    );
}

#[test]
fn run_rejects_non_array_args() {
    // Build a tiny valid wasm so the read step succeeds and we exercise the
    // --args validation specifically.
    let wasm = wat::parse_str(r#"(module (func (export "noop")))"#).expect("compile");
    let tmp = tempfile::tempdir().expect("tempdir");
    let p = tmp.path().join("m.wasm");
    std::fs::write(&p, &wasm).expect("write");
    let out = Command::new(bali_bin())
        .args([
            "run",
            p.to_str().unwrap(),
            "--export",
            "noop",
            "--args",
            "{\"not\": \"array\"}",
        ])
        .output()
        .expect("spawn bali run");
    assert!(
        !out.status.success(),
        "expected non-zero exit for non-array --args"
    );
}
