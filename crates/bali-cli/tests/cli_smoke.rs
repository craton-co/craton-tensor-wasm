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
