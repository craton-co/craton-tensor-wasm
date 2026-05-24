// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `insta` snapshot tests pinning the help text of every `tensor-wasm` subcommand.
//!
//! These guard the user-facing CLI surface: if a flag is renamed, added, or
//! removed, the snapshot review (`cargo insta review`) makes the diff
//! explicit instead of silently shipping. Run with `INSTA_UPDATE=auto cargo
//! test --test help_snapshots` to refresh on intentional changes.

use assert_cmd::Command;

/// Capture stdout of `tensor-wasm <args>` as a UTF-8 string. Panics on non-success
/// since `--help` is expected to exit 0.
fn help(args: &[&str]) -> String {
    let mut cmd = Command::cargo_bin("tensor-wasm").expect("tensor-wasm binary built");
    cmd.env_remove("TENSOR_WASM_TOKEN").env_remove("TENSOR_WASM_LOG");
    let assertion = cmd.args(args).assert().success();
    String::from_utf8(assertion.get_output().stdout.clone())
        .expect("help output is UTF-8")
}

#[test]
fn root_help() {
    insta::assert_snapshot!("root", help(&["--help"]));
}

#[test]
fn run_help() {
    insta::assert_snapshot!("run", help(&["run", "--help"]));
}

#[test]
fn deploy_help() {
    insta::assert_snapshot!("deploy", help(&["deploy", "--help"]));
}

#[test]
fn invoke_help() {
    insta::assert_snapshot!("invoke", help(&["invoke", "--help"]));
}

#[test]
fn bench_help() {
    insta::assert_snapshot!("bench", help(&["bench", "--help"]));
}

#[test]
fn metrics_help() {
    insta::assert_snapshot!("metrics", help(&["metrics", "--help"]));
}

#[test]
fn snapshot_help() {
    insta::assert_snapshot!("snapshot", help(&["snapshot", "--help"]));
}

#[test]
fn snapshot_save_help() {
    insta::assert_snapshot!("snapshot_save", help(&["snapshot", "save", "--help"]));
}

#[test]
fn snapshot_restore_help() {
    insta::assert_snapshot!(
        "snapshot_restore",
        help(&["snapshot", "restore", "--help"])
    );
}

#[test]
fn completions_help() {
    insta::assert_snapshot!("completions", help(&["completions", "--help"]));
}
