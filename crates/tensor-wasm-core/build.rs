// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Build script for `tensor-wasm-core`.
//!
//! Surfaces a small set of compile-time environment variables so the
//! `tensor_wasm_build_info` Prometheus gauge has stable identity labels:
//!
//! - `TENSOR_WASM_GIT_SHA` — `git rev-parse HEAD` output, or `"unknown"`
//!   if `git` is not on `PATH`, the working tree is a source tarball, or
//!   the invocation fails for any other reason.
//! - `TENSOR_WASM_RUSTC_VERSION` — first line of `rustc --version`, or
//!   `"unknown"` if `rustc` is not on `PATH` (it always is during a real
//!   cargo build, but defensive code costs nothing and keeps hermetic /
//!   sandboxed CI happy).
//! - `TENSOR_WASM_PROFILE` and `TENSOR_WASM_TARGET` — forwarded from the
//!   `PROFILE` and `TARGET` variables cargo sets *for the build script
//!   itself*. They are not visible to the crate's `env!()` calls at
//!   compile time unless the build script re-emits them via
//!   `cargo:rustc-env=`, which is exactly what this script does.
//!
//! Hard requirement: this script MUST NOT panic on hosts without `git`
//! or `rustc` on `PATH`. Source tarballs (no `.git` directory) and
//! hermetic build systems both exercise that path.

use std::env;
use std::process::Command;

fn main() {
    // A new commit (HEAD moves) should re-trigger this script so the
    // baked-in SHA stays accurate. We rerun on `.git/HEAD` rather than
    // every file under `.git/`; that captures `git commit`, `git checkout`,
    // and `git reset` without rebuilding on every dangling reflog edit.
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=build.rs");

    let git_sha = run_capture("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let rustc_version = run_capture("rustc", &["--version"]).unwrap_or_else(|| "unknown".into());

    // PROFILE and TARGET are set by cargo for the build script's
    // execution environment. Re-emit them so the crate proper can
    // `env!("TENSOR_WASM_PROFILE")` / `env!("TENSOR_WASM_TARGET")`.
    //
    // Route both through the same single-line filter `git_sha` /
    // `rustc_version` get (`.lines().next()`). Defense-in-depth: an
    // embedded newline in either value would otherwise terminate the
    // `cargo:rustc-env=` directive and let the trailing bytes emit a
    // second, attacker-controlled cargo directive.
    let profile = env_single_line("PROFILE");
    let target = env_single_line("TARGET");

    println!("cargo:rustc-env=TENSOR_WASM_GIT_SHA={}", git_sha);
    println!(
        "cargo:rustc-env=TENSOR_WASM_RUSTC_VERSION={}",
        rustc_version
    );
    println!("cargo:rustc-env=TENSOR_WASM_PROFILE={}", profile);
    println!("cargo:rustc-env=TENSOR_WASM_TARGET={}", target);
}

/// Read environment variable `name` and return its first line, trimmed.
///
/// Falls back to the literal `"unknown"` when the variable is unset or
/// its first line is empty. Mirrors the single-line discipline
/// [`run_capture`] applies to `git` / `rustc` output so a value carrying
/// an embedded newline cannot terminate the `cargo:rustc-env=` directive
/// early and emit a second cargo directive from the trailing bytes.
fn env_single_line(name: &str) -> String {
    let raw = env::var(name).unwrap_or_default();
    let first = raw.lines().next().unwrap_or("").trim();
    if first.is_empty() {
        "unknown".into()
    } else {
        first.to_string()
    }
}

/// Run `cmd args...` and return trimmed stdout on success.
///
/// Returns `None` if the binary is missing from `PATH`, the process
/// fails to spawn, exits non-zero, or produces non-UTF-8 output. The
/// caller substitutes a literal `"unknown"` for the `None` case so
/// the resulting `cargo:rustc-env=` line is always present and the
/// downstream `env!()` never fails the build.
fn run_capture(cmd: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(cmd).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8(output.stdout).ok()?;
    let trimmed = s.lines().next().unwrap_or("").trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}
