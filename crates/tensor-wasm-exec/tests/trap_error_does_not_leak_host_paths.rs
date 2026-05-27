// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression guard for exec S-9 (host-path / pointer leak via error envelopes).
//!
//! Before the fix, `From<ExecError> for TensorWasmError` rendered the inner
//! `wasmtime::Error` via `format!("{err:#}")`, the *alternate* form that walks
//! the entire `#[source]` chain. Wasmtime traps and Cranelift compile errors
//! routinely include host pointer addresses, host file system paths, and
//! internal stack-frame / module names — none of which are safe to expose to
//! an untrusted caller.
//!
//! This test loads a tiny wasm module that traps at runtime via `unreachable`,
//! drives it through the executor, converts the resulting `ExecError` into the
//! public `TensorWasmError`, and asserts that NEITHER `Display` nor the
//! alternate `Display` (`{:#}`) of the converted error contains any of the
//! known leak markers.

use std::sync::Arc;

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::TenantId;
use tensor_wasm_exec::engine::TensorWasmEngine;
use tensor_wasm_exec::executor::{SpawnConfig, TensorWasmExecutor};

/// Substrings that must never appear in a client-facing error payload. Each
/// represents a class of host-internal detail that the pre-fix `format!("{err:#}")`
/// path was observed to (or could plausibly) emit.
const FORBIDDEN_SUBSTRINGS: &[&str] = &[
    "0x",            // any hex pointer address (also covers e.g. "0x1234abcd")
    "/usr/",         // POSIX system paths
    "\\\\?\\C:\\",   // Windows extended-length device paths
    "wasmtime/runtime", // wasmtime internal module path
    "__libc_",       // glibc symbol names from stack traces
    "cranelift",     // Cranelift compiler internals
];

fn trapping_wasm() -> Vec<u8> {
    wat::parse_str(r#"(module (func (export "go") (unreachable)))"#)
        .expect("trapping wat parses")
}

fn assert_no_leaks(label: &str, rendered: &str) {
    let lower = rendered.to_ascii_lowercase();
    for needle in FORBIDDEN_SUBSTRINGS {
        let needle_lower = needle.to_ascii_lowercase();
        assert!(
            !lower.contains(&needle_lower),
            "{label}: rendered error contains forbidden substring {needle:?}; \
             full rendered text was:\n---\n{rendered}\n---",
        );
    }
}

#[tokio::test]
async fn trap_error_does_not_leak_host_paths_or_pointers() {
    let engine = Arc::new(TensorWasmEngine::new().expect("engine"));
    let exec = TensorWasmExecutor::new(engine);

    let id = exec
        .spawn_instance(SpawnConfig::for_tenant(TenantId(1)), &trapping_wasm())
        .await
        .expect("spawn trapping module");

    let exec_err = exec
        .call_export(id, "go")
        .await
        .expect_err("unreachable must trap");

    // Convert into the public `TensorWasmError` — this is the boundary that
    // S-9 covers. The exec-internal `ExecError` may carry the raw wasmtime
    // error for server-side logging; only the converted form is what crosses
    // a trust boundary.
    let public_err: TensorWasmError = exec_err.into();

    let plain = format!("{public_err}");
    let alt = format!("{public_err:#}");

    // Sanity: the converted error must be classified as a trap. If this
    // ever flips to `WasmCompile` something is very wrong with the
    // classifier.
    assert!(
        matches!(public_err, TensorWasmError::WasmTrap(_)),
        "expected WasmTrap, got {public_err:?}",
    );

    assert_no_leaks("Display", &plain);
    assert_no_leaks("Display (alt)", &alt);

    // Defensive cleanup — the executor must still be in a usable state.
    exec.terminate(id).await.expect("terminate after trap");
}
