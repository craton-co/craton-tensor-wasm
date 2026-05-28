// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! T27 perf-pass: the MPS decision is cached after the first probe.
//!
//! This file is its own integration test binary, which means its
//! process has a fresh `OnceLock<MpsDecision>` cache. The test
//! controls the env var BEFORE the first call, then mutates it to
//! something incompatible BEFORE the second call. With the cache in
//! place, the second call observes the first decision.
//!
//! Rationale lives on `TenantRegistry::mps_or_fallback`: the MPS
//! daemon pipe is process-global host state, so re-probing on every
//! call wastes a `getenv` + `stat(2)` on a hot lookup path.

use std::fs::File;

use tensor_wasm_tenant::{MpsDecision, TenantRegistry, MPS_PIPE_DIRECTORY_ENV};
use tempfile::tempdir;

#[test]
fn mps_decision_is_cached_after_first_probe() {
    // Prime the cache via a `control`-bearing tempdir so the first call
    // returns `Mps(...)`. Note: each integration test file is its own
    // binary, and this is the only `#[test]` here, so no other thread in
    // this process is racing the env var.
    let primed = tempdir().expect("tempdir");
    File::create(primed.path().join("control")).expect("touch control");
    // SAFETY: single test, single thread observing the env in this
    // binary. See module docs.
    std::env::set_var(MPS_PIPE_DIRECTORY_ENV, primed.path());

    let first = TenantRegistry::mps_or_fallback().clone();
    assert!(
        matches!(first, MpsDecision::Mps(_)),
        "first probe must observe the primed control pipe; got {first:?}"
    );

    // Now flip the env var to a directory WITHOUT a control file. If
    // the cache works, the second call still returns the first
    // decision; if it doesn't, the second call returns `Fallback` and
    // the test fails.
    let empty = tempdir().expect("tempdir");
    std::env::set_var(MPS_PIPE_DIRECTORY_ENV, empty.path());

    let second = TenantRegistry::mps_or_fallback().clone();
    assert_eq!(
        first, second,
        "T27 cache contract: second call must return the FIRST decision"
    );

    // And one more, after removing the env var entirely.
    std::env::remove_var(MPS_PIPE_DIRECTORY_ENV);
    let third = TenantRegistry::mps_or_fallback().clone();
    assert_eq!(
        first, third,
        "T27 cache contract: third call (env removed) must still return the FIRST decision"
    );
}
