// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! `mps_or_fallback` must honour `CUDA_MPS_PIPE_DIRECTORY` and then
//! cache the decision for the lifetime of the process (T27).
//!
//! Each Rust integration test binary runs in its own process, so the
//! `set_var` / `remove_var` calls below cannot race with the other
//! `tests/` files. Inside this binary the test must hold all env-var
//! mutation in a single `#[test]` function (the `OnceLock` cache is
//! process-global, so once any test in this binary touches
//! `mps_or_fallback` the answer is frozen for the rest of the
//! process).
//!
//! The check itself is a filesystem probe — we point the env var at a
//! `tempfile::tempdir`, touch a `control` file inside, and assert the
//! registry reports `Mps(dir)`. The cache then makes subsequent calls
//! return that same value even when the env var is mutated underneath
//! it; this is the documented T27 contract (the MPS daemon is a
//! process-global piece of host state and is not expected to be
//! reconfigured on the fly underneath a running registry).

use std::fs::File;

use tempfile::tempdir;
use tensor_wasm_tenant::{MpsDecision, TenantRegistry, MPS_PIPE_DIRECTORY_ENV};

#[test]
fn mps_pipe_env_var_precedence_and_then_cached() {
    // Stage 1: point the env var at a directory containing `control` so
    // the FIRST observation of the cache is the MPS branch with a known
    // path. Everything we assert from here on uses that fact.
    let dir = tempdir().expect("tempdir");
    let control = dir.path().join("control");
    File::create(&control).expect("touch control file");

    // SAFETY: integration tests each compile into their own binary, and
    // this is the only `#[test]` in this file, so no concurrent reader
    // of the env var exists in this process. The `OnceLock` cache is
    // also untouched at this point — the call below is what initialises
    // it.
    std::env::set_var(MPS_PIPE_DIRECTORY_ENV, dir.path());

    let first = TenantRegistry::mps_or_fallback();
    match first {
        MpsDecision::Mps(captured) => {
            assert_eq!(
                captured,
                dir.path(),
                "MpsDecision::Mps must carry the absolute directory that contained the control pipe",
            );
        }
        MpsDecision::Fallback => {
            panic!(
                "expected Mps({:?}) — control file exists at {:?}",
                dir.path(),
                control
            );
        }
    }

    // Stage 2: mutate the env var to a directory that does NOT contain
    // `control`. With T27 caching in place, the second call must
    // observe the FIRST decision regardless — the cache wins. Without
    // the cache, this assertion would flip to Fallback and fail.
    let empty_dir = tempdir().expect("empty tempdir");
    std::env::set_var(MPS_PIPE_DIRECTORY_ENV, empty_dir.path());
    let second = TenantRegistry::mps_or_fallback();
    assert_eq!(
        first, second,
        "T27: second call must return the FIRST cached decision, not re-probe the env",
    );
    // And the path inside the cached `Mps` variant must still point at
    // the ORIGINAL temp dir (not the empty one).
    if let MpsDecision::Mps(captured) = second {
        assert_eq!(
            captured,
            dir.path(),
            "cached MPS root must be the directory probed on the first call",
        );
    } else {
        panic!("second call returned a different variant than the first");
    }

    // Stage 3: remove the env var entirely. The cache must still win.
    std::env::remove_var(MPS_PIPE_DIRECTORY_ENV);
    let third = TenantRegistry::mps_or_fallback();
    assert_eq!(
        first, third,
        "T27: removing the env var after the cache initialised must NOT change the observed decision",
    );
}
