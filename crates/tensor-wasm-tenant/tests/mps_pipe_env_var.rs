// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! `mps_or_fallback` must honour `CUDA_MPS_PIPE_DIRECTORY`.
//!
//! Each Rust integration test binary runs in its own process, so the
//! `set_var` / `remove_var` calls below cannot race with the other
//! `tests/` files. Inside this binary the tests still run on threads,
//! so we keep all env-var mutation in a single `#[test]` function to
//! avoid the well-known `std::env::set_var` data-race trap.
//!
//! The check itself is a filesystem probe — we point the env var at a
//! `tempfile::tempdir`, touch a `control` file inside, and assert the
//! registry reports `Mps`. After dropping the env var, the registry
//! must fall back to the historical `/tmp/nvidia-mps` path, which on
//! this test host (Windows or a clean Linux runner) is guaranteed not
//! to exist, so the answer is `Fallback`.

use std::fs::File;
use std::path::Path;

use tensor_wasm_tenant::{MpsDecision, TenantRegistry, MPS_CONTROL_PATH, MPS_PIPE_DIRECTORY_ENV};
use tempfile::tempdir;

#[test]
fn mps_pipe_env_var_precedence_and_fallback() {
    // If the test host happens to be a CUDA box with a real MPS daemon
    // running at the default location, the "after drop" assertion below
    // would be Mps, not Fallback. We don't assert the fallback branch in
    // that case — the documented contract is "fall back to the default
    // path", which we can't disprove when the default path is itself live.
    let default_live = Path::new(MPS_CONTROL_PATH).join("control").exists();

    // --- Case 1: env var points at a directory containing `control` ---
    let dir = tempdir().expect("tempdir");
    let control = dir.path().join("control");
    File::create(&control).expect("touch control file");

    // SAFETY: integration tests each compile into their own binary, and
    // this is the only `#[test]` in this file, so no concurrent reader
    // of the env var exists in this process.
    std::env::set_var(MPS_PIPE_DIRECTORY_ENV, dir.path());
    assert_eq!(
        TenantRegistry::mps_or_fallback(),
        MpsDecision::Mps,
        "env-var-pointed control file must be detected as MPS",
    );

    // --- Case 2: env var points at a directory with no `control` file ---
    let empty_dir = tempdir().expect("empty tempdir");
    std::env::set_var(MPS_PIPE_DIRECTORY_ENV, empty_dir.path());
    if !default_live {
        assert_eq!(
            TenantRegistry::mps_or_fallback(),
            MpsDecision::Fallback,
            "env var present but no control file inside the directory \
             must not be reported as MPS-up",
        );
    }

    // --- Case 3: env var removed → fall back to the default path ---
    std::env::remove_var(MPS_PIPE_DIRECTORY_ENV);
    if !default_live {
        assert_eq!(
            TenantRegistry::mps_or_fallback(),
            MpsDecision::Fallback,
            "without the env var and without a default control file, \
             the probe must report Fallback",
        );
    }
}
