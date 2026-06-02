// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Integration coverage for [`AppConfig::from_env`] and the snapshot
//! HMAC env-var schema.
//!
//! Each Rust integration test binary runs in its own process, so the
//! `set_var` / `remove_var` calls below cannot race with the other
//! `tests/` files. Inside this binary the cases still run on the test
//! threadpool, so every case lives inside a single `#[test]` function
//! to avoid the well-known `std::env::set_var` data-race trap (the same
//! convention `mps_pipe_env_var.rs` uses elsewhere in the workspace).
//!
//! The future `/snapshot/save` and `/snapshot/restore` routes are not
//! yet wired into `build_router` — see the doc comment on
//! `tensor_wasm_api::config` and `crates/tensor-wasm-cli/src/cmd/snapshot.rs`.
//! This file asserts the config knob those routes will consume, not the
//! routes themselves.

use tensor_wasm_api::{
    AppConfig, ConfigError, HexParseReason, ENV_SNAPSHOT_HMAC_KEY, ENV_SNAPSHOT_REQUIRE_SIGNATURE,
    SNAPSHOT_HMAC_KEY_LEN,
};

#[test]
fn snapshot_hmac_env_var_round_trip() {
    // Start from a known-clean state. We restore the same on the way out
    // so a wedged earlier test doesn't poison this one (and vice versa).
    let prior_key = std::env::var_os(ENV_SNAPSHOT_HMAC_KEY);
    let prior_require = std::env::var_os(ENV_SNAPSHOT_REQUIRE_SIGNATURE);

    // SAFETY: integration tests each compile into their own binary, and
    // every case in this file runs inside this single `#[test]` so no
    // concurrent reader of the env var exists in this process.
    std::env::remove_var(ENV_SNAPSHOT_HMAC_KEY);
    std::env::remove_var(ENV_SNAPSHOT_REQUIRE_SIGNATURE);

    // --- Case 1: unset env vars → all defaults --------------------------
    let cfg = AppConfig::from_env().expect("defaults parse");
    assert!(cfg.snapshot_hmac_key.is_none(), "unset key must be None");
    assert!(!cfg.snapshot_require_signature, "unset bool must be false");

    // --- Case 2: well-formed lowercase hex key --------------------------
    let hex = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
    assert_eq!(hex.len(), SNAPSHOT_HMAC_KEY_LEN * 2);
    std::env::set_var(ENV_SNAPSHOT_HMAC_KEY, hex);
    let cfg = AppConfig::from_env().expect("valid hex parses");
    let key = cfg.snapshot_hmac_key.expect("key is populated");
    assert_eq!(key[0], 0x00);
    assert_eq!(key[1], 0x11);
    assert_eq!(key[15], 0xff);
    assert_eq!(key[31], 0xff);

    // --- Case 3: uppercase hex also accepted ---------------------------
    let hex_upper = "DEADBEEFCAFEBABE0123456789ABCDEFDEADBEEFCAFEBABE0123456789ABCDEF";
    std::env::set_var(ENV_SNAPSHOT_HMAC_KEY, hex_upper);
    let cfg = AppConfig::from_env().expect("uppercase hex parses");
    let key = cfg.snapshot_hmac_key.expect("key populated");
    assert_eq!(&key[..4], &[0xde, 0xad, 0xbe, 0xef]);

    // --- Case 4: surrounding whitespace stripped -----------------------
    std::env::set_var(ENV_SNAPSHOT_HMAC_KEY, format!("  {hex}\n"));
    let cfg = AppConfig::from_env().expect("trimmed hex parses");
    assert!(
        cfg.snapshot_hmac_key.is_some(),
        "trim should accept padding"
    );

    // --- Case 5: short hex string → hard error -------------------------
    std::env::set_var(ENV_SNAPSHOT_HMAC_KEY, "deadbeef");
    let err = AppConfig::from_env().expect_err("short hex must fail");
    match err {
        ConfigError::InvalidHexKey {
            var,
            reason: HexParseReason::WrongLength { actual: 8 },
        } => assert_eq!(var, ENV_SNAPSHOT_HMAC_KEY),
        other => panic!("expected WrongLength{{actual:8}}, got {other:?}"),
    }

    // --- Case 6: garbage character (correct length) → hard error ------
    let bad = "g0112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
    assert_eq!(bad.len(), 64);
    std::env::set_var(ENV_SNAPSHOT_HMAC_KEY, bad);
    let err = AppConfig::from_env().expect_err("non-hex char must fail");
    assert!(
        matches!(
            err,
            ConfigError::InvalidHexKey {
                reason: HexParseReason::InvalidCharacter,
                ..
            }
        ),
        "expected InvalidCharacter, got {err:?}"
    );

    // --- Case 7: empty (set but empty) silently degrades to None ------
    std::env::set_var(ENV_SNAPSHOT_HMAC_KEY, "");
    let cfg = AppConfig::from_env().expect("empty parses");
    assert!(cfg.snapshot_hmac_key.is_none(), "empty string == unset");

    // --- Case 8: require-signature toggle accepts `true`/`false` ------
    std::env::set_var(ENV_SNAPSHOT_HMAC_KEY, hex);
    std::env::set_var(ENV_SNAPSHOT_REQUIRE_SIGNATURE, "true");
    let cfg = AppConfig::from_env().expect("true parses");
    assert!(
        cfg.snapshot_require_signature,
        "true must turn the toggle on"
    );

    std::env::set_var(ENV_SNAPSHOT_REQUIRE_SIGNATURE, "FALSE");
    let cfg = AppConfig::from_env().expect("FALSE parses");
    assert!(
        !cfg.snapshot_require_signature,
        "FALSE must turn the toggle off"
    );

    // --- Case 9: garbage bool → hard error ----------------------------
    std::env::set_var(ENV_SNAPSHOT_REQUIRE_SIGNATURE, "yes");
    let err = AppConfig::from_env().expect_err("non-bool must fail");
    assert!(
        matches!(err, ConfigError::InvalidBool { .. }),
        "got {err:?}"
    );

    // --- Case 10: require=true with no key emits warn but still loads -
    std::env::remove_var(ENV_SNAPSHOT_HMAC_KEY);
    std::env::set_var(ENV_SNAPSHOT_REQUIRE_SIGNATURE, "true");
    let cfg = AppConfig::from_env().expect("require-without-key is allowed");
    assert!(cfg.snapshot_require_signature);
    assert!(cfg.snapshot_hmac_key.is_none());

    // --- Case 11: hidden builder sets the field without env vars ------
    std::env::remove_var(ENV_SNAPSHOT_HMAC_KEY);
    std::env::remove_var(ENV_SNAPSHOT_REQUIRE_SIGNATURE);
    let explicit_key = [0x42u8; SNAPSHOT_HMAC_KEY_LEN];
    let cfg = AppConfig::default()
        .with_snapshot_hmac_key(explicit_key)
        .with_snapshot_require_signature(true);
    assert_eq!(cfg.snapshot_hmac_key, Some(explicit_key));
    assert!(cfg.snapshot_require_signature);

    // --- Restore caller's env so other tests in this process see the
    // same view they started with. The `match` keeps the restoration
    // honest in the (unlikely) case that the variable was unset.
    match prior_key {
        Some(v) => std::env::set_var(ENV_SNAPSHOT_HMAC_KEY, v),
        None => std::env::remove_var(ENV_SNAPSHOT_HMAC_KEY),
    }
    match prior_require {
        Some(v) => std::env::set_var(ENV_SNAPSHOT_REQUIRE_SIGNATURE, v),
        None => std::env::remove_var(ENV_SNAPSHOT_REQUIRE_SIGNATURE),
    }
}
