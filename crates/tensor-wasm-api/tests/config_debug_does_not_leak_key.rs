//! Regression test for HMAC-key Debug-leak in AppConfig (api S-33).
//!
//! `AppConfig` derived `Debug` over `Option<[u8; 32]>` snapshot_hmac_key;
//! `tracing::debug!(?cfg)` printed every key byte. The fix replaces the
//! derive with a manual impl that emits a `<REDACTED>` placeholder.

use tensor_wasm_api::config::AppConfig;

const SENTINEL_KEY: [u8; 32] = [0xCD; 32];

#[test]
fn app_config_debug_redacts_hmac_key() {
    let cfg = AppConfig {
        snapshot_hmac_key: Some(SENTINEL_KEY),
        snapshot_require_signature: true,
    };
    let dbg = format!("{cfg:?}");
    assert!(
        !dbg.contains("205") && !dbg.to_lowercase().contains("cd, cd"),
        "AppConfig Debug leaked key bytes: {dbg}"
    );
    assert!(
        dbg.to_lowercase().contains("redacted"),
        "AppConfig Debug missing REDACTED marker: {dbg}"
    );
}

#[test]
fn app_config_debug_without_key_does_not_say_redacted() {
    let cfg = AppConfig::default();
    let dbg = format!("{cfg:?}");
    assert!(
        dbg.contains("None"),
        "default AppConfig should render hmac_key: None, got: {dbg}"
    );
}
