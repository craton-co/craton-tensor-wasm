// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Top-level application config knobs that don't fit naturally into any
//! single middleware-scoped config struct.
//!
//! The historical `tensor-wasm-api` pattern co-locates configuration with
//! the middleware it parametrises (see
//! [`crate::middleware::AuthConfig`], [`crate::middleware::TenantConfig`],
//! [`crate::rate_limit::RateLimitConfig`], [`crate::audit::AuditConfig`]).
//! That works fine for knobs that flow into exactly one layer, but breaks
//! down for cross-cutting concerns whose consumer spans more than one
//! route — e.g. the snapshot HMAC key, which is parsed at server startup
//! and consumed by the `/snapshot/save` and `/snapshot/restore` route
//! handlers.
//!
//! This module hosts that small set of cross-cutting knobs as
//! [`AppConfig`], keeping the env-var schema in one place.
//!
//! ## Env-var schema
//!
//! **Both variables below are LIVE (M5).** They are parsed and validated
//! at startup by [`AppConfig::from_env`], which
//! [`crate::server::build_router`] now calls and threads onto the shared
//! [`crate::routes::AppState`] (via `AppState::with_app_config`). The
//! `/snapshot/save` and `/snapshot/restore` routes consume the resulting
//! [`AppConfig`] to sign and verify snapshot blobs.
//!
//! | Variable                                       | Format         | Default | Meaning                                                                                                            |
//! |------------------------------------------------|----------------|---------|--------------------------------------------------------------------------------------------------------------------|
//! | `TENSOR_WASM_API_SNAPSHOT_HMAC_KEY`            | hex (64 chars) | unset   | `/snapshot/save` HMAC-SHA256-signs the returned blob and `/snapshot/restore` verifies it. Unset ⇒ both routes return `503 snapshot_signing_not_configured`. Malformed values are a hard startup error. |
//! | `TENSOR_WASM_API_SNAPSHOT_REQUIRE_SIGNATURE`   | `true`/`false` | `false` | Strict-restore posture: refuse unsigned (v2) snapshots. `/snapshot/restore` enforces signature verification unconditionally as its hardened default; this knob is the operator surface for that posture. |
//!
//! ## Status
//!
//! The `/snapshot/save` and `/snapshot/restore` HTTP routes are wired into
//! [`crate::server::build_router`], which reads
//! [`AppConfig::from_env`] at startup and installs it on the
//! [`crate::routes::AppState`]. The key is therefore a **live knob** at
//! runtime (closes finding M5): with it set, save returns a signed blob
//! and restore verifies the HMAC; with it unset, both routes report
//! `503 snapshot_signing_not_configured`. A malformed key / toggle is a
//! hard startup failure (`build_router` panics) — the gateway refuses to
//! come up serving snapshot routes under a misconfigured signing key
//! rather than silently downgrading restore integrity.

use std::fmt;

/// Environment variable carrying the hex-encoded HMAC-SHA256 key used for
/// signing and verifying snapshot blobs.
///
/// **LIVE (M5).** Consumed by the `/snapshot/save` and `/snapshot/restore`
/// routes: [`AppConfig::from_env`] is read by
/// [`crate::server::build_router`] and threaded onto the
/// [`crate::routes::AppState`]. With the key set, save returns an
/// HMAC-SHA256-signed blob and restore verifies the signature; with it
/// unset both routes return `503 snapshot_signing_not_configured`.
///
/// 64 lowercase or uppercase hex characters (32 bytes). Any other length
/// or non-hex character is a hard parse error from
/// [`AppConfig::from_env`] (and a startup panic from `build_router`).
pub const ENV_SNAPSHOT_HMAC_KEY: &str = "TENSOR_WASM_API_SNAPSHOT_HMAC_KEY";

/// Environment variable selecting the strict-restore posture: when set to
/// `true` (case-insensitive) snapshot restore refuses v2 (unsigned)
/// snapshots. Defaults to `false`.
///
/// **LIVE (M5).** Carried on the [`crate::routes::AppState`] alongside
/// [`ENV_SNAPSHOT_HMAC_KEY`]. The `/snapshot/restore` route enforces
/// signature verification unconditionally as its hardened default (the
/// gateway only ever writes signed blobs), so this knob is the documented
/// operator surface for that posture; an operator who sets it gets a
/// confirming log line on restore. A one-shot startup warning still fires
/// when it is `true` with no key set.
pub const ENV_SNAPSHOT_REQUIRE_SIGNATURE: &str = "TENSOR_WASM_API_SNAPSHOT_REQUIRE_SIGNATURE";

/// Byte length of the HMAC-SHA256 key. Fixed by the algorithm.
pub const SNAPSHOT_HMAC_KEY_LEN: usize = 32;

/// Top-level configuration knobs read from the process environment at
/// server startup.
///
/// Today this only carries snapshot-related knobs that don't have a
/// natural home in any of the middleware-scoped config structs. As the
/// API surface grows, additional cross-cutting knobs land here so the
/// env-var schema stays discoverable.
///
/// The `snapshot_hmac_key` field is a raw `[u8; 32]` rather than a
/// `tensor_wasm_snapshot::HmacKey` (or similar) so this crate does NOT
/// take a hard dependency on `tensor-wasm-snapshot`. When the snapshot
/// route handlers land they can hand the bytes to
/// `SnapshotWriter::with_hmac_sha256_key(key)` /
/// `SnapshotReader::with_hmac_sha256_key(key)` directly.
///
/// `Debug` is implemented manually so that `snapshot_hmac_key` renders as
/// a redacted placeholder. A derived `Debug` would print all 32 key bytes
/// any time a caller writes `tracing::debug!(?cfg)` or similar.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct AppConfig {
    /// Optional HMAC-SHA256 signing key for snapshot save/restore.
    ///
    /// `None` means snapshots are written in the v2 (unsigned) format and
    /// restore accepts both signed and unsigned blobs. `Some(key)` means
    /// snapshots are written in the signed v3 format and restore verifies
    /// any signature present (rejecting tampered blobs).
    ///
    /// See [`Self::snapshot_require_signature`] for the strict-restore
    /// knob that additionally rejects unsigned v2 blobs.
    pub snapshot_hmac_key: Option<[u8; SNAPSHOT_HMAC_KEY_LEN]>,

    /// When `true`, snapshot restore refuses v2 (unsigned) blobs even if
    /// a key is configured. Defaults to `false` so existing v2 archives
    /// keep working through the migration window.
    pub snapshot_require_signature: bool,
}

impl std::fmt::Debug for AppConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppConfig")
            .field(
                "snapshot_hmac_key",
                &self
                    .snapshot_hmac_key
                    .as_ref()
                    .map(|_| "<REDACTED 32-byte HMAC key>"),
            )
            .field(
                "snapshot_require_signature",
                &self.snapshot_require_signature,
            )
            .finish()
    }
}

impl AppConfig {
    /// Load from the process environment.
    ///
    /// Reads [`ENV_SNAPSHOT_HMAC_KEY`] (hex, 64 chars) and
    /// [`ENV_SNAPSHOT_REQUIRE_SIGNATURE`] (`true`/`false`,
    /// case-insensitive).
    ///
    /// Returns [`ConfigError`] when:
    ///
    /// * `TENSOR_WASM_API_SNAPSHOT_HMAC_KEY` is set but is not exactly 64
    ///   hex characters (each character must be `0-9`, `a-f`, or `A-F`).
    /// * `TENSOR_WASM_API_SNAPSHOT_REQUIRE_SIGNATURE` is set to a value
    ///   that is neither `true` nor `false` (case-insensitive).
    ///
    /// An unset variable in either case is *not* an error: the resulting
    /// `AppConfig` carries `None` / `false` for the corresponding fields.
    /// This matches the silent-default behaviour every other
    /// `*Config::from_env` in this crate exhibits for unset variables —
    /// we deliberately diverge from that pattern only for *malformed*
    /// values, where a typo in a secret would otherwise be silently
    /// swallowed.
    pub fn from_env() -> Result<Self, ConfigError> {
        let snapshot_hmac_key = match std::env::var(ENV_SNAPSHOT_HMAC_KEY) {
            Ok(raw) => {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(parse_hex_key(trimmed)?)
                }
            }
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(ConfigError::NonUnicode {
                    var: ENV_SNAPSHOT_HMAC_KEY,
                });
            }
        };

        let snapshot_require_signature = match std::env::var(ENV_SNAPSHOT_REQUIRE_SIGNATURE) {
            Ok(raw) => parse_bool(raw.trim())?,
            Err(std::env::VarError::NotPresent) => false,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(ConfigError::NonUnicode {
                    var: ENV_SNAPSHOT_REQUIRE_SIGNATURE,
                });
            }
        };

        if snapshot_require_signature && snapshot_hmac_key.is_none() {
            tracing::warn!(
                target: "tensor_wasm_api::config",
                env_key = ENV_SNAPSHOT_HMAC_KEY,
                env_require = ENV_SNAPSHOT_REQUIRE_SIGNATURE,
                "{} is true but {} is unset; the /snapshot/* routes will \
                 return 503 snapshot_signing_not_configured — this is almost \
                 certainly a misconfiguration",
                ENV_SNAPSHOT_REQUIRE_SIGNATURE,
                ENV_SNAPSHOT_HMAC_KEY,
            );
        }

        if snapshot_hmac_key.is_some() {
            tracing::info!(
                target: "tensor_wasm_api::config",
                require_signature = snapshot_require_signature,
                "snapshot HMAC-SHA256 key configured ({} chars hex)",
                SNAPSHOT_HMAC_KEY_LEN * 2,
            );
        }

        Ok(Self {
            snapshot_hmac_key,
            snapshot_require_signature,
        })
    }

    /// Test-only builder that installs an explicit HMAC key without
    /// touching the process environment.
    ///
    /// Hidden from rustdoc because production code paths should always
    /// flow through [`Self::from_env`] — this exists so tests of the
    /// future snapshot route handlers can drive the config directly
    /// without env-var poisoning.
    #[doc(hidden)]
    pub fn with_snapshot_hmac_key(mut self, key: [u8; SNAPSHOT_HMAC_KEY_LEN]) -> Self {
        self.snapshot_hmac_key = Some(key);
        self
    }

    /// Test-only builder for the require-signature toggle. See
    /// [`Self::with_snapshot_hmac_key`] for the rationale.
    #[doc(hidden)]
    pub fn with_snapshot_require_signature(mut self, require: bool) -> Self {
        self.snapshot_require_signature = require;
        self
    }
}

/// Errors returned by [`AppConfig::from_env`] for malformed values.
///
/// Unset variables are NOT errors — see [`AppConfig::from_env`] for the
/// silent-default behaviour.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// The variable held bytes that were not valid UTF-8.
    NonUnicode {
        /// Name of the offending environment variable.
        var: &'static str,
    },
    /// The variable was set but did not parse as a 64-character lowercase
    /// or uppercase hex string.
    InvalidHexKey {
        /// Name of the offending environment variable.
        var: &'static str,
        /// Human-readable reason (length / non-hex character).
        reason: HexParseReason,
    },
    /// The variable was set but is not `true` or `false` (case-insensitive).
    InvalidBool {
        /// Name of the offending environment variable.
        var: &'static str,
        /// The raw value that failed to parse (trimmed).
        value: String,
    },
}

/// Specific reason a hex-encoded key failed to parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HexParseReason {
    /// The string had the wrong number of characters. The expected length
    /// is always `SNAPSHOT_HMAC_KEY_LEN * 2`.
    WrongLength {
        /// The number of characters actually present.
        actual: usize,
    },
    /// The string contained a character that is not `0-9`, `a-f`, or `A-F`.
    InvalidCharacter,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::NonUnicode { var } => {
                write!(f, "{var} is set but its value is not valid UTF-8")
            }
            ConfigError::InvalidHexKey { var, reason } => match reason {
                HexParseReason::WrongLength { actual } => write!(
                    f,
                    "{var} must be exactly {expected} hex characters \
                     (32 bytes); got {actual}",
                    expected = SNAPSHOT_HMAC_KEY_LEN * 2,
                ),
                HexParseReason::InvalidCharacter => {
                    write!(f, "{var} must contain only hex characters (0-9, a-f, A-F)",)
                }
            },
            ConfigError::InvalidBool { var, value } => write!(
                f,
                "{var} must be `true` or `false` (case-insensitive); got `{value}`",
            ),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Parse a hex-encoded `[u8; SNAPSHOT_HMAC_KEY_LEN]` from `s`.
///
/// `s` must be already-trimmed (the caller is responsible for
/// whitespace handling so the error message reports the right length).
fn parse_hex_key(s: &str) -> Result<[u8; SNAPSHOT_HMAC_KEY_LEN], ConfigError> {
    let expected = SNAPSHOT_HMAC_KEY_LEN * 2;
    if s.len() != expected {
        return Err(ConfigError::InvalidHexKey {
            var: ENV_SNAPSHOT_HMAC_KEY,
            reason: HexParseReason::WrongLength { actual: s.len() },
        });
    }

    let mut out = [0u8; SNAPSHOT_HMAC_KEY_LEN];
    let bytes = s.as_bytes();
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = hex_nibble(bytes[i * 2])?;
        let lo = hex_nibble(bytes[i * 2 + 1])?;
        *slot = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_nibble(c: u8) -> Result<u8, ConfigError> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(ConfigError::InvalidHexKey {
            var: ENV_SNAPSHOT_HMAC_KEY,
            reason: HexParseReason::InvalidCharacter,
        }),
    }
}

fn parse_bool(s: &str) -> Result<bool, ConfigError> {
    if s.eq_ignore_ascii_case("true") {
        Ok(true)
    } else if s.eq_ignore_ascii_case("false") || s.is_empty() {
        // Treat empty / whitespace as `false` (matches the documented
        // default for unset). Callers that care about "set but empty"
        // can read the raw env var themselves.
        Ok(false)
    } else {
        Err(ConfigError::InvalidBool {
            var: ENV_SNAPSHOT_REQUIRE_SIGNATURE,
            value: s.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_all_none_or_false() {
        let cfg = AppConfig::default();
        assert!(cfg.snapshot_hmac_key.is_none());
        assert!(!cfg.snapshot_require_signature);
    }

    #[test]
    fn parse_hex_key_round_trips_lowercase() {
        let s = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
        let key = parse_hex_key(s).expect("valid");
        assert_eq!(key[0], 0x00);
        assert_eq!(key[1], 0x11);
        assert_eq!(key[15], 0xff);
        assert_eq!(key[31], 0xff);
    }

    #[test]
    fn parse_hex_key_accepts_uppercase() {
        let s = "00112233445566778899AABBCCDDEEFF00112233445566778899AABBCCDDEEFF";
        let key = parse_hex_key(s).expect("valid");
        assert_eq!(key[1], 0x11);
        assert_eq!(key[15], 0xff);
    }

    #[test]
    fn parse_hex_key_rejects_short() {
        let err = parse_hex_key("deadbeef").expect_err("too short");
        assert!(matches!(
            err,
            ConfigError::InvalidHexKey {
                reason: HexParseReason::WrongLength { actual: 8 },
                ..
            }
        ));
    }

    #[test]
    fn parse_hex_key_rejects_non_hex() {
        // 64 chars, but with a 'g'.
        let s = "g0112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";
        assert_eq!(s.len(), 64);
        let err = parse_hex_key(s).expect_err("non-hex character");
        assert!(matches!(
            err,
            ConfigError::InvalidHexKey {
                reason: HexParseReason::InvalidCharacter,
                ..
            }
        ));
    }

    #[test]
    fn parse_bool_accepts_case_insensitive() {
        assert!(parse_bool("true").unwrap());
        assert!(parse_bool("TRUE").unwrap());
        assert!(parse_bool("True").unwrap());
        assert!(!parse_bool("false").unwrap());
        assert!(!parse_bool("FALSE").unwrap());
        assert!(!parse_bool("").unwrap());
    }

    #[test]
    fn parse_bool_rejects_garbage() {
        let err = parse_bool("yes").expect_err("not a bool");
        assert!(matches!(err, ConfigError::InvalidBool { .. }));
    }

    #[test]
    fn with_snapshot_hmac_key_builder_sets_field() {
        let key = [0x42u8; SNAPSHOT_HMAC_KEY_LEN];
        let cfg = AppConfig::default().with_snapshot_hmac_key(key);
        assert_eq!(cfg.snapshot_hmac_key, Some(key));
    }

    #[test]
    fn with_snapshot_require_signature_builder_sets_field() {
        let cfg = AppConfig::default().with_snapshot_require_signature(true);
        assert!(cfg.snapshot_require_signature);
    }

    #[test]
    fn config_error_display_messages_are_descriptive() {
        // Spot-check each variant renders something humans can act on.
        let e = ConfigError::InvalidHexKey {
            var: ENV_SNAPSHOT_HMAC_KEY,
            reason: HexParseReason::WrongLength { actual: 4 },
        };
        let msg = e.to_string();
        assert!(msg.contains(ENV_SNAPSHOT_HMAC_KEY));
        assert!(msg.contains("64"));
        assert!(msg.contains('4'));

        let e = ConfigError::InvalidHexKey {
            var: ENV_SNAPSHOT_HMAC_KEY,
            reason: HexParseReason::InvalidCharacter,
        };
        assert!(e.to_string().contains("hex"));

        let e = ConfigError::InvalidBool {
            var: ENV_SNAPSHOT_REQUIRE_SIGNATURE,
            value: "yes".to_owned(),
        };
        let msg = e.to_string();
        assert!(msg.contains(ENV_SNAPSHOT_REQUIRE_SIGNATURE));
        assert!(msg.contains("yes"));
    }
}
