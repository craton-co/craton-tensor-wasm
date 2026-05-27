// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Wire-format version tags and signature-kind discriminants.
//!
//! Hosts every constant that lives on the wire and needs to be referenced
//! independently by the writer, the reader, and downstream callers (the CLI
//! and API crates inspect [`SignatureKind`] to surface human-readable error
//! messages). See `FORMAT.md` for the byte layout these constants control.

/// Wire-format tag for the unsigned snapshot envelope.
///
/// A v2 blob is exactly the zstd-compressed bincode payload defined by
/// [`crate::writer::Snapshot`]; nothing follows the compressed frame. Any
/// reader without an HMAC key set, and any writer without
/// [`crate::writer::SnapshotWriter::with_hmac_sha256_key`] called on it,
/// produces and consumes this version.
pub const SNAPSHOT_VERSION_V2: u32 = 2;

/// Wire-format tag for the HMAC-SHA256-signed snapshot envelope.
///
/// A v3 blob is a v2-shaped prefix (the same compressed bincode payload, but
/// with the inner `version` field bumped to `3`) followed by a single
/// signature-kind byte and the trailing signature bytes. The trailer is
/// **not** zstd-compressed — it sits *after* the zstd frame, so existing
/// tooling that decompresses with a single-frame decoder sees only the
/// authenticated prefix and the trailer is observed by the reader directly.
pub const SNAPSHOT_VERSION_V3: u32 = 3;

/// Trailer discriminant for a HMAC-SHA256 signature.
///
/// Stored as a single `u8` immediately after the zstd frame in a v3 blob.
/// Followed by the 32-byte HMAC-SHA256 output (see [`HMAC_SHA256_SIG_LEN`]).
/// The reader rejects any unknown discriminant rather than silently treating
/// it as unsigned, so a future signature algorithm cannot be downgraded by
/// stripping its trailer.
pub const SIGNATURE_KIND_HMAC_SHA256: u8 = 1;

/// Length in bytes of an HMAC-SHA256 signature (the digest size of SHA-256).
pub const HMAC_SHA256_SIG_LEN: usize = 32;

/// Total length in bytes of a v3 trailer (`[signature_kind: u8][signature]`).
///
/// Today this is always `1 + 32 = 33` because the only defined
/// [`SignatureKind`] is HMAC-SHA256. The reader uses this constant to split
/// the v2-shaped prefix from the trailer before verifying the signature.
pub const SIGNATURE_TRAILER_LEN: usize = 1 + HMAC_SHA256_SIG_LEN;

/// Enumeration of the signature algorithms understood by the reader.
///
/// `#[non_exhaustive]` so adding a future variant (e.g. Ed25519) is not a
/// breaking change for downstream `match` arms — they must handle the
/// catch-all today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[repr(u8)]
pub enum SignatureKind {
    /// HMAC-SHA256 over the v2-shaped prefix (zstd frame bytes, inclusive
    /// of magic, version, payload, and CRC32 — i.e. everything between the
    /// start of input and the trailer). 32-byte signature.
    HmacSha256 = SIGNATURE_KIND_HMAC_SHA256,
}

impl SignatureKind {
    /// Parse a wire-format discriminant byte into a [`SignatureKind`].
    ///
    /// Returns `None` if the byte does not correspond to any known variant.
    /// The reader translates `None` into a `Serialization("unknown signature_kind")`
    /// error so a forward-compatible writer cannot silently downgrade an
    /// older reader.
    #[must_use]
    pub const fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            SIGNATURE_KIND_HMAC_SHA256 => Some(Self::HmacSha256),
            _ => None,
        }
    }

    /// Length in bytes of the signature produced by this kind.
    #[must_use]
    pub const fn signature_len(self) -> usize {
        match self {
            Self::HmacSha256 => HMAC_SHA256_SIG_LEN,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_constants_are_distinct() {
        assert_ne!(SNAPSHOT_VERSION_V2, SNAPSHOT_VERSION_V3);
        assert_eq!(SNAPSHOT_VERSION_V2, 2);
        assert_eq!(SNAPSHOT_VERSION_V3, 3);
    }

    #[test]
    fn signature_kind_round_trips() {
        let k = SignatureKind::HmacSha256;
        assert_eq!(k as u8, SIGNATURE_KIND_HMAC_SHA256);
        assert_eq!(SignatureKind::from_byte(k as u8), Some(k));
        assert_eq!(k.signature_len(), HMAC_SHA256_SIG_LEN);
    }

    #[test]
    fn unknown_signature_kind_is_none() {
        assert_eq!(SignatureKind::from_byte(0), None);
        assert_eq!(SignatureKind::from_byte(0xFF), None);
        assert_eq!(SignatureKind::from_byte(2), None);
    }

    #[test]
    fn trailer_length_matches_hmac_sha256() {
        assert_eq!(SIGNATURE_TRAILER_LEN, 1 + HMAC_SHA256_SIG_LEN);
        assert_eq!(SIGNATURE_TRAILER_LEN, 33);
    }
}
