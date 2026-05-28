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
/// Stored as a single `u8` immediately after the [`V3_TRAILER_MAGIC`] in a
/// v3 blob. Followed by the 32-byte HMAC-SHA256 output (see
/// [`HMAC_SHA256_SIG_LEN`]). The reader rejects any unknown discriminant
/// rather than silently treating it as unsigned, so a future signature
/// algorithm cannot be downgraded by stripping its trailer.
pub const SIGNATURE_KIND_HMAC_SHA256: u8 = 1;

/// Length in bytes of an HMAC-SHA256 signature (the digest size of SHA-256).
pub const HMAC_SHA256_SIG_LEN: usize = 32;

/// 4-byte magic prefix that begins every v3 trailer.
///
/// Chosen as ASCII `b"S3T1"` ("Snapshot 3 Trailer v1"): four printable
/// bytes whose lexicographic appearance in a hex dump unambiguously marks
/// the start of a v3 trailer, and whose value is chosen so that a chance
/// collision with the trailing bytes of a zstd frame epilogue is
/// vanishingly improbable (~1/2^32 versus the ~1/256 false-positive rate
/// of the pre-T8 single-byte sniff).
///
/// Why a magic prefix at all? Prior to T8 the reader classified a blob
/// as v3 by checking whether `bytes[len - SIGNATURE_TRAILER_LEN]` equalled
/// [`SIGNATURE_KIND_HMAC_SHA256`] (`1`). Because that byte sits inside
/// the zstd frame epilogue of a legitimate v2 blob, a v2 capture would
/// be misclassified as v3 with ~1/256 probability. The reader then
/// dispatched HMAC verification on the v2 payload, which always failed
/// — but the downgrade-shaped error message and the wasted HMAC work
/// were both observable side channels. The 4-byte magic shrinks the
/// false-positive rate to ~1/2^32 (~2.3e-10), well below the per-blob
/// CRC32 mismatch rate and effectively eliminating the spurious
/// classification path.
///
/// This is a **breaking** change for any v3 snapshot produced by a
/// pre-T8 writer: the old trailer was `[kind][sig]` (33 bytes), the
/// new trailer is `[magic][kind][sig]` (37 bytes). The reader no
/// longer accepts the legacy 33-byte form — operators with archived
/// v3 captures must re-sign with a current writer. v2 snapshots are
/// unaffected; their wire format is unchanged.
pub const V3_TRAILER_MAGIC: [u8; 4] = *b"S3T1";

/// Length in bytes of [`V3_TRAILER_MAGIC`] — exposed so the reader and
/// writer can use a single named constant for trailer-offset arithmetic
/// rather than hard-coding `4` at every call site.
pub const V3_TRAILER_MAGIC_LEN: usize = V3_TRAILER_MAGIC.len();

/// Total length in bytes of a v3 trailer
/// (`[magic: 4][signature_kind: u8][signature]`).
///
/// Today this is always `4 + 1 + 32 = 37` because the only defined
/// [`SignatureKind`] is HMAC-SHA256. The reader uses this constant to
/// split the v2-shaped prefix from the trailer before verifying the
/// signature.
///
/// **Wire-format note (T8):** prior to the magic-prefix change this
/// constant was `1 + 32 = 33` and the trailer began with the
/// signature-kind byte directly. The 4-byte [`V3_TRAILER_MAGIC`] is
/// now prepended to the trailer, growing it to 37 bytes. The
/// `SNAPSHOT_VERSION_V3` revision number is **not** bumped — the
/// magic prefix is itself the discriminator and the inner bincode
/// payload is byte-identical to the pre-T8 v3 shape.
pub const SIGNATURE_TRAILER_LEN: usize = V3_TRAILER_MAGIC_LEN + 1 + HMAC_SHA256_SIG_LEN;

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
        // T8: trailer now begins with V3_TRAILER_MAGIC (4 bytes) so the
        // total length is `magic + kind + sig = 4 + 1 + 32 = 37`. The
        // pre-T8 layout was `kind + sig = 1 + 32 = 33`; the bump is
        // intentional and gated by the BREAKING note in CHANGELOG.md.
        assert_eq!(
            SIGNATURE_TRAILER_LEN,
            V3_TRAILER_MAGIC_LEN + 1 + HMAC_SHA256_SIG_LEN,
        );
        assert_eq!(SIGNATURE_TRAILER_LEN, 37);
    }

    #[test]
    fn v3_trailer_magic_constants_are_consistent() {
        // The magic must be exactly 4 bytes and the exposed length must
        // match the array length — these are tied together at the type
        // level but a regression on one would surface as a confusing
        // off-by-N at the reader, so check both explicitly.
        assert_eq!(V3_TRAILER_MAGIC.len(), V3_TRAILER_MAGIC_LEN);
        assert_eq!(V3_TRAILER_MAGIC_LEN, 4);
        // Sanity: documented bytes are `S3T1` (Snapshot 3 Trailer v1).
        assert_eq!(&V3_TRAILER_MAGIC, b"S3T1");
    }
}
