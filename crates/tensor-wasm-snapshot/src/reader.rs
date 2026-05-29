// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `SnapshotReader`: restore a [`Snapshot`] from a compressed byte blob.
//!
//! The reader is the strict counterpart to [`crate::writer::SnapshotWriter`]:
//! it never panics on malformed input. Magic and version mismatches, truncated
//! zstd frames, broken bincode payloads, oversized decompressed streams, and
//! oversized declared `Vec<u8>` lengths are all surfaced as
//! [`TensorWasmError::Serialization`]. The hot path is a streaming zstd decode
//! capped at [`limits::MAX_DECOMPRESSED_BYTES`] followed by a size-limited
//! bincode deserialise; no I/O is performed.
//!
//! NOTE: cuda-feature code in this file is compile-tested on CUDA hosts only;
//! on no-CUDA hosts only the `#[cfg(not(feature = "cuda"))]` branches are
//! exercised. The cuda branches use the `cust` 0.3.x unified-memory APIs.

use std::io::Read;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tensor_wasm_core::error::{Result, TensorWasmError};
use tracing::{debug, instrument};

use crate::format::{SNAPSHOT_VERSION_V2, SNAPSHOT_VERSION_V3};
use crate::writer::{
    check_blob_size, limits, payload_crc32, Snapshot, SnapshotMetadata, SNAPSHOT_MAGIC,
};

#[cfg(feature = "signed-snapshots")]
use crate::format::{SignatureKind, V3_TRAILER_MAGIC, V3_TRAILER_MAGIC_LEN};
#[cfg(feature = "signed-snapshots")]
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
#[cfg(feature = "artifact-backing")]
use tensor_wasm_artifacts::ArtifactStore;
#[cfg(feature = "signed-snapshots")]
use zeroize::Zeroizing;

/// Reverse of [`SnapshotWriter`](crate::writer::SnapshotWriter) — turns a
/// compressed byte blob back into an in-memory [`Snapshot`].
///
/// Stateless (the fields are configuration, not buffers) and `Send + Sync`;
/// one instance can serve many concurrent restores. Construction is `const`,
/// so this is suitable for `static` singletons.
///
/// The default `max_decompressed` is [`limits::MAX_DECOMPRESSED_BYTES`] (256
/// MiB). Override via [`SnapshotReader::with_max_decompressed`] for trusted
/// snapshots that legitimately exceed the default.
///
/// By default the reader accepts both v2 (unsigned) and v3 (HMAC-SHA256
/// signed) blobs:
/// - A v2 blob is accepted as-is (legacy behaviour).
/// - A v3 blob is **rejected** unless [`SnapshotReader::with_hmac_sha256_key`]
///   has been called to install the verification key.
///
/// Call [`SnapshotReader::require_signature`] to also refuse unsigned v2
/// blobs (defence-in-depth for production deployments).
///
/// `Debug` is implemented manually to redact `hmac_key` — a derived `Debug`
/// would print all 32 key bytes via `{:?}` and expose the signing secret
/// any time a caller writes `tracing::debug!(?reader)` or similar.
///
/// `Copy` is intentionally NOT derived when the `signed-snapshots` feature
/// is enabled: the HMAC verification key is wrapped in
/// [`zeroize::Zeroizing`] so its backing bytes are scrubbed on drop, and
/// `Zeroizing<T>` is never `Copy` (a `Copy` would silently duplicate the
/// secret and skip the scrub on the original). The no-feature build keeps
/// `Copy` for backward compatibility — no secret material is present
/// there.
#[cfg_attr(not(feature = "signed-snapshots"), derive(Copy))]
#[derive(Clone)]
pub struct SnapshotReader {
    /// Hard ceiling on bytes the streaming zstd decoder is allowed to emit
    /// before being aborted. Bounds the attacker's memory budget independent
    /// of what the bincode `Vec<u8>` length fields claim.
    max_decompressed: usize,
    /// HMAC-SHA256 key used to verify v3 signatures. `None` -> v3 inputs
    /// are rejected.
    ///
    /// Wrapped in [`zeroize::Zeroizing`] so the 32 key bytes are
    /// overwritten when the reader is dropped — best-effort defence
    /// against the key surviving in swap-backed memory or in the
    /// allocator's freelist after the reader has gone out of scope.
    #[cfg(feature = "signed-snapshots")]
    hmac_key: Option<Zeroizing<[u8; 32]>>,
    /// Ed25519 *public* verifying key used to verify v3 blobs whose trailer
    /// carries `signature_kind = 2`. `None` -> such blobs are rejected.
    ///
    /// This is the asymmetric counterpart to [`Self::hmac_key`]: a publisher
    /// signs with the private key and any number of readers verify with only
    /// this public key, which cannot forge new snapshots. The key is public
    /// by nature, so it is not wrapped in `Zeroizing`.
    #[cfg(feature = "signed-snapshots")]
    ed25519_verifying_key: Option<VerifyingKey>,
    /// When `true`, v2 (unsigned) inputs are rejected even if otherwise
    /// well-formed. Allows operators to enforce signature-only restores
    /// without compiling a separate binary.
    require_signature: bool,
    /// Replay defence: when `Some(n)`, [`SnapshotReader::restore`] rejects a
    /// blob whose `metadata.sequence_no` is `< n`. `None` (the default)
    /// disables the floor. See [`SnapshotReader::with_min_sequence_no`].
    min_sequence_no: Option<u64>,
    /// Replay defence: when `Some(nonce)`, [`SnapshotReader::restore`]
    /// rejects a blob whose `metadata.nonce` is not exactly `Some(nonce)`.
    /// `None` (the default) disables the nonce check. See
    /// [`SnapshotReader::with_expected_nonce`].
    expected_nonce: Option<[u8; 16]>,
    /// T9 freshness check: when `Some(d)`, a restored snapshot whose
    /// `metadata.created_unix_ms` is older than `now - d` is rejected
    /// with [`TensorWasmError::SnapshotTooOld`]. `None` (the default)
    /// disables the check and preserves the v0.3.x behaviour of
    /// accepting arbitrarily old captures.
    ///
    /// See [`SnapshotReader::with_max_age`] for the public surface.
    max_age: Option<Duration>,
}

impl Default for SnapshotReader {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SnapshotReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("SnapshotReader");
        d.field("max_decompressed", &self.max_decompressed);
        #[cfg(feature = "signed-snapshots")]
        d.field(
            "hmac_key",
            &self
                .hmac_key
                .as_ref()
                .map(|_| "<REDACTED 32-byte HMAC key>"),
        );
        #[cfg(feature = "signed-snapshots")]
        d.field(
            "ed25519_verifying_key",
            &self.ed25519_verifying_key.as_ref().map(|_| "<set>"),
        );
        d.field("require_signature", &self.require_signature);
        d.field("min_sequence_no", &self.min_sequence_no);
        d.field("expected_nonce", &self.expected_nonce.as_ref().map(|_| "<set>"));
        d.field("max_age", &self.max_age);
        d.finish()
    }
}

impl SnapshotReader {
    /// Construct a fresh reader with the default decompressed-size cap
    /// ([`limits::MAX_DECOMPRESSED_BYTES`]). Equivalent to
    /// [`SnapshotReader::default`] but `const`.
    ///
    /// The freshness check is **disabled** by default (`max_age = None`);
    /// see [`SnapshotReader::with_max_age`] to opt in.
    pub const fn new() -> Self {
        Self {
            max_decompressed: limits::MAX_DECOMPRESSED_BYTES,
            #[cfg(feature = "signed-snapshots")]
            hmac_key: None,
            #[cfg(feature = "signed-snapshots")]
            ed25519_verifying_key: None,
            require_signature: false,
            min_sequence_no: None,
            expected_nonce: None,
            max_age: None,
        }
    }

    /// Configure HMAC-SHA256 verification with a 32-byte key.
    ///
    /// Required before the reader can accept v3 blobs. Without a key, any v3
    /// input is rejected with a `Serialization` error ("snapshot is signed
    /// (v3) but reader has no HMAC key"). v2 inputs continue to be accepted
    /// unless [`SnapshotReader::require_signature`] has also been called.
    ///
    /// The key is held by value and never logged or surfaced in error
    /// messages. Treat the reader as secret once this method has been
    /// called: it is `Clone + Copy`.
    #[cfg(feature = "signed-snapshots")]
    #[cfg_attr(docsrs, doc(cfg(feature = "signed-snapshots")))]
    #[must_use]
    pub fn with_hmac_sha256_key(mut self, key: [u8; 32]) -> Self {
        // Wrap in `Zeroizing` so the 32 bytes are scrubbed when the reader
        // drops. `Zeroizing::new` is not `const`, so this constructor is no
        // longer `const fn`. All existing call-sites are runtime contexts.
        self.hmac_key = Some(Zeroizing::new(key));
        self
    }

    /// Configure Ed25519 *asymmetric* verification with a public key.
    ///
    /// Required before the reader can accept a v3 blob whose trailer carries
    /// `signature_kind = 2`. Without it, such a blob is rejected with a
    /// `Serialization` error. v2 inputs continue to be accepted unless
    /// [`SnapshotReader::require_signature`] has also been called, and HMAC
    /// (`signature_kind = 1`) blobs are still handled by
    /// [`SnapshotReader::with_hmac_sha256_key`].
    ///
    /// This is the verifier half of the asymmetric scheme: pass the public
    /// key obtained from the publisher's
    /// `signing_key.verifying_key()`. The reader can verify but never sign,
    /// so distributing this key to many restorers does not let any of them
    /// forge a snapshot — the property HMAC cannot provide. Verification is
    /// performed by `ed25519_dalek`, which checks the signature without
    /// data-dependent branching on the secret-derived state.
    #[cfg(feature = "signed-snapshots")]
    #[cfg_attr(docsrs, doc(cfg(feature = "signed-snapshots")))]
    #[must_use]
    pub fn with_ed25519_verifying_key(mut self, key: VerifyingKey) -> Self {
        self.ed25519_verifying_key = Some(key);
        self
    }

    /// Reject any blob whose `metadata.nonce` is not exactly `Some(nonce)`.
    ///
    /// Pairs with [`crate::writer::SnapshotWriter::with_nonce`]: the
    /// restorer pins the snapshot to a single expected challenge value and
    /// rejects everything else — a blob with no nonce (`None`), a blob with a
    /// different nonce, or a replayed older capture that carried a stale
    /// nonce. The check runs after authentication and the structural checks,
    /// so for a v3 blob the signature has already certified the nonce
    /// (it lives inside the signed bincode payload) and an attacker cannot
    /// substitute a matching nonce without re-signing.
    ///
    /// Disabled by default (`None`), preserving backward compatibility.
    #[must_use]
    pub const fn with_expected_nonce(mut self, nonce: [u8; 16]) -> Self {
        self.expected_nonce = Some(nonce);
        self
    }

    /// Reject any blob whose `metadata.sequence_no` is `< floor`
    /// (rollback / replay defence).
    ///
    /// Pairs with [`crate::writer::SnapshotWriter::with_sequence_no`]. The
    /// intended operator usage is a **per-signing-key "track highest seen
    /// sequence_no"** pattern: maintain a persistent `last_seen` value for
    /// each signing key, construct the reader with
    /// `with_min_sequence_no(last_seen + 1)` (or `last_seen` if equal values
    /// should be accepted), and after a successful `restore` update
    /// `last_seen = max(last_seen, restored.metadata.sequence_no)`. A
    /// replayed older snapshot then carries a `sequence_no` below the floor
    /// and is rejected, closing the rollback window that timestamp-based
    /// freshness ([`SnapshotReader::with_max_age`]) alone cannot — an
    /// attacker who replays a once-valid capture within the `max_age`
    /// window still trips the sequence floor.
    ///
    /// The check runs after authentication, so for a v3 blob the signature
    /// has already certified `sequence_no` (it is inside the signed payload)
    /// and cannot be rewritten without re-signing. Disabled by default
    /// (`None`).
    #[must_use]
    pub const fn with_min_sequence_no(mut self, floor: u64) -> Self {
        self.min_sequence_no = Some(floor);
        self
    }

    /// Refuse v2 (unsigned) snapshots, accepting only v3 with a valid HMAC.
    ///
    /// Combined with [`SnapshotReader::with_hmac_sha256_key`], this is the
    /// production-grade defaults-against-downgrade configuration: an attacker
    /// cannot bypass signature verification by stripping the v3 trailer and
    /// re-encoding the inner payload as v2, because the reader refuses v2
    /// outright. Without an HMAC key configured, this reader will reject
    /// *every* well-formed input (v2 by this flag, v3 by the missing key) —
    /// callers must set both for a usable configuration.
    #[must_use]
    pub const fn require_signature(mut self) -> Self {
        self.require_signature = true;
        self
    }

    /// Override the decompressed-stream cap.
    ///
    /// Use when restoring trusted snapshots that legitimately decompress past
    /// the 256 MiB default. The reader still refuses inputs whose compressed
    /// size exceeds [`limits::MAX_INPUT_BYTES`].
    ///
    /// Semantic note (bincode 2.x migration): in the 1.x era this knob also
    /// bounded the bincode allocator via `Options::with_limit(max)` — that
    /// limit was a runtime value. bincode 2.x's allocator limit is a
    /// *compile-time* `const` generic, so it can no longer be tied to a
    /// per-instance runtime override. The reader instead uses a static
    /// allocator ceiling of [`limits::MAX_TOTAL_PAYLOAD_BYTES`] (the sum of
    /// the per-blob caps plus envelope slack). This runtime knob continues
    /// to bound the *decompressed buffer* via `Read::take`, and the per-blob
    /// caps in [`limits`] still reject any oversized declared length after
    /// deserialisation — so the practical guarantees (no zip-bomb, no
    /// length-prefix abuse) are unchanged. Only the layer that catches the
    /// length-prefix abuse has shifted from "bincode allocator cap" to
    /// "static bincode allocator cap + per-blob check".
    #[must_use]
    pub const fn with_max_decompressed(mut self, max: usize) -> Self {
        self.max_decompressed = max;
        self
    }

    /// Maximum decompressed-stream size this reader will accept, in bytes.
    #[must_use]
    pub const fn max_decompressed(&self) -> usize {
        self.max_decompressed
    }

    /// Enable the T9 freshness check with `max_age` as the maximum
    /// accepted age of a captured snapshot.
    ///
    /// When enabled, the reader compares the snapshot's
    /// `metadata.created_unix_ms` against the host's wall clock at the
    /// moment of `restore` and rejects the blob with
    /// [`TensorWasmError::SnapshotTooOld`] if `now - created > max_age`.
    /// The check is **opt-in**: a reader constructed via
    /// [`SnapshotReader::new`] (or [`SnapshotReader::default`]) has
    /// `max_age = None` and accepts arbitrarily old snapshots, preserving
    /// backward compatibility with v0.3.x callers and with the v0.3.x
    /// wire format (snapshots written by pre-T9 writers that emitted
    /// `created_unix_ms = 0` on `SystemTime` failure will fail every
    /// `max_age` check — operators opting into the check must re-emit).
    ///
    /// The HMAC trailer on v3 snapshots transitively authenticates
    /// `created_unix_ms` (the timestamp sits inside the bincode payload
    /// that the HMAC covers), so an attacker who replays a stale v3
    /// blob must keep its original timestamp; the reader's `max_age`
    /// check then rejects it. Combined with
    /// [`SnapshotReader::with_hmac_sha256_key`] and
    /// [`SnapshotReader::require_signature`], this closes the
    /// indefinite-replay window without needing the still-reserved
    /// `sequence_no` / `nonce` fields (those are v0.4 work).
    ///
    /// Clock-skew note: the check uses `SystemTime::now()` on the
    /// reader host. A reader whose clock is *behind* the writer's
    /// clock at capture time would see `now < created` and treat the
    /// resulting underflow as "fresh" (the check ignores future-dated
    /// snapshots rather than rejecting them, since clock skew is
    /// typically a transient condition and operators prefer accepting
    /// a slightly-future-dated snapshot over a hard rejection). A
    /// reader whose `SystemTime::now()` fails (clock before
    /// `UNIX_EPOCH`) also accepts the snapshot — the failure does not
    /// promote a clock-broken host to a brick.
    #[must_use]
    pub const fn with_max_age(mut self, max_age: Duration) -> Self {
        self.max_age = Some(max_age);
        self
    }

    /// Current freshness-check budget, if any. `None` means the check is
    /// disabled (the default).
    #[must_use]
    pub const fn max_age(&self) -> Option<Duration> {
        self.max_age
    }

    /// Decode `bytes` previously produced by
    /// [`SnapshotWriter::capture`](crate::writer::SnapshotWriter::capture).
    ///
    /// Returns [`TensorWasmError::Serialization`] for every malformed input — bad
    /// zstd frame, bad bincode bytes, wrong magic, wrong version, oversized
    /// input, oversized decompressed stream, oversized declared `Vec<u8>`
    /// length, or CRC32 mismatch. The function never panics, so callers can
    /// safely feed it untrusted bytes from disk or the network.
    ///
    /// Validation order is intentional and follows "authenticate then parse":
    /// (1) the cheap input-size cap rejects oversized blobs without touching
    /// zstd; (2) for any v3 blob (detected by the trailing signature-kind
    /// byte) the HMAC trailer is verified over the compressed prefix
    /// **before** any decompression or bincode decoding runs, so an attacker
    /// who has not forged a valid signature cannot exercise the zstd or
    /// bincode decoders at all; (3) only after authentication (or after v2 is
    /// confirmed under a `require_signature == false` reader) do we
    /// decompress, decode, and run the remaining structural and integrity
    /// checks (magic, version-consistency, per-blob caps, CRC32, total bytes).
    /// Decompression is streamed through a hard byte cap so a "zip bomb"
    /// payload cannot allocate past [`SnapshotReader::max_decompressed`] even
    /// if its compressed footprint fits under [`limits::MAX_INPUT_BYTES`].
    #[instrument(skip(self, bytes), fields(input_len = bytes.len()))]
    pub fn restore(&self, bytes: &[u8]) -> Result<Snapshot> {
        // Cap the raw input first to bound the attacker's memory budget before
        // zstd ever runs. A snapshot that decompresses to gigabytes is fine, as
        // long as the *compressed* slice itself is below this ceiling.
        if bytes.len() > limits::MAX_INPUT_BYTES {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot input too large: {} bytes exceeds cap of {} bytes",
                    bytes.len(),
                    limits::MAX_INPUT_BYTES,
                )
                .into(),
            ));
        }

        // T40 default-cutover: detect the v0.4 unified artifact-store
        // envelope by its leading 16-byte magic. This is the cheapest
        // possible discriminator — `bytes[..16] == ARTIFACT_MAGIC` is a
        // single fixed-length comparison before any keyed work runs.
        // Foreign blobs (legacy v3/v2) fall through to the existing
        // trailer-magic detector below.
        //
        // The fall-through is deliberately driven by `ArtifactError::BadMagic`
        // (and by the magic mismatch on the leading 16 bytes that
        // produces it): a v3 blob carries `zstd` framing in its first
        // bytes, which never equals `b"twasm-artifact01"`. Any other
        // artifact-envelope failure (bad version, bad HMAC, hash
        // mismatch) is reported as an error rather than silently
        // re-classified — those signal a tampered or wrong-key v4
        // blob and shouldn't be confused with a legacy-format input.
        //
        // H2 invariant — `require_signature` is honoured on the v4 path
        // *without* an explicit gate here. A v4 envelope is only ever
        // returned `Some(..)` by `try_restore_artifact_envelope` after the
        // artifact crate's `decode_envelope_from_bytes_with_cap` has verified
        // the envelope's HMAC-SHA256 trailer (it requires `self.hmac_key` to
        // be present and returns `Err` on any HMAC mismatch). An
        // HMAC-authenticated v4 envelope is therefore "signed" in the sense
        // `require_signature` cares about, so it legitimately satisfies the
        // gate. Conversely no v4 blob can reach this early return *without*
        // HMAC verification having run, so the early return cannot be used to
        // bypass the strict `require_signature` check below: a wrong-key or
        // tampered v4 blob returns `Err` here (not `Ok(None)`), and a blob
        // that is not v4 at all returns `Ok(None)` and falls through to the
        // v3/v2 gate. The `?` propagates the `Err` case.
        #[cfg(all(feature = "artifact-backing", feature = "signed-snapshots"))]
        {
            if let Some(snapshot) = self.try_restore_artifact_envelope(bytes)? {
                return Ok(snapshot);
            }
        }

        // Detect a v3 (signed) blob by peeking at the trailer position,
        // *before* decompression, so signature verification can authenticate
        // the prefix bytes before zstd or bincode see them — the
        // "authenticate then parse" property. Keying off a 4-byte magic
        // prefix (rather than the pre-T8 single-byte kind sniff) keeps the v2
        // false-positive rate at ~1/2^32.
        //
        // The trailer is now variable-length: HMAC-SHA256 carries a 37-byte
        // trailer (`[magic: 4][kind: 1][sig: 32]`), Ed25519 a 69-byte one
        // (`[magic: 4][kind: 1][sig: 64]`). `detect_v3_trailer` probes both
        // candidate lengths — for each it checks that the 4-byte magic sits at
        // `len - trailer_len(kind)` AND the kind byte immediately after the
        // magic matches that kind — and returns the matched
        // `(SignatureKind, prefix_len)`. Keying off both the magic and the
        // self-consistent kind byte keeps the ~1/2^32 false-positive guarantee
        // and unambiguously selects the right prefix split for the two trailer
        // sizes.
        #[cfg(feature = "signed-snapshots")]
        let detected = detect_v3_trailer(bytes);
        #[cfg(feature = "signed-snapshots")]
        let is_v3 = detected.is_some();
        #[cfg(not(feature = "signed-snapshots"))]
        let is_v3 = false;

        // STEP 2.5 — AUTHENTICATE FIRST.
        // Verify the signature over the compressed prefix before any
        // decompression or bincode decode runs. On failure we return
        // immediately so an attacker cannot use the zstd or bincode decoders
        // as oracles. The prefix is `bytes[..len - kind.trailer_len()]`.
        #[cfg(feature = "signed-snapshots")]
        let prefix_len = if let Some((kind, p)) = detected {
            self.verify_v3_trailer(bytes, p, kind)?;
            p
        } else {
            bytes.len()
        };
        #[cfg(not(feature = "signed-snapshots"))]
        let prefix_len = bytes.len();

        // `require_signature` enforcement applies to anything that wasn't
        // recognised as v3 above (i.e. legacy v2 or malformed). Together with
        // the early HMAC verification, this closes the strip-trailer-and-
        // rewrite-version downgrade attack: an attacker who removes the
        // signature trailer ends up in this branch, and a reader configured
        // with `require_signature` rejects without ever decoding the payload.
        if !is_v3 && self.require_signature {
            return Err(TensorWasmError::Serialization(
                "snapshot is unsigned (v2) but signature is required".into(),
            ));
        }

        // From here on we are working on authenticated bytes (for v3) or on a
        // v2 input that the operator has chosen to accept unsigned. Strip the
        // v3 trailer from the slice we hand to zstd so the decoder sees only
        // the compressed frame and `cursor.position()` lines up with the end
        // of that frame.
        let auth_prefix: &[u8] = &bytes[..prefix_len];

        // Streaming zstd decode with a hard ceiling. `Read::take` aborts the
        // decoder once `max_decompressed + 1` bytes are emitted, so a zip-bomb
        // payload cannot grow the destination buffer past the cap. We probe
        // one byte past the cap so we can distinguish "decompresses to exactly
        // cap bytes" (allowed) from "decompresses to >cap bytes" (rejected).
        //
        // The decoder is wrapped around a `Cursor<&[u8]>` (which is itself a
        // `BufRead`) via `with_buffer`, bypassing zstd's default `BufReader`
        // wrap. We then read `cursor.position()` to confirm the decoder
        // consumed exactly the authenticated prefix (no junk left over inside
        // it). `single_frame()` stops the decoder at the first frame end
        // rather than treating any unexpected trailing bytes as the start of
        // a second concatenated frame (which would otherwise fail with a
        // misleading "zstd init" error).
        let cap = self.max_decompressed;
        let probe_limit = u64::try_from(cap)
            .ok()
            .and_then(|c| c.checked_add(1))
            .unwrap_or(u64::MAX);
        let mut cursor = std::io::Cursor::new(auth_prefix);
        let mut decoder = zstd::stream::read::Decoder::with_buffer(&mut cursor)
            .map_err(|e| TensorWasmError::Serialization(format!("zstd init: {e}").into()))?
            .single_frame();
        // Pre-size the buffer to a small constant (1 MiB, capped by `cap`) to
        // avoid grow-by-doubling reallocs. We refuse to trust an
        // attacker-supplied frame-size hint, and the `Take` ceiling guarantees
        // we cannot grow past `cap + 1` bytes regardless of the input.
        let initial_capacity = cap.min(1024 * 1024);
        let mut decompressed: Vec<u8> = Vec::with_capacity(initial_capacity);
        (&mut decoder)
            .take(probe_limit)
            .read_to_end(&mut decompressed)
            .map_err(|e| TensorWasmError::Serialization(format!("zstd decode: {e}").into()))?;
        drop(decoder);
        // Bytes the zstd decoder consumed from the authenticated prefix. For
        // a well-formed v2 input this equals `bytes.len()`; for a well-formed
        // v3 input this equals `prefix_len` (i.e. `auth_prefix.len()`).
        let zstd_consumed = usize::try_from(cursor.position()).unwrap_or(usize::MAX);
        if decompressed.len() > cap {
            return Err(TensorWasmError::Serialization(
                format!("snapshot decompressed payload too large: > {cap} bytes exceeds cap")
                    .into(),
            ));
        }

        // Bound the bincode allocator separately: even within `cap` bytes of
        // decompressed input, a malicious `Vec<u8>` length prefix could ask
        // for a much larger allocation. In bincode 2.x the allocator limit is
        // a const generic, so we use a static upper bound — the sum of every
        // per-blob cap plus envelope slack. Any single allocation that would
        // push past this static ceiling is refused by bincode before the
        // backing buffer is touched. The per-blob `check_blob_size` calls
        // below catch anything that survives this gate.
        //
        // `legacy()` keeps the on-wire encoding (LE, fixint) byte-identical
        // to bincode 1.x's `DefaultOptions::new().with_fixint_encoding().with_little_endian()`.
        // `decode_from_slice` returns `(T, consumed_bytes)` and ignores any
        // trailing bytes by default, replacing the explicit `.allow_trailing_bytes()`
        // opt-in from 1.x.
        let cfg = bincode::config::legacy()
            .with_limit::<{ crate::writer::limits::MAX_TOTAL_PAYLOAD_BYTES }>();
        let (snapshot, _read): (Snapshot, usize) =
            bincode::serde::decode_from_slice(decompressed.as_slice(), cfg).map_err(|e| {
                TensorWasmError::Serialization(format!("bincode decode: {e}").into())
            })?;

        if snapshot.magic != SNAPSHOT_MAGIC {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot magic mismatch: expected {:#X}, got {:#X}",
                    SNAPSHOT_MAGIC, snapshot.magic,
                )
                .into(),
            ));
        }

        // Version dispatch (post-authentication). At this point HMAC has
        // already either verified the trailer (`is_v3 == true`) or confirmed
        // the input carries no trailer (`is_v3 == false`). The inner `version`
        // field must agree with the trailer-derived classification: a v3 blob
        // whose inner version is *not* v3 has been tampered with after
        // signing; a v2 blob whose inner version *is* v3 is a downgrade
        // attempt (someone stripped the trailer and left the inner version
        // bumped) — closed by the `require_signature` branch above and by
        // the consistency reject here.
        match (is_v3, snapshot.version) {
            (true, SNAPSHOT_VERSION_V3) => {
                // Defence in depth: the zstd frame must end exactly at the
                // trailer offset. The writer never inserts a gap between
                // `encoder.finish()` and the appended trailer, so any tail
                // junk inside the authenticated prefix would only show up
                // under a writer bug (the HMAC has already certified those
                // bytes as authentic, so it's not an attack — but it's not
                // a shape the writer should ever emit either). This branch
                // is unreachable when the `signed-snapshots` feature is off
                // (because `is_v3` is then always `false`); the check still
                // type-checks under both configurations.
                if zstd_consumed != auth_prefix.len() {
                    return Err(TensorWasmError::Serialization(
                        format!(
                            "snapshot v3 has unexpected bytes between zstd frame and trailer: \
                             {} byte(s) past zstd frame",
                            auth_prefix.len().saturating_sub(zstd_consumed),
                        )
                        .into(),
                    ));
                }
            }
            (false, SNAPSHOT_VERSION_V2) => {
                // v2 forbids any trailing bytes after the zstd frame —
                // otherwise an attacker could append a chosen 33-byte tail
                // and observe the reader's reaction. `auth_prefix == bytes`
                // for v2, so this comparison covers the entire input.
                if zstd_consumed != auth_prefix.len() {
                    return Err(TensorWasmError::Serialization(
                        format!(
                            "snapshot v2 has unexpected trailing bytes: {} byte(s) past zstd frame",
                            auth_prefix.len().saturating_sub(zstd_consumed),
                        )
                        .into(),
                    ));
                }
            }
            (false, SNAPSHOT_VERSION_V3) => {
                // Inner payload claims v3 but no trailer was detected. Either
                // the `signed-snapshots` feature is not compiled in (so we
                // could not even classify the trailer), or the trailer was
                // stripped (downgrade attack), or the writer produced a
                // malformed v3 blob. Each case is rejected; the message
                // distinguishes feature-off from a runtime tamper so an
                // operator can tell at a glance which knob to flip.
                #[cfg(feature = "signed-snapshots")]
                return Err(TensorWasmError::Serialization(
                    "snapshot v3 trailer missing".into(),
                ));
                #[cfg(not(feature = "signed-snapshots"))]
                return Err(TensorWasmError::Serialization(
                    "snapshot is signed (v3) but the `signed-snapshots` feature is not compiled in"
                        .into(),
                ));
            }
            (true, _) => {
                // HMAC verified successfully, but the inner version field is
                // not v3. This should be impossible for any blob produced by
                // our writer: the signing path always bumps `version` to v3
                // before computing the HMAC. Surface it as an explicit
                // version-mismatch rather than letting it pass silently.
                return Err(TensorWasmError::Serialization(
                    format!(
                        "snapshot v3 inner version mismatch: expected {}, got {}",
                        SNAPSHOT_VERSION_V3, snapshot.version,
                    )
                    .into(),
                ));
            }
            (false, other) => {
                return Err(TensorWasmError::Serialization(
                    format!(
                        "snapshot version mismatch: expected {} or {}, got {}",
                        SNAPSHOT_VERSION_V2, SNAPSHOT_VERSION_V3, other,
                    )
                    .into(),
                ));
            }
        }

        // Per-blob caps catch a tampered `Vec<u8>` length that survived bincode.
        check_blob_size(
            "wasm_memory",
            snapshot.wasm_memory.len(),
            limits::MAX_WASM_MEMORY_BYTES,
        )?;
        check_blob_size(
            "gpu_memory",
            snapshot.gpu_memory.len(),
            limits::MAX_GPU_MEMORY_BYTES,
        )?;
        check_blob_size(
            "registers",
            snapshot.registers.len(),
            limits::MAX_REGISTERS_BYTES,
        )?;

        // CRC32 is the integrity check — it catches in-place byte flips that
        // happen to survive zstd and bincode framing.
        let expected = payload_crc32(
            &snapshot.wasm_memory,
            &snapshot.gpu_memory,
            &snapshot.registers,
        );
        if snapshot.crc32 != expected {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot crc32 mismatch: expected {:#010X}, got {:#010X}",
                    expected, snapshot.crc32,
                )
                .into(),
            ));
        }

        // Cross-check the declared `total_uncompressed_bytes` against the
        // actual blob sums. Each `.len()` is already known to fit under its
        // per-blob cap (checked above), so the `checked_add`s here cannot
        // realistically overflow on a 64-bit host — but we use them anyway to
        // keep the arithmetic robust against future cap changes, and surface
        // any overflow as a format error rather than a wrap-around. Callers
        // that trust the metadata to size buffers or report compression ratios
        // would otherwise be misled by a tampered field.
        let actual_total = snapshot
            .wasm_memory
            .len()
            .checked_add(snapshot.gpu_memory.len())
            .and_then(|s| s.checked_add(snapshot.registers.len()))
            .ok_or_else(|| {
                TensorWasmError::Serialization("snapshot blob length sum overflowed usize".into())
            })?;
        if (actual_total as u64) != snapshot.metadata.total_uncompressed_bytes {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot metadata.total_uncompressed_bytes mismatch: \
                     expected {} (wasm_memory={} + gpu_memory={} + registers={}), got {}",
                    actual_total,
                    snapshot.wasm_memory.len(),
                    snapshot.gpu_memory.len(),
                    snapshot.registers.len(),
                    snapshot.metadata.total_uncompressed_bytes,
                )
                .into(),
            ));
        }

        // T9 freshness check. Runs *after* HMAC, CRC, and the structural
        // checks so a malformed blob is not silently re-categorised as
        // "stale". The check is opt-in via `with_max_age`; the default
        // `max_age = None` preserves the v0.3.x behaviour of accepting
        // arbitrarily old snapshots. For v3 blobs the HMAC has already
        // certified `created_unix_ms` (the timestamp is inside the
        // bincode-encoded payload that the HMAC covers), so an attacker
        // who replays a stale v3 blob cannot lie about its age — the
        // reader's clock is authoritative.
        self.check_freshness(snapshot.metadata.created_unix_ms)?;

        // Replay/rollback defence (opt-in via `with_min_sequence_no` /
        // `with_expected_nonce`). Runs last, on a fully-authenticated and
        // structurally-valid snapshot.
        self.check_replay(&snapshot.metadata)?;

        debug!(
            decompressed = decompressed.len(),
            wasm = snapshot.wasm_memory.len(),
            gpu = snapshot.gpu_memory.len(),
            regs = snapshot.registers.len(),
            version = snapshot.version,
            "snapshot restored",
        );
        Ok(snapshot)
    }

    /// Streaming restore: authenticate `bytes`, then hand each restored
    /// memory blob to `sink` one at a time, freeing each blob's backing
    /// allocation **before** the next is decoded out of the owned
    /// [`Snapshot`]. Returns the snapshot's [`SnapshotMetadata`] — every
    /// field except the three large `Vec<u8>` payloads.
    ///
    /// # Why this exists (lower peak memory)
    ///
    /// [`SnapshotReader::restore`] returns the whole [`Snapshot`] by value,
    /// so the caller transitively holds `wasm_memory + gpu_memory +
    /// registers` resident simultaneously — for a multi-GiB GPU snapshot
    /// that is a large peak. `restore_streaming` instead moves each blob
    /// into the sink and drops the reader's copy immediately afterward, so
    /// the reader never holds more than one blob beyond what the sink itself
    /// chooses to retain. A sink that writes straight to disk, a GPU
    /// `cuMemcpyHtoD`, or an `mmap`'d region keeps the reader-side resident
    /// set bounded by a single blob rather than the full snapshot.
    ///
    /// # Residual unavoidable copy
    ///
    /// The wire format stores `wasm_memory`/`gpu_memory`/`registers` as
    /// owned, length-prefixed `serde_bytes` fields, and `bincode` decodes
    /// each into an owned `Vec<u8>` in one shot — there is no incremental
    /// "decode the next N bytes of this field" hook in the `serde`/`bincode`
    /// data model. So the full decoded [`Snapshot`] does momentarily exist
    /// before the first blob is handed off. What this method eliminates is
    /// the *downstream* redundancy: the caller no longer has to keep the
    /// whole struct alive after consuming it, and each blob is released as
    /// soon as the sink has taken it (via [`std::mem::take`]) rather than at
    /// the end of a `restore` caller's scope. Where the buffered
    /// decompression `Vec` could be dropped earlier, it is (it is consumed
    /// by the bincode decode and dropped before the sink runs).
    ///
    /// # Verify-before-expose
    ///
    /// This method delegates the entire authenticate-then-parse pipeline to
    /// [`SnapshotReader::restore`] (HMAC/Ed25519 verification, the v4
    /// artifact-envelope HMAC, the zip-bomb cap, magic/version consistency,
    /// per-blob caps, CRC32, freshness, and replay). The sink is invoked
    /// **only** on the `Ok(Snapshot)` returned by that pipeline — no byte of
    /// any blob is passed to `sink` until every integrity and signature
    /// check has already passed. A tampered or wrong-key blob returns `Err`
    /// from `restore` and the sink is never touched.
    #[instrument(skip(self, bytes, sink), fields(input_len = bytes.len()))]
    pub fn restore_streaming<S: SnapshotSink>(
        &self,
        bytes: &[u8],
        sink: &mut S,
    ) -> Result<SnapshotMetadata> {
        // Full authenticate-then-parse pipeline. Nothing below this line runs
        // unless every signature / integrity check has already passed, so the
        // verify-before-expose invariant is inherited verbatim from `restore`.
        let mut snapshot = self.restore(bytes)?;

        // Stream each blob to the sink and drop the reader's copy right after.
        // `mem::take` swaps in an empty `Vec` (no allocation) so the owned
        // bytes move into the sink call and are freed as soon as the sink
        // returns, rather than all three living until the `Snapshot` drops.
        sink.wasm_memory(std::mem::take(&mut snapshot.wasm_memory))?;
        sink.gpu_memory(std::mem::take(&mut snapshot.gpu_memory))?;
        sink.registers(std::mem::take(&mut snapshot.registers))?;

        debug!(version = snapshot.version, "snapshot restored (streaming)");
        Ok(snapshot.metadata)
    }

    /// Streaming restore that writes the three memory blobs, in canonical
    /// order (`wasm_memory`, then `gpu_memory`, then `registers`), to a
    /// single [`std::io::Write`] sink.
    ///
    /// Thin convenience wrapper over [`SnapshotReader::restore_streaming`]
    /// for the common "concatenate the payload to a file / socket" case.
    /// The blobs are written back-to-back with no framing — the caller is
    /// expected to already know each blob's length from the returned
    /// [`SnapshotMetadata`] (or from an out-of-band manifest). Each blob is
    /// dropped immediately after it is written, so the reader-side peak is
    /// one blob rather than the whole snapshot.
    ///
    /// The verify-before-expose ordering is identical to
    /// [`SnapshotReader::restore_streaming`]: `out` is written to only after
    /// the full authentication pipeline in [`SnapshotReader::restore`] has
    /// succeeded.
    #[instrument(skip(self, bytes, out), fields(input_len = bytes.len()))]
    pub fn restore_to_writer<W: std::io::Write>(
        &self,
        bytes: &[u8],
        out: &mut W,
    ) -> Result<SnapshotMetadata> {
        let mut sink = WriteSink { out };
        self.restore_streaming(bytes, &mut sink)
    }

    /// Compare `created_unix_ms` against the host's wall clock and the
    /// configured `max_age`. Returns
    /// [`TensorWasmError::SnapshotTooOld`] if the snapshot is older than
    /// `max_age`; returns `Ok(())` when the check is disabled
    /// (`max_age == None`), when the host clock cannot be read, or when
    /// the snapshot is future-dated (clock skew between writer and
    /// reader hosts is typically transient and operators prefer accept
    /// over reject in that direction — see [`SnapshotReader::with_max_age`]
    /// docs).
    fn check_freshness(&self, created_unix_ms: u64) -> Result<()> {
        let max_age = match self.max_age {
            Some(d) => d,
            None => return Ok(()),
        };
        // Host clock unreadable -> accept rather than brick the reader.
        // This is the symmetric counterpart to the writer's
        // `SystemTime::now()` propagation: if the reader's clock is
        // broken the operator wants a single clear "fix the clock"
        // signal at capture time, not a cascade of failed restores.
        let now_unix_ms = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(d) => match u64::try_from(d.as_millis()) {
                Ok(ms) => ms,
                Err(_) => return Ok(()),
            },
            Err(_) => return Ok(()),
        };
        // Future-dated snapshot (writer clock ahead of reader clock):
        // accept. Saturating subtraction below means `age_ms == 0` in
        // that case, which always satisfies any non-zero `max_age`.
        let age_ms = now_unix_ms.saturating_sub(created_unix_ms);
        let max_age_ms = u64::try_from(max_age.as_millis()).unwrap_or(u64::MAX);
        if age_ms > max_age_ms {
            return Err(TensorWasmError::SnapshotTooOld {
                created_unix_ms,
                now_unix_ms,
                max_age_ms,
            });
        }
        Ok(())
    }

    /// Replay/rollback defence: enforce the optional `min_sequence_no` floor
    /// and `expected_nonce` match against the snapshot's metadata.
    ///
    /// Runs *after* authentication, CRC, and the structural checks (so a
    /// malformed blob is never re-categorised as a replay) and after
    /// `check_freshness`. For a v3 blob the signature has already certified
    /// both fields (they live inside the signed bincode payload), so an
    /// attacker who replays a stale capture cannot rewrite the sequence
    /// number or nonce to slip past these checks without re-signing.
    ///
    /// Both checks are opt-in; a reader built via [`SnapshotReader::new`]
    /// leaves them disabled and accepts any `sequence_no` / `nonce`,
    /// preserving backward compatibility.
    ///
    /// Rejections are surfaced as [`TensorWasmError::Serialization`] with a
    /// distinct, greppable message per check (`"snapshot sequence_no ... below
    /// floor ..."` / `"snapshot nonce mismatch"` / `"snapshot nonce missing"`)
    /// so dashboards can pin replay-attempt rejections apart from generic
    /// format errors without a new core-crate error variant.
    fn check_replay(&self, metadata: &crate::writer::SnapshotMetadata) -> Result<()> {
        if let Some(floor) = self.min_sequence_no {
            if metadata.sequence_no < floor {
                return Err(TensorWasmError::Serialization(
                    format!(
                        "snapshot sequence_no {} below floor {} (replay/rollback rejected)",
                        metadata.sequence_no, floor,
                    )
                    .into(),
                ));
            }
        }
        if let Some(expected) = self.expected_nonce {
            match metadata.nonce {
                Some(actual) if actual == expected => {}
                Some(_) => {
                    return Err(TensorWasmError::Serialization(
                        "snapshot nonce mismatch (replay rejected)".into(),
                    ));
                }
                None => {
                    return Err(TensorWasmError::Serialization(
                        "snapshot nonce missing but a nonce was required (replay rejected)".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Validate the trailing `[magic][signature_kind][signature]` bytes of
    /// a v3 blob.
    ///
    /// Called by [`SnapshotReader::restore`] **before** any decompression or
    /// bincode decode runs, as soon as the input has been classified as v3
    /// by the trailer-position magic. `prefix_len` is the number of bytes
    /// that precede the trailer — for the v3 envelope this is always
    /// `bytes.len() - SIGNATURE_TRAILER_LEN`, and the HMAC is computed
    /// over `bytes[..prefix_len]` concatenated with the trailer magic and
    /// the signature-kind byte. Authenticating here, before exposing the
    /// compressed prefix to zstd, is what gives the reader the
    /// "authenticate then parse" property: a forged or tampered blob
    /// cannot drive the zstd, bincode, or per-blob validation paths as a
    /// side channel.
    ///
    /// **Wire format:** the trailer is `[magic: 4][kind: 1][sig: N]` where
    /// `N` is 32 for HMAC-SHA256 (37-byte trailer) or 64 for Ed25519
    /// (69-byte trailer). The classifier in [`SnapshotReader::restore`] has
    /// already matched the magic and the self-consistent kind byte before
    /// dispatching here and passes the decided `kind`; we re-read and
    /// re-validate the magic and kind from the slice (rather than trusting
    /// the classifier) so this function remains correct under future
    /// refactors that hoist the check.
    ///
    /// Errors are deliberately generic: we never include the expected or
    /// observed signature bytes in the error message, since either could
    /// leak information about the secret key under a side-channel attacker.
    /// The constant-time `ct_eq` from `subtle` is used to compare the
    /// recomputed HMAC against the stored bytes so a timing oracle cannot
    /// recover the signature byte-by-byte; Ed25519 verification is delegated
    /// to `ed25519_dalek`, which is constant-time with respect to the key.
    #[cfg(feature = "signed-snapshots")]
    fn verify_v3_trailer(
        &self,
        bytes: &[u8],
        prefix_len: usize,
        kind: SignatureKind,
    ) -> Result<()> {
        // The trailer must be exactly `[magic: 4][kind: 1][sig: N]` for the
        // detected kind. Anything else is a truncation or junk after the
        // signature (refused rather than silently accepted).
        let trailer = bytes
            .get(prefix_len..)
            .ok_or_else(|| TensorWasmError::Serialization("snapshot v3 trailer missing".into()))?;
        if trailer.len() != kind.trailer_len() {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot v3 trailer length mismatch: expected {} bytes, got {}",
                    kind.trailer_len(),
                    trailer.len(),
                )
                .into(),
            ));
        }
        // Layout: trailer[0..4] = magic, trailer[4] = kind, trailer[5..] = sig.
        // The classifier already checked the magic and kind, but re-validate
        // here for defence in depth (a future refactor that hoists detection
        // upstream must not be able to skip authentication).
        let magic_bytes = &trailer[..V3_TRAILER_MAGIC_LEN];
        if magic_bytes != V3_TRAILER_MAGIC {
            return Err(TensorWasmError::Serialization(
                "snapshot v3 trailer magic mismatch".into(),
            ));
        }
        let kind_byte = trailer[V3_TRAILER_MAGIC_LEN];
        let sig_bytes = &trailer[V3_TRAILER_MAGIC_LEN + 1..];
        let parsed_kind = SignatureKind::from_byte(kind_byte).ok_or_else(|| {
            TensorWasmError::Serialization(format!("unknown signature_kind: {kind_byte}").into())
        })?;
        // The kind byte in the slice must agree with the kind the classifier
        // selected the trailer length from — otherwise the offset arithmetic
        // and the signature length would be inconsistent.
        if parsed_kind != kind {
            return Err(TensorWasmError::Serialization(
                "snapshot v3 trailer kind inconsistent with detected length".into(),
            ));
        }
        debug_assert_eq!(sig_bytes.len(), kind.signature_len());

        match kind {
            SignatureKind::Ed25519 => {
                let verifying_key = self.ed25519_verifying_key.as_ref().ok_or_else(|| {
                    TensorWasmError::Serialization(
                        "snapshot is Ed25519-signed (v3) but reader has no Ed25519 verifying key"
                            .into(),
                    )
                })?;
                // Reconstruct the 64-byte signature. `Signature::from_slice`
                // only fails on a wrong length, which we have already
                // guaranteed via the trailer-length check above.
                let sig = Signature::from_slice(sig_bytes).map_err(|_| {
                    TensorWasmError::Serialization("snapshot Ed25519 signature malformed".into())
                })?;
                // Reconstruct the exact signed message:
                // `prefix || V3_TRAILER_MAGIC || [kind_byte]`. This mirrors
                // the writer and authenticates the trailer header, so an
                // attacker cannot rewrite the kind byte (e.g. to claim HMAC)
                // without invalidating the signature.
                let mut message =
                    Vec::with_capacity(prefix_len + V3_TRAILER_MAGIC_LEN + 1);
                message.extend_from_slice(&bytes[..prefix_len]);
                message.extend_from_slice(&V3_TRAILER_MAGIC);
                message.push(kind_byte);
                verifying_key.verify(&message, &sig).map_err(|_| {
                    TensorWasmError::Serialization("snapshot Ed25519 signature mismatch".into())
                })?;
            }
            SignatureKind::HmacSha256 => {
                let key = self.hmac_key.as_ref().ok_or_else(|| {
                    TensorWasmError::Serialization(
                        "snapshot is signed (v3) but reader has no HMAC key".into(),
                    )
                })?;
                use hmac::{Hmac, Mac};
                use sha2::Sha256;
                use subtle::ConstantTimeEq;

                // `key` is `&Zeroizing<[u8; 32]>`. `Zeroizing` is
                // `Deref<Target = [u8; 32]>`, so `&key[..]` borrows the full
                // 32-byte slice without copying the secret out of the
                // zeroizing wrapper.
                let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(&key[..]).map_err(|_| {
                    // [u8; 32] is always a valid HMAC-SHA256 key length; this
                    // branch is unreachable but we translate rather than
                    // panic and keep the key out of the message.
                    TensorWasmError::Serialization("HMAC init failed".into())
                })?;
                // HMAC covers every byte the zstd decoder consumed (the v2-shaped
                // prefix), i.e. transitively magic, version, payload, and CRC32.
                mac.update(&bytes[..prefix_len]);
                // T8: authenticate the 4-byte trailer magic. The writer
                // mixes the same bytes into its HMAC input, so a tampered
                // magic prefix would produce a mismatch here — closing
                // the "rewrite the trailer magic to disguise the blob"
                // attack vector at signature-verification time.
                mac.update(&V3_TRAILER_MAGIC);
                // snapshot 1.1: also authenticate the signature-kind byte
                // so a future second variant cannot be substituted via
                // trailer rewrite (would otherwise be a downgrade primitive
                // once two kinds coexist). The writer adds the same byte
                // to its HMAC input; verification must match.
                mac.update(&[kind_byte]);
                let expected = mac.finalize().into_bytes();
                // ConstantTimeEq returns `Choice` (0 or 1) without short-circuiting.
                let ok: bool = expected.as_slice().ct_eq(sig_bytes).unwrap_u8() == 1;
                if !ok {
                    return Err(TensorWasmError::Serialization(
                        "snapshot HMAC mismatch".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Restore a [`Snapshot`] previously written via
    /// [`crate::writer::SnapshotWriter::capture_to_artifact_store`].
    ///
    /// The artifact store owns the outer envelope (magic, BLAKE3 content
    /// hash, zstd compression, HMAC-SHA256 trailer, key-fingerprinted
    /// filename); this method just bincode-decodes the payload bytes
    /// returned by [`tensor_wasm_artifacts::DiskArtifactStore::get`] and
    /// re-applies the per-blob size caps and CRC32 cross-check that the
    /// legacy [`SnapshotReader::restore`] path runs.
    ///
    /// Validation order:
    /// 1. Artifact store performs its own authentication (magic, version,
    ///    HMAC, content-hash) — a tampered or wrong-key blob returns an
    ///    error here, mapped to `TensorWasmError::Serialization`.
    /// 2. The decoded byte payload is run through bincode with the same
    ///    static [`limits::MAX_TOTAL_PAYLOAD_BYTES`] allocator ceiling
    ///    the legacy path uses, so a malicious length-prefix inside the
    ///    bincode payload cannot drive a runaway allocation.
    /// 3. The inner magic and version are checked, the three byte blobs
    ///    are validated against their per-blob caps, and the CRC32 and
    ///    `metadata.total_uncompressed_bytes` cross-checks run exactly
    ///    as on the legacy path. These checks are defence-in-depth on
    ///    top of the artifact store's own integrity guarantees: a writer
    ///    bug that produced a Snapshot with a stale CRC32 should still
    ///    be rejected even though the artifact store happily authenticated
    ///    the payload.
    ///
    /// The reader's `max_decompressed`, `hmac_key`, and `require_signature`
    /// fields are intentionally **not** consulted on this path — the
    /// artifact store owns those concerns. Callers that want
    /// operator-tunable HMAC keys for snapshots should construct the
    /// [`tensor_wasm_artifacts::DiskArtifactStore`] with the appropriate
    /// key.
    #[cfg(feature = "artifact-backing")]
    #[cfg_attr(docsrs, doc(cfg(feature = "artifact-backing")))]
    #[instrument(skip(self, store), fields(content_hash = %hash))]
    pub fn restore_from_artifact_store(
        &self,
        store: &tensor_wasm_artifacts::DiskArtifactStore,
        hash: &tensor_wasm_artifacts::ContentHash,
    ) -> Result<Snapshot> {
        let bytes = store.get(hash).map_err(|e| {
            // Forward the artifact store's `Display` output through the
            // generic Serialization variant — `ArtifactError` is not part
            // of the `TensorWasmError` enum, and the messages are already
            // operator-facing (no key bytes leak).
            TensorWasmError::Serialization(format!("artifact store get: {e}").into())
        })?;

        // Static allocator ceiling for bincode, identical to the legacy
        // restore path. The artifact store already bounds memory by its
        // own decompression cap, but this gate catches any in-payload
        // length-prefix abuse before the backing buffer is touched.
        let cfg = bincode::config::legacy()
            .with_limit::<{ crate::writer::limits::MAX_TOTAL_PAYLOAD_BYTES }>();
        let (snapshot, _read): (Snapshot, usize) =
            bincode::serde::decode_from_slice(bytes.as_slice(), cfg).map_err(|e| {
                TensorWasmError::Serialization(format!("bincode decode: {e}").into())
            })?;

        if snapshot.magic != SNAPSHOT_MAGIC {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot magic mismatch: expected {:#X}, got {:#X}",
                    SNAPSHOT_MAGIC, snapshot.magic,
                )
                .into(),
            ));
        }

        // M1: the artifact-backed write path emits *only* the v2 inner
        // discriminant — the outer envelope already supplies authentication
        // (HMAC + content hash) and the inner payload carries no v3 signature
        // trailer. Accept inner v2 ONLY here. Inner v3 is deliberately
        // rejected rather than accepted "for forward compatibility": no v3
        // trailer is written or verified on this path, so a v3 discriminant
        // would wave through an unverifiable signed-inner claim. The
        // conservative choice (v2 only) holds until a signed-inner-v4 format
        // exists with its own inner-trailer verification.
        if snapshot.version != SNAPSHOT_VERSION_V2 {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot artifact-store inner version mismatch: expected {} (inner v3 is not \
                     accepted on the artifact path — no inner trailer is verified there), got {}",
                    SNAPSHOT_VERSION_V2, snapshot.version,
                )
                .into(),
            ));
        }

        check_blob_size(
            "wasm_memory",
            snapshot.wasm_memory.len(),
            limits::MAX_WASM_MEMORY_BYTES,
        )?;
        check_blob_size(
            "gpu_memory",
            snapshot.gpu_memory.len(),
            limits::MAX_GPU_MEMORY_BYTES,
        )?;
        check_blob_size(
            "registers",
            snapshot.registers.len(),
            limits::MAX_REGISTERS_BYTES,
        )?;

        let expected = payload_crc32(
            &snapshot.wasm_memory,
            &snapshot.gpu_memory,
            &snapshot.registers,
        );
        if snapshot.crc32 != expected {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot crc32 mismatch: expected {:#010X}, got {:#010X}",
                    expected, snapshot.crc32,
                )
                .into(),
            ));
        }

        let actual_total = snapshot
            .wasm_memory
            .len()
            .checked_add(snapshot.gpu_memory.len())
            .and_then(|s| s.checked_add(snapshot.registers.len()))
            .ok_or_else(|| {
                TensorWasmError::Serialization("snapshot blob length sum overflowed usize".into())
            })?;
        if (actual_total as u64) != snapshot.metadata.total_uncompressed_bytes {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot metadata.total_uncompressed_bytes mismatch: \
                     expected {} (wasm_memory={} + gpu_memory={} + registers={}), got {}",
                    actual_total,
                    snapshot.wasm_memory.len(),
                    snapshot.gpu_memory.len(),
                    snapshot.registers.len(),
                    snapshot.metadata.total_uncompressed_bytes,
                )
                .into(),
            ));
        }

        // T9 freshness check (same opt-in semantics as the legacy
        // restore path — see `check_freshness`). The artifact store's
        // outer HMAC has already certified the bincode payload, so
        // `created_unix_ms` cannot have been tampered with after the
        // snapshot was sealed.
        self.check_freshness(snapshot.metadata.created_unix_ms)?;
        self.check_replay(&snapshot.metadata)?;

        debug!(
            wasm = snapshot.wasm_memory.len(),
            gpu = snapshot.gpu_memory.len(),
            regs = snapshot.registers.len(),
            version = snapshot.version,
            "snapshot restored from artifact store",
        );
        Ok(snapshot)
    }

    /// T40: detect a v0.4 unified artifact-store envelope by its leading
    /// magic, and if present, decode the inner bincode-encoded
    /// [`Snapshot`] payload.
    ///
    /// Returns `Ok(None)` when the leading 16 bytes do not match
    /// [`tensor_wasm_artifacts::ARTIFACT_MAGIC`] — the caller falls
    /// through to the legacy v3/v2 path. Returns `Ok(Some(snapshot))`
    /// when the envelope verifies (HMAC, content hash) and the inner
    /// payload passes the same per-blob caps / CRC32 / total-bytes
    /// cross-check as the legacy path. Returns `Err` for any
    /// *artifact-envelope* failure other than `BadMagic` (HMAC
    /// mismatch, hash mismatch, malformed inner payload, etc.) so a
    /// tampered v4 blob is not silently mistaken for a malformed v3
    /// blob.
    ///
    /// The HMAC key consulted is the one configured via
    /// [`SnapshotReader::with_hmac_sha256_key`] — same key that
    /// verifies the legacy v3 trailer. Without a key configured, a
    /// v4 envelope is rejected the same way a v3 trailer is (the
    /// outer envelope requires HMAC by construction).
    ///
    /// H1: the reader's [`SnapshotReader::with_max_decompressed`] cap is
    /// honoured on this v4 path — it is threaded into the artifact crate's
    /// `decode_envelope_from_bytes_with_cap` as the zstd zip-bomb ceiling,
    /// so the per-reader cap applies on **both** the v4 (default) and the
    /// legacy v3/v2 paths rather than the v4 path silently using the
    /// artifact crate's hardcoded 1 GiB default.
    ///
    /// M1: only an inner `Snapshot::version` of `SNAPSHOT_VERSION_V2` is
    /// accepted here. Inner v3 is rejected — no inner v3 trailer is written
    /// or verified inside the envelope, so accepting it would assert an
    /// unverifiable signed-inner format.
    #[cfg(all(feature = "artifact-backing", feature = "signed-snapshots"))]
    fn try_restore_artifact_envelope(&self, bytes: &[u8]) -> Result<Option<Snapshot>> {
        use tensor_wasm_artifacts::{ArtifactError, ARTIFACT_MAGIC};
        // Cheap magic check — if the leading 16 bytes are not the
        // artifact magic we let the caller fall through to v3/v2
        // detection. This is the load-bearing classifier; everything
        // below assumes the envelope is at least claiming to be v4.
        if bytes.len() < ARTIFACT_MAGIC.len() || bytes[..ARTIFACT_MAGIC.len()] != ARTIFACT_MAGIC {
            return Ok(None);
        }
        let key = self.hmac_key.as_ref().ok_or_else(|| {
            TensorWasmError::Serialization(
                "snapshot is artifact-envelope (v4) but reader has no HMAC key".into(),
            )
        })?;
        // The artifact crate's pure decode helper does magic + version +
        // HMAC + zstd + content-hash checks in one pass. We map its
        // errors into `TensorWasmError::Serialization` for forward to
        // the snapshot caller; the messages are already operator-facing
        // (no key bytes leak).
        // `key: &Zeroizing<[u8; 32]>` deref-coerces to `&[u8; 32]` at
        // the call site below — no explicit `&**key` dance needed, and
        // the underlying 32 bytes are never copied out of the
        // zeroizing wrapper.
        //
        // H1: pass `self.max_decompressed` through the cap-parameterised
        // variant so the reader's `with_max_decompressed` knob bounds the
        // zip-bomb ceiling on the v4 (default) path exactly as it does on the
        // legacy v3/v2 path. `decode_envelope_from_bytes` (the wrapper) would
        // instead hardcode the artifact crate's 1 GiB `MAX_DECOMPRESSED_LEN`,
        // silently overriding a reader configured for a tighter (or looser)
        // budget.
        let payload = tensor_wasm_artifacts::decode_envelope_from_bytes_with_cap(
            bytes,
            key,
            self.max_decompressed,
        )
        .map_err(|e| match e {
            // `BadMagic` should be impossible here — we already
            // checked the leading 16 bytes — but treat it as a
            // hard failure rather than a fall-through. Otherwise
            // a writer that produced a malformed envelope (e.g.
            // truncated below the minimum length) could escape
            // through the legacy reader and surface a confusing
            // "zstd init" error.
            ArtifactError::BadMagic => TensorWasmError::Serialization(
                "snapshot artifact envelope: minimum-length / magic check failed".into(),
            ),
            other => TensorWasmError::Serialization(
                format!("snapshot artifact envelope: {other}").into(),
            ),
        })?;

        // Decode the bincode payload using the same static allocator
        // ceiling the legacy path uses. The envelope's HMAC has already
        // certified these bytes, so length-prefix abuse cannot reach
        // here without a key leak — the cap is defence-in-depth.
        let cfg = bincode::config::legacy()
            .with_limit::<{ crate::writer::limits::MAX_TOTAL_PAYLOAD_BYTES }>();
        let (snapshot, _read): (Snapshot, usize) =
            bincode::serde::decode_from_slice(payload.as_slice(), cfg).map_err(|e| {
                TensorWasmError::Serialization(format!("bincode decode: {e}").into())
            })?;

        if snapshot.magic != SNAPSHOT_MAGIC {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot magic mismatch: expected {:#X}, got {:#X}",
                    SNAPSHOT_MAGIC, snapshot.magic,
                )
                .into(),
            ));
        }
        // M1: the artifact-backed write path emits *only* the v2 inner
        // discriminant — the outer v4 envelope already supplies
        // authentication (HMAC + content hash), so the inner payload carries
        // no v3 signature trailer. Accept inner v2 ONLY here. We deliberately
        // do *not* accept inner v3 "for forward compatibility": there is no
        // v3 trailer to verify on this path (it was never written), so
        // accepting a v3 discriminant would wave through a value whose
        // claimed signed-inner format is unverifiable. Until a signed-inner-v4
        // format exists with its own inner-trailer verification, the
        // conservative choice is to reject inner v3 outright. The envelope's
        // HMAC has already authenticated these bytes, so this is a
        // format-shape assertion, not an additional integrity gate.
        if snapshot.version != SNAPSHOT_VERSION_V2 {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot artifact envelope inner version mismatch: expected {} (inner v3 \
                     is not accepted on the v4 path — no inner trailer is verified there), got {}",
                    SNAPSHOT_VERSION_V2, snapshot.version,
                )
                .into(),
            ));
        }
        check_blob_size(
            "wasm_memory",
            snapshot.wasm_memory.len(),
            limits::MAX_WASM_MEMORY_BYTES,
        )?;
        check_blob_size(
            "gpu_memory",
            snapshot.gpu_memory.len(),
            limits::MAX_GPU_MEMORY_BYTES,
        )?;
        check_blob_size(
            "registers",
            snapshot.registers.len(),
            limits::MAX_REGISTERS_BYTES,
        )?;
        let expected_crc = payload_crc32(
            &snapshot.wasm_memory,
            &snapshot.gpu_memory,
            &snapshot.registers,
        );
        if snapshot.crc32 != expected_crc {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot crc32 mismatch: expected {:#010X}, got {:#010X}",
                    expected_crc, snapshot.crc32,
                )
                .into(),
            ));
        }
        let actual_total = snapshot
            .wasm_memory
            .len()
            .checked_add(snapshot.gpu_memory.len())
            .and_then(|s| s.checked_add(snapshot.registers.len()))
            .ok_or_else(|| {
                TensorWasmError::Serialization("snapshot blob length sum overflowed usize".into())
            })?;
        if (actual_total as u64) != snapshot.metadata.total_uncompressed_bytes {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot metadata.total_uncompressed_bytes mismatch: \
                     expected {} (wasm_memory={} + gpu_memory={} + registers={}), got {}",
                    actual_total,
                    snapshot.wasm_memory.len(),
                    snapshot.gpu_memory.len(),
                    snapshot.registers.len(),
                    snapshot.metadata.total_uncompressed_bytes,
                )
                .into(),
            ));
        }
        self.check_freshness(snapshot.metadata.created_unix_ms)?;
        self.check_replay(&snapshot.metadata)?;

        debug!(
            wasm = snapshot.wasm_memory.len(),
            gpu = snapshot.gpu_memory.len(),
            regs = snapshot.registers.len(),
            version = snapshot.version,
            "snapshot restored via artifact envelope (T40 default)",
        );
        Ok(Some(snapshot))
    }
}

/// Receiver for the memory blobs produced by
/// [`SnapshotReader::restore_streaming`].
///
/// Each method is called **at most once**, in canonical order
/// (`wasm_memory`, then `gpu_memory`, then `registers`), and only after the
/// reader has fully authenticated and validated the snapshot — see the
/// verify-before-expose note on [`SnapshotReader::restore_streaming`]. The
/// blob is moved into the call by value so the implementation can take
/// ownership (write it to disk, copy it to the GPU, hand it to an `mmap`'d
/// region…) without an extra copy; the reader drops its own reference the
/// instant the method returns.
///
/// Returning `Err` from any method aborts the restore and propagates the
/// error to the [`SnapshotReader::restore_streaming`] caller. Subsequent
/// blob callbacks are skipped.
pub trait SnapshotSink {
    /// Consume the Wasm linear-memory blob.
    fn wasm_memory(&mut self, bytes: Vec<u8>) -> Result<()>;
    /// Consume the GPU device-memory blob.
    fn gpu_memory(&mut self, bytes: Vec<u8>) -> Result<()>;
    /// Consume the register-file blob.
    fn registers(&mut self, bytes: Vec<u8>) -> Result<()>;
}

/// [`SnapshotSink`] adapter that concatenates the three blobs, in canonical
/// order, into a single [`std::io::Write`]. Backs
/// [`SnapshotReader::restore_to_writer`].
struct WriteSink<'a, W: std::io::Write> {
    out: &'a mut W,
}

impl<W: std::io::Write> WriteSink<'_, W> {
    fn write_blob(&mut self, bytes: Vec<u8>) -> Result<()> {
        // Write, then let `bytes` drop at the end of this call so the blob's
        // allocation is released before the next blob is decoded out of the
        // owned `Snapshot`.
        self.out.write_all(&bytes).map_err(|e| {
            TensorWasmError::Serialization(format!("restore_to_writer: {e}").into())
        })
    }
}

impl<W: std::io::Write> SnapshotSink for WriteSink<'_, W> {
    fn wasm_memory(&mut self, bytes: Vec<u8>) -> Result<()> {
        self.write_blob(bytes)
    }
    fn gpu_memory(&mut self, bytes: Vec<u8>) -> Result<()> {
        self.write_blob(bytes)
    }
    fn registers(&mut self, bytes: Vec<u8>) -> Result<()> {
        self.write_blob(bytes)
    }
}

// Memory-mapped restore input path (non-default `mmap` feature). These
// methods map the snapshot file read-only and feed the mapped bytes through
// the unchanged verify-then-decode pipeline, avoiding a full read-into-`Vec`
// of the on-disk blob. Per-method docs carry the full rationale, the
// verify-before-expose note, and the mmap soundness contract.
#[cfg(feature = "mmap")]
impl SnapshotReader {
    /// Restore a snapshot by memory-mapping the file at `path` instead of
    /// reading it wholly into a `Vec<u8>` first, then feed the mapped bytes
    /// through the unchanged verify-then-decode pipeline.
    ///
    /// Available only under the non-default `mmap` feature.
    ///
    /// # Lower peak memory on the input side
    ///
    /// [`SnapshotReader::restore`] takes a `&[u8]`, so a file-based caller
    /// would normally `std::fs::read` the whole compressed blob into an
    /// owned `Vec<u8>`. This wrapper maps the file read-only and hands the
    /// region straight to [`SnapshotReader::restore`]: the OS pages the
    /// compressed bytes in on demand and can evict clean pages under memory
    /// pressure, so the compressed input no longer has to be wholly resident
    /// as an owned allocation. The decompressed payload and decoded
    /// [`Snapshot`] still follow the usual restore path — this only removes
    /// the read-into-`Vec` copy on the input side.
    ///
    /// # Verify-before-expose
    ///
    /// The mapped bytes flow into [`SnapshotReader::restore`] unchanged, so
    /// the full authenticate-then-parse pipeline runs before any payload byte
    /// is exposed, exactly as on the in-memory path.
    ///
    /// # Safety / soundness
    ///
    /// `memmap2::Mmap` is sound only while the underlying file is not mutated
    /// by another process for the duration of this call (a concurrent
    /// truncation can fault the reader). Treat the snapshot file as immutable
    /// — which matches how blobs are produced (atomic-rename of a finished
    /// tempfile) and consumed (read-only restore). The map is dropped before
    /// this method returns.
    #[cfg_attr(docsrs, doc(cfg(feature = "mmap")))]
    #[instrument(skip(self), fields(path = %path.as_ref().display()))]
    pub fn restore_from_path_mmap<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<Snapshot> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| {
            TensorWasmError::Serialization(format!("restore_from_path_mmap open: {e}").into())
        })?;
        // SAFETY: we require (and document) that the snapshot file is not
        // mutated by another process for the duration of this call. The map
        // is read-only and dropped before we return.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
            TensorWasmError::Serialization(format!("restore_from_path_mmap mmap: {e}").into())
        })?;
        // Feed the mapped bytes through the unchanged verify-then-decode
        // pipeline. Authentication runs before any payload byte is exposed.
        self.restore(&mmap)
    }

    /// Streaming counterpart to [`SnapshotReader::restore_from_path_mmap`]:
    /// map the file and stream the decoded blobs to `sink`. Combines the
    /// input-side mmap saving with the output-side per-blob streaming of
    /// [`SnapshotReader::restore_streaming`]. Returns the snapshot metadata.
    ///
    /// Verify-before-expose is preserved: the mapped bytes go through the
    /// full authentication pipeline before any blob reaches `sink`.
    #[cfg_attr(docsrs, doc(cfg(feature = "mmap")))]
    #[instrument(skip(self, sink), fields(path = %path.as_ref().display()))]
    pub fn restore_streaming_from_path_mmap<P: AsRef<std::path::Path>, S: SnapshotSink>(
        &self,
        path: P,
        sink: &mut S,
    ) -> Result<SnapshotMetadata> {
        let file = std::fs::File::open(path.as_ref()).map_err(|e| {
            TensorWasmError::Serialization(
                format!("restore_streaming_from_path_mmap open: {e}").into(),
            )
        })?;
        // SAFETY: same immutability contract as `restore_from_path_mmap`.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| {
            TensorWasmError::Serialization(
                format!("restore_streaming_from_path_mmap mmap: {e}").into(),
            )
        })?;
        self.restore_streaming(&mmap, sink)
    }
}

/// Detect a v3 (signed) trailer at the tail of `bytes` and, if present,
/// return the `(SignatureKind, prefix_len)` it implies.
///
/// The v3 trailer is `[V3_TRAILER_MAGIC: 4][signature_kind: 1][sig: N]`,
/// where `N` depends on the kind (32 for HMAC-SHA256, 64 for Ed25519). The
/// detector probes each known kind's trailer length: for candidate `kind`
/// it checks whether the 4-byte magic sits at `len - kind.trailer_len()`
/// **and** the byte immediately after the magic equals that kind's
/// discriminant. Requiring the kind byte to be self-consistent with the
/// trailer length it was probed at disambiguates the two trailer sizes — a
/// stray `S3T1` inside an Ed25519 signature's bytes cannot masquerade as an
/// HMAC trailer because its kind byte would not read as `1` — while
/// preserving the ~1/2^32 magic false-positive guarantee. Ed25519 (the
/// longer trailer) is probed first.
///
/// As a backward-compatibility fallback, if neither self-consistent probe
/// matches but the 4-byte magic *is* present at the HMAC offset
/// (`len - 37`), the blob is still classified as an HMAC-kind trailer so
/// that a corrupted/unknown kind byte at that position is routed to
/// [`SnapshotReader::verify_v3_trailer`] and rejected there with the precise
/// `"unknown signature_kind"` / length-mismatch error rather than being
/// silently re-classified as v2. The signature is verified later; this
/// function only classifies.
///
/// Returns `None` for any blob that carries no recognised trailer magic at
/// either candidate offset — the caller treats it as v2 (or rejects it under
/// `require_signature`).
#[cfg(feature = "signed-snapshots")]
fn detect_v3_trailer(bytes: &[u8]) -> Option<(SignatureKind, usize)> {
    // First pass: a self-consistent magic+kind match at either candidate
    // trailer length unambiguously selects the kind and the prefix split.
    for kind in [SignatureKind::Ed25519, SignatureKind::HmacSha256] {
        let trailer_len = kind.trailer_len();
        if bytes.len() < trailer_len {
            continue;
        }
        let trailer_start = bytes.len() - trailer_len;
        let magic_ok =
            bytes[trailer_start..trailer_start + V3_TRAILER_MAGIC_LEN] == V3_TRAILER_MAGIC;
        let kind_ok = bytes[trailer_start + V3_TRAILER_MAGIC_LEN] == kind as u8;
        if magic_ok && kind_ok {
            return Some((kind, trailer_start));
        }
    }
    // Fallback: magic present at the HMAC offset but the kind byte did not
    // read as a self-consistent discriminant. Classify as HMAC so
    // `verify_v3_trailer` can surface the precise kind/length error. (The
    // Ed25519 offset has no analogous fallback: a corrupted kind byte there
    // with an intact magic is indistinguishable from a coincidental magic
    // inside payload bytes, so we leave it to the v2 path / version-
    // consistency check, exactly as a stripped trailer would be handled.)
    let hmac_trailer_len = SignatureKind::HmacSha256.trailer_len();
    if bytes.len() >= hmac_trailer_len {
        let trailer_start = bytes.len() - hmac_trailer_len;
        if bytes[trailer_start..trailer_start + V3_TRAILER_MAGIC_LEN] == V3_TRAILER_MAGIC {
            return Some((SignatureKind::HmacSha256, trailer_start));
        }
    }
    None
}

/// In-memory representation of a snapshot whose `gpu_memory` blob has been
/// materialised into a CUDA `UnifiedBuffer` and prefetched to the target
/// device. Only available when the `cuda` feature is enabled.
///
/// The wasm-memory, registers, and metadata fields are owned exactly like
/// the corresponding fields on [`Snapshot`]; the GPU buffer is a fresh
/// allocation backed by managed memory, ready to be handed to a kernel.
#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
pub struct RestoredOnGpu {
    /// Raw bytes of the Wasm linear memory at capture time (host-side copy).
    pub wasm_memory: Vec<u8>,
    /// GPU device-side memory blob, now resident in unified memory.
    pub gpu_memory: cust::memory::UnifiedBuffer<u8>,
    /// Register-file snapshot (PTX-level state captured by the JIT).
    pub registers: Vec<u8>,
    /// Free-form metadata describing the snapshot's provenance.
    pub metadata: crate::writer::SnapshotMetadata,
}

/// Restore `bytes` and stage the `gpu_memory` payload onto the GPU at
/// `device_index` via `cuMemPrefetchAsync` on a fresh non-blocking stream.
///
/// **M2:** this convenience wrapper restores through a *default*
/// [`SnapshotReader::new`], so it applies the default decompressed-size cap
/// and accepts unsigned v2 blobs. Callers that need to apply hardening —
/// [`SnapshotReader::with_max_decompressed`], [`SnapshotReader::require_signature`],
/// [`SnapshotReader::with_hmac_sha256_key`], [`SnapshotReader::with_max_age`] —
/// must use [`restore_to_gpu_with`] and pass their own configured reader.
/// This wrapper is retained for backward compatibility and delegates to it.
///
/// On success the returned [`RestoredOnGpu`] owns a populated
/// `UnifiedBuffer<u8>` whose pages have been requested to migrate to the
/// target device. The stream is synchronised before return so the buffer
/// is observably ready (no half-prefetched state leaks to the caller).
///
/// Requires the `cuda` feature; on no-CUDA builds this symbol does not
/// exist, and callers should fall back to [`SnapshotReader::restore`]
/// followed by a manual host-to-device copy.
#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
#[instrument(skip(bytes), fields(input_len = bytes.len(), device_index = device_index))]
pub fn restore_to_gpu(bytes: &[u8], device_index: u32) -> Result<RestoredOnGpu> {
    restore_to_gpu_with(&SnapshotReader::new(), bytes, device_index)
}

/// Restore `bytes` through the caller-supplied `reader` and stage the
/// `gpu_memory` payload onto the GPU at `device_index` via
/// `cuMemPrefetchAsync` on a fresh non-blocking stream.
///
/// **M2:** this is the hardening-aware counterpart to [`restore_to_gpu`].
/// Because the `reader` is supplied by the caller, every reader knob —
/// [`SnapshotReader::with_max_decompressed`], [`SnapshotReader::require_signature`],
/// [`SnapshotReader::with_hmac_sha256_key`], [`SnapshotReader::with_max_age`] —
/// is honoured on the GPU restore path exactly as it is on the plain
/// [`SnapshotReader::restore`] path. The GPU-staging behaviour is otherwise
/// identical to [`restore_to_gpu`].
///
/// Requires the `cuda` feature; on no-CUDA builds this symbol does not
/// exist, and callers should fall back to [`SnapshotReader::restore`]
/// followed by a manual host-to-device copy.
#[cfg(feature = "cuda")]
#[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
#[instrument(skip(reader, bytes), fields(input_len = bytes.len(), device_index = device_index))]
pub fn restore_to_gpu_with(
    reader: &SnapshotReader,
    bytes: &[u8],
    device_index: u32,
) -> Result<RestoredOnGpu> {
    use cust::memory::UnifiedBuffer;
    use cust::stream::{Stream, StreamFlags};

    let snapshot = reader.restore(bytes)?;

    // UnifiedBuffer::new requires a non-zero capacity to actually allocate;
    // a zero-length snapshot is allowed — we just produce an empty buffer.
    let mut gpu_buf: UnifiedBuffer<u8> = if snapshot.gpu_memory.is_empty() {
        // SAFETY: capacity 0 -> no allocation, no uninitialised reads possible.
        unsafe { UnifiedBuffer::uninitialized(0) }.map_err(|e| {
            TensorWasmError::CudaError(format!("UnifiedBuffer::uninitialized(0): {e:?}"))
        })?
    } else {
        UnifiedBuffer::new(&0u8, snapshot.gpu_memory.len())
            .map_err(|e| TensorWasmError::CudaError(format!("UnifiedBuffer::new: {e:?}")))?
    };

    if !snapshot.gpu_memory.is_empty() {
        gpu_buf.as_mut_slice().copy_from_slice(&snapshot.gpu_memory);

        let device = cust::device::Device::get_device(device_index as i32).map_err(|e| {
            TensorWasmError::CudaError(format!("Device::get_device({device_index}): {e:?}"))
        })?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)
            .map_err(|e| TensorWasmError::CudaError(format!("Stream::new: {e:?}")))?;
        gpu_buf
            .prefetch_to_device(&stream, &device)
            .map_err(|e| TensorWasmError::CudaError(format!("prefetch_to_device: {e:?}")))?;
        stream
            .synchronize()
            .map_err(|e| TensorWasmError::CudaError(format!("stream.synchronize: {e:?}")))?;
    }

    Ok(RestoredOnGpu {
        wasm_memory: snapshot.wasm_memory,
        gpu_memory: gpu_buf,
        registers: snapshot.registers,
        metadata: snapshot.metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "signed-snapshots")]
    use crate::format::{HMAC_SHA256_SIG_LEN, SIGNATURE_KIND_HMAC_SHA256, SIGNATURE_TRAILER_LEN};
    use crate::writer::{InstanceState, SnapshotWriter, SNAPSHOT_VERSION};
    use tensor_wasm_core::types::{InstanceId, TenantId};

    #[test]
    fn malformed_bytes_return_error_without_panicking() {
        let reader = SnapshotReader::new();
        for bad in [
            &b""[..],
            &b"not zstd"[..],
            &[0x28, 0xB5, 0x2F, 0xFD, 0x00, 0x00][..], // zstd magic with truncated frame
            &[0xFF; 64][..],
        ] {
            let err = reader.restore(bad).expect_err("must reject");
            assert!(matches!(err, TensorWasmError::Serialization(_)));
        }
    }

    #[test]
    fn valid_round_trip_succeeds() {
        let wasm = vec![1u8, 2, 3, 4, 5];
        let gpu = vec![9u8; 1024];
        let regs = vec![0x42u8; 16];
        let bytes = SnapshotWriter::new()
            .capture(InstanceState {
                tenant_id: TenantId(11),
                instance_id: InstanceId(22),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            })
            .expect("capture");
        let restored = SnapshotReader::new().restore(&bytes).expect("restore");
        assert_eq!(restored.magic, SNAPSHOT_MAGIC);
        assert_eq!(restored.version, SNAPSHOT_VERSION);
        assert_eq!(restored.wasm_memory, wasm);
        assert_eq!(restored.gpu_memory, gpu);
        assert_eq!(restored.registers, regs);
    }

    #[test]
    fn truncated_blob_is_error() {
        let bytes = SnapshotWriter::new()
            .capture(InstanceState {
                tenant_id: TenantId(1),
                instance_id: InstanceId(1),
                wasm_memory: &[1, 2, 3],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        // Chop off the trailing half of the compressed payload.
        let truncated = &bytes[..bytes.len() / 2];
        let err = SnapshotReader::new()
            .restore(truncated)
            .expect_err("must reject");
        assert!(matches!(err, TensorWasmError::Serialization(_)));
    }

    #[test]
    fn with_max_decompressed_overrides_default() {
        let reader = SnapshotReader::new().with_max_decompressed(1024);
        assert_eq!(reader.max_decompressed(), 1024);
    }

    /// `require_signature` causes well-formed v2 blobs to be rejected with a
    /// message that mentions the unsigned envelope.
    #[test]
    fn require_signature_rejects_v2() {
        let bytes = SnapshotWriter::new()
            .capture(InstanceState {
                tenant_id: TenantId(1),
                instance_id: InstanceId(1),
                wasm_memory: &[1, 2, 3],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        let err = SnapshotReader::new()
            .require_signature()
            .restore(&bytes)
            .expect_err("v2 must be rejected when signature required");
        match err {
            TensorWasmError::Serialization(m) => assert!(
                m.contains("unsigned") && m.contains("required"),
                "unexpected message: {m}",
            ),
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    /// A reader with no HMAC key configured refuses every v3 blob, even one
    /// that was produced by a writer using the same key (the reader has no
    /// way to verify the signature so the only safe answer is to reject).
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn v3_without_key_is_rejected() {
        let key = [0x11u8; 32];
        let bytes = SnapshotWriter::new()
            .with_hmac_sha256_key(key)
            .capture(InstanceState {
                tenant_id: TenantId(2),
                instance_id: InstanceId(2),
                wasm_memory: &[7; 64],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        let err = SnapshotReader::new()
            .restore(&bytes)
            .expect_err("v3 must be rejected when reader has no key");
        match err {
            TensorWasmError::Serialization(m) => assert!(
                m.contains("signed") && m.contains("HMAC key"),
                "unexpected message: {m}",
            ),
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    /// Round-trip a v3 blob with a matching key, then flip a byte in the
    /// stored signature and confirm the reader rejects it with the generic
    /// "HMAC mismatch" message (no key material leaks into the error).
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn v3_tampered_signature_is_rejected() {
        let key = [0xA5u8; 32];
        let mut bytes = SnapshotWriter::new()
            .with_hmac_sha256_key(key)
            .capture(InstanceState {
                tenant_id: TenantId(3),
                instance_id: InstanceId(3),
                wasm_memory: &[1, 2, 3, 4, 5, 6, 7, 8],
                gpu_memory: &[0xCC; 32],
                registers: &[0xDD; 8],
            })
            .expect("capture");
        // Flip the last byte of the signature.
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        let err = SnapshotReader::new()
            .with_hmac_sha256_key(key)
            .restore(&bytes)
            .expect_err("tampered signature must be rejected");
        match err {
            TensorWasmError::Serialization(m) => {
                assert!(m.contains("HMAC mismatch"), "unexpected message: {m}");
                // Defence in depth: confirm the key never leaks into the
                // error string (no hex characters from the 0xA5 pattern).
                assert!(!m.contains("A5"), "error must not leak key bytes: {m}");
            }
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    /// A v3 blob signed with key A is rejected by a reader configured with
    /// key B — the classic wrong-key case.
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn v3_wrong_key_is_rejected() {
        let key_a = [0x01u8; 32];
        let key_b = [0x02u8; 32];
        let bytes = SnapshotWriter::new()
            .with_hmac_sha256_key(key_a)
            .capture(InstanceState {
                tenant_id: TenantId(4),
                instance_id: InstanceId(4),
                wasm_memory: &[42; 16],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        let err = SnapshotReader::new()
            .with_hmac_sha256_key(key_b)
            .restore(&bytes)
            .expect_err("wrong key must be rejected");
        match err {
            TensorWasmError::Serialization(m) => {
                assert!(m.contains("HMAC mismatch"), "unexpected message: {m}");
            }
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    /// Overwriting the signature-kind byte with an unknown discriminant
    /// (under the T8 layout, the kind byte sits at
    /// `len - SIGNATURE_TRAILER_LEN + V3_TRAILER_MAGIC_LEN`) leaves the
    /// trailer magic intact, so the classifier still routes the blob to
    /// `verify_v3_trailer`. There the unknown kind is caught by
    /// `SignatureKind::from_byte` and the reader rejects with
    /// `"unknown signature_kind: <byte>"` — a more precise error than the
    /// pre-T8 "trailer missing" wording (which was an artefact of the
    /// kind byte doubling as the v3 detection sentinel).
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn v3_kind_byte_rewritten_is_rejected() {
        let key = [0x77u8; 32];
        let mut bytes = SnapshotWriter::new()
            .with_hmac_sha256_key(key)
            .capture(InstanceState {
                tenant_id: TenantId(5),
                instance_id: InstanceId(5),
                wasm_memory: &[],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        // T8 trailer layout: `[magic: 4][kind: 1][sig: 32]`. The kind
        // byte sits at `len - SIGNATURE_TRAILER_LEN + V3_TRAILER_MAGIC_LEN`
        // (i.e. immediately after the 4-byte magic prefix).
        let kind_pos = bytes.len() - SIGNATURE_TRAILER_LEN + V3_TRAILER_MAGIC_LEN;
        bytes[kind_pos] = 0xFE; // not a known SignatureKind
        let err = SnapshotReader::new()
            .with_hmac_sha256_key(key)
            .restore(&bytes)
            .expect_err("rewritten kind byte must be rejected");
        match err {
            TensorWasmError::Serialization(m) => assert!(
                m.contains("unknown signature_kind"),
                "unexpected message: {m}",
            ),
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    /// T8: rewriting the v3 trailer magic to anything other than
    /// `V3_TRAILER_MAGIC` makes the classifier treat the blob as v2.
    /// The inner bincode payload still claims `version = 3`, so the
    /// post-decode version-consistency check trips and the reader
    /// rejects with the missing-trailer error. This is the new
    /// equivalent of the pre-T8 "rewrite the kind sentinel" test —
    /// the discriminator is now the 4-byte magic rather than the
    /// single kind byte.
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn v3_trailer_magic_rewritten_falls_through_to_trailer_missing() {
        let key = [0x77u8; 32];
        let mut bytes = SnapshotWriter::new()
            .with_hmac_sha256_key(key)
            .capture(InstanceState {
                tenant_id: TenantId(6),
                instance_id: InstanceId(6),
                wasm_memory: &[],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        // Corrupt the trailer magic — any value other than `V3_TRAILER_MAGIC`.
        let magic_pos = bytes.len() - SIGNATURE_TRAILER_LEN;
        bytes[magic_pos] ^= 0xFF;
        let err = SnapshotReader::new()
            .with_hmac_sha256_key(key)
            .restore(&bytes)
            .expect_err("rewritten trailer magic must be rejected");
        match err {
            TensorWasmError::Serialization(m) => {
                assert!(m.contains("v3 trailer missing"), "unexpected message: {m}",)
            }
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    /// Sanity: the HMAC trailer length equals exactly the 4-byte magic
    /// plus the kind discriminant plus the SHA-256 digest size, so the
    /// trailer offset arithmetic above is correct. Pre-T8 this assertion
    /// was `SIGNATURE_TRAILER_LEN == 1 + HMAC_SHA256_SIG_LEN`; the bump
    /// to 37 bytes is the BREAKING change announced in CHANGELOG.md.
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn trailer_constants_are_consistent() {
        assert_eq!(
            SIGNATURE_TRAILER_LEN,
            V3_TRAILER_MAGIC_LEN + 1 + HMAC_SHA256_SIG_LEN,
        );
        assert_eq!(SIGNATURE_TRAILER_LEN, 37);
    }

    /// T8 regression: a v2 blob whose final byte happens to be `0x01`
    /// (the pre-T8 v3 detection sentinel) must NOT be misclassified as
    /// v3. The new detector keys off a 4-byte magic prefix, so the
    /// single-byte coincidence — which previously hit with ~1/256
    /// probability against legitimate v2 zstd-frame epilogues — is no
    /// longer enough to flip the classifier.
    ///
    /// Strategy: round-trip a v2 capture, brute-force-search a payload
    /// shape whose final byte is `0x01`, and confirm the reader still
    /// accepts it without ever calling the HMAC path. To make the
    /// outcome deterministic, we forcibly rewrite the last byte of a
    /// real v2 capture to `0x01` AFTER capture, which puts the byte
    /// inside the zstd frame epilogue and so almost always corrupts the
    /// frame (the zstd decoder rejects it). To exercise the
    /// classification path without changing payload bytes, we instead
    /// build a synthetic v2 blob whose tail byte is `0x01` by appending
    /// a single byte to a fresh capture: that byte sits past the zstd
    /// frame and would have tripped the pre-T8 v3 sniff, but under T8
    /// it does not match the 4-byte magic and the reader correctly
    /// treats the blob as v2-with-trailing-junk (rejected by the
    /// post-decode "unexpected trailing bytes" check). The point of
    /// the test is that the classifier never reaches `verify_v3_trailer`,
    /// so a reader configured WITHOUT an HMAC key still produces the
    /// trailing-bytes error rather than the "signed (v3) but reader has
    /// no HMAC key" error — i.e. the blob was correctly classified as
    /// v2.
    #[test]
    fn v2_tail_byte_01_is_not_misclassified_as_v3() {
        let bytes = SnapshotWriter::new()
            .capture(InstanceState {
                tenant_id: TenantId(101),
                instance_id: InstanceId(101),
                wasm_memory: &[1, 2, 3, 4, 5, 6, 7, 8],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        // Append a single `0x01` byte so the tail byte at `len - 1`
        // (which under the pre-T8 layout was the kind sentinel position
        // when `SIGNATURE_TRAILER_LEN == 33` and the blob was exactly
        // 33 bytes long, but more importantly is the byte that pre-T8
        // single-byte sniff would have looked at on the inner v2
        // frame's epilogue) is `0x01`. The key behaviour is the
        // classifier outcome: a reader with NO HMAC key must NOT
        // produce the "signed (v3) but reader has no HMAC key" error,
        // because that would mean the blob was misclassified as v3.
        let mut tampered = bytes.clone();
        tampered.push(0x01);
        // Sanity: the new classifier window is at
        // `len - SIGNATURE_TRAILER_LEN..len - SIGNATURE_TRAILER_LEN + 4`,
        // which on a small v2 blob with one appended byte lands inside
        // the zstd frame — exactly the scenario the pre-T8 detector
        // was wrong about.
        let reader = SnapshotReader::new();
        let err = reader
            .restore(&tampered)
            .expect_err("malformed v2 must still be rejected (but as v2, not v3)");
        let TensorWasmError::Serialization(msg) = err else {
            panic!("expected Serialization");
        };
        let msg = msg.to_string();
        // The critical assertion: the rejection must NOT come from
        // the v3 classification path (which would produce a message
        // mentioning the HMAC key or v3 trailer). Any v2-shaped error
        // is acceptable here — zstd decode failure, unexpected
        // trailing bytes, or magic/version mismatch — what matters
        // is that the magic-prefix detector did not flip on a stray
        // `0x01` byte.
        assert!(
            !msg.contains("signed (v3)") && !msg.contains("HMAC"),
            "v2 blob misclassified as v3: {msg}",
        );
    }

    /// Companion to the above: a v2 blob whose **trailer-magic window**
    /// (i.e. the 4 bytes at offset `len - SIGNATURE_TRAILER_LEN`)
    /// happens to start with `0x01` (the pre-T8 sentinel) but does NOT
    /// equal `V3_TRAILER_MAGIC` must be classified as v2. This is the
    /// hardened-against-T8-regression version of the test above: it
    /// guarantees the reader looks at all 4 magic bytes, not just the
    /// first.
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn v2_with_kind_byte_in_trailer_window_is_not_v3() {
        let bytes = SnapshotWriter::new()
            .capture(InstanceState {
                tenant_id: TenantId(102),
                instance_id: InstanceId(102),
                wasm_memory: &[9, 9, 9, 9, 9, 9, 9, 9],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        // We need a v2 blob whose tail 4 bytes are exactly `[0x01, X, Y, Z]`
        // with X/Y/Z not matching the T8 magic. Append a synthetic
        // 4-byte tail to a real v2 capture; the appended bytes are
        // beyond the zstd frame so the decoder will reject the input
        // (v2 forbids trailing bytes), but the rejection must come from
        // the v2 path, not from the v3 HMAC check.
        let mut tampered = bytes.clone();
        // Make sure the magic-window does NOT match V3_TRAILER_MAGIC.
        // Choose four bytes whose first byte is `0x01` (the legacy
        // sentinel) but whose remaining bytes mismatch on the magic
        // (S3T1 = [0x53, 0x33, 0x54, 0x31]).
        debug_assert_ne!(V3_TRAILER_MAGIC[0], 0x01);
        tampered.extend_from_slice(&[0x01, 0x02, 0x03, 0x04]);
        // Pad up so the magic window is large enough — append zeros
        // until we are confidently past the original zstd frame and
        // the classifier reads the appended `0x01` at position
        // `len - SIGNATURE_TRAILER_LEN`. The simplest reliable shape:
        // pad to exactly `original_len + SIGNATURE_TRAILER_LEN`, so
        // the classifier's 4-byte window starts at the boundary we
        // wrote `0x01` to.
        let target_len = bytes.len() + SIGNATURE_TRAILER_LEN;
        while tampered.len() < target_len {
            tampered.push(0x00);
        }
        // Now bytes[len - SIGNATURE_TRAILER_LEN..len - SIGNATURE_TRAILER_LEN + 4]
        // starts with `0x01` but does not equal `S3T1`.
        let trailer_start = tampered.len() - SIGNATURE_TRAILER_LEN;
        assert_eq!(tampered[trailer_start], 0x01, "test setup invariant");
        assert_ne!(
            &tampered[trailer_start..trailer_start + V3_TRAILER_MAGIC_LEN],
            &V3_TRAILER_MAGIC,
            "test setup invariant: window must NOT equal v3 magic",
        );
        // Reader with HMAC key configured: if the blob were
        // misclassified as v3, we would hit "HMAC mismatch". Under T8
        // the classifier correctly treats it as v2 and the rejection
        // comes from the v2 trailing-bytes path.
        let reader = SnapshotReader::new().with_hmac_sha256_key([0xAAu8; 32]);
        let err = reader
            .restore(&tampered)
            .expect_err("padded v2 must be rejected via the v2 path");
        let TensorWasmError::Serialization(msg) = err else {
            panic!("expected Serialization");
        };
        let msg = msg.to_string();
        assert!(
            !msg.contains("HMAC mismatch"),
            "v2 blob with `0x01` in magic window was misclassified as v3: {msg}",
        );
    }

    /// T8 round-trip: a v3 capture from the post-T8 writer parses
    /// successfully through the post-T8 reader. Distinct from the
    /// existing `signed_writer_emits_v3_and_round_trips` test in
    /// `writer.rs` in that it lives next to the classifier code it
    /// guards, so a future reader refactor that breaks the magic-prefix
    /// path trips this test even if the writer-side test is skipped.
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn v3_round_trip_through_magic_prefix_trailer() {
        let key = [0x5Au8; 32];
        let wasm = vec![0xAAu8; 256];
        let gpu = vec![0xBBu8; 512];
        let regs = vec![0xCCu8; 32];
        let bytes = SnapshotWriter::new()
            .with_hmac_sha256_key(key)
            .capture(InstanceState {
                tenant_id: TenantId(7),
                instance_id: InstanceId(7),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            })
            .expect("capture v3");
        // The emitted blob must end with `[V3_TRAILER_MAGIC][kind=1][32-byte sig]`.
        assert!(bytes.len() >= SIGNATURE_TRAILER_LEN);
        let trailer_start = bytes.len() - SIGNATURE_TRAILER_LEN;
        assert_eq!(
            &bytes[trailer_start..trailer_start + V3_TRAILER_MAGIC_LEN],
            &V3_TRAILER_MAGIC,
            "writer must emit the v3 trailer magic",
        );
        assert_eq!(
            bytes[trailer_start + V3_TRAILER_MAGIC_LEN],
            SIGNATURE_KIND_HMAC_SHA256,
            "writer must emit the kind byte after the magic",
        );

        let restored = SnapshotReader::new()
            .with_hmac_sha256_key(key)
            .restore(&bytes)
            .expect("v3 must round-trip through magic-prefix reader");
        assert_eq!(restored.wasm_memory, wasm);
        assert_eq!(restored.gpu_memory, gpu);
        assert_eq!(restored.registers, regs);
        assert_eq!(restored.version, crate::format::SNAPSHOT_VERSION_V3);
    }

    // ----- Ed25519 asymmetric signatures -----

    /// Ed25519 sign → verify round-trip: a writer signs with the private
    /// key, a reader holding only the public key restores it.
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn ed25519_round_trip() {
        use ed25519_dalek::SigningKey;
        let signing_key = SigningKey::from_bytes(&[0x21u8; 32]);
        let verifying_key = signing_key.verifying_key();
        let wasm = vec![0xAAu8; 200];
        let gpu = vec![0xBBu8; 100];
        let regs = vec![0xCCu8; 8];
        let bytes = SnapshotWriter::new()
            .with_ed25519_signing_key(signing_key)
            .capture(InstanceState {
                tenant_id: TenantId(7),
                instance_id: InstanceId(7),
                wasm_memory: &wasm,
                gpu_memory: &gpu,
                registers: &regs,
            })
            .expect("capture");
        let restored = SnapshotReader::new()
            .with_ed25519_verifying_key(verifying_key)
            .restore(&bytes)
            .expect("ed25519 round-trip");
        assert_eq!(restored.wasm_memory, wasm);
        assert_eq!(restored.gpu_memory, gpu);
        assert_eq!(restored.registers, regs);
    }

    /// A blob signed with key A is rejected by a reader holding key B's
    /// public key.
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn ed25519_wrong_key_is_rejected() {
        use ed25519_dalek::SigningKey;
        let signing_key_a = SigningKey::from_bytes(&[0x01u8; 32]);
        let verifying_key_b = SigningKey::from_bytes(&[0x02u8; 32]).verifying_key();
        let bytes = SnapshotWriter::new()
            .with_ed25519_signing_key(signing_key_a)
            .capture(InstanceState {
                tenant_id: TenantId(4),
                instance_id: InstanceId(4),
                wasm_memory: &[42; 16],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        let err = SnapshotReader::new()
            .with_ed25519_verifying_key(verifying_key_b)
            .restore(&bytes)
            .expect_err("wrong key must be rejected");
        match err {
            TensorWasmError::Serialization(m) => {
                assert!(m.contains("Ed25519"), "unexpected message: {m}");
            }
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    /// A reader configured with an Ed25519 key but no HMAC key cannot verify
    /// an HMAC-signed blob, and vice-versa: the kind byte distinguishes the
    /// two so a blob signed under one scheme cannot be presented as the other
    /// (downgrade / cross-scheme rejection).
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn ed25519_vs_hmac_kind_byte_downgrade_rejected() {
        use ed25519_dalek::SigningKey;
        let signing_key = SigningKey::from_bytes(&[0x33u8; 32]);
        let verifying_key = signing_key.verifying_key();
        let hmac_key = [0x44u8; 32];

        // Ed25519 blob presented to an HMAC-only reader → rejected (the
        // reader has no Ed25519 key for the kind=2 trailer).
        let ed_bytes = SnapshotWriter::new()
            .with_ed25519_signing_key(signing_key)
            .capture(InstanceState {
                tenant_id: TenantId(5),
                instance_id: InstanceId(5),
                wasm_memory: &[1, 2, 3, 4],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture ed25519");
        let err = SnapshotReader::new()
            .with_hmac_sha256_key(hmac_key)
            .restore(&ed_bytes)
            .expect_err("ed25519 blob must not verify against an HMAC-only reader");
        assert!(
            matches!(err, TensorWasmError::Serialization(_)),
            "expected Serialization",
        );

        // HMAC blob presented to an Ed25519-only reader → rejected.
        let hmac_bytes = SnapshotWriter::new()
            .with_hmac_sha256_key(hmac_key)
            .with_legacy_envelope()
            .capture(InstanceState {
                tenant_id: TenantId(6),
                instance_id: InstanceId(6),
                wasm_memory: &[5, 6, 7, 8],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture hmac");
        let err = SnapshotReader::new()
            .with_ed25519_verifying_key(verifying_key)
            .restore(&hmac_bytes)
            .expect_err("hmac blob must not verify against an ed25519-only reader");
        match err {
            TensorWasmError::Serialization(m) => assert!(
                m.contains("no HMAC key"),
                "expected the HMAC-key-missing error, got: {m}",
            ),
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    /// Flipping the kind byte of an Ed25519 trailer to claim HMAC (=1) must
    /// be rejected — the asymmetric signature authenticates the kind byte,
    /// and the trailer-length classifier no longer self-consistently matches.
    #[cfg(feature = "signed-snapshots")]
    #[test]
    fn ed25519_kind_byte_rewrite_to_hmac_rejected() {
        use ed25519_dalek::SigningKey;
        let signing_key = SigningKey::from_bytes(&[0x55u8; 32]);
        let verifying_key = signing_key.verifying_key();
        let mut bytes = SnapshotWriter::new()
            .with_ed25519_signing_key(signing_key)
            .capture(InstanceState {
                tenant_id: TenantId(8),
                instance_id: InstanceId(8),
                wasm_memory: &[9; 32],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        // Rewrite the kind byte (at the Ed25519 trailer's magic+4 offset)
        // from 2 (Ed25519) to 1 (HMAC).
        let kind_pos = bytes.len() - crate::format::ED25519_TRAILER_LEN + V3_TRAILER_MAGIC_LEN;
        assert_eq!(bytes[kind_pos], crate::format::SIGNATURE_KIND_ED25519);
        bytes[kind_pos] = SIGNATURE_KIND_HMAC_SHA256;
        let err = SnapshotReader::new()
            .with_ed25519_verifying_key(verifying_key)
            .with_hmac_sha256_key([0x55u8; 32])
            .restore(&bytes)
            .expect_err("kind-byte downgrade must be rejected");
        assert!(
            matches!(err, TensorWasmError::Serialization(_)),
            "expected Serialization",
        );
    }

    // ----- Replay protection: nonce -----

    /// Matching nonce is accepted; mismatched / missing nonce is rejected.
    #[test]
    fn nonce_match_and_mismatch() {
        let nonce = [0x7Eu8; 16];
        let bytes = SnapshotWriter::new()
            .with_nonce(nonce)
            .capture(InstanceState {
                tenant_id: TenantId(1),
                instance_id: InstanceId(1),
                wasm_memory: &[1, 2, 3],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");

        // Match → accepted.
        SnapshotReader::new()
            .with_expected_nonce(nonce)
            .restore(&bytes)
            .expect("matching nonce must be accepted");

        // Mismatch → rejected.
        let err = SnapshotReader::new()
            .with_expected_nonce([0x00u8; 16])
            .restore(&bytes)
            .expect_err("mismatched nonce must be rejected");
        match err {
            TensorWasmError::Serialization(m) => {
                assert!(m.contains("nonce mismatch"), "unexpected message: {m}")
            }
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    /// A reader requiring a nonce rejects a blob that carries none.
    #[test]
    fn nonce_required_but_absent_is_rejected() {
        let bytes = SnapshotWriter::new()
            .capture(InstanceState {
                tenant_id: TenantId(1),
                instance_id: InstanceId(1),
                wasm_memory: &[1, 2, 3],
                gpu_memory: &[],
                registers: &[],
            })
            .expect("capture");
        let err = SnapshotReader::new()
            .with_expected_nonce([0x01u8; 16])
            .restore(&bytes)
            .expect_err("absent nonce must be rejected when required");
        match err {
            TensorWasmError::Serialization(m) => {
                assert!(m.contains("nonce missing"), "unexpected message: {m}")
            }
            other => panic!("expected Serialization, got {other:?}"),
        }
    }

    // ----- Replay protection: sequence_no floor -----

    /// A snapshot at or above the floor is accepted; one below is rejected.
    #[test]
    fn sequence_no_floor_accept_and_reject() {
        let make = |seq: u64| {
            SnapshotWriter::new()
                .with_sequence_no(seq)
                .capture(InstanceState {
                    tenant_id: TenantId(1),
                    instance_id: InstanceId(1),
                    wasm_memory: &[1, 2, 3],
                    gpu_memory: &[],
                    registers: &[],
                })
                .expect("capture")
        };

        // At the floor → accepted.
        let at_floor = make(10);
        SnapshotReader::new()
            .with_min_sequence_no(10)
            .restore(&at_floor)
            .expect("seq == floor must be accepted");

        // Above the floor → accepted.
        let above = make(11);
        SnapshotReader::new()
            .with_min_sequence_no(10)
            .restore(&above)
            .expect("seq > floor must be accepted");

        // Below the floor → rejected (rollback/replay).
        let below = make(9);
        let err = SnapshotReader::new()
            .with_min_sequence_no(10)
            .restore(&below)
            .expect_err("seq < floor must be rejected");
        match err {
            TensorWasmError::Serialization(m) => assert!(
                m.contains("sequence_no") && m.contains("below floor"),
                "unexpected message: {m}",
            ),
            other => panic!("expected Serialization, got {other:?}"),
        }
    }
}
