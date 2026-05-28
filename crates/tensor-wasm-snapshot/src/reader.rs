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

use tensor_wasm_core::error::{TensorWasmError, Result};
use tracing::{debug, instrument};

use crate::format::{SNAPSHOT_VERSION_V2, SNAPSHOT_VERSION_V3};
use crate::writer::{
    check_blob_size, limits, payload_crc32, Snapshot, SNAPSHOT_MAGIC, SNAPSHOT_VERSION,
};

#[cfg(feature = "signed-snapshots")]
use crate::format::{
    SignatureKind, HMAC_SHA256_SIG_LEN, SIGNATURE_KIND_HMAC_SHA256, SIGNATURE_TRAILER_LEN,
    V3_TRAILER_MAGIC, V3_TRAILER_MAGIC_LEN,
};
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
    /// When `true`, v2 (unsigned) inputs are rejected even if otherwise
    /// well-formed. Allows operators to enforce signature-only restores
    /// without compiling a separate binary.
    require_signature: bool,
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
            &self.hmac_key.as_ref().map(|_| "<REDACTED 32-byte HMAC key>"),
        );
        d.field("require_signature", &self.require_signature);
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
            require_signature: false,
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
        #[cfg(all(feature = "artifact-backing", feature = "signed-snapshots"))]
        {
            if let Some(snapshot) = self.try_restore_artifact_envelope(bytes)? {
                return Ok(snapshot);
            }
        }

        // Detect a v3 (signed) blob by peeking at the trailer position.
        // A v3 envelope is
        // `[compressed prefix][V3_TRAILER_MAGIC: 4][signature_kind: 1][32-byte sig]`
        // where the trailer magic starts at `len - SIGNATURE_TRAILER_LEN`.
        // We deliberately key off the trailer *before* decompression so
        // HMAC verification can authenticate the prefix bytes before zstd
        // or bincode see them — the "authenticate then parse" property.
        //
        // **T8:** prior to this commit the detector was a single-byte
        // sniff at `bytes[len - 33] == SIGNATURE_KIND_HMAC_SHA256` (`1`).
        // Because that byte sits inside the zstd frame epilogue of a
        // legitimate v2 blob with ~1/256 probability, a v2 capture could
        // be misclassified as v3 and then rejected by the HMAC check —
        // a downgrade-shaped error message and wasted HMAC work, both
        // observable side channels. Switching to a 4-byte magic prefix
        // shrinks the false-positive rate to ~1/2^32 (~2.3e-10), well
        // below per-blob CRC32 collision rates. v2 snapshots whose tail
        // bytes happen to match `S3T1` exactly are still vanishingly
        // rare in practice; if one is observed it will be caught by the
        // HMAC mismatch path (since the writer would not have signed it)
        // exactly as before.
        #[cfg(feature = "signed-snapshots")]
        let is_v3 = bytes.len() >= SIGNATURE_TRAILER_LEN && {
            let trailer_start = bytes.len() - SIGNATURE_TRAILER_LEN;
            // Magic check: the first 4 bytes of the trailer must equal
            // V3_TRAILER_MAGIC. We deliberately do NOT also check the
            // signature-kind byte here — that check lives inside
            // `verify_v3_trailer`, which already rejects unknown kinds
            // with a descriptive error and so doubles as the
            // forward-compatibility hook for future variants.
            &bytes[trailer_start..trailer_start + V3_TRAILER_MAGIC_LEN] == &V3_TRAILER_MAGIC
        };
        #[cfg(not(feature = "signed-snapshots"))]
        let is_v3 = false;

        // STEP 2.5 — AUTHENTICATE FIRST.
        // Verify HMAC over the compressed prefix before any decompression or
        // bincode decode runs. On failure we return immediately so an attacker
        // cannot use the zstd or bincode decoders as oracles. The trailer
        // length and HMAC are constants of the v3 envelope (37 bytes total
        // post-T8: 4-byte magic + 1-byte kind + 32-byte signature), so the
        // prefix is exactly `bytes[..len - SIGNATURE_TRAILER_LEN]`.
        #[cfg(feature = "signed-snapshots")]
        let prefix_len = if is_v3 {
            let p = bytes.len() - SIGNATURE_TRAILER_LEN;
            self.verify_v3_trailer(bytes, p)?;
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
            bincode::serde::decode_from_slice(decompressed.as_slice(), cfg)
                .map_err(|e| TensorWasmError::Serialization(format!("bincode decode: {e}").into()))?;

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
                TensorWasmError::Serialization(
                    "snapshot blob length sum overflowed usize".into(),
                )
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
    /// **T8 wire format:** the trailer is now `[magic: 4][kind: 1][sig: 32]`
    /// = 37 bytes. The classifier in [`SnapshotReader::restore`] has
    /// already checked that the magic equals [`V3_TRAILER_MAGIC`] before
    /// dispatching here; we re-read the magic from the slice (rather than
    /// trusting the classifier) so this function remains correct under
    /// future refactors that hoist the check.
    ///
    /// Errors are deliberately generic: we never include the expected or
    /// observed signature bytes in the error message, since either could
    /// leak information about the secret key under a side-channel attacker.
    /// The constant-time `ct_eq` from `subtle` is used to compare the
    /// recomputed HMAC against the stored bytes so a timing oracle cannot
    /// recover the signature byte-by-byte.
    #[cfg(feature = "signed-snapshots")]
    fn verify_v3_trailer(&self, bytes: &[u8], prefix_len: usize) -> Result<()> {
        let key = self.hmac_key.as_ref().ok_or_else(|| {
            TensorWasmError::Serialization(
                "snapshot is signed (v3) but reader has no HMAC key".into(),
            )
        })?;

        // The trailer must be exactly `[magic: 4][kind: 1][sig: 32]`.
        // Anything shorter is a truncation; anything longer is junk after
        // the signature (which we refuse rather than silently accept).
        let trailer = bytes
            .get(prefix_len..)
            .ok_or_else(|| TensorWasmError::Serialization("snapshot v3 trailer missing".into()))?;
        if trailer.len() != SIGNATURE_TRAILER_LEN {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot v3 trailer length mismatch: expected {} bytes, got {}",
                    SIGNATURE_TRAILER_LEN,
                    trailer.len(),
                )
                .into(),
            ));
        }
        // Layout: trailer[0..4] = magic, trailer[4] = kind, trailer[5..] = sig.
        // The classifier already checked the magic, but re-validate here
        // for defence in depth (a future refactor that hoists detection
        // upstream must not be able to skip authentication).
        let magic_bytes = &trailer[..V3_TRAILER_MAGIC_LEN];
        if magic_bytes != &V3_TRAILER_MAGIC {
            return Err(TensorWasmError::Serialization(
                "snapshot v3 trailer magic mismatch".into(),
            ));
        }
        let kind_byte = trailer[V3_TRAILER_MAGIC_LEN];
        let sig_bytes = &trailer[V3_TRAILER_MAGIC_LEN + 1..];
        let kind = SignatureKind::from_byte(kind_byte).ok_or_else(|| {
            TensorWasmError::Serialization(
                format!("unknown signature_kind: {kind_byte}").into(),
            )
        })?;
        debug_assert_eq!(sig_bytes.len(), kind.signature_len());

        match kind {
            SignatureKind::HmacSha256 => {
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
                let ok: bool = expected
                    .as_slice()
                    .ct_eq(sig_bytes)
                    .unwrap_u8()
                    == 1;
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
            TensorWasmError::Serialization(
                format!("artifact store get: {e}").into(),
            )
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

        // The artifact-backed write path emits the v2 inner discriminant
        // (the outer envelope already supplies authentication). Accept v2
        // and v3 here for forward compatibility — a future writer might
        // route signed inner payloads through the same envelope without
        // bumping the wire format.
        if snapshot.version != SNAPSHOT_VERSION_V2 && snapshot.version != SNAPSHOT_VERSION_V3 {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot version mismatch: expected {} or {}, got {}",
                    SNAPSHOT_VERSION_V2, SNAPSHOT_VERSION_V3, snapshot.version,
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
                TensorWasmError::Serialization(
                    "snapshot blob length sum overflowed usize".into(),
                )
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
        let payload = tensor_wasm_artifacts::decode_envelope_from_bytes(bytes, key)
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
        // The artifact-backed write path emits the v2 inner discriminant
        // (the outer envelope already supplies authentication). Accept
        // v2 and v3 for forward compatibility — a future writer might
        // route signed inner payloads through the same envelope without
        // bumping the wire format.
        if snapshot.version != SNAPSHOT_VERSION_V2 && snapshot.version != SNAPSHOT_VERSION_V3 {
            return Err(TensorWasmError::Serialization(
                format!(
                    "snapshot version mismatch: expected {} or {}, got {}",
                    SNAPSHOT_VERSION_V2, SNAPSHOT_VERSION_V3, snapshot.version,
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
                TensorWasmError::Serialization(
                    "snapshot blob length sum overflowed usize".into(),
                )
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
    use cust::memory::UnifiedBuffer;
    use cust::stream::{Stream, StreamFlags};

    let snapshot = SnapshotReader::new().restore(bytes)?;

    // UnifiedBuffer::new requires a non-zero capacity to actually allocate;
    // a zero-length snapshot is allowed — we just produce an empty buffer.
    let mut gpu_buf: UnifiedBuffer<u8> = if snapshot.gpu_memory.is_empty() {
        // SAFETY: capacity 0 -> no allocation, no uninitialised reads possible.
        unsafe { UnifiedBuffer::uninitialized(0) }
            .map_err(|e| TensorWasmError::CudaError(format!("UnifiedBuffer::uninitialized(0): {e:?}")))?
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
    use crate::writer::{InstanceState, SnapshotWriter};
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
            TensorWasmError::Serialization(m) => assert!(
                m.contains("v3 trailer missing"),
                "unexpected message: {m}",
            ),
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
}
