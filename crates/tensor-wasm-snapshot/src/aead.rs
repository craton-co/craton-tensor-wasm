// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! AEAD encryption-at-rest envelope for snapshot blobs (non-default
//! `aead-at-rest` feature).
//!
//! The default snapshot pipeline *authenticates* a blob (CRC32 + optional
//! HMAC-SHA256 / Ed25519 signature) but leaves the payload in the clear — the
//! zstd-compressed bytes are readable by anyone who can read the file. For
//! callers that need the payload to be **confidential** at rest as well, this
//! module wraps the already-framed snapshot bytes in a ChaCha20-Poly1305 AEAD
//! envelope:
//!
//! ```text
//! AEAD_MAGIC(8) || nonce(12) || ChaCha20Poly1305(key, nonce, ad = AEAD_MAGIC, blob)
//! ```
//!
//! The trailing ChaCha20-Poly1305 output is `ciphertext || tag(16)`, where the
//! ciphertext is the same length as the inner snapshot blob. The 8-byte magic
//! is bound into the AEAD as associated data so the framing cannot be stripped
//! or swapped without failing the tag check.
//!
//! ## Layering with authentication
//!
//! AEAD here provides *confidentiality + integrity of the ciphertext*, but it
//! is keyed independently of the snapshot signature. The recommended layering
//! is **sign-then-encrypt**: produce a signed v3/v4 blob via
//! [`crate::writer::SnapshotWriter::capture`] (with an HMAC or Ed25519 key),
//! then encrypt it here. On the read side, decrypt first
//! ([`crate::reader::SnapshotReader::restore_decrypted`]) and then run the
//! normal verify-then-decode pipeline on the recovered plaintext, so the
//! snapshot's own signature is still checked. The AEAD tag tells you the
//! ciphertext is intact under the AEAD key; the snapshot signature tells you
//! the *contents* were produced by a holder of the signing key.
//!
//! ## Nonce discipline
//!
//! ChaCha20-Poly1305 requires a **unique** 96-bit nonce per (key, message).
//! This module takes the nonce from the caller rather than generating one, to
//! avoid pulling an RNG into the crate's dependency surface and to match the
//! deterministic-by-request philosophy of the writer (see
//! [`crate::writer::SnapshotWriter::with_created_unix_ms`]). **Reusing a nonce
//! under the same key is catastrophic** for ChaCha20-Poly1305 — callers must
//! supply a fresh random nonce (or a strictly-monotonic counter) for every
//! encryption under a given key.

use tensor_wasm_core::error::{Result, TensorWasmError};

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};

/// 8-byte magic prefix that marks a ChaCha20-Poly1305 AEAD-at-rest envelope.
///
/// ASCII `b"twsAEAD1"` ("tensor-wasm snapshot AEAD v1"). Bound into the AEAD as
/// associated data, so an attacker cannot strip or rewrite the framing without
/// invalidating the Poly1305 tag.
pub const AEAD_MAGIC: [u8; 8] = *b"twsAEAD1";

/// Length in bytes of the ChaCha20-Poly1305 nonce (96 bits).
pub const AEAD_NONCE_LEN: usize = 12;

/// Length in bytes of the ChaCha20-Poly1305 authentication tag (128 bits).
pub const AEAD_TAG_LEN: usize = 16;

/// Total length of the AEAD envelope header (magic + nonce) that precedes the
/// ciphertext.
pub const AEAD_HEADER_LEN: usize = AEAD_MAGIC.len() + AEAD_NONCE_LEN;

/// Encrypt `blob` under `key` with the caller-supplied `nonce`, producing the
/// AEAD-at-rest envelope `AEAD_MAGIC || nonce || ciphertext || tag`.
///
/// `blob` is the already-framed snapshot bytes (v2/v3/v4 — whatever
/// [`crate::writer::SnapshotWriter::capture`] emitted). The 8-byte magic is fed
/// to ChaCha20-Poly1305 as associated data so the framing is authenticated.
///
/// SECURITY: `nonce` MUST be unique per (key, message). See the module-level
/// nonce-discipline note — reuse under the same key breaks confidentiality.
pub(crate) fn encrypt_blob(
    blob: &[u8],
    key: &[u8; 32],
    nonce: &[u8; AEAD_NONCE_LEN],
) -> Result<Vec<u8>> {
    let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
    let ciphertext = cipher
        .encrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: blob,
                aad: &AEAD_MAGIC,
            },
        )
        .map_err(|_| {
            // `aead::Error` is opaque by design (it carries no detail to avoid
            // leaking anything about the key or plaintext); translate to a
            // generic, key-free message.
            TensorWasmError::Serialization("snapshot AEAD encrypt failed".into())
        })?;

    let mut out = Vec::with_capacity(AEAD_HEADER_LEN + ciphertext.len());
    out.extend_from_slice(&AEAD_MAGIC);
    out.extend_from_slice(nonce);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Returns `true` if `bytes` begins with [`AEAD_MAGIC`] — i.e. it is (claims to
/// be) an AEAD-at-rest envelope rather than a plaintext snapshot blob.
#[must_use]
pub(crate) fn has_aead_magic(bytes: &[u8]) -> bool {
    bytes.len() >= AEAD_MAGIC.len() && bytes[..AEAD_MAGIC.len()] == AEAD_MAGIC
}

/// Decrypt an AEAD-at-rest envelope produced by [`encrypt_blob`], returning the
/// inner snapshot blob.
///
/// Verifies the Poly1305 tag (over the ciphertext, with the 8-byte magic as
/// associated data) before returning any plaintext, so a tampered envelope or a
/// wrong key fails here without exposing decrypted bytes. The recovered blob is
/// the original framed snapshot; the caller is expected to feed it through the
/// normal verify-then-decode pipeline so the snapshot's own signature is also
/// checked.
pub(crate) fn decrypt_blob(bytes: &[u8], key: &[u8; 32]) -> Result<Vec<u8>> {
    if !has_aead_magic(bytes) {
        return Err(TensorWasmError::Serialization(
            "snapshot AEAD envelope: bad magic (not an encrypted snapshot)".into(),
        ));
    }
    // Need at least header + tag to hold a (possibly empty) ciphertext.
    if bytes.len() < AEAD_HEADER_LEN + AEAD_TAG_LEN {
        return Err(TensorWasmError::Serialization(
            "snapshot AEAD envelope: truncated (shorter than header + tag)".into(),
        ));
    }
    let nonce = &bytes[AEAD_MAGIC.len()..AEAD_HEADER_LEN];
    let ciphertext = &bytes[AEAD_HEADER_LEN..];

    let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
    cipher
        .decrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: ciphertext,
                aad: &AEAD_MAGIC,
            },
        )
        .map_err(|_| {
            // Opaque on purpose — do not distinguish "wrong key" from "tampered
            // ciphertext" in the message (both are an authentication failure)
            // and never echo key or nonce bytes.
            TensorWasmError::Serialization("snapshot AEAD decrypt/authentication failed".into())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let key = [0x11u8; 32];
        let nonce = [0x22u8; AEAD_NONCE_LEN];
        let blob = b"the framed snapshot bytes".to_vec();
        let env = encrypt_blob(&blob, &key, &nonce).expect("encrypt");
        assert!(has_aead_magic(&env));
        assert_eq!(&env[..AEAD_MAGIC.len()], &AEAD_MAGIC);
        let back = decrypt_blob(&env, &key).expect("decrypt");
        assert_eq!(back, blob);
    }

    #[test]
    fn empty_blob_round_trips() {
        let key = [0x33u8; 32];
        let nonce = [0x44u8; AEAD_NONCE_LEN];
        let env = encrypt_blob(&[], &key, &nonce).expect("encrypt empty");
        // header + tag, no ciphertext body.
        assert_eq!(env.len(), AEAD_HEADER_LEN + AEAD_TAG_LEN);
        let back = decrypt_blob(&env, &key).expect("decrypt empty");
        assert!(back.is_empty());
    }

    #[test]
    fn wrong_key_is_rejected() {
        let nonce = [0x55u8; AEAD_NONCE_LEN];
        let env = encrypt_blob(b"secret", &[0x01u8; 32], &nonce).expect("encrypt");
        let err = decrypt_blob(&env, &[0x02u8; 32]).expect_err("wrong key must fail");
        let TensorWasmError::Serialization(m) = err else {
            panic!("expected Serialization");
        };
        assert!(m.contains("authentication failed"), "unexpected: {m}");
    }

    #[test]
    fn tampered_ciphertext_is_rejected() {
        let key = [0x66u8; 32];
        let nonce = [0x77u8; AEAD_NONCE_LEN];
        let mut env = encrypt_blob(b"hello world payload", &key, &nonce).expect("encrypt");
        // Flip a byte in the ciphertext body.
        let last = env.len() - 1;
        env[last] ^= 0x80;
        assert!(decrypt_blob(&env, &key).is_err());
    }

    #[test]
    fn rewritten_magic_is_rejected() {
        let key = [0x88u8; 32];
        let nonce = [0x99u8; AEAD_NONCE_LEN];
        let mut env = encrypt_blob(b"payload", &key, &nonce).expect("encrypt");
        env[0] ^= 0x01; // corrupt the magic
        let err = decrypt_blob(&env, &key).expect_err("bad magic must fail");
        let TensorWasmError::Serialization(m) = err else {
            panic!("expected Serialization");
        };
        assert!(m.contains("bad magic"), "unexpected: {m}");
    }
}
