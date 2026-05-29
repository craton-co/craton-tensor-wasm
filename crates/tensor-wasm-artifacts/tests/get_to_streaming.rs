// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `get_to` streaming-output API: round-trip into a writer plus tamper
//! rejection that proves no unverified bytes are emitted.

use std::io::{Read, Write};

use tensor_wasm_artifacts::{
    ArtifactError, ArtifactStore, DiskArtifactStore, InMemoryArtifactStore, ARTIFACT_HEADER_LEN,
    ARTIFACT_HMAC_LEN,
};

/// Locate the single `.bin` blob file the store wrote under `dir`.
fn sole_blob_file(dir: &std::path::Path) -> std::path::PathBuf {
    let mut hits = Vec::new();
    for entry in std::fs::read_dir(dir).expect("read_dir").flatten() {
        let name = entry.file_name().into_string().unwrap_or_default();
        if name.ends_with(".bin") {
            hits.push(entry.path());
        }
    }
    assert_eq!(hits.len(), 1, "expected one blob, found {hits:?}");
    hits.into_iter().next().unwrap()
}

#[test]
fn disk_get_to_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x12; 32]);

    // A payload big enough that the body spans several stream buffers.
    let payload: Vec<u8> = (0..200_000u32).map(|n| (n % 251) as u8).collect();
    let hash = store.put(&payload).expect("put");

    let mut sink: Vec<u8> = Vec::new();
    let written = store.get_to(&hash, &mut sink).expect("get_to");
    assert_eq!(written, payload.len() as u64, "byte count matches");
    assert_eq!(sink, payload, "streamed bytes match the payload");

    // `get_to` and `get` agree.
    assert_eq!(store.get(&hash).expect("get"), sink);
}

#[test]
fn in_memory_get_to_round_trip() {
    let store = InMemoryArtifactStore::new([0x34; 32]);
    let payload = b"in-memory streaming payload".to_vec();
    let hash = store.put(&payload).expect("put");

    let mut sink: Vec<u8> = Vec::new();
    let written = store.get_to(&hash, &mut sink).expect("get_to");
    assert_eq!(written, payload.len() as u64);
    assert_eq!(sink, payload);
}

#[test]
fn disk_get_to_missing_is_not_found() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x56; 32]);
    let hash = tensor_wasm_artifacts::ContentHash::from_bytes([0x99; 32]);
    let mut sink: Vec<u8> = Vec::new();
    let err = store.get_to(&hash, &mut sink).expect_err("must miss");
    assert!(matches!(err, ArtifactError::NotFound(_)), "got {err:?}");
    assert!(sink.is_empty(), "no bytes written on a miss");
}

/// A writer that records every byte handed to it, so we can prove a
/// tampered blob produces `BadHmac` *without* any decoded byte reaching
/// the sink (the verify pass runs before any decode-to-writer pass).
struct RecordingWriter {
    bytes: Vec<u8>,
}

impl Write for RecordingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn disk_get_to_tamper_emits_no_unverified_bytes() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x7A; 32]);
    let payload = b"the quick brown fox jumps over the lazy dog, repeated enough \
                    times that zstd has a real body to chew through and the HMAC \
                    covers more than one stream buffer of compressed output here";
    let hash = store.put(payload).expect("put");

    // Sanity: clean read streams the payload.
    let mut clean = RecordingWriter { bytes: Vec::new() };
    store.get_to(&hash, &mut clean).expect("clean get_to");
    assert_eq!(clean.bytes, payload);

    // Flip a byte inside the zstd body (past the header, before the tag).
    let path = sole_blob_file(tmp.path());
    let mut bytes = Vec::new();
    std::fs::File::open(&path)
        .expect("open")
        .read_to_end(&mut bytes)
        .expect("read");
    assert!(bytes.len() > ARTIFACT_HEADER_LEN + ARTIFACT_HMAC_LEN);
    bytes[ARTIFACT_HEADER_LEN + 2] ^= 0xFF;
    std::fs::write(&path, &bytes).expect("write tampered");

    // The verify pass authenticates the whole blob BEFORE the decode pass
    // streams anything, so a tampered body must surface as BadHmac with an
    // empty sink — no unverified bytes leaked to the writer.
    let mut sink = RecordingWriter { bytes: Vec::new() };
    let err = store.get_to(&hash, &mut sink).expect_err("must reject tamper");
    assert!(matches!(err, ArtifactError::BadHmac), "got {err:?}");
    assert!(
        sink.bytes.is_empty(),
        "no unverified bytes may reach the writer on tamper; got {} bytes",
        sink.bytes.len()
    );
}

#[test]
fn disk_get_to_tampered_tag_rejected_with_empty_sink() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let store = DiskArtifactStore::new(tmp.path().to_path_buf(), [0x9C; 32]);
    let hash = store.put(b"sign me").expect("put");

    let path = sole_blob_file(tmp.path());
    let mut bytes = std::fs::read(&path).expect("read");
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01; // corrupt the trailing HMAC tag
    std::fs::write(&path, &bytes).expect("write");

    let mut sink = RecordingWriter { bytes: Vec::new() };
    let err = store.get_to(&hash, &mut sink).expect_err("must reject");
    assert!(matches!(err, ArtifactError::BadHmac), "got {err:?}");
    assert!(sink.bytes.is_empty(), "no bytes on tampered-tag reject");
}
