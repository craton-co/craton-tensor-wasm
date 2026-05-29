// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Registration-drift guard for the bench crate.
//!
//! Every `benches/*.rs` file in this crate is a `harness = false`
//! Criterion bench and must be declared as a `[[bench]]` in
//! `crates/tensor-wasm-bench/Cargo.toml`. If a bench file is added to
//! `benches/` but never registered, `cargo bench` silently auto-discovers
//! it with the default libtest harness — which clashes with the file's
//! `criterion_main!` and fails to compile, or worse, runs a bench under the
//! wrong harness and emits no Criterion output. The earlier
//! `call_export_args` / `streaming_invoke` drift (both added to `benches/`
//! before their `[[bench]]` stanzas) is exactly the failure mode this test
//! pins.
//!
//! The test reads both the `benches/` directory listing and `Cargo.toml`
//! at run time via [`std::fs`] + `CARGO_MANIFEST_DIR` (no build-time
//! codegen, no extra deps), parses the `[[bench]] name = "..."` entries and
//! the `benches/*.rs` file stems, and asserts the two sets match exactly.
//! A `mod.rs` / `common.rs` shared-helper file (if one is ever added under
//! `benches/`) is ignored by convention — those are includable modules, not
//! bench targets, and Cargo does not auto-discover them.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// Bench-file stems that are shared helper modules rather than bench
/// targets. Cargo does not auto-discover these, so they need no
/// `[[bench]]` registration. Kept as a tiny allow-list so the convention
/// is explicit rather than implied by a naming-prefix regex.
const HELPER_STEMS: &[&str] = &["mod", "common"];

/// Collect the file stems of every `benches/*.rs` bench target, skipping
/// the documented helper-module names.
fn bench_files(benches_dir: &Path) -> BTreeSet<String> {
    let mut stems = BTreeSet::new();
    let entries = fs::read_dir(benches_dir).unwrap_or_else(|e| {
        panic!(
            "failed to read benches dir {}: {e}",
            benches_dir.display()
        )
    });
    for entry in entries {
        let entry = entry.expect("readable benches/ dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .expect("bench file has a UTF-8 stem")
            .to_string();
        if HELPER_STEMS.contains(&stem.as_str()) {
            continue;
        }
        stems.insert(stem);
    }
    stems
}

/// Parse the `name = "..."` value out of every `[[bench]]` table in
/// `Cargo.toml`. Deliberately a hand-rolled line scanner rather than a TOML
/// dep: the bench crate's manifest is hand-edited and this test must run
/// without pulling a `toml` crate into the workspace. It tolerates the
/// inline comments that decorate several `[[bench]]` stanzas (lines like
/// `# F3 — ...` between the `[[bench]]` header and its `name`).
fn registered_benches(cargo_toml: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_bench_table = false;
    for raw in cargo_toml.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            // A new table header. We're inside a bench stanza only while the
            // current table is exactly `[[bench]]`.
            in_bench_table = line == "[[bench]]";
            continue;
        }
        if !in_bench_table {
            continue;
        }
        // Within a `[[bench]]` table, capture the first `name = "..."`.
        if let Some(rest) = line.strip_prefix("name") {
            let rest = rest.trim_start();
            if let Some(value) = rest.strip_prefix('=') {
                let value = value.trim();
                let name = value.trim_matches('"');
                names.insert(name.to_string());
                // Stay in the table; a `[[bench]]` has a single name.
            }
        }
    }
    names
}

#[test]
fn every_bench_file_is_registered() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let benches_dir = Path::new(manifest_dir).join("benches");
    let cargo_toml_path = Path::new(manifest_dir).join("Cargo.toml");

    let cargo_toml = fs::read_to_string(&cargo_toml_path).unwrap_or_else(|e| {
        panic!("failed to read {}: {e}", cargo_toml_path.display())
    });

    let files = bench_files(&benches_dir);
    let registered = registered_benches(&cargo_toml);

    assert!(
        !files.is_empty(),
        "no bench files discovered under {} — the test would vacuously \
         pass; check the path",
        benches_dir.display()
    );

    let unregistered: Vec<&String> = files.difference(&registered).collect();
    let orphaned: Vec<&String> = registered.difference(&files).collect();

    assert!(
        unregistered.is_empty(),
        "bench file(s) in benches/ have no matching [[bench]] in \
         Cargo.toml: {unregistered:?}. Add a `[[bench]] name = \"<stem>\" \
         harness = false` stanza (this is the call_export_args / \
         streaming_invoke drift this test guards against).",
    );

    assert!(
        orphaned.is_empty(),
        "[[bench]] name(s) in Cargo.toml have no matching benches/<name>.rs \
         file: {orphaned:?}. Remove the stale registration or restore the \
         file.",
    );
}
