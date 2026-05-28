// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! OpenAPI YAML <-> JSON drift regression.
//!
//! `openapi/tensor-wasm-api.yaml` is the canonical source of truth for
//! the tensor-wasm-api HTTP contract; `crates/tensor-wasm-api/openapi.json`
//! is a regenerated companion kept around for tooling that does not parse
//! YAML (some typed-client generators, operator-facing UIs). The
//! `openapi_validation_test` integration test already pins the YAML
//! against the live router, but nothing pinned the JSON sibling against
//! the YAML -- the two had drifted into different `paths` sets by the
//! time the v0.3.7 audit ran (see continue_prompt.md s4.6).
//!
//! This test loads both files, parses each into a `serde_json::Value`
//! (the YAML via a tiny hand-rolled `paths:` scanner so we don't pull
//! `serde_yaml` into the workspace just for this test), and asserts the
//! set of `paths:` keys is identical. Schema-level deep equality would
//! be ideal but is out of scope -- two equivalent OpenAPI documents can
//! differ in incidental formatting (key ordering, scalar block-vs-flow
//! style, trailing newlines in `description` strings) without being
//! semantically distinct, and a structural diff would either need a
//! YAML parser or a hand-rolled normaliser larger than the value it
//! delivers. Path-key parity is the high-signal property: every route
//! the public contract describes must appear in both files, and any
//! addition / removal goes through the regen script.
//!
//! If this test fails, do NOT hand-edit `openapi.json`. Edit the YAML
//! and re-run `scripts/regen-openapi-json.sh` (or the .ps1 sibling)
//! to refresh the JSON from the YAML.

use std::collections::BTreeSet;
use std::path::PathBuf;

use serde_json::Value;

/// Workspace root resolved from the api crate's `CARGO_MANIFEST_DIR`.
/// The YAML lives at `<root>/openapi/tensor-wasm-api.yaml`; the JSON
/// lives at `<root>/crates/tensor-wasm-api/openapi.json`.
fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(PathBuf::from)
        .expect("workspace root resolves from CARGO_MANIFEST_DIR")
}

fn yaml_path() -> PathBuf {
    workspace_root().join("openapi/tensor-wasm-api.yaml")
}

fn json_path() -> PathBuf {
    workspace_root().join("crates/tensor-wasm-api/openapi.json")
}

/// Walk the YAML once and pull every top-level path key (the keys at
/// indent level 2 directly under the `paths:` block) into a sorted set.
/// Mirrors the scanner in `openapi_validation_test.rs` -- duplicated
/// here so this drift test stands alone (the validation test asserts
/// YAML <-> router parity; this one asserts YAML <-> JSON parity, a
/// different axis of comparison).
fn parse_yaml_path_keys(yaml: &str) -> BTreeSet<String> {
    let mut paths = BTreeSet::new();
    let mut in_paths = false;
    for raw_line in yaml.lines() {
        let line = strip_trailing_comment(raw_line);
        if line.trim().is_empty() {
            continue;
        }
        if !line.starts_with(' ') {
            in_paths = line == "paths:";
            continue;
        }
        if !in_paths {
            continue;
        }
        if let Some(rest) = line.strip_prefix("  ") {
            if !rest.starts_with(' ') && rest.starts_with('/') && rest.ends_with(':') {
                let key = rest.trim_end_matches(':').trim().to_string();
                paths.insert(key);
            }
        }
    }
    paths
}

/// Drop a trailing `# comment` from a YAML line, taking care not to
/// trim inside quoted strings.
fn strip_trailing_comment(line: &str) -> &str {
    if line.contains('"') || line.contains('\'') {
        return line.trim_end();
    }
    match line.find('#') {
        Some(idx) => line[..idx].trim_end(),
        None => line.trim_end(),
    }
}

fn read_yaml() -> String {
    let p = yaml_path();
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read openapi YAML at {p:?}: {e}"))
}

fn read_json() -> Value {
    let p = json_path();
    let raw = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("read openapi JSON at {p:?}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse openapi JSON at {p:?}: {e}"))
}

#[test]
fn json_carries_do_not_edit_banner() {
    // The JSON copy is regenerated from the YAML; hand-edits get
    // silently overwritten and shouldn't be made. The regen script
    // injects a `_comment` key at the root warning editors off; if
    // that key is missing, someone has either bypassed the script or
    // started authoring the JSON directly. Fail loud.
    let json = read_json();
    let comment = json
        .get("_comment")
        .and_then(Value::as_str)
        .unwrap_or_else(|| {
            panic!(
                "openapi.json is missing the `_comment` regen banner. \
                 Re-run scripts/regen-openapi-json.sh (or .ps1) to refresh \
                 the JSON from openapi/tensor-wasm-api.yaml."
            )
        });
    assert!(
        comment.contains("Generated from openapi/tensor-wasm-api.yaml")
            && comment.contains("Do NOT edit by hand"),
        "openapi.json `_comment` does not look like the regen banner: {comment:?}",
    );
}

#[test]
fn json_paths_match_yaml_paths() {
    // The high-signal drift check: every path documented in the YAML
    // must appear in the JSON sibling and vice versa. A divergence
    // here means either (a) the YAML was updated without re-running
    // the regen script, or (b) the JSON was hand-edited. Either way
    // run scripts/regen-openapi-json.{sh,ps1} to resolve.
    let yaml = read_yaml();
    let yaml_paths = parse_yaml_path_keys(&yaml);
    assert!(
        !yaml_paths.is_empty(),
        "YAML scanner returned zero paths -- spec file shape changed? \
         see openapi/tensor-wasm-api.yaml header for the assumed layout.",
    );

    let json = read_json();
    let json_paths_obj = json
        .get("paths")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("openapi.json is missing the `paths` object"));
    let json_paths: BTreeSet<String> = json_paths_obj.keys().cloned().collect();

    assert_eq!(
        json_paths, yaml_paths,
        "openapi.json `paths` keys disagree with openapi/tensor-wasm-api.yaml. \
         The YAML is authoritative -- run scripts/regen-openapi-json.sh \
         (or scripts/regen-openapi-json.ps1 on Windows) to refresh the JSON.",
    );
}

#[test]
fn json_info_title_and_version_match_yaml() {
    // Title and version are part of the contract that downstream
    // tooling reads. Even if `paths:` parity passes, a stale
    // `info.version` would mislead consumers. Pin both.
    let yaml = read_yaml();
    let yaml_title = scan_info_scalar(&yaml, "title")
        .unwrap_or_else(|| panic!("YAML missing info.title"));
    let yaml_version = scan_info_scalar(&yaml, "version")
        .unwrap_or_else(|| panic!("YAML missing info.version"));

    let json = read_json();
    let info = json
        .get("info")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("openapi.json is missing `info` object"));
    let json_title = info
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("openapi.json info.title is not a string"));
    let json_version = info
        .get("version")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("openapi.json info.version is not a string"));

    assert_eq!(
        json_title, yaml_title,
        "openapi.json info.title disagrees with the YAML; run scripts/regen-openapi-json.{{sh,ps1}}",
    );
    assert_eq!(
        json_version, yaml_version,
        "openapi.json info.version disagrees with the YAML; run scripts/regen-openapi-json.{{sh,ps1}}",
    );
}

/// Pull a single-line scalar value out of the YAML `info:` block.
/// Looks for `  key: value` at indent 2 directly under a top-level
/// `info:` key. Returns `None` if the key isn't present or its value
/// is multi-line (block scalar) -- the api YAML keeps title/version
/// as single-line scalars, so this is sufficient here.
fn scan_info_scalar(yaml: &str, key: &str) -> Option<String> {
    let mut in_info = false;
    for raw_line in yaml.lines() {
        let line = strip_trailing_comment(raw_line);
        if line.trim().is_empty() {
            continue;
        }
        if !line.starts_with(' ') {
            in_info = line == "info:";
            continue;
        }
        if !in_info {
            continue;
        }
        if let Some(rest) = line.strip_prefix("  ") {
            if !rest.starts_with(' ') {
                let needle = format!("{key}:");
                if let Some(value) = rest.strip_prefix(&needle) {
                    let trimmed = value.trim().trim_matches(|c| c == '"' || c == '\'');
                    if trimmed.is_empty() {
                        // Block scalar follow-on -- not handled here.
                        return None;
                    }
                    return Some(trimmed.to_string());
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Scanner self-tests
// ---------------------------------------------------------------------------

#[test]
fn yaml_path_scanner_extracts_keys() {
    let yaml = "\
paths:
  /a:
    get:
      summary: x
  /b/{id}:
    post:
      summary: y
components:
  schemas:
    Foo:
      type: object
";
    let paths = parse_yaml_path_keys(yaml);
    let expected: BTreeSet<String> = ["/a", "/b/{id}"].iter().map(|s| s.to_string()).collect();
    assert_eq!(paths, expected);
}

#[test]
fn yaml_info_scanner_extracts_scalars() {
    let yaml = "\
openapi: 3.1.0
info:
  title: My API
  version: 1.2.3
paths:
  /a:
    get: {}
";
    assert_eq!(scan_info_scalar(yaml, "title"), Some("My API".to_string()));
    assert_eq!(scan_info_scalar(yaml, "version"), Some("1.2.3".to_string()));
    assert_eq!(scan_info_scalar(yaml, "missing"), None);
}
