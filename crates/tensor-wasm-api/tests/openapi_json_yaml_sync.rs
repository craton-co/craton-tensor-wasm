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

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde_json::Value;

/// HTTP methods an OpenAPI Operation Object can sit under, per the spec.
/// We scan for these (and only these) at the per-path indent level so a
/// sibling key like `parameters:` or `summary:` on a Path Item Object is
/// never mistaken for an operation.
const HTTP_METHODS: &[&str] = &[
    "get", "put", "post", "delete", "options", "head", "patch", "trace",
];

/// Normalised form of an operation's `security` value, comparable across
/// the YAML and JSON specs regardless of incidental formatting.
///
/// The empty-vs-absent distinction is load-bearing in OpenAPI and is
/// preserved here:
///
/// * `Absent`        -- no operation-level `security` key. The operation
///   inherits the root `security` (here `[BearerAuth]`).
/// * `Empty`         -- `security: []`. Explicitly clears auth for this
///   operation ("no auth"), overriding the root.
/// * `Schemes(set)`  -- a non-empty list; each requirement object's
///   scheme name(s) collected into a sorted set. Two
///   requirement objects `{A:[]}` and `{B:[]}` flatten
///   into `{A, B}`. (The api specs use only `Absent`
///   and `Empty` today; `Schemes` keeps the assertion
///   meaningful if a scheme is ever added to a route.)
///
/// `Absent != Empty` by construction (distinct enum variants), which is
/// exactly the drift the `/metrics` regression exposed: `security: []`
/// present in one file and missing in the other read as "no auth" vs.
/// "inherit bearer auth" -- a real authz divergence, not a formatting
/// nit.
#[derive(Debug, Clone, PartialEq, Eq)]
enum SecuritySpec {
    Absent,
    Empty,
    Schemes(BTreeSet<String>),
}

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

/// Walk the YAML `paths:` block and build a map keyed by `"METHOD PATH"`
/// (e.g. `"get /metrics"`) to the operation's normalised [`SecuritySpec`].
///
/// The scanner reuses the same indentation contract the path-key scanner
/// relies on (two-space indent, paths at column 2, operations at column
/// 4, operation fields at column 6; see the YAML file header). It handles
/// the two `security` encodings the api spec actually uses plus the
/// general block-sequence form:
///
/// * `security: []`            -> [`SecuritySpec::Empty`]
/// * (key absent)              -> [`SecuritySpec::Absent`]
/// * a block list of
///   `- SchemeName: [...]`     -> [`SecuritySpec::Schemes`] (scheme names)
///
/// Operation keys are normalised to lowercase so `"get"` matches the JSON
/// object key regardless of YAML casing.
fn parse_yaml_operation_security(yaml: &str) -> BTreeMap<String, SecuritySpec> {
    let mut out: BTreeMap<String, SecuritySpec> = BTreeMap::new();
    let mut in_paths = false;
    let mut cur_path: Option<String> = None;
    let mut cur_method: Option<String> = None;
    // When we step into an operation-level `security:` block sequence we
    // accumulate scheme names here until the next sibling/dedent key.
    let mut in_security_block = false;
    let mut security_schemes: BTreeSet<String> = BTreeSet::new();

    // Flush an in-progress block-sequence `security:` into the map.
    let flush_block = |out: &mut BTreeMap<String, SecuritySpec>,
                       path: &Option<String>,
                       method: &Option<String>,
                       schemes: &mut BTreeSet<String>| {
        if let (Some(p), Some(m)) = (path, method) {
            let key = format!("{m} {p}");
            let spec = if schemes.is_empty() {
                // `security:` with no list items underneath -- treat
                // as an empty requirement set ("no auth"), same as
                // the inline `[]` form.
                SecuritySpec::Empty
            } else {
                SecuritySpec::Schemes(std::mem::take(schemes))
            };
            out.insert(key, spec);
        }
        schemes.clear();
    };

    for raw_line in yaml.lines() {
        let line = strip_trailing_comment(raw_line);
        if line.trim().is_empty() {
            continue;
        }

        // Indentation of this (non-blank) line.
        let indent = line.len() - line.trim_start().len();

        // Top-level key: enter/leave the paths block.
        if indent == 0 {
            if in_security_block {
                flush_block(&mut out, &cur_path, &cur_method, &mut security_schemes);
                in_security_block = false;
            }
            in_paths = line == "paths:";
            cur_path = None;
            cur_method = None;
            continue;
        }

        if !in_paths {
            continue;
        }

        // While inside a `security:` block sequence, consume list items
        // (`- SchemeName: []`) at deeper indent than the `security:` key
        // (which sits at indent 6, so its items sit at indent 8).
        if in_security_block {
            let trimmed = line.trim_start();
            if indent >= 8 && trimmed.starts_with('-') {
                // `- BearerAuth: []` -> scheme name `BearerAuth`.
                let item = trimmed[1..].trim();
                let name = item.split(':').next().unwrap_or("").trim();
                if !name.is_empty() {
                    security_schemes.insert(name.to_string());
                }
                continue;
            }
            // Any line at indent <= 6 ends the block sequence; fall
            // through to re-process it as a normal key below.
            flush_block(&mut out, &cur_path, &cur_method, &mut security_schemes);
            in_security_block = false;
        }

        // Path key: `  /foo:` at indent 2.
        if indent == 2 {
            let rest = line.trim();
            if rest.starts_with('/') && rest.ends_with(':') {
                cur_path = Some(rest.trim_end_matches(':').trim().to_string());
                cur_method = None;
            }
            continue;
        }

        // Operation key: `    get:` at indent 4.
        if indent == 4 {
            let rest = line.trim();
            let name = rest.trim_end_matches(':').trim().to_lowercase();
            if rest.ends_with(':') && HTTP_METHODS.contains(&name.as_str()) {
                cur_method = Some(name);
                // Register the operation up-front with a default of `Absent`
                // so the YAML map's key set matches the JSON map's, which
                // records every operation (defaulting missing `security` to
                // `Absent`; see `json_operation_security`). An operation-level
                // `security:` line at indent 6 overrides this default below.
                // Without this, operations that inherit the root `security`
                // (no explicit key) would be silently dropped from the YAML
                // side, so `assert_eq!(json_ops, yaml_ops)` could never hold
                // once any operation relies on root inheritance.
                if let (Some(p), Some(m)) = (&cur_path, &cur_method) {
                    out.entry(format!("{m} {p}"))
                        .or_insert(SecuritySpec::Absent);
                }
            } else {
                // Non-operation Path Item field (e.g. `parameters:`,
                // `summary:`); leave cur_method untouched but don't treat
                // its children as operation fields.
                cur_method = None;
            }
            continue;
        }

        // Operation field: `      security: ...` at indent 6.
        if indent == 6 {
            let rest = line.trim();
            if let Some(after) = rest.strip_prefix("security:") {
                let val = after.trim();
                let key = match (&cur_path, &cur_method) {
                    (Some(p), Some(m)) => format!("{m} {p}"),
                    _ => continue,
                };
                if val == "[]" {
                    out.insert(key, SecuritySpec::Empty);
                } else if val.is_empty() {
                    // Block sequence follows on subsequent indented lines.
                    in_security_block = true;
                    security_schemes.clear();
                } else {
                    // Inline non-empty flow sequence, e.g.
                    // `security: [{BearerAuth: []}]`. Pull bare word
                    // tokens that look like scheme names.
                    let mut set = BTreeSet::new();
                    for tok in val
                        .trim_matches(|c| c == '[' || c == ']' || c == '{' || c == '}')
                        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
                    {
                        let t = tok.trim();
                        if !t.is_empty() {
                            set.insert(t.to_string());
                        }
                    }
                    out.insert(
                        key,
                        if set.is_empty() {
                            SecuritySpec::Empty
                        } else {
                            SecuritySpec::Schemes(set)
                        },
                    );
                }
            }
            continue;
        }
    }

    // Trailing block sequence at EOF.
    if in_security_block {
        flush_block(&mut out, &cur_path, &cur_method, &mut security_schemes);
    }

    out
}

/// Normalise a JSON operation object's `security` member into a
/// [`SecuritySpec`], mirroring [`parse_yaml_operation_security`].
fn json_operation_security(op: &Value) -> SecuritySpec {
    match op.get("security") {
        None => SecuritySpec::Absent,
        Some(Value::Array(arr)) if arr.is_empty() => SecuritySpec::Empty,
        Some(Value::Array(arr)) => {
            // Each element is a Security Requirement Object: a map of
            // scheme-name -> scopes. Flatten the scheme names into a set.
            let mut set = BTreeSet::new();
            for req in arr {
                if let Some(obj) = req.as_object() {
                    for name in obj.keys() {
                        set.insert(name.clone());
                    }
                }
            }
            if set.is_empty() {
                SecuritySpec::Empty
            } else {
                SecuritySpec::Schemes(set)
            }
        }
        // `security` present but not an array -- malformed; surface it as
        // a distinct value so the assertion fails loudly rather than
        // silently coercing to Absent.
        Some(_) => {
            SecuritySpec::Schemes(["<non-array security>".to_string()].into_iter().collect())
        }
    }
}

/// Build the same `"METHOD PATH" -> SecuritySpec` map from the parsed
/// JSON spec.
fn json_operation_security_map(json: &Value) -> BTreeMap<String, SecuritySpec> {
    let mut out = BTreeMap::new();
    let Some(paths) = json.get("paths").and_then(Value::as_object) else {
        return out;
    };
    for (path, item) in paths {
        let Some(item_obj) = item.as_object() else {
            continue;
        };
        for (method, op) in item_obj {
            let m = method.to_lowercase();
            if !HTTP_METHODS.contains(&m.as_str()) {
                continue;
            }
            out.insert(format!("{m} {path}"), json_operation_security(op));
        }
    }
    out
}

/// Walk the YAML `paths:` block and map `"METHOD PATH"` to the
/// operation's `operationId` scalar (indent-6 field, same contract as the
/// security scanner). Operations without an `operationId` are omitted.
fn parse_yaml_operation_ids(yaml: &str) -> BTreeMap<String, String> {
    let mut out: BTreeMap<String, String> = BTreeMap::new();
    let mut in_paths = false;
    let mut cur_path: Option<String> = None;
    let mut cur_method: Option<String> = None;

    for raw_line in yaml.lines() {
        let line = strip_trailing_comment(raw_line);
        if line.trim().is_empty() {
            continue;
        }
        let indent = line.len() - line.trim_start().len();

        if indent == 0 {
            in_paths = line == "paths:";
            cur_path = None;
            cur_method = None;
            continue;
        }
        if !in_paths {
            continue;
        }

        if indent == 2 {
            let rest = line.trim();
            if rest.starts_with('/') && rest.ends_with(':') {
                cur_path = Some(rest.trim_end_matches(':').trim().to_string());
                cur_method = None;
            }
            continue;
        }
        if indent == 4 {
            let rest = line.trim();
            let name = rest.trim_end_matches(':').trim().to_lowercase();
            if rest.ends_with(':') && HTTP_METHODS.contains(&name.as_str()) {
                cur_method = Some(name);
            } else {
                cur_method = None;
            }
            continue;
        }
        if indent == 6 {
            let rest = line.trim();
            if let Some(after) = rest.strip_prefix("operationId:") {
                if let (Some(p), Some(m)) = (&cur_path, &cur_method) {
                    let id = after.trim().trim_matches(|c| c == '"' || c == '\'');
                    if !id.is_empty() {
                        out.insert(format!("{m} {p}"), id.to_string());
                    }
                }
            }
            continue;
        }
    }
    out
}

/// Build the `"METHOD PATH" -> operationId` map from the parsed JSON spec.
fn json_operation_ids_map(json: &Value) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let Some(paths) = json.get("paths").and_then(Value::as_object) else {
        return out;
    };
    for (path, item) in paths {
        let Some(item_obj) = item.as_object() else {
            continue;
        };
        for (method, op) in item_obj {
            let m = method.to_lowercase();
            if !HTTP_METHODS.contains(&m.as_str()) {
                continue;
            }
            if let Some(id) = op.get("operationId").and_then(Value::as_str) {
                out.insert(format!("{m} {path}"), id.to_string());
            }
        }
    }
    out
}

fn read_yaml() -> String {
    let p = yaml_path();
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read openapi YAML at {p:?}: {e}"))
}

fn read_json() -> Value {
    let p = json_path();
    let raw =
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read openapi JSON at {p:?}: {e}"));
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
fn json_operation_security_matches_yaml() {
    // Per-operation `security` parity. Path-key parity (above) does NOT
    // catch a `security` block that is present in one file and missing in
    // the other -- exactly the `/metrics` drift that slipped through
    // until the v0.3.7 fix (security: [] in the YAML, absent in the JSON,
    // or vice versa). That divergence is a real authz difference:
    //
    //   * `security: []` -> "no auth" (open endpoint)
    //   * (absent)       -> inherit root `security: [BearerAuth]`
    //
    // so the two are NOT interchangeable. We compare a normalised
    // [`SecuritySpec`] per `METHOD PATH` and require both specs to agree.
    let yaml = read_yaml();
    let yaml_sec = parse_yaml_operation_security(&yaml);
    assert!(
        !yaml_sec.is_empty(),
        "YAML security scanner returned zero operations -- spec file shape \
         changed? see openapi/tensor-wasm-api.yaml header for the assumed \
         indentation layout.",
    );

    let json = read_json();
    let json_sec = json_operation_security_map(&json);

    // Operation sets must match first, otherwise a missing operation
    // would masquerade as `Absent` on one side. (Path-key parity is
    // checked separately, but `paths` parity does not imply per-method
    // parity -- e.g. /kernels has both get and post.)
    let yaml_ops: BTreeSet<&String> = yaml_sec.keys().collect();
    let json_ops: BTreeSet<&String> = json_sec.keys().collect();
    assert_eq!(
        json_ops, yaml_ops,
        "openapi.json and openapi/tensor-wasm-api.yaml describe different \
         (method, path) operation sets. The YAML is authoritative -- run \
         scripts/regen-openapi-json.{{sh,ps1}} to refresh the JSON.",
    );

    // Now compare the security spec for every operation.
    let mut mismatches: Vec<String> = Vec::new();
    for (op, yaml_spec) in &yaml_sec {
        let json_spec = json_sec
            .get(op)
            .expect("operation sets already asserted equal");
        if yaml_spec != json_spec {
            mismatches.push(format!("  {op}: yaml={yaml_spec:?} json={json_spec:?}"));
        }
    }
    assert!(
        mismatches.is_empty(),
        "per-operation `security` drift between openapi/tensor-wasm-api.yaml \
         and openapi.json (security: [] means \"no auth\"; absent means \
         \"inherit global BearerAuth\" -- these are distinct). The YAML is \
         authoritative; run scripts/regen-openapi-json.{{sh,ps1}}:\n{}",
        mismatches.join("\n"),
    );
}

#[test]
fn json_operation_ids_match_yaml() {
    // Cheap bonus parity: `operationId` is what typed-client generators
    // turn into method names, so a drift here silently renames generated
    // API surface. Compare the `METHOD PATH -> operationId` maps.
    let yaml = read_yaml();
    let yaml_ids = parse_yaml_operation_ids(&yaml);
    assert!(
        !yaml_ids.is_empty(),
        "YAML operationId scanner returned nothing -- spec shape changed?",
    );

    let json = read_json();
    let json_ids = json_operation_ids_map(&json);

    assert_eq!(
        json_ids, yaml_ids,
        "per-operation `operationId` drift between openapi/tensor-wasm-api.yaml \
         and openapi.json. The YAML is authoritative; run \
         scripts/regen-openapi-json.{{sh,ps1}}.",
    );
}

#[test]
fn json_info_title_and_version_match_yaml() {
    // Title and version are part of the contract that downstream
    // tooling reads. Even if `paths:` parity passes, a stale
    // `info.version` would mislead consumers. Pin both.
    let yaml = read_yaml();
    let yaml_title =
        scan_info_scalar(&yaml, "title").unwrap_or_else(|| panic!("YAML missing info.title"));
    let yaml_version =
        scan_info_scalar(&yaml, "version").unwrap_or_else(|| panic!("YAML missing info.version"));

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

#[test]
fn yaml_security_scanner_distinguishes_empty_absent_and_schemes() {
    let yaml = "\
paths:
  /open:
    get:
      summary: x
      security: []
  /inherit:
    get:
      summary: y
  /scoped:
    post:
      summary: z
      security:
        - BearerAuth: []
        - ApiKey: []
components:
  schemas:
    Foo:
      type: object
      security: []
";
    let sec = parse_yaml_operation_security(yaml);
    assert_eq!(sec.get("get /open"), Some(&SecuritySpec::Empty));
    // Inherit -> absent from the map entirely.
    assert_eq!(sec.get("get /inherit"), None);
    let schemes: BTreeSet<String> = ["ApiKey", "BearerAuth"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        sec.get("post /scoped"),
        Some(&SecuritySpec::Schemes(schemes))
    );
    // The `security: []` buried under components.schemas.Foo must NOT
    // leak in -- the scanner only walks the paths block.
    assert_eq!(sec.len(), 2);
}

#[test]
fn json_security_normaliser_distinguishes_empty_absent_and_schemes() {
    let absent: Value = serde_json::json!({ "summary": "x" });
    assert_eq!(json_operation_security(&absent), SecuritySpec::Absent);

    let empty: Value = serde_json::json!({ "security": [] });
    assert_eq!(json_operation_security(&empty), SecuritySpec::Empty);

    let scoped: Value = serde_json::json!({
        "security": [{ "BearerAuth": [] }, { "ApiKey": [] }]
    });
    let schemes: BTreeSet<String> = ["ApiKey", "BearerAuth"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        json_operation_security(&scoped),
        SecuritySpec::Schemes(schemes)
    );
}

#[test]
fn yaml_operation_id_scanner_extracts_ids() {
    let yaml = "\
paths:
  /a:
    get:
      operationId: getA
      summary: x
  /b:
    post:
      summary: y
";
    let ids = parse_yaml_operation_ids(yaml);
    assert_eq!(ids.get("get /a"), Some(&"getA".to_string()));
    // No operationId -> omitted.
    assert_eq!(ids.get("post /b"), None);
    assert_eq!(ids.len(), 1);
}
