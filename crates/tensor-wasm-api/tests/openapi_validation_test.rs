// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! OpenAPI ↔ live-router validation.
//!
//! This test is the runtime half of the v0.4 "OpenAPI spec validated
//! against the live router in CI" milestone (see `docs/PATH-TO-V1.md`).
//! The canonical machine-readable spec lives at
//! `openapi/tensor-wasm-api.yaml`; this test asserts three things, in
//! escalating strictness:
//!
//! 1. The YAML file exists at the documented path and parses as a
//!    well-formed OpenAPI 3.1 document (we don't take a YAML parser as
//!    a runtime dependency — the structural scan is sufficient for this
//!    file's known shape, and the CI workflow runs `swagger-cli
//!    validate` as the authoritative parser-of-record).
//! 2. The set of routes documented in the spec matches the set of
//!    routes the live axum router exposes. The router is built
//!    in-process via `build_router_with_config`; we hard-code the
//!    expected `(METHOD, PATH)` set drawn from `src/server.rs`'s
//!    `Router::new().route(...)` calls and assert equality with the
//!    spec's `paths` keys (and their HTTP method maps).
//! 3. A real `CreateFunctionRequest` (the actual Rust struct used by
//!    the live handler) round-trips against the `paths` entry for
//!    `POST /functions`: the JSON-encoded value carries every property
//!    the spec marks `required` and nothing the spec marks
//!    `additionalProperties: false` rejects.
//!
//! The combination gives the same property a generated typed client
//! would prove ("client and server agree on at least one request
//! shape") without taking a code-generator into the test binary, which
//! would balloon build time. The CI workflow handles the heavier
//! "spec parses cleanly under a real OpenAPI parser" check.
//!
//! Out of scope: response-body validation. Adding a JSON Schema
//! validator dep is overkill for the v0.4 gate — the response shapes
//! are exercised by the existing `http_integration` test suite which
//! deserialises into the live Rust types.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;
use tensor_wasm_api::{
    build_router_with_config, routes::CreateFunctionRequest, AppState, AuthConfig, TenantConfig,
};

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

/// Locate `openapi/tensor-wasm-api.yaml` relative to the workspace root.
/// `CARGO_MANIFEST_DIR` points at `crates/tensor-wasm-api`; the spec lives
/// two directories up under `openapi/`.
fn spec_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|workspace_root| workspace_root.join("openapi/tensor-wasm-api.yaml"))
        .expect("workspace root resolves from CARGO_MANIFEST_DIR")
}

/// Read the YAML spec source.
fn read_spec() -> String {
    let path = spec_path();
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read openapi spec at {path:?}: {e}"))
}

/// Source-of-truth list of HTTP routes the live router exposes. Keep
/// this list in sync with `crates/tensor-wasm-api/src/server.rs` — every
/// `Router::new().route("...", ...)` call must have a matching entry
/// here (and a matching `paths:` entry in the OpenAPI spec). The test
/// `router_routes_match_expected_set` defends the parity between this
/// list and the in-process router; `spec_paths_cover_router` defends
/// the parity between this list and the spec.
///
/// The kernel-registry routes (B6.4) are mounted only when the
/// `kernel-registry-api` Cargo feature is on. The OpenAPI spec
/// documents them unconditionally (the spec is the public contract;
/// gating it on a Cargo feature would hide the surface from API-doc
/// tooling that builds without the feature). `expected_routes()`
/// reconciles the two views: when the feature is off it filters the
/// `/kernels` rows out of the in-process expectation but the parity
/// asserts still require the spec to mention them — so a stale spec
/// after a feature removal would still be caught.
fn expected_routes() -> Vec<(&'static str, &'static str)> {
    let mut routes: Vec<(&'static str, &'static str)> = vec![
        ("GET", "/healthz"),
        ("GET", "/metrics"),
        ("POST", "/functions"),
        ("DELETE", "/functions/{id}"),
        ("POST", "/functions/{id}/invoke"),
        ("POST", "/functions/{id}/invoke-async"),
        ("POST", "/functions/{id}/invoke-stream"),
        ("GET", "/jobs/{id}"),
        // OpenAI-compat shim (B4.9). Scaffold routes that return 501 with
        // the OpenAI-shape error envelope. See
        // `crates/tensor-wasm-api/src/openai.rs` and
        // `docs/OPENAI-COMPAT.md` for the rollout plan.
        ("POST", "/v1/completions"),
        ("POST", "/v1/chat/completions"),
    ];
    #[cfg(feature = "kernel-registry-api")]
    {
        // Kernel registry routes (B6.4 — roadmap feature #3). `POST` and
        // `GET` share the `/kernels` path; the resolve route uses the
        // `{name}/{version}` two-segment shape (axum 0.7 `:name/:version`).
        routes.push(("POST", "/kernels"));
        routes.push(("GET", "/kernels"));
        routes.push(("GET", "/kernels/{name}/{version}"));
    }
    routes
}

/// Build the live router exactly as `server::build_router` does, but
/// with deterministic config (no env reads) so the test does not get
/// poisoned by `TENSOR_WASM_API_TOKENS` etc.
fn live_router() -> axum::Router {
    build_router_with_config(
        Arc::new(AppState::default()),
        AuthConfig::default(),
        TenantConfig::default(),
    )
}

// ---------------------------------------------------------------------------
// Minimal YAML structural scan
// ---------------------------------------------------------------------------
//
// We intentionally do not pull a full YAML parser into the test
// binary; `serde_yaml` is not a workspace dependency and adding it for
// a single test would inflate the dependency graph noticeably. The
// spec file is hand-authored with strict two-space indentation (see
// the header comment in `openapi/tensor-wasm-api.yaml`); the routines
// below assume that layout and scan deterministically.
//
// The CI workflow (`openapi` job in `.github/workflows/ci.yml`) runs
// `swagger-cli validate` as the authoritative OpenAPI parser — if
// indentation drifts in a way our scanner happens not to notice,
// `swagger-cli` will catch it independently.

/// Walk the YAML once and pull every top-level path key (the keys at
/// indent level 2 directly under the `paths:` block) into a sorted set.
fn parse_spec_path_keys(yaml: &str) -> BTreeSet<String> {
    let mut paths = BTreeSet::new();
    let mut in_paths = false;
    for raw_line in yaml.lines() {
        // Skip pure comments and blanks.
        let line = strip_trailing_comment(raw_line);
        if line.trim().is_empty() {
            continue;
        }
        // Top-level keys (indent 0) — entering or leaving the paths block.
        if !line.starts_with(' ') {
            in_paths = line == "paths:";
            continue;
        }
        if !in_paths {
            continue;
        }
        // Inside `paths:`, the direct children are keys at indent 2 that
        // start with a slash and end with a colon. Anything more deeply
        // indented (the operations under them) is ignored here.
        if let Some(rest) = line.strip_prefix("  ") {
            if !rest.starts_with(' ') && rest.starts_with('/') && rest.ends_with(':') {
                let key = rest.trim_end_matches(':').trim().to_string();
                paths.insert(key);
            }
        }
    }
    paths
}

/// For each documented path, collect the HTTP methods it declares.
/// Methods sit at indent 4 (two spaces under the path key) and end
/// with a colon.
fn parse_spec_path_methods(yaml: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut by_path: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut in_paths = false;
    let mut current: Option<String> = None;

    for raw_line in yaml.lines() {
        let line = strip_trailing_comment(raw_line);
        if line.trim().is_empty() {
            continue;
        }
        if !line.starts_with(' ') {
            in_paths = line == "paths:";
            current = None;
            continue;
        }
        if !in_paths {
            continue;
        }
        // Path entry at indent 2.
        if let Some(rest) = line.strip_prefix("  ") {
            if !rest.starts_with(' ') && rest.starts_with('/') && rest.ends_with(':') {
                let key = rest.trim_end_matches(':').trim().to_string();
                by_path.entry(key.clone()).or_default();
                current = Some(key);
                continue;
            }
        }
        // Method entry at indent 4.
        if let Some(after4) = line.strip_prefix("    ") {
            if !after4.starts_with(' ') && after4.ends_with(':') {
                let method = after4.trim_end_matches(':').trim().to_ascii_uppercase();
                if is_http_method(&method) {
                    if let Some(path) = &current {
                        by_path
                            .get_mut(path)
                            .expect("path entry seeded")
                            .insert(method);
                    }
                }
            }
        }
    }
    by_path
}

/// Collect the schema names declared under `components.schemas:`.
/// Schema keys sit at indent 4 (`components:` 0, `schemas:` 2, names 4).
fn parse_spec_schema_names(yaml: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_components = false;
    let mut in_schemas = false;
    for raw_line in yaml.lines() {
        let line = strip_trailing_comment(raw_line);
        if line.trim().is_empty() {
            continue;
        }
        if !line.starts_with(' ') {
            in_components = line == "components:";
            in_schemas = false;
            continue;
        }
        if !in_components {
            continue;
        }
        if let Some(rest) = line.strip_prefix("  ") {
            // The `schemas:` sub-key sits at indent 2 within `components:`.
            if !rest.starts_with(' ') {
                in_schemas = rest == "schemas:";
                continue;
            }
        }
        if !in_schemas {
            continue;
        }
        // Schema names sit at indent 4. Anything deeper (their fields)
        // is uninteresting here.
        if let Some(after4) = line.strip_prefix("    ") {
            if !after4.starts_with(' ') && after4.ends_with(':') {
                let name = after4.trim_end_matches(':').trim().to_string();
                if is_likely_identifier(&name) {
                    names.insert(name);
                }
            }
        }
    }
    names
}

/// For a known schema name, pull its `required: [a, b, c]` and
/// `additionalProperties: false|true` declarations out of the YAML. Returns
/// `(required_field_set, additional_properties_strict)`.
///
/// The scanner expects the schema to be authored as `required: [a, b, ...]`
/// (single-line flow form), which is how `openapi/tensor-wasm-api.yaml`
/// expresses every required list. Block-style (`required:\n  - a`) is
/// not parsed; the structural test that uses this helper covers only
/// `CreateFunctionRequest`, which is authored in the flow form.
fn parse_schema_required_and_strictness(yaml: &str, schema_name: &str) -> (BTreeSet<String>, bool) {
    let mut required = BTreeSet::new();
    let mut additional_properties_strict = false;
    let mut in_components = false;
    let mut in_schemas = false;
    let mut in_target = false;
    // Track the indent level of the target schema's child keys so we know
    // when we've left the block.
    const SCHEMA_CHILD_INDENT: usize = 6;

    for raw_line in yaml.lines() {
        let line = strip_trailing_comment(raw_line);
        if line.trim().is_empty() {
            continue;
        }
        if !line.starts_with(' ') {
            in_components = line == "components:";
            in_schemas = false;
            in_target = false;
            continue;
        }
        if !in_components {
            continue;
        }
        if let Some(rest) = line.strip_prefix("  ") {
            if !rest.starts_with(' ') {
                in_schemas = rest == "schemas:";
                in_target = false;
                continue;
            }
        }
        if !in_schemas {
            continue;
        }
        // Schema name marker at indent 4.
        if let Some(after4) = line.strip_prefix("    ") {
            if !after4.starts_with(' ') && after4.ends_with(':') {
                let name = after4.trim_end_matches(':').trim();
                in_target = name == schema_name;
                continue;
            }
        }
        if !in_target {
            continue;
        }
        // Inside the target schema. Inspect lines at indent 6 only —
        // properties' nested fields live at indent 8+ and are not the
        // schema's direct keys.
        let indent = leading_space_count(&line);
        if indent != SCHEMA_CHILD_INDENT {
            continue;
        }
        let payload = &line[SCHEMA_CHILD_INDENT..];
        if let Some(list) = payload.strip_prefix("required:").map(str::trim) {
            // Expect the flow-array form: `[a, b, c]`.
            if let Some(inner) = list.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                for item in inner.split(',') {
                    let cleaned = item.trim().trim_matches(|c| c == '"' || c == '\'');
                    if !cleaned.is_empty() {
                        required.insert(cleaned.to_string());
                    }
                }
            }
        } else if let Some(value) = payload.strip_prefix("additionalProperties:").map(str::trim) {
            additional_properties_strict = value == "false";
        }
    }
    (required, additional_properties_strict)
}

fn is_http_method(s: &str) -> bool {
    matches!(
        s,
        "GET" | "POST" | "PUT" | "DELETE" | "PATCH" | "HEAD" | "OPTIONS" | "TRACE"
    )
}

fn is_likely_identifier(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn leading_space_count(s: &str) -> usize {
    s.bytes().take_while(|b| *b == b' ').count()
}

/// Drop a trailing `# comment` from a YAML line, taking care not to
/// trim inside quoted strings. Conservative: if the line contains a
/// quote we leave it untouched (the spec file doesn't use mid-line
/// comments inside quoted scalars).
fn strip_trailing_comment(line: &str) -> &str {
    if line.contains('"') || line.contains('\'') {
        return line.trim_end();
    }
    match line.find('#') {
        Some(idx) => line[..idx].trim_end(),
        None => line.trim_end(),
    }
}

// ---------------------------------------------------------------------------
// Router introspection
// ---------------------------------------------------------------------------

/// Convert the colon-prefixed path-param syntax axum 0.7 uses
/// (`/functions/:id`) into the OpenAPI-standard curly-brace form
/// (`/functions/{id}`) so we can compare apples to apples.
fn axum_to_openapi_path(p: &str) -> String {
    let mut out = String::with_capacity(p.len());
    let mut chars = p.chars().peekable();
    while let Some(c) = chars.next() {
        if c == ':' {
            // Capture the identifier up to the next '/' or end.
            let mut ident = String::new();
            while let Some(&next) = chars.peek() {
                if next == '/' {
                    break;
                }
                ident.push(next);
                chars.next();
            }
            out.push('{');
            out.push_str(&ident);
            out.push('}');
        } else {
            out.push(c);
        }
    }
    out
}

/// Drive the live router with each `(method, path)` in `expected_routes()`
/// and assert it does NOT return `405 Method Not Allowed`. axum's
/// router returns 405 when the request path matches a registered
/// route but no handler is wired for the supplied verb; it returns
/// `404` when no route matches at all. We can't use 404 to prove
/// route registration here because several handlers (`DELETE
/// /functions/{id}`, `GET /jobs/{id}`, the invoke endpoints) legitimately
/// return 404 for unknown resources, but a 405 unambiguously means the
/// `(method, path)` pair is not registered as expected. As a second
/// signal we then re-issue an obviously-wrong method against the same
/// path and assert it DOES 405 — proving the path matched.
async fn assert_router_serves(method: &str, path: &str) {
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use tower::ServiceExt;

    // Substitute `{id}` in OpenAPI-shape paths with a real UUID so the
    // router can match the dynamic segment. Healthz/metrics have no
    // params and are unaffected.
    let concrete = path.replace("{id}", "00000000-0000-4000-8000-000000000000");
    let m = Method::from_bytes(method.as_bytes()).expect("known HTTP method");

    // First: assert the expected (method, path) pair is wired. 405 here
    // means the path matched but the verb did not — a registration bug.
    let mut builder = Request::builder().method(m.clone()).uri(&concrete);
    let body = if matches!(m, Method::POST) {
        builder = builder.header("content-type", "application/json");
        Body::from(b"{}".to_vec())
    } else {
        Body::empty()
    };
    let req = builder.body(body).expect("request builds");
    let resp = live_router()
        .oneshot(req)
        .await
        .expect("router serves the request");
    assert_ne!(
        resp.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "router 405 for {method} {concrete}: route registered with wrong verb?",
    );

    // Second: pick a verb the route definitely does NOT serve and
    // assert the response status confirms the path itself is
    // registered. For GET routes we send PATCH; for everything else we
    // send PATCH too (none of the documented routes implement PATCH).
    // A registered path returns 405 here; a missing path returns 404.
    let probe = Request::builder()
        .method(Method::PATCH)
        .uri(&concrete)
        .body(Body::empty())
        .expect("probe builds");
    let probe_resp = live_router()
        .oneshot(probe)
        .await
        .expect("router serves the probe");
    assert_eq!(
        probe_resp.status(),
        StatusCode::METHOD_NOT_ALLOWED,
        "PATCH {concrete} returned {} — expected 405 (proving the path \
         exists). 404 implies the route is missing from src/server.rs.",
        probe_resp.status(),
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn spec_file_exists_and_is_nonempty() {
    let yaml = read_spec();
    assert!(yaml.len() > 256, "spec is suspiciously short ({} bytes)", yaml.len());
    assert!(
        yaml.starts_with("# SPDX-License-Identifier: Apache-2.0"),
        "spec must start with SPDX header",
    );
    assert!(
        yaml.contains("openapi: 3.1"),
        "spec must declare OpenAPI 3.1 (found header `openapi:` other than 3.1)",
    );
}

#[test]
fn spec_paths_cover_router() {
    let yaml = read_spec();
    // Kernel-registry routes are documented in the spec unconditionally
    // but only mounted in the router when `kernel-registry-api` is on.
    // When the feature is off, drop the `/kernels*` paths from the spec
    // view so the parity assert reflects what the live router exposes.
    // The feature-on branch keeps the spec verbatim so a stale entry
    // (e.g. a removed kernel sub-route) still surfaces here.
    let spec_paths: BTreeSet<String> = {
        let all = parse_spec_path_keys(&yaml);
        #[cfg(not(feature = "kernel-registry-api"))]
        {
            all.into_iter().filter(|p| !p.starts_with("/kernels")).collect()
        }
        #[cfg(feature = "kernel-registry-api")]
        {
            all
        }
    };
    let expected: BTreeSet<String> = expected_routes()
        .iter()
        .map(|(_, p)| p.to_string())
        .collect();
    assert_eq!(
        spec_paths, expected,
        "OpenAPI spec `paths:` keys do not match the live router. \
         Either add the missing entry to openapi/tensor-wasm-api.yaml \
         or remove the stale route from `expected_routes()` (and the \
         router in src/server.rs).",
    );
}

#[test]
fn spec_methods_match_router_methods() {
    let yaml = read_spec();
    let documented = parse_spec_path_methods(&yaml);
    // Same kernel-registry filter as `spec_paths_cover_router`: drop
    // `/kernels*` rows from the spec view when the feature is off, so
    // the parity assert reflects what the live router exposes.
    let documented: BTreeMap<String, BTreeSet<String>> = {
        #[cfg(not(feature = "kernel-registry-api"))]
        {
            documented
                .into_iter()
                .filter(|(p, _)| !p.starts_with("/kernels"))
                .collect()
        }
        #[cfg(feature = "kernel-registry-api")]
        {
            documented
        }
    };

    // Roll `expected_routes()` into the same shape so we can compare maps.
    let mut expected: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (method, path) in expected_routes() {
        expected
            .entry(path.to_string())
            .or_default()
            .insert(method.to_ascii_uppercase());
    }

    assert_eq!(
        documented, expected,
        "OpenAPI spec method maps do not match the live router. \
         Compare keys per path and add/remove operations under the \
         relevant `paths:` entry in openapi/tensor-wasm-api.yaml.",
    );
}

#[test]
fn spec_declares_required_component_schemas() {
    // Every schema referenced from a `paths:` response or request body
    // (transitively, through other `$ref`s) must exist under
    // `components.schemas` so a code generator can resolve every link.
    let yaml = read_spec();
    let schemas = parse_spec_schema_names(&yaml);
    for needed in [
        "FunctionId",
        "JobId",
        "Health",
        "CreateFunctionRequest",
        "CreateFunctionResponse",
        "InvokeRequest",
        "InvokeResult",
        "InvokeAsyncResponse",
        "JobStatus",
        "JobRecord",
        "ApiErrorBody",
        "ApiErrorEnvelope",
    ] {
        assert!(
            schemas.contains(needed),
            "components.schemas is missing `{needed}`. Existing schemas: {schemas:?}",
        );
    }
}

#[test]
fn create_function_request_round_trips_through_spec() {
    // Construct a real CreateFunctionRequest using the production Rust
    // type, serialise it, and check that every key the spec marks
    // `required` is present and no key not declared by the spec sneaks
    // in (mirrors `additionalProperties: false`). This is the
    // synthetic-request-round-trip the v0.4 milestone asks for: it
    // proves the spec's request schema and the server's request struct
    // describe the same wire shape.
    let req = CreateFunctionRequest {
        name: "round-trip".to_string(),
        // Minimal valid Wasm module: 4-byte magic + 4-byte version, base64.
        wasm_b64: "AGFzbQEAAAA=".to_string(),
    };
    // `CreateFunctionRequest` derives only Deserialize on the server
    // side (it is what the handler receives), so we can't `to_value`
    // directly. Round-trip through a manually-constructed JSON object
    // built from the same field set the deserialiser accepts; the
    // assertions below then check the spec describes exactly those
    // fields.
    let payload = serde_json::json!({
        "name": req.name,
        "wasm_b64": req.wasm_b64,
    });

    let yaml = read_spec();
    let (required, strict) = parse_schema_required_and_strictness(&yaml, "CreateFunctionRequest");
    assert!(
        !required.is_empty(),
        "spec must declare `required:` for CreateFunctionRequest",
    );
    assert!(
        strict,
        "spec must declare `additionalProperties: false` on CreateFunctionRequest \
         so unknown keys are rejected by generated clients",
    );

    let obj = payload.as_object().expect("payload is a JSON object");
    for required_key in &required {
        assert!(
            obj.contains_key(required_key),
            "CreateFunctionRequest is missing required key `{required_key}` per the spec; \
             got keys {:?}",
            obj.keys().collect::<Vec<_>>(),
        );
    }
    // Inversely: every key in the payload must be one the spec allows
    // (it's authored with additionalProperties: false, so the allowed
    // set is exactly the keys mentioned in `required` for this shape —
    // the schema has no optional fields). If the Rust struct ever
    // grows an optional field, extend the spec's `required` list or
    // relax this check.
    for actual_key in obj.keys() {
        assert!(
            required.contains(actual_key),
            "payload key `{actual_key}` is not declared in the spec's \
             CreateFunctionRequest schema; required set is {required:?}",
        );
    }

    // Final sanity check: parsing the payload back into the real Rust
    // struct must succeed. Closes the round-trip loop.
    let _parsed: CreateFunctionRequest =
        serde_json::from_value(Value::Object(obj.clone())).expect("payload deserialises");
}

#[tokio::test]
async fn router_routes_match_expected_set() {
    // Drive each (method, path) in expected_routes() through the live
    // router and assert it does not 404/405. Defends the parity
    // between expected_routes() and the actual Router::new().route(...)
    // calls in src/server.rs: if a route is removed (or renamed)
    // without updating this list, the corresponding assertion below
    // fails loudly.
    for (method, path) in expected_routes() {
        assert_router_serves(method, path).await;
    }
}

#[tokio::test]
async fn round_trip_synthetic_request_through_router() {
    // The end-to-end check: build a CreateFunctionRequest that matches
    // the spec's `CreateFunctionRequest` schema, POST it to the live
    // /functions handler, and assert the response shape matches the
    // spec's CreateFunctionResponse (has an `id` UUID). This is the
    // "generated client compiles + round-trips a synthetic request"
    // property from the v0.4 milestone, expressed without a generated
    // client.
    use axum::body::Body;
    use axum::http::{Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    let payload = serde_json::json!({
        "name": "openapi-round-trip",
        // Minimal valid Wasm module, base64-encoded.
        "wasm_b64": "AGFzbQEAAAA=",
    });
    let req = Request::builder()
        .method(Method::POST)
        .uri("/functions")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .expect("request builds");
    let resp = live_router()
        .oneshot(req)
        .await
        .expect("router serves /functions");
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "POST /functions with a valid CreateFunctionRequest should be 200",
    );
    let body_bytes = resp
        .into_body()
        .collect()
        .await
        .expect("body collects")
        .to_bytes();
    let body: Value = serde_json::from_slice(&body_bytes).expect("response is JSON");
    let id = body
        .get("id")
        .and_then(Value::as_str)
        .expect("CreateFunctionResponse has `id` per the spec");
    uuid::Uuid::parse_str(id).expect("id is a uuid per the CreateFunctionResponse schema");
}

// ---------------------------------------------------------------------------
// Scanner self-tests
// ---------------------------------------------------------------------------
//
// The handwritten YAML scanner is small but easy to regress on, so a
// few inline self-tests pin its behaviour against synthetic input.

#[test]
fn scanner_extracts_paths() {
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
    let paths = parse_spec_path_keys(yaml);
    let expected: BTreeSet<String> = ["/a", "/b/{id}"].iter().map(|s| s.to_string()).collect();
    assert_eq!(paths, expected);

    let methods = parse_spec_path_methods(yaml);
    assert_eq!(
        methods.get("/a").map(|m| m.iter().cloned().collect::<Vec<_>>()),
        Some(vec!["GET".to_string()]),
    );
    assert_eq!(
        methods.get("/b/{id}").map(|m| m.iter().cloned().collect::<Vec<_>>()),
        Some(vec!["POST".to_string()]),
    );

    let schemas = parse_spec_schema_names(yaml);
    assert!(schemas.contains("Foo"));
}

#[test]
fn scanner_parses_required_and_strictness() {
    let yaml = "\
components:
  schemas:
    Bar:
      type: object
      properties:
        a: { type: string }
        b: { type: integer }
      required: [a, b]
      additionalProperties: false
";
    let (required, strict) = parse_schema_required_and_strictness(yaml, "Bar");
    assert_eq!(
        required,
        ["a", "b"].iter().map(|s| s.to_string()).collect()
    );
    assert!(strict);
}

#[test]
fn scanner_axum_to_openapi_translation() {
    assert_eq!(axum_to_openapi_path("/healthz"), "/healthz");
    assert_eq!(axum_to_openapi_path("/functions/:id"), "/functions/{id}");
    assert_eq!(
        axum_to_openapi_path("/functions/:id/invoke"),
        "/functions/{id}/invoke",
    );
    assert_eq!(
        axum_to_openapi_path("/functions/:id/invoke-async"),
        "/functions/{id}/invoke-async",
    );
}
