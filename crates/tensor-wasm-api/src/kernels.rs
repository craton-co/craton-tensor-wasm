// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! `/kernels` HTTP endpoints (roadmap feature #3 — server-side path).
//!
//! Exposes the kernel registry to authenticated operators via three
//! routes:
//!
//!   * `POST /kernels` — publish a signed [`KernelManifest`] + PTX text.
//!   * `GET  /kernels` — list manifests (PTX omitted).
//!   * `GET  /kernels/{name}/{version}` — resolve a manifest + PTX.
//!
//! All three routes sit behind `bearer_auth` + `tenant_scope` (so the
//! tenant extension is always present and every caller has cleared the
//! token allowlist). The two GET routes admit any authenticated
//! tenant. `POST /kernels` additionally requires the
//! **kernel-publish** scope:
//!
//!   * **Dev mode** (`TENSOR_WASM_API_TOKENS` unset/empty) — every
//!     `POST /kernels` is rejected with `403
//!     kernel_publish_disabled_in_dev_mode`. Dev mode already grants
//!     every caller wildcard tenant scope, so allowing publish there
//!     would let unauthenticated traffic seed the registry. Operators
//!     who genuinely want a dev-time publish path must configure a
//!     token allowlist *and* opt the publishing token into
//!     [`TENSOR_WASM_API_KERNEL_PUBLISH_TOKENS`](crate::middleware::ENV_KERNEL_PUBLISH_TOKENS).
//!   * **Production mode** — the caller's bearer token's
//!     [`crate::rate_limit::TokenId`] must appear in
//!     [`KernelPublishTokens`](crate::middleware::KernelPublishTokens),
//!     the allowlist parsed from
//!     [`TENSOR_WASM_API_KERNEL_PUBLISH_TOKENS`](crate::middleware::ENV_KERNEL_PUBLISH_TOKENS).
//!     A token in the main allowlist but not in the publish allowlist
//!     gets `403 kernel_publish_scope_required`.
//!
//! The server-side [`InMemoryRegistry`] is held in
//! [`AppState::kernel_registry`] as an `Option<Arc<dyn KernelRegistry>>`
//! and constructed at startup from the `TENSOR_WASM_API_KERNEL_HMAC_KEY`
//! environment variable (64-hex 32-byte key). When the env var is unset,
//! the routes return `503 kernel_registry_not_configured` so client
//! tools can detect feature availability without inspecting the URL
//! surface.
//!
//! ## URL syntax
//!
//! `axum 0.7` resolves path parameters via the colon-prefix syntax
//! (`:name`, `:version`); the literal `{name}/{version}` shape used by
//! the OpenAPI spec maps to the `:name/:version` axum route. The CLI
//! issues `GET /kernels/<name>/<version>` (slash-separated) rather than
//! the `name@version` form used by the in-memory key — the at-sign in a
//! path segment is awkward to encode and most HTTP clients eagerly
//! percent-encode it, so the canonical wire form takes the two-segment
//! shape and the handler concatenates internally before the registry
//! lookup. See `docs/KERNEL-REGISTRY.md` §"v0.4 rollout plan".
//!
//! ## Error envelope
//!
//! Every failure surfaces the same `{error: {kind, message}}` shape the
//! native routes use (see [`crate::routes::ApiError`]). The kinds
//! introduced by this module:
//!
//! | kind                                  | status | meaning                                                                   |
//! |---------------------------------------|--------|---------------------------------------------------------------------------|
//! | `kernel_registry_not_configured`      | 503    | `TENSOR_WASM_API_KERNEL_HMAC_KEY` is unset; routes are wired but disabled |
//! | `kernel_registry_storage_error`       | 503    | T35: disk-backed registry I/O error during publish                        |
//! | `kernel_publish_disabled_in_dev_mode` | 403    | `POST /kernels` rejected because the gateway is in dev mode (no tokens)   |
//! | `kernel_publish_scope_required`       | 403    | Caller's bearer token is not in `TENSOR_WASM_API_KERNEL_PUBLISH_TOKENS`   |
//! | `publisher_not_allowed`               | 403    | T35: `manifest.publisher` is not in the registry's publisher allowlist    |
//! | `bad_signature`                       | 403    | HMAC verification under the configured key failed                         |
//! | `digest_mismatch`                     | 400    | BLAKE3(`ptx_text`) does not match `manifest.digest`                       |
//! | `already_registered`                  | 409    | A manifest with the same `name@version` is already present                |
//! | `invalid_request`                     | 400    | Catch-all for other `RegistryError` variants (forward-compat)             |
//! | `not_found`                           | 404    | The selector did not resolve to a registered manifest                     |

use std::sync::Arc;

use axum::{
    extract::{Extension, Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};

use crate::middleware::KernelPublishTokens;
use crate::rate_limit::{AuthContext, TokenId};
use crate::routes::{ApiError, AppState};
use tensor_wasm_core::types::TenantId;
use tensor_wasm_jit::registry::KernelRegistry;

/// Request body for `POST /kernels`.
///
/// `manifest` carries the signed envelope (name, version, sm_version,
/// digest, signature, advisory metadata); `ptx_text` is the kernel
/// source whose BLAKE3 digest must match `manifest.digest`. The server
/// re-verifies both the digest match and the HMAC signature under the
/// configured key before persisting — see
/// [`tensor_wasm_jit::registry::InMemoryRegistry::publish`] for the
/// verification order.
#[derive(Debug, Deserialize)]
pub struct PublishKernelRequest {
    /// The signed manifest. Must be HMAC-SHA256-signed under the same key
    /// the server was configured with.
    pub manifest: tensor_wasm_jit::registry::KernelManifest,
    /// PTX text. The server computes BLAKE3 over the UTF-8 bytes and
    /// requires a match with `manifest.digest` before any signature
    /// check (cheaper failure for corrupted uploads).
    pub ptx_text: String,
}

/// Response body for `GET /kernels`.
#[derive(Debug, Serialize)]
pub struct ListKernelsResponse {
    /// Page of registered manifests, in unspecified order. PTX text is NOT
    /// included — callers that need the kernel source must issue a
    /// follow-up `GET /kernels/{name}/{version}`. The size of this page
    /// is at most `limit` (clamped to 1000 server-side); a shorter page
    /// indicates the end of the keyspace was reached.
    pub manifests: Vec<tensor_wasm_jit::registry::KernelManifest>,
    /// Echo of the effective `offset` the handler used. Matches the
    /// inbound query parameter, defaulting to 0.
    pub offset: usize,
    /// Echo of the effective `limit` the handler used (after clamping).
    /// Matches the inbound query parameter, defaulting to 100, clamped
    /// to 1000.
    pub limit: usize,
}

/// Query parameters for `GET /kernels`.
///
/// Both fields are optional; defaults mirror the registry's pagination
/// constants (`offset = 0`, `limit = 100`). `limit` is silently clamped
/// to 1000 on the registry side so a caller asking for an unbounded
/// page still receives a bounded response.
#[derive(Debug, Default, Deserialize)]
pub struct ListKernelsQuery {
    /// Index of the first manifest to return; defaults to 0.
    #[serde(default)]
    pub offset: Option<usize>,
    /// Maximum number of manifests in the returned page; defaults to
    /// 100. Server-side cap: 1000.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Response body for `GET /kernels/{name}/{version}`.
#[derive(Debug, Serialize)]
pub struct ResolveKernelResponse {
    /// The verified manifest as previously published.
    pub manifest: tensor_wasm_jit::registry::KernelManifest,
    /// PTX text whose BLAKE3 matches `manifest.digest`.
    pub ptx_text: String,
}

/// `POST /kernels` — publish a signed manifest + PTX text.
///
/// The handler:
///
/// 1. Verifies the caller has the **kernel-publish** scope:
///    * In dev mode (`TokenId::DEV` — empty `TENSOR_WASM_API_TOKENS`)
///      the call is rejected with `403
///      kernel_publish_disabled_in_dev_mode`. Closing the T1 finding:
///      previously, dev mode silently admitted every anonymous caller.
///    * Otherwise the caller's [`TokenId`] must appear in
///      [`KernelPublishTokens`] (parsed from
///      [`TENSOR_WASM_API_KERNEL_PUBLISH_TOKENS`](crate::middleware::ENV_KERNEL_PUBLISH_TOKENS)).
///      A token in the main API allowlist but missing from the publish
///      allowlist gets `403 kernel_publish_scope_required`.
/// 2. Pulls the registry out of [`AppState`]; returns `503` when unset.
/// 3. Hands the (manifest, ptx_text) pair to
///    [`InMemoryRegistry::publish`](tensor_wasm_jit::registry::InMemoryRegistry::publish),
///    which performs digest + HMAC verification before insertion.
/// 4. Maps each [`RegistryError`](tensor_wasm_jit::registry::RegistryError)
///    variant to a stable `(status, kind)` pair on the wire.
///
/// On success, returns `201 Created` with `{name, version}` echoed back
/// so the CLI can confirm the canonical key.
///
/// The `_tenant` extension is required (the route sits under
/// `tenant_scope` so it is always present). It is captured here for
/// future use by audit/per-tenant key-scope tables without re-shaping
/// the handler signature; today it is intentionally unused beyond
/// proving its presence — the kernel registry is operator-scope.
///
/// The [`KernelPublishTokens`] extension is optional only because tests
/// that drive the handler directly may not install one; production
/// routers always layer it on. An absent extension is treated as the
/// empty allowlist (deny by default), which is the same posture as
/// `TENSOR_WASM_API_KERNEL_PUBLISH_TOKENS` being unset.
pub async fn publish_kernel(
    State(state): State<Arc<AppState>>,
    Extension(_tenant): Extension<TenantId>,
    Extension(auth): Extension<AuthContext>,
    publish_tokens: Option<Extension<KernelPublishTokens>>,
    Json(req): Json<PublishKernelRequest>,
) -> Result<(StatusCode, Json<serde_json::Value>), ApiError> {
    // Step 1: enforce the kernel-publish scope BEFORE any registry work.
    // The HMAC verification that runs inside `InMemoryRegistry::publish`
    // is an integrity check, NOT an authorization check — without this
    // gate, any allowlisted token (including a tenant-1 token) could
    // seed the deployment-wide registry.
    if auth.token_id == TokenId::DEV {
        return Err(ApiError::forbidden(
            "kernel_publish_disabled_in_dev_mode",
            "POST /kernels is disabled when TENSOR_WASM_API_TOKENS is empty; \
             configure an API token allowlist and add the publishing token to \
             TENSOR_WASM_API_KERNEL_PUBLISH_TOKENS",
        ));
    }
    let allow = publish_tokens
        .map(|Extension(t)| t)
        .unwrap_or_default();
    if !allow.allows(auth.token_id) {
        return Err(ApiError::forbidden(
            "kernel_publish_scope_required",
            "bearer token is not allowlisted for POST /kernels; \
             add it to TENSOR_WASM_API_KERNEL_PUBLISH_TOKENS",
        ));
    }

    let registry = state
        .kernel_registry
        .as_ref()
        .ok_or_else(|| {
            ApiError::service_unavailable(
                "kernel_registry_not_configured",
                "set TENSOR_WASM_API_KERNEL_HMAC_KEY to enable /kernels",
            )
        })?;
    // The InMemoryRegistry::publish does signature + digest verification.
    // Snapshot the (name, version) BEFORE handing the manifest off — the
    // call takes the manifest by value so we cannot read its fields on
    // the response side without an extra clone.
    let echo_name = req.manifest.name.clone();
    let echo_version = req.manifest.version.clone();
    registry
        .publish(req.manifest, req.ptx_text)
        .map_err(|e| match e {
            tensor_wasm_jit::registry::RegistryError::BadSignature(_) => {
                ApiError::forbidden("bad_signature", e.to_string())
            }
            tensor_wasm_jit::registry::RegistryError::DigestMismatch(_) => {
                ApiError::bad_request("digest_mismatch", e.to_string())
            }
            tensor_wasm_jit::registry::RegistryError::AlreadyRegistered(_) => {
                ApiError::conflict("already_registered", e.to_string())
            }
            // T35: the disk-backed registry can additionally refuse a
            // publish when the manifest's `publisher` field is not in
            // the operator's configured allowlist. Surface as 403
            // (authorization failure) — distinct from `bad_signature`
            // (signature didn't verify under any key) and from
            // `kernel_publish_scope_required` (the route-level scope
            // gate from T1, which fires before we even reach the
            // registry call).
            tensor_wasm_jit::registry::RegistryError::PublisherNotAllowed(_) => {
                ApiError::forbidden("publisher_not_allowed", e.to_string())
            }
            // T35: artifact-store I/O failure (disk full, permissions,
            // tampered envelope, etc.). The store layer already
            // logged the underlying cause; surface to the client as
            // 503 so the operator's tooling can distinguish a
            // transient backend failure from a malformed request.
            // Storage failures during publish are non-retriable from
            // the client's perspective without operator action — same
            // posture as `kernel_registry_not_configured`.
            tensor_wasm_jit::registry::RegistryError::Storage(_) => {
                ApiError::service_unavailable("kernel_registry_storage_error", e.to_string())
            }
            // T35: bincode encode failure during publish. Should only
            // fire on truly pathological input (it's the encode path
            // for a manifest we just verified, so the bytes are well-
            // formed). Map to 500 internal — there is no client action
            // that resolves this.
            tensor_wasm_jit::registry::RegistryError::Codec(_) => {
                ApiError::internal(e.to_string())
            }
            // Forward-compat catch-all: future RegistryError variants
            // (e.g. NotFound, which `publish` cannot actually produce
            // today) collapse to 400 invalid_request rather than masking
            // a real error as 500. The match is exhaustive at compile
            // time; the catch-all is here for v0.4 extension points.
            _ => ApiError::bad_request("invalid_request", e.to_string()),
        })?;
    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "name": echo_name,
            "version": echo_version,
        })),
    ))
}

/// `GET /kernels?offset=N&limit=M` — list registered manifests.
///
/// Returns `200 OK` with `{manifests: [...], offset, limit}`. PTX text
/// is intentionally omitted; callers that need a kernel's source must
/// resolve it individually so a tenant listing every manifest does not
/// transfer hundreds of megabytes of source over the wire.
///
/// `offset` defaults to 0, `limit` defaults to 100, and `limit` is
/// silently clamped to 1000 server-side via
/// [`tensor_wasm_jit::registry::DISK_REGISTRY_MAX_LIMIT`]. The response
/// echoes the effective values so clients can drive the next-page
/// request without a second round of math.
///
/// Authorization: any authenticated tenant may list. The route sits
/// under `bearer_auth` + `tenant_scope`, so an unauthenticated caller
/// is already 401'd before reaching this handler. Capturing the tenant
/// extension here is belt-and-braces — it documents the routing
/// expectation and ensures the handler will fail to mount on a router
/// that bypasses `tenant_scope`.
pub async fn list_kernels(
    State(state): State<Arc<AppState>>,
    Extension(_tenant): Extension<TenantId>,
    Query(query): Query<ListKernelsQuery>,
) -> Result<Json<ListKernelsResponse>, ApiError> {
    let registry = state
        .kernel_registry
        .as_ref()
        .ok_or_else(|| {
            ApiError::service_unavailable(
                "kernel_registry_not_configured",
                "set TENSOR_WASM_API_KERNEL_HMAC_KEY to enable /kernels",
            )
        })?;
    let offset = query.offset.unwrap_or(0);
    let raw_limit = query
        .limit
        .unwrap_or(tensor_wasm_jit::registry::DISK_REGISTRY_DEFAULT_LIMIT);
    let effective_limit =
        raw_limit.min(tensor_wasm_jit::registry::DISK_REGISTRY_MAX_LIMIT);
    Ok(Json(ListKernelsResponse {
        manifests: registry.list_paginated(offset, effective_limit),
        offset,
        limit: effective_limit,
    }))
}

/// `GET /kernels/{name}/{version}` — resolve a manifest + PTX.
///
/// Returns `200 OK` with `{manifest, ptx_text}` on hit, `404 not_found`
/// when the selector does not match any registered manifest. Path
/// segments are accepted verbatim; the CLI must percent-encode any
/// reserved characters in `name` or `version` before issuing the
/// request.
///
/// Authorization: any authenticated tenant may resolve. See
/// [`list_kernels`] for the routing-stack rationale on the captured
/// tenant extension.
pub async fn resolve_kernel(
    State(state): State<Arc<AppState>>,
    Extension(_tenant): Extension<TenantId>,
    Path((name, version)): Path<(String, String)>,
) -> Result<Json<ResolveKernelResponse>, ApiError> {
    let registry = state
        .kernel_registry
        .as_ref()
        .ok_or_else(|| {
            ApiError::service_unavailable(
                "kernel_registry_not_configured",
                "set TENSOR_WASM_API_KERNEL_HMAC_KEY to enable /kernels",
            )
        })?;
    let entry = registry
        .get(&name, &version)
        .map_err(|_| ApiError::not_found(format!("kernel {name}@{version} not found")))?;
    // The registry returns Arc<(manifest, ptx_text)>; clone the inner
    // pair into the response body rather than dragging the Arc through
    // serde. Two String clones per resolve is well within the noise
    // budget for an admin-class API.
    Ok(Json(ResolveKernelResponse {
        manifest: entry.0.clone(),
        ptx_text: entry.1.clone(),
    }))
}
