// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Axum router builder and listener.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::routing::{delete, get, post};
use axum::Router;
use tower::ServiceBuilder;

use crate::middleware::{
    bearer_auth, body_limit_layer, concurrency_limit_layer, tenant_scope, timeout_layer,
    trace_layer_with_propagation, AuthConfig, TenantConfig, MAX_REQUEST_BODY_BYTES,
};
use crate::routes::{
    create_function, delete_function, get_job, healthz, invoke_function, invoke_function_async,
    metrics, AppState,
};

/// Build the axum Router with all routes and middleware applied.
///
/// Reads [`AuthConfig`] (`$TENSOR_WASM_API_TOKENS`) and [`TenantConfig`]
/// (`$TENSOR_WASM_API_REQUIRE_TENANT`) from the process environment. Empty /
/// unset `TENSOR_WASM_API_TOKENS` puts the gateway in dev mode (auth disabled
/// with a startup warning).
pub fn build_router(state: Arc<AppState>) -> Router {
    let auth = AuthConfig::from_env();
    let tenant = TenantConfig::from_env();
    build_router_with_config(state, auth, tenant)
}

/// Build the router with explicitly supplied auth / tenant config. Used by
/// integration tests so they can drive the gateway without poisoning the
/// process environment.
pub fn build_router_with_config(
    state: Arc<AppState>,
    auth: AuthConfig,
    tenant: TenantConfig,
) -> Router {
    // The outer ServiceBuilder is layered top-to-bottom: tracing is the
    // outermost layer (so it covers timeouts and rejections), the body
    // limit guards every downstream layer from oversized payloads,
    // followed by the per-request timeout and the global concurrency cap.
    // Auth and tenant resolution are `from_fn` middleware that run after
    // the size cap (so a request that would be rejected with 413 does
    // not consume an auth slot).
    let global_layers = ServiceBuilder::new()
        .layer(trace_layer_with_propagation())
        .layer(body_limit_layer(MAX_REQUEST_BODY_BYTES))
        .layer(timeout_layer(Duration::from_secs(30)))
        .layer(concurrency_limit_layer(64))
        // Pass the config into the request extensions so the `from_fn`
        // middleware below can pick it up without capturing through a
        // type-erased closure.
        .layer(axum::Extension(auth))
        .layer(axum::Extension(tenant))
        .layer(axum::middleware::from_fn(bearer_auth))
        .layer(axum::middleware::from_fn(tenant_scope));

    Router::new()
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics))
        .route("/functions", post(create_function))
        .route("/functions/:id", delete(delete_function))
        .route("/functions/:id/invoke", post(invoke_function))
        .route("/functions/:id/invoke-async", post(invoke_function_async))
        .route("/jobs/:id", get(get_job))
        .layer(global_layers)
        .with_state(state)
}

/// Bind and serve the router on the given address.
pub async fn serve(state: Arc<AppState>, addr: SocketAddr) -> anyhow::Result<()> {
    let router = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(target: "tensor_wasm_api::server", %addr, "tensor-wasm-api listening");
    axum::serve(listener, router).await?;
    Ok(())
}
