//! Axum router builder and listener.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::routing::{delete, get, post};
use axum::Router;

use crate::middleware::{concurrency_limit_layer, timeout_layer, trace_layer_with_propagation};
use crate::routes::{
    create_function, delete_function, get_job, healthz, invoke_function, invoke_function_async,
    metrics, AppState,
};

/// Build the axum Router with all routes and middleware applied.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics))
        .route("/functions", post(create_function))
        .route("/functions/:id", delete(delete_function))
        .route("/functions/:id/invoke", post(invoke_function))
        .route("/functions/:id/invoke-async", post(invoke_function_async))
        .route("/jobs/:id", get(get_job))
        .layer(timeout_layer(Duration::from_secs(30)))
        .layer(concurrency_limit_layer(64))
        .layer(trace_layer_with_propagation())
        .with_state(state)
}

/// Bind and serve the router on the given address.
pub async fn serve(state: Arc<AppState>, addr: SocketAddr) -> anyhow::Result<()> {
    let router = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(target: "bali_api::server", %addr, "bali-api listening");
    axum::serve(listener, router).await?;
    Ok(())
}
