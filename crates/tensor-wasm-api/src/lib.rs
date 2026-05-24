// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! HTTP serverless API gateway built on axum 0.7.
//!
//! Exposes deploy/invoke/metrics/healthz endpoints with structured JSON
//! errors. Application state is a `DashMap<Uuid, FunctionRecord>` shared
//! via `Arc<AppState>`. The synchronous and async invoke paths both drive
//! `tensor_wasm_exec::executor::TensorWasmExecutor`.
//!
//! ## Security surface
//!
//! * **Body limit.** Every request is capped at 64 MiB by
//!   [`tower_http::limit::RequestBodyLimitLayer`]; oversized bodies are
//!   rejected with `413 Payload Too Large` before any handler runs.
//! * **Bearer auth.** Reads `TENSOR_WASM_API_TOKENS` (comma-separated allowlist)
//!   at startup. Empty/unset means dev mode (pass-through with warning);
//!   otherwise requests must carry `Authorization: Bearer <token>`.
//! * **Tenant scoping.** The `X-TensorWasm-Tenant: <u64>` header is parsed and
//!   threaded through to the executor. Set `TENSOR_WASM_API_REQUIRE_TENANT=1`
//!   to make the header mandatory.
//!
//! See [`API.md`](../API.md) for the wire-format contract.
#![deny(missing_docs)]

pub mod middleware;
pub mod routes;
pub mod server;

pub use middleware::{
    AuthConfig, TenantConfig, ENV_API_TOKENS, ENV_REQUIRE_TENANT, HEADER_TENANT,
    MAX_REQUEST_BODY_BYTES,
};
pub use routes::{ApiError, AppState, FunctionRecord, JobRecord, JobStatus};
pub use server::{build_router, build_router_with_config, serve};
