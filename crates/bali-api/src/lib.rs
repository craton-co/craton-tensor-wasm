//! HTTP serverless API gateway built on axum 0.7.
//!
//! Exposes deploy/invoke/metrics/healthz endpoints with structured JSON
//! errors. Application state is a `DashMap<Uuid, FunctionRecord>` shared
//! via `Arc<AppState>`. Real Wasm execution wiring (driving `bali-exec`)
//! lands in a follow-up; S17 lands the HTTP surface, request validation,
//! and error envelope.
#![warn(missing_docs)]

pub mod middleware;
pub mod routes;
pub mod server;

pub use routes::{ApiError, AppState, FunctionRecord, JobRecord, JobStatus};
pub use server::{build_router, serve};
