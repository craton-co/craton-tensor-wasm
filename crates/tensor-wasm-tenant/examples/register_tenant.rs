// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! Minimal example: register a tenant, look it up, account some bytes
//! against its quota, release them, and print the final isolation kind.
//!
//! Run with `cargo run --example register_tenant -p tensor-wasm-tenant`.

use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::{IsolationKind, TenantContext, TenantRegistry};

#[tokio::main]
async fn main() {
    let registry = TenantRegistry::new();

    let ctx = TenantContext::builder(TenantId(42))
        .with_isolation(IsolationKind::StreamIsolated)
        .with_stream_id(1)
        .with_memory_quota_bytes(64 * 1024 * 1024) // 64 MiB
        .build();

    let tenant = registry.register(ctx).expect("registration");
    println!("registered: {}", tenant.id());

    let same = registry
        .get(TenantId(42))
        .expect("just-registered tenant must be findable");
    println!("isolation: {}", same.isolation());

    same.consume_bytes(4096).expect("within quota");
    println!("after consume: {} bytes in use", same.bytes_in_use());

    same.release_bytes(4096);
    println!("after release: {} bytes in use", same.bytes_in_use());

    println!("has_real_context: {}", same.has_real_context());
}
