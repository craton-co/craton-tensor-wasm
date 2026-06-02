// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Regression test for core H1: tenant-info leak via error `Display`.
//!
//! `TensorWasmError::TenantIsolationViolation` carries the offending
//! `tenant_id` and a free-form `resource` string (often a host filesystem
//! path such as `/dev/shm/<other-tenant>/...`). The API layer surfaces
//! `Display` of this error as part of 4xx response bodies, so anything the
//! `Display` impl reveals is visible to the *requesting* tenant — i.e. the
//! one that just tried (and failed) to escape its sandbox.
//!
//! That means `Display` must NOT reveal:
//!   * the violator's tenant id (would confirm/identify an internal handle),
//!   * the resource path (typically encodes another tenant's on-disk layout).
//!
//! The structured fields stay on the variant and remain reachable through the
//! derived `Debug` impl, which is what `tracing::error!(?err, ...)` uses for
//! server-side operator logs.

use std::io;

use tensor_wasm_core::error::TensorWasmError;
use tensor_wasm_core::types::TenantId;

/// The exact phrase the API layer (and operator dashboards) grep for. If this
/// constant is ever renamed, downstream log-search queries break — keep the
/// change deliberate.
const STABLE_GREP_TOKEN: &str = "isolation";

#[test]
fn display_does_not_leak_tenant_or_resource() {
    let err = TensorWasmError::TenantIsolationViolation {
        tenant_id: TenantId(42),
        resource: "/dev/shm/secret-path".into(),
    };
    let rendered = err.to_string();

    // Forbidden substrings — each represents a different shape of leak.
    for needle in ["42", "T#42", "dev", "shm", "secret"] {
        assert!(
            !rendered.contains(needle),
            "Display leaked {needle:?} to tenant-facing output: {rendered:?}",
        );
    }

    // The rendered string must still carry a stable identifier so that
    // operators can correlate a 4xx response body with their server-side logs.
    assert!(
        rendered.contains(STABLE_GREP_TOKEN) || rendered.contains("violation"),
        "Display must contain a stable identifier (\"isolation\" or \
         \"violation\") so callers can pivot to server-side logs: {rendered:?}",
    );
}

#[test]
fn display_io_does_not_leak_path_or_os_string() {
    // `TensorWasmError::Io` wraps `std::io::Error`, whose `Display` impl
    // echoes the underlying OS error string verbatim. When the error came
    // from a filesystem operation that string very often includes the
    // offending path (e.g. `/secret/path` for a `NotFound` on an
    // unauthorised lookup). Surfacing that into a tenant-facing 4xx body
    // would leak host layout. The sanitised `Display` must keep the
    // `ErrorKind` (a closed, non-sensitive enum) and drop the inner
    // string.
    let err: TensorWasmError = io::Error::new(io::ErrorKind::NotFound, "/secret/path").into();
    let rendered = format!("{}", err);

    assert!(
        !rendered.contains("/secret/path"),
        "Display leaked filesystem path to tenant-facing output: {rendered:?}",
    );

    // Positive contract: the ErrorKind variant name MUST appear so
    // operator dashboards can pivot on it.
    assert!(
        rendered.contains("NotFound"),
        "Display must surface the ErrorKind variant name: {rendered:?}",
    );
}

#[test]
fn debug_io_still_carries_inner_string_for_operators() {
    // `Debug` is what `tracing::error!(?err)` records on the server side;
    // the inner OS string MUST still be reachable there so operators can
    // diagnose I/O failures. Only `Display` is sanitised.
    let err: TensorWasmError = io::Error::new(io::ErrorKind::NotFound, "/secret/path").into();
    let rendered = format!("{:?}", err);
    assert!(
        rendered.contains("/secret/path"),
        "Debug must still expose the inner OS string for server-side triage: {rendered:?}",
    );
}

#[test]
fn debug_still_carries_structured_fields() {
    // Sanity-check the other half of the contract: `Debug` is what the host
    // pipes into `tracing::error!(?err, ...)`, so it MUST keep exposing the
    // structured fields for operator triage. If a future refactor strips
    // them (e.g. by switching to a manual `Debug` impl), this test catches
    // the regression before it ships.
    let err = TensorWasmError::TenantIsolationViolation {
        tenant_id: TenantId(42),
        resource: "/dev/shm/secret-path".into(),
    };
    let rendered = format!("{err:?}");

    assert!(
        rendered.contains("42"),
        "Debug must still expose tenant_id for server-side triage: {rendered:?}",
    );
    assert!(
        rendered.contains("/dev/shm/secret-path"),
        "Debug must still expose resource for server-side triage: {rendered:?}",
    );
}
