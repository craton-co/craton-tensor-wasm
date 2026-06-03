// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Fuzz target for the [`tensor_wasm_api::AuditRecord`] JSON wire format.
//!
//! ## Goal
//!
//! The audit log is a public contract: downstream consumers (SIEMs,
//! log shippers, compliance tooling) parse the JSONL stream and expect
//! a stable field shape. The crate intentionally **does not** derive
//! `Deserialize` on `AuditRecord` to avoid promoting it to an input
//! type, so the existing unit-test coverage only round-trips through
//! `serde_json::Value` — it cannot catch a regression where, say,
//! `AuditActor::token_id` is renamed but only the Serialize derive is
//! touched.
//!
//! This harness defends the wire-format contract from *both* sides:
//!
//! 1. Build an arbitrary [`AuditRecordFixture`] (mirrors the public
//!    shape, derives `Serialize + Deserialize + PartialEq +
//!    arbitrary::Arbitrary`).
//! 2. Round-trip it through `serde_json` and assert equality. This
//!    catches a divergence in the fixture's own derives — useful when
//!    we add fields to the wrapper to match upstream changes.
//! 3. Project the fixture into a real [`AuditRecord`], serialize it
//!    via the **production** `Serialize` impl, then deserialize the
//!    resulting JSON back into the fixture wrapper and assert it
//!    matches. This catches a divergence between `AuditRecord`'s
//!    serde derives and the public contract the wrapper encodes.
//!
//! ## Why a wrapper instead of `#[derive(Arbitrary)]` on `AuditRecord`
//!
//! `AuditRecord` does not derive `Deserialize`, and adding it would
//! widen the public surface in a way the audit module explicitly
//! avoids (see the existing `record_round_trips_through_serde_json`
//! test rationale in `audit.rs`). The fixture wrapper is therefore
//! the right home for both `Arbitrary` (input generation) and
//! `Deserialize` (round-trip parsing).
//!
//! ## Property assertions
//!
//! For every iteration:
//!
//! * Fixture → JSON → fixture is a no-op (`PartialEq`).
//! * Fixture → [`AuditRecord`] → JSON → fixture is a no-op.
//! * The intermediate JSON parses as a JSON object (`serde_json::Value`
//!   is `Object`) — a sanity check that catches a Serialize impl that
//!   somehow emits an array or scalar at the top level.
//!
//! ## Running
//!
//! ```sh
//! cargo +nightly fuzz run audit_json_round_trip -- -max_total_time=86400
//! ```

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use tensor_wasm_api::audit::{
    AuditAction as RealAction, AuditActor as RealActor, AuditActorKind as RealActorKind,
    AuditOutcome as RealOutcome, AuditRecord as RealRecord, AuditResource as RealResource,
    TokenScopeView as RealScopeView,
};
use tensor_wasm_api::rate_limit::TokenId;
use tensor_wasm_core::types::TenantId;

// ---------------------------------------------------------------------------
// Fixture wrapper — mirrors the public JSON shape of AuditRecord.
//
// Field names and enum tags MUST match the production `Serialize` impl
// exactly; that is the entire point of the round-trip assertion.
// ---------------------------------------------------------------------------

// Per-field `skip_serializing_if` flags below MUST match the production
// `Serialize` derive on `AuditRecord` and friends in
// `tensor-wasm-api/src/audit.rs`:
//
//   AuditActor.token_id              — NO skip  (emits null when None)
//   AuditResource.function_id        — skip if None
//   AuditResource.tenant_id          — skip if None
//   AuditOutcome.error_kind          — skip if None
//   AuditRecord.peer_addr            — NO skip  (emits null when None)
//   AuditRecord.client_cert_subject  — NO skip  (emits null when None)
//
// If the fixture and the production type drift here, the
// "real → JSON → fixture" round-trip below will still parse (serde
// happily reads a `null` into an absent field) but a later subtle bug
// (e.g. a consumer relying on key presence) would slip through. The
// `serde_default` on the Option fields keeps the deserialize side
// permissive across either choice, which is the desired behaviour for
// a round-trip property test.

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Arbitrary)]
struct AuditRecordFixture {
    ts_unix_ms: u64,
    request_id: Uuid,
    actor: AuditActorFixture,
    action: AuditActionFixture,
    resource: AuditResourceFixture,
    outcome: AuditOutcomeFixture,
    latency_ms: u64,
    #[serde(default)]
    peer_addr: Option<String>,
    #[serde(default)]
    client_cert_subject: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Arbitrary)]
struct AuditActorFixture {
    kind: AuditActorKindFixture,
    #[serde(default)]
    token_id: Option<u64>,
    scope: TokenScopeViewFixture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Arbitrary)]
#[serde(rename_all = "snake_case")]
enum AuditActorKindFixture {
    Bearer,
    Dev,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Arbitrary)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TokenScopeViewFixture {
    Wildcard,
    TenantSet { tenants: Vec<u64> },
    Dev,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Arbitrary)]
#[serde(rename_all = "snake_case")]
enum AuditActionFixture {
    CreateFunction,
    DeleteFunction,
    InvokeFunction,
    InvokeFunctionAsync,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Arbitrary)]
struct AuditResourceFixture {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    function_id: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tenant_id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Arbitrary)]
struct AuditOutcomeFixture {
    status_code: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error_kind: Option<String>,
}

// ---------------------------------------------------------------------------
// Projection: fixture → real AuditRecord
//
// The fixture's tenant ids are sorted on output by the production
// `TokenScopeView::from_scope`, which we mirror here so the round-trip
// equality holds. We also normalise the fixture itself (sort the
// tenants vec) before serializing it, so that "fixture → real → JSON →
// fixture" matches "fixture → JSON → fixture".
// ---------------------------------------------------------------------------

impl AuditRecordFixture {
    /// Bring the fixture into the canonical shape the production
    /// serializer emits. Today this means sorting the `tenants` list
    /// inside `TokenScopeView::TenantSet` so the round-trip equality
    /// is meaningful.
    fn canonicalize(&mut self) {
        if let TokenScopeViewFixture::TenantSet { tenants } = &mut self.actor.scope {
            tenants.sort_unstable();
            // Production `TokenScopeView::from_scope` runs through a
            // HashSet first, which de-duplicates. Match it so the
            // fixture-vs-real round-trip cannot fail on a duplicate
            // tenant id supplied by arbitrary.
            tenants.dedup();
        }
    }

    /// Project into the real [`AuditRecord`]. Mirrors the production
    /// constructors used inside the audit middleware.
    fn to_real(&self) -> RealRecord {
        let scope = match &self.actor.scope {
            TokenScopeViewFixture::Wildcard => RealScopeView::Wildcard,
            TokenScopeViewFixture::TenantSet { tenants } => RealScopeView::TenantSet {
                tenants: tenants.clone(),
            },
            TokenScopeViewFixture::Dev => RealScopeView::Dev,
        };
        let actor = RealActor {
            kind: match self.actor.kind {
                AuditActorKindFixture::Bearer => RealActorKind::Bearer,
                AuditActorKindFixture::Dev => RealActorKind::Dev,
            },
            token_id: self.actor.token_id.map(TokenId),
            scope,
        };
        let action = match self.action {
            AuditActionFixture::CreateFunction => RealAction::CreateFunction,
            AuditActionFixture::DeleteFunction => RealAction::DeleteFunction,
            AuditActionFixture::InvokeFunction => RealAction::InvokeFunction,
            AuditActionFixture::InvokeFunctionAsync => RealAction::InvokeFunctionAsync,
        };
        let resource = RealResource {
            function_id: self.resource.function_id,
            tenant_id: self.resource.tenant_id.map(TenantId),
        };
        let outcome = RealOutcome {
            status_code: self.outcome.status_code,
            error_kind: self.outcome.error_kind.clone(),
        };
        RealRecord {
            ts_unix_ms: self.ts_unix_ms,
            request_id: self.request_id,
            actor,
            action,
            resource,
            outcome,
            latency_ms: self.latency_ms,
            peer_addr: self.peer_addr.clone(),
            client_cert_subject: self.client_cert_subject.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Fuzz entry point
// ---------------------------------------------------------------------------

fuzz_target!(|data: &[u8]| {
    // Bound input length so we don't spend a whole iteration on a
    // 64 KiB error_kind string; the wire-format properties we care
    // about are exercised by short inputs.
    if data.len() > 16 * 1024 {
        return;
    }
    let mut u = Unstructured::new(data);
    let Ok(mut fixture) = AuditRecordFixture::arbitrary(&mut u) else {
        return;
    };
    fixture.canonicalize();

    // ---- Property 1: fixture round-trips through its own derives. ----
    let fixture_json = match serde_json::to_string(&fixture) {
        Ok(s) => s,
        // Serde does not fail to serialise simple owned types; any error
        // here is unexpected. Surface as a panic so libfuzzer captures
        // the input.
        Err(e) => panic!("fixture serialise failed: {e}"),
    };
    let fixture_back: AuditRecordFixture = match serde_json::from_str(&fixture_json) {
        Ok(v) => v,
        Err(e) => panic!(
            "fixture round-trip failed: serialize→deserialize diverged.\n\
             error={e}\n\
             json={fixture_json}",
        ),
    };
    assert_eq!(
        fixture, fixture_back,
        "fixture self round-trip mismatch.\nbefore={fixture:?}\nafter={fixture_back:?}\njson={fixture_json}",
    );

    // ---- Property 2: production `AuditRecord` Serialize lands in the
    //                  same wire shape the fixture's Deserialize accepts. ----
    let real = fixture.to_real();
    let real_json = match serde_json::to_string(&real) {
        Ok(s) => s,
        Err(e) => panic!("real AuditRecord serialise failed: {e}\nfixture={fixture:?}"),
    };

    // ---- Property 3: the production JSON is a top-level object. ----
    let as_value: serde_json::Value = serde_json::from_str(&real_json)
        .unwrap_or_else(|e| panic!("real JSON is not valid JSON: {e}\njson={real_json}"));
    assert!(
        as_value.is_object(),
        "AuditRecord top-level JSON must be an object; got {real_json}",
    );

    // ---- Property 4: real JSON parses back into the fixture wrapper. ----
    let real_back: AuditRecordFixture = match serde_json::from_str(&real_json) {
        Ok(v) => v,
        Err(e) => panic!(
            "production AuditRecord JSON does not parse into the wire-format fixture.\n\
             this means Serialize on AuditRecord has diverged from the documented contract.\n\
             error={e}\n\
             json={real_json}\n\
             fixture={fixture:?}",
        ),
    };
    assert_eq!(
        fixture, real_back,
        "production AuditRecord round-trip mismatch.\nfixture={fixture:?}\nreal_json={real_json}\nreal_back={real_back:?}",
    );
});
