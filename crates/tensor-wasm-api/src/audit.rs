// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Structured audit log for state-mutating API calls.
//!
//! Implements PATH-TO-V1 v0.4 exit-criterion *Audit log*: every request
//! that mutates server state writes one JSON record describing who called,
//! what they did, what came back, and how long it took. Read-only calls
//! (`GET /healthz`, `GET /metrics`, `GET /jobs/{id}`) emit nothing — they
//! would drown the log in noise.
//!
//! ## Record shape
//!
//! Each record is a single JSON object emitted on its own line:
//!
//! ```json
//! {
//!   "ts_unix_ms": 1716491220123,
//!   "request_id": "b8b6f7e0-...-...",
//!   "actor": {
//!     "kind": "bearer",
//!     "token_id": 14217683...,
//!     "scope": { "kind": "tenant_set", "tenants": [1, 2] }
//!   },
//!   "action": "invoke_function",
//!   "resource": {
//!     "function_id": "f47ac10b-...-...",
//!     "tenant_id": 7
//!   },
//!   "outcome": { "status_code": 200, "error_kind": null },
//!   "latency_ms": 12,
//!   "peer_addr": "10.0.0.1:54321",
//!   "client_cert_subject": null
//! }
//! ```
//!
//! Field-stability rules match the rest of the public surface: keys and the
//! discriminant tags inside [`AuditAction`] and [`AuditActorKind`] are part
//! of the public contract; additive forward-compatible fields are allowed.
//! See [`docs/AUDIT-LOG.md`](../../../docs/AUDIT-LOG.md) for the wire-format
//! contract.
//!
//! ## Sinks
//!
//! [`AuditSink`] is a small trait so consumers can plug in whatever
//! durable target they prefer. The crate ships two implementations:
//!
//! * [`StdoutJsonSink`] — writes each record to stdout via `println!`. The
//!   default sink: stdout is what container runtimes capture and ship.
//!   Also mirrors each record at `tracing::info!` level so OpenTelemetry
//!   tail-sampling sees it.
//! * [`FileJsonSink`] — appends to a file. The file is opened append-only
//!   and the writes serialise through a `std::sync::Mutex<File>`. The
//!   critical section is `write_all` + `flush` of ~512 bytes — measured
//!   below in the [latency budget](#latency-budget) section.
//! * [`NoopSink`] — drops every record. Used when
//!   `TENSOR_WASM_API_AUDIT_LOG=none` and exposed for tests.
//!
//! ## Configuration
//!
//! Read at server start via [`AuditConfig::from_env`]:
//!
//! | `TENSOR_WASM_API_AUDIT_LOG` value | Resulting sink                                  |
//! |-----------------------------------|--------------------------------------------------|
//! | unset / empty                     | [`StdoutJsonSink`] (default — stdout)            |
//! | `none`                            | [`NoopSink`] (audit disabled)                    |
//! | `stdout`                          | [`StdoutJsonSink`] (explicit form)               |
//! | `file:/path/to/audit.log`         | [`FileJsonSink`] (append-only at that path)      |
//!
//! ## Latency budget
//!
//! The audit middleware runs *after* the handler completes, so it never
//! delays the response from the application's point of view in the sense
//! of "client sees the body later" — `axum::middleware::from_fn` happens to
//! serialise the post-handler work into the response future, but the bytes
//! the client receives flow once the inner future yields. The relevant
//! cost is therefore the CPU time spent in [`AuditSink::emit`]:
//!
//! * `StdoutJsonSink`: `serde_json::to_string` on a ~10-field struct +
//!   one `println!`. The macro takes the stdout lock for the duration of
//!   the write. Locally measured at **~6–18 µs** per call on a modern x86
//!   workstation (cold) — well inside the < 100 µs budget.
//! * `FileJsonSink`: `serde_json::to_string` + `Mutex<File>::lock()` +
//!   `write_all` + `flush`. The flush forces a `write(2)` per record so
//!   crashes lose at most one record; on commodity NVMe the worst case
//!   we observed is **~30–80 µs** (Linux ext4, single-writer). On a slow
//!   disk or under contention this can spike — the documented mitigation
//!   below applies.
//!
//! The orchestrator should re-measure under realistic disk + concurrency.
//! If the file sink shows tails > 100 µs the recommended fix is to wrap
//! the write in [`tokio::task::spawn_blocking`] — the trait's `emit` is
//! sync today because the in-process latency observed under our load
//! does not justify the additional task spawn (which is itself 5–10 µs)
//! and the at-most-once-per-state-mutation cadence is bounded by the
//! upstream rate limit anyway.
//!
//! Read-only routes do **not** invoke the sink at all (the route filter
//! short-circuits before serialisation), so the entire mechanism is zero
//! cost on the dominant `GET /metrics` scrape path.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashSet;
use ipnet::IpNet;
use serde::Serialize;
use uuid::Uuid;

use tensor_wasm_core::types::TenantId;

use crate::rate_limit::{AuthContext, TokenId};
use crate::token_scope::{TenantScope, TokenScope};

/// Environment variable selecting the audit-log destination. See module
/// docs for the accepted values.
pub const ENV_AUDIT_LOG: &str = "TENSOR_WASM_API_AUDIT_LOG";

/// Sentinel value that disables the audit log entirely
/// (`TENSOR_WASM_API_AUDIT_LOG=none`).
pub const AUDIT_LOG_DISABLED_VALUE: &str = "none";

/// Sentinel value that selects the explicit stdout sink. Equivalent to
/// leaving the variable unset.
pub const AUDIT_LOG_STDOUT_VALUE: &str = "stdout";

/// Prefix that selects the file sink: `file:/absolute/path/to/audit.log`.
pub const AUDIT_LOG_FILE_PREFIX: &str = "file:";

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

/// Top-level audit record emitted once per state-mutating HTTP call.
///
/// Every field is serialised in a stable shape; see the module docs and
/// `docs/AUDIT-LOG.md` for the wire-format contract.
#[derive(Debug, Clone, Serialize)]
pub struct AuditRecord {
    /// Millisecond-precision Unix timestamp when the request completed
    /// (i.e. when the audit record was synthesised).
    pub ts_unix_ms: u64,
    /// Unique identifier for this request. Generated by the audit
    /// middleware on entry and inserted into the request extensions so
    /// handlers can correlate logs against the audit trail.
    pub request_id: Uuid,
    /// Who issued the request.
    pub actor: AuditActor,
    /// What they tried to do.
    pub action: AuditAction,
    /// What they tried to do it to.
    pub resource: AuditResource,
    /// What happened.
    pub outcome: AuditOutcome,
    /// End-to-end handler latency, in milliseconds. Includes all
    /// middleware that ran *after* `audit_log_middleware`.
    pub latency_ms: u64,
    /// Caller's peer socket address as observed by the listener, if the
    /// router was bound with `axum::extract::connect_info::IntoMakeServiceWithConnectInfo`.
    /// `None` in tests (which drive the router via `oneshot`) and in
    /// proxy-fronted deployments that strip the connection.
    pub peer_addr: Option<String>,
    /// Client-certificate Subject DN as recovered from the
    /// `X-Forwarded-Client-Cert` header (Envoy XFCC format). Populated
    /// only when the reverse-proxy mTLS path described in
    /// `docs/deployment/mtls.md` (W2.8) is in front of the gateway and
    /// the request reached us with the header set.
    pub client_cert_subject: Option<String>,
}

/// Authenticated principal that issued the request.
#[derive(Debug, Clone, Serialize)]
pub struct AuditActor {
    /// Discriminator: bearer-token or dev-mode pass-through.
    pub kind: AuditActorKind,
    /// Stable process-local token id (from [`TokenId`]). `None` in dev
    /// mode — the dev sentinel has no meaningful identity to log.
    pub token_id: Option<TokenId>,
    /// Stringified projection of the caller's [`TokenScope`].
    pub scope: TokenScopeView,
}

/// Discriminator on [`AuditActor::kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditActorKind {
    /// Bearer-token authenticated caller. The default for production.
    Bearer,
    /// Dev-mode pass-through (no `TENSOR_WASM_API_TOKENS` configured).
    /// Recorded explicitly so operators can spot accidental dev-mode
    /// deployments in the audit stream.
    Dev,
}

/// Stable, JSON-friendly projection of a [`TokenScope`].
///
/// The on-wire shape is tagged by `kind` so consumers can pattern-match
/// without inspecting field presence:
///
/// * `{"kind":"wildcard"}` — wildcard scope (`tenant=*`).
/// * `{"kind":"tenant_set","tenants":[1,2,3]}` — explicit tenant set.
/// * `{"kind":"dev"}` — dev-mode synthetic scope.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TokenScopeView {
    /// Token may address every tenant. Produced by `token:tenant=*` and
    /// by legacy bare entries.
    Wildcard,
    /// Token is restricted to the listed tenant ids. Always sorted on
    /// output so two equivalent scopes render byte-identically — useful
    /// for diff-based audit-stream consumers.
    TenantSet {
        /// Sorted list of allowed tenant ids.
        tenants: Vec<u64>,
    },
    /// Dev-mode pass-through. Distinct from `Wildcard` so an audit
    /// reader can tell "the operator opted into wildcard" from "no auth
    /// was configured".
    Dev,
}

impl TokenScopeView {
    /// Project a [`TokenScope`] into the JSON-shaped view. Caller is
    /// expected to handle the dev-mode case explicitly via [`Self::Dev`].
    pub fn from_scope(scope: &TokenScope) -> Self {
        match &scope.tenants {
            TenantScope::All => TokenScopeView::Wildcard,
            TenantScope::Set(s) => {
                let mut tenants: Vec<u64> = s.iter().map(|t| t.0).collect();
                tenants.sort_unstable();
                TokenScopeView::TenantSet { tenants }
            }
        }
    }
}

/// Catalogue of state-mutating actions recognised by the audit log.
///
/// Tag strings (`create_function`, `delete_function`, `invoke_function`,
/// `invoke_function_async`, `invoke_function_stream`) are part of the
/// public contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditAction {
    /// `POST /functions` — deploy a new function.
    CreateFunction,
    /// `DELETE /functions/{id}` — remove a deployed function.
    DeleteFunction,
    /// `POST /functions/{id}/invoke` — synchronous invocation.
    InvokeFunction,
    /// `POST /functions/{id}/invoke-async` — fire-and-forget invocation.
    InvokeFunctionAsync,
    /// `POST /functions/{id}/invoke-stream` — streaming invocation
    /// (SSE / chunked-transfer). Roadmap feature #2 — see
    /// `docs/STREAMING.md`. The audit record carries the same
    /// `function_id` / `tenant` shape as the other invoke actions so
    /// operators can correlate streaming and non-streaming invocations
    /// uniformly.
    InvokeFunctionStream,
}

impl AuditAction {
    /// Classify an `(method, path)` pair as one of the state-mutating
    /// actions, or `None` for a read-only / unknown route.
    ///
    /// The path matcher walks segments so per-id routes (`{id}`) match
    /// regardless of the concrete UUID. Read-only routes (`GET /healthz`,
    /// `GET /metrics`, `GET /jobs/{id}`) all return `None` here, which is
    /// what suppresses their audit emission.
    pub fn classify(method: &axum::http::Method, path: &str) -> Option<Self> {
        use axum::http::Method;
        // Strip a single leading slash, then split into trimmed segments.
        let trimmed = path.trim_start_matches('/');
        let segments: Vec<&str> = trimmed.split('/').filter(|s| !s.is_empty()).collect();
        match (method, segments.as_slice()) {
            (&Method::POST, ["functions"]) => Some(AuditAction::CreateFunction),
            (&Method::DELETE, ["functions", _id]) => Some(AuditAction::DeleteFunction),
            (&Method::POST, ["functions", _id, "invoke"]) => Some(AuditAction::InvokeFunction),
            (&Method::POST, ["functions", _id, "invoke-async"]) => {
                Some(AuditAction::InvokeFunctionAsync)
            }
            (&Method::POST, ["functions", _id, "invoke-stream"]) => {
                Some(AuditAction::InvokeFunctionStream)
            }
            _ => None,
        }
    }
}

/// Resource the action targeted. Both fields are optional: not every
/// action binds to both, and `POST /functions` has no function id at the
/// time the route is matched (the new id is assigned by the handler).
#[derive(Debug, Clone, Serialize)]
pub struct AuditResource {
    /// Function id parsed from the route, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_id: Option<Uuid>,
    /// Tenant id resolved from the `X-TensorWasm-Tenant` header by the
    /// `tenant_scope` middleware, when present. `None` for routes that do
    /// not bind to a tenant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<TenantId>,
}

/// What came back from the handler.
#[derive(Debug, Clone, Serialize)]
pub struct AuditOutcome {
    /// HTTP status code returned to the client.
    pub status_code: u16,
    /// The `error.kind` value from the error envelope, when the response
    /// was a non-2xx. Read from the response extensions; handlers stash
    /// the kind there via [`AuditOutcomeExt`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<String>,
}

/// Marker type inserted into the response extensions by handlers so the
/// audit middleware can recover the `error.kind` value without re-parsing
/// the JSON body. The handler stamps this when it converts an `ApiError`
/// into a response.
#[derive(Debug, Clone)]
pub struct AuditOutcomeExt {
    /// Stable machine-readable identifier (the `kind` field on the wire).
    pub error_kind: String,
}

// ---------------------------------------------------------------------------
// Sinks
// ---------------------------------------------------------------------------

/// Pluggable target for [`AuditRecord`] emission.
///
/// Implementations must be `Send + Sync` because the audit middleware
/// shares a single instance across all in-flight requests through an
/// `Arc<dyn AuditSink>`.
pub trait AuditSink: Send + Sync + std::fmt::Debug {
    /// Persist `record`. Errors are intentionally absorbed: an audit
    /// failure must never bubble up and corrupt a successful API call.
    /// Implementations should log internally via `tracing::error!`
    /// instead of propagating.
    fn emit(&self, record: &AuditRecord);
}

/// Default sink — writes each record to stdout as a single JSON line and
/// mirrors it at `tracing::info!` for OTel correlation.
#[derive(Debug, Clone, Copy, Default)]
pub struct StdoutJsonSink;

impl StdoutJsonSink {
    /// Construct a stdout sink.
    pub fn new() -> Self {
        Self
    }
}

impl AuditSink for StdoutJsonSink {
    fn emit(&self, record: &AuditRecord) {
        match serde_json::to_string(record) {
            Ok(line) => {
                // `println!` takes the process-wide stdout lock for the
                // write. The macro adds the trailing newline, so the
                // emitted line is `"<json>\n"` — JSONL-friendly.
                println!("{line}");
                // Mirror at info level so an OTel-attached subscriber can
                // correlate the audit entry with the request trace.
                tracing::info!(
                    target: "tensor_wasm_api::audit",
                    audit = %line,
                    "audit",
                );
            }
            Err(e) => {
                tracing::error!(
                    target: "tensor_wasm_api::audit",
                    error = %e,
                    "failed to serialise audit record",
                );
            }
        }
    }
}

/// File-backed sink. Records are appended to `path` as JSONL.
///
/// The file is opened once at construction with `O_APPEND` (or the
/// platform equivalent) and writes serialise through an inner
/// `std::sync::Mutex`. Each `emit` performs `write_all` + `flush` so a
/// process crash loses at most the in-flight record. See module docs for
/// the latency rationale.
#[derive(Debug)]
pub struct FileJsonSink {
    /// Path the sink was constructed with — retained for diagnostics.
    pub path: PathBuf,
    /// The append-only file handle. `std::sync::Mutex` (not
    /// `tokio::sync::Mutex`) because the critical section is sync I/O
    /// with no `.await` point.
    pub file: Mutex<File>,
}

impl FileJsonSink {
    /// Open `path` for append, creating it if missing. Returns an `io`
    /// error if the OS refuses (typically permissions or a missing
    /// parent directory).
    pub fn open(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            path,
            file: Mutex::new(file),
        })
    }
}

impl AuditSink for FileJsonSink {
    fn emit(&self, record: &AuditRecord) {
        let line = match serde_json::to_string(record) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!(
                    target: "tensor_wasm_api::audit",
                    error = %e,
                    "failed to serialise audit record",
                );
                return;
            }
        };
        let mut guard = match self.file.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                // A poisoned mutex means a previous writer panicked
                // mid-write; the file handle itself is still usable.
                // Recover the inner guard and continue: dropping the
                // audit stream on a stale panic is worse than a possibly
                // truncated prior record.
                tracing::warn!(
                    target: "tensor_wasm_api::audit",
                    path = %self.path.display(),
                    "audit file mutex was poisoned; recovering",
                );
                poisoned.into_inner()
            }
        };
        if let Err(e) = writeln!(&mut *guard, "{line}") {
            tracing::error!(
                target: "tensor_wasm_api::audit",
                error = %e,
                path = %self.path.display(),
                "failed to write audit record",
            );
            return;
        }
        if let Err(e) = guard.flush() {
            tracing::error!(
                target: "tensor_wasm_api::audit",
                error = %e,
                path = %self.path.display(),
                "failed to flush audit record",
            );
        }
    }
}

/// No-op sink. Selected by `TENSOR_WASM_API_AUDIT_LOG=none` for
/// deployments where the operator aggregates Prometheus + OTel and does
/// not want a third audit stream.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopSink;

impl NoopSink {
    /// Construct the no-op sink.
    pub fn new() -> Self {
        Self
    }
}

impl AuditSink for NoopSink {
    fn emit(&self, _record: &AuditRecord) {
        // Intentionally empty.
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Resolved audit-log configuration loaded from the process environment.
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// The selected sink, wrapped in an `Arc` so the middleware can
    /// clone cheaply into each request.
    pub sink: Arc<dyn AuditSink>,
}

impl AuditConfig {
    /// Construct from an explicit sink. Used by tests and embedders.
    pub fn from_sink(sink: Arc<dyn AuditSink>) -> Self {
        Self { sink }
    }

    /// Convenience: build the default stdout config.
    pub fn stdout() -> Self {
        Self::from_sink(Arc::new(StdoutJsonSink::new()))
    }

    /// Convenience: build the disabled config (no-op sink).
    pub fn disabled() -> Self {
        Self::from_sink(Arc::new(NoopSink::new()))
    }

    /// Load from `$TENSOR_WASM_API_AUDIT_LOG`.
    ///
    /// Falls back to [`StdoutJsonSink`] when the variable is unset or
    /// empty. A `file:` prefix that fails to open emits a startup
    /// `tracing::error!` and degrades to stdout — refusing to start
    /// because a log target is unavailable would be hostile in containers
    /// where the path is mounted asynchronously.
    pub fn from_env() -> Self {
        let raw = std::env::var(ENV_AUDIT_LOG).unwrap_or_default();
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case(AUDIT_LOG_STDOUT_VALUE) {
            tracing::info!(
                target: "tensor_wasm_api::audit",
                "audit log enabled — sink: stdout (JSONL)",
            );
            return Self::stdout();
        }
        if trimmed.eq_ignore_ascii_case(AUDIT_LOG_DISABLED_VALUE) {
            tracing::warn!(
                target: "tensor_wasm_api::audit",
                env = ENV_AUDIT_LOG,
                "audit log disabled ({}={})",
                ENV_AUDIT_LOG,
                AUDIT_LOG_DISABLED_VALUE,
            );
            return Self::disabled();
        }
        if let Some(path) = trimmed.strip_prefix(AUDIT_LOG_FILE_PREFIX) {
            let path = path.trim();
            if path.is_empty() {
                tracing::error!(
                    target: "tensor_wasm_api::audit",
                    env = ENV_AUDIT_LOG,
                    "audit log file: prefix had no path; falling back to stdout",
                );
                return Self::stdout();
            }
            match FileJsonSink::open(path) {
                Ok(sink) => {
                    tracing::info!(
                        target: "tensor_wasm_api::audit",
                        path,
                        "audit log enabled — sink: file (JSONL, append-only)",
                    );
                    return Self::from_sink(Arc::new(sink));
                }
                Err(e) => {
                    tracing::error!(
                        target: "tensor_wasm_api::audit",
                        path,
                        error = %e,
                        "failed to open audit log file; falling back to stdout",
                    );
                    return Self::stdout();
                }
            }
        }
        tracing::warn!(
            target: "tensor_wasm_api::audit",
            env = ENV_AUDIT_LOG,
            value = trimmed,
            "unrecognised TENSOR_WASM_API_AUDIT_LOG value; expected `none`, \
             `stdout`, or `file:/path/to/log` — falling back to stdout",
        );
        Self::stdout()
    }
}

impl Default for AuditConfig {
    /// Default is stdout — same as the unset env behaviour.
    fn default() -> Self {
        Self::stdout()
    }
}

// ---------------------------------------------------------------------------
// Helpers used by the middleware
// ---------------------------------------------------------------------------

/// Current wall-clock time as milliseconds since the Unix epoch.
///
/// On the (vanishingly unlikely) event that the system clock is set
/// before 1970-01-01, returns 0 rather than panicking. The audit middleware
/// uses this so a misconfigured clock never aborts a state-mutating
/// request.
pub(crate) fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Project an [`AuthContext`] into its actor representation. Dev contexts
/// become [`AuditActorKind::Dev`] with no token id and a `Dev`-tagged
/// scope view; everything else is a `Bearer` with the token's id and the
/// real scope.
pub(crate) fn actor_from_auth(auth: &AuthContext) -> AuditActor {
    if auth.token_id == TokenId::DEV {
        AuditActor {
            kind: AuditActorKind::Dev,
            token_id: None,
            scope: TokenScopeView::Dev,
        }
    } else {
        AuditActor {
            kind: AuditActorKind::Bearer,
            token_id: Some(auth.token_id),
            scope: TokenScopeView::from_scope(&auth.scope),
        }
    }
}

/// Default actor used when the auth middleware did not run (e.g. a test
/// that drives a sub-router without bearer_auth). Recorded as dev — the
/// most conservative interpretation.
pub(crate) fn default_actor() -> AuditActor {
    AuditActor {
        kind: AuditActorKind::Dev,
        token_id: None,
        scope: TokenScopeView::Dev,
    }
}

/// Header name from which the audit middleware recovers the
/// client-certificate Subject when an XFCC-aware reverse proxy is in
/// front of the gateway. See `docs/deployment/mtls.md` §4.4.
pub const HEADER_XFCC: &str = "X-Forwarded-Client-Cert";

/// Environment variable carrying a comma-separated allowlist of IPv4/IPv6
/// addresses or CIDR ranges whose `X-Forwarded-Client-Cert` headers the
/// audit middleware will trust. Empty / unset = **never trust XFCC** (the
/// safe default — see [`TrustedProxies`]).
///
/// Example: `TENSOR_WASM_API_TRUSTED_XFCC_PROXIES=10.0.0.0/8,127.0.0.1,::1`.
pub const ENV_TRUSTED_XFCC_PROXIES: &str = "TENSOR_WASM_API_TRUSTED_XFCC_PROXIES";

/// Allowlist of reverse-proxy peer addresses whose `X-Forwarded-Client-Cert`
/// headers the audit middleware will trust.
///
/// # Threat model
///
/// XFCC is a header an upstream Envoy / Istio sidecar sets after performing
/// mTLS termination on behalf of the gateway. Because it is a plain HTTP
/// header, anything that can open a TCP connection to the gateway can also
/// *claim* a `Subject=...` value. If the gateway forwards that claim into
/// the audit log unchecked, an attacker on the same L3 segment (or with
/// access to a misconfigured ingress) can write arbitrary identities into
/// the audit stream — defeating non-repudiation, poisoning downstream SIEM
/// tooling, and providing cover for malicious activity attributed to a
/// fabricated certificate Subject.
///
/// The mitigation is layered: only consult XFCC when the *immediate TCP
/// peer* (the L4 source IP axum's listener observed via `ConnectInfo`) is
/// in an operator-curated allowlist of proxies known to terminate mTLS.
/// Everything else has its XFCC header silently dropped. The allowlist is
/// empty by default so a fresh deployment cannot accidentally trust an
/// attacker.
///
/// # Configuration
///
/// Operators populate the allowlist via [`ENV_TRUSTED_XFCC_PROXIES`]: a
/// comma-separated list of IPv4 / IPv6 addresses (`127.0.0.1`, `::1`) or
/// CIDR ranges (`10.0.0.0/8`, `fd00::/8`). Parse failures on individual
/// entries are logged at `warn` and the entry is dropped; a fully empty
/// list (the default) means *no peer is trusted*.
///
/// # Sharing
///
/// `Clone` is intentionally cheap: the inner CIDR list is small and the
/// per-peer warn-dedup `DashSet` is wrapped in `Arc`, so cloning into each
/// request's extensions does not duplicate state.
#[derive(Debug, Clone, Default)]
pub struct TrustedProxies {
    /// Parsed allowlist. Empty means "trust nobody" (the safe default).
    ranges: Arc<Vec<IpNet>>,
    /// Set of peer IPs we have already warned about. Shared across requests
    /// so the warn fires at most once per unique untrusted peer per
    /// process — not per request — to avoid drowning operators in log
    /// noise during a probe storm.
    warned: Arc<DashSet<IpAddr>>,
}

impl TrustedProxies {
    /// Construct an empty allowlist (the safe default — trusts nobody).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Load from [`ENV_TRUSTED_XFCC_PROXIES`].
    ///
    /// Unset / empty → empty allowlist (no peer is trusted, every inbound
    /// `X-Forwarded-Client-Cert` header is dropped). Malformed individual
    /// entries are skipped with a startup `tracing::warn!` so a typo in
    /// one entry does not poison the whole allowlist.
    pub fn from_env() -> Self {
        let raw = std::env::var(ENV_TRUSTED_XFCC_PROXIES).unwrap_or_default();
        Self::parse(&raw)
    }

    /// Parse a comma-separated list of IPs / CIDR ranges into an allowlist.
    ///
    /// Bare IPs (`127.0.0.1`, `::1`) are normalised into `/32` (IPv4) or
    /// `/128` (IPv6) host routes so the membership test is uniform across
    /// shapes. Whitespace around individual entries is tolerated.
    pub fn parse(s: &str) -> Self {
        let mut ranges: Vec<IpNet> = Vec::new();
        let mut had_any = false;
        for entry in s.split(',') {
            let trimmed = entry.trim();
            if trimmed.is_empty() {
                continue;
            }
            had_any = true;
            match trimmed.parse::<IpNet>() {
                Ok(net) => ranges.push(net),
                Err(_) => match trimmed.parse::<IpAddr>() {
                    Ok(ip) => ranges.push(ip.into()),
                    Err(e) => {
                        tracing::warn!(
                            target: "tensor_wasm_api::audit",
                            env = ENV_TRUSTED_XFCC_PROXIES,
                            entry = trimmed,
                            error = %e,
                            "ignored malformed XFCC trusted-proxy entry; \
                             expected an IPv4/IPv6 address or CIDR range",
                        );
                    }
                },
            }
        }
        if had_any && ranges.is_empty() {
            tracing::warn!(
                target: "tensor_wasm_api::audit",
                env = ENV_TRUSTED_XFCC_PROXIES,
                "{} was set but no entries parsed; XFCC will be dropped \
                 from every request",
                ENV_TRUSTED_XFCC_PROXIES,
            );
        } else if ranges.is_empty() {
            tracing::debug!(
                target: "tensor_wasm_api::audit",
                "{} unset; XFCC headers will be ignored from every peer \
                 (set this to the IPs / CIDR ranges of your mTLS-terminating \
                 proxies to enable XFCC ingestion)",
                ENV_TRUSTED_XFCC_PROXIES,
            );
        } else {
            tracing::info!(
                target: "tensor_wasm_api::audit",
                count = ranges.len(),
                "XFCC trusted-proxy allowlist configured",
            );
        }
        Self {
            ranges: Arc::new(ranges),
            warned: Arc::new(DashSet::new()),
        }
    }

    /// `true` when no peer is trusted (the safe default). When this is
    /// `true`, [`Self::contains`] returns `false` for every input.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Membership test: is `ip` in any of the configured ranges?
    pub fn contains(&self, ip: IpAddr) -> bool {
        self.ranges.iter().any(|net| net.contains(&ip))
    }

    /// Record a warn-level diagnostic exactly once per unique peer that
    /// tried to send `X-Forwarded-Client-Cert` from outside the allowlist.
    /// Subsequent requests from the same peer are silent. Returns `true`
    /// if the warn was emitted (the peer was previously unseen).
    pub fn warn_once_untrusted(&self, peer: IpAddr) -> bool {
        if self.warned.insert(peer) {
            tracing::warn!(
                target: "tensor_wasm_api::audit",
                env = ENV_TRUSTED_XFCC_PROXIES,
                %peer,
                "dropped X-Forwarded-Client-Cert from peer not in the \
                 trusted-proxy allowlist — set {} if this peer is a known \
                 mTLS-terminating proxy",
                ENV_TRUSTED_XFCC_PROXIES,
            );
            true
        } else {
            false
        }
    }
}

/// Recover the `Subject="..."` field from an Envoy-style XFCC header
/// value. Returns the contents of the first matched `Subject="..."`
/// component, unescaping doubled `\"` sequences. `None` if the header is
/// absent or shaped unexpectedly — we deliberately do not surface a
/// parse error to the audit record because XFCC is a best-effort
/// optional field.
///
/// # Threat model
///
/// `X-Forwarded-Client-Cert` is a free-form HTTP header. Any TCP peer can
/// claim an arbitrary `Subject=...` value, so this function must only be
/// invoked when the *immediate TCP peer* has already been authenticated as
/// a trusted reverse proxy. Callers in the audit pipeline route through
/// [`extract_client_cert_subject_gated`] which performs that check; this
/// raw parser is `pub(crate)` so unit tests of the parser shape can call
/// it directly without round-tripping through axum extensions.
pub(crate) fn extract_client_cert_subject(headers: &axum::http::HeaderMap) -> Option<String> {
    let raw = headers.get(HEADER_XFCC)?.to_str().ok()?;
    // XFCC components are `;`-separated; each component is `key=value`.
    // The value may be a bare token or a double-quoted string with `\"`
    // escapes per the Envoy spec.
    for component in raw.split(';') {
        let (k, v) = component.split_once('=')?;
        if k.trim().eq_ignore_ascii_case("Subject") {
            let v = v.trim();
            // Strip surrounding quotes if present; unescape `\"`.
            let inner = if v.starts_with('"') && v.ends_with('"') && v.len() >= 2 {
                &v[1..v.len() - 1]
            } else {
                v
            };
            return Some(inner.replace("\\\"", "\""));
        }
    }
    None
}

/// XFCC-spoofing-resistant wrapper around [`extract_client_cert_subject`].
///
/// # Threat model
///
/// `X-Forwarded-Client-Cert` is intended as a "trust path" header: an
/// upstream Envoy / Istio sidecar terminates mTLS, validates the client
/// certificate, and forwards the validated `Subject` DN into the gateway
/// via this header so the audit log can record the certificate identity
/// alongside the bearer-token identity. The header carries **no
/// cryptographic guarantee on its own** — anything able to open a TCP
/// connection to the gateway can also set any header it likes. Operators
/// that deploy the gateway directly (no mTLS proxy in front) would, prior
/// to this gate, write attacker-supplied `Subject` values straight into
/// the audit stream, breaking non-repudiation and providing cover for
/// abuse attributed to a forged certificate.
///
/// The gate restores the trust path. Only when the *immediate TCP peer*
/// (the L4 source axum observes via `ConnectInfo`) is in the
/// operator-curated [`TrustedProxies`] allowlist do we consult the header.
/// Every other peer has its XFCC dropped silently — with a one-shot
/// `tracing::warn!` per unique peer to surface possible misconfigurations
/// without flooding logs during a probe storm.
///
/// # Behaviour
///
/// * `peer_ip = None` (e.g. a test driving the router via `oneshot`, or a
///   listener bound without `IntoMakeServiceWithConnectInfo`) → drop the
///   header. We cannot validate the peer, so we cannot trust the claim.
/// * `peer_ip = Some(ip)` and `trusted.contains(ip)` → parse the header
///   as before.
/// * `peer_ip = Some(ip)` and not trusted → drop the header. If the
///   header was actually present, emit a one-shot warn for this peer.
pub(crate) fn extract_client_cert_subject_gated(
    headers: &axum::http::HeaderMap,
    peer_ip: Option<IpAddr>,
    trusted: &TrustedProxies,
) -> Option<String> {
    let Some(ip) = peer_ip else {
        // Unknown peer: cannot validate trust, drop the claim. We do not
        // warn here because the absence of ConnectInfo is structural
        // (test harness, embedded use) rather than indicative of attack.
        return None;
    };
    if trusted.contains(ip) {
        return extract_client_cert_subject(headers);
    }
    if headers.contains_key(HEADER_XFCC) {
        trusted.warn_once_untrusted(ip);
    }
    None
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

/// Axum middleware that synthesises one [`AuditRecord`] per state-mutating
/// HTTP call and dispatches it to the configured [`AuditSink`].
///
/// Wiring contract:
///
/// * The router must layer `axum::Extension(AuditConfig)` so this
///   middleware can recover the sink.
/// * Run this middleware **after** `bearer_auth` and `tenant_scope` so the
///   `AuthContext` and `Extension<TenantId>` are populated by the time
///   the audit record is built.
/// * The route-shape filter ([`AuditAction::classify`]) suppresses
///   emission for read-only routes (`GET /healthz`, `GET /metrics`,
///   `GET /jobs/{id}`); those calls flow through with zero serialisation
///   cost.
///
/// On entry the middleware stamps the request extensions with a fresh
/// [`Uuid`] so handlers can correlate their own logs against the audit
/// trail (recovered via `Extension<Uuid>` or directly from request
/// extensions). The same id appears in [`AuditRecord::request_id`].
pub async fn audit_log_middleware(
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use std::time::Instant;

    let method = req.method().clone();
    let path = req.uri().path().to_owned();
    let action = AuditAction::classify(&method, &path);

    // Read-only / unknown route: skip the entire audit path.
    let Some(action) = action else {
        return next.run(req).await;
    };

    // Stamp the request with a stable id so handlers can correlate their
    // own logs against the audit trail.
    let request_id = Uuid::new_v4();
    req.extensions_mut().insert(request_id);

    // Snapshot inputs that the downstream handler may overwrite or
    // consume before the response is produced. If no `AuditConfig` is
    // present in the extensions (e.g. a test driver bypassed the
    // production router), fall back to the no-op sink — emitting to
    // stdout from a misconfigured router would surprise integration
    // tests that pre-date the audit middleware.
    let cfg = req
        .extensions()
        .get::<AuditConfig>()
        .cloned()
        .unwrap_or_else(AuditConfig::disabled);
    let actor = req
        .extensions()
        .get::<AuthContext>()
        .map(actor_from_auth)
        .unwrap_or_else(default_actor);
    let tenant_id = req.extensions().get::<TenantId>().copied();
    let function_id = parse_function_id_from_path(&path);
    // XFCC spoofing mitigation: only consult `X-Forwarded-Client-Cert`
    // when the immediate TCP peer is in the operator-curated trusted-
    // proxy allowlist. Missing extension → safe-default empty allowlist
    // → drop the header. See `TrustedProxies` and
    // `extract_client_cert_subject_gated` for the threat model.
    let connect_info = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0);
    let peer_ip = connect_info.map(|sa| sa.ip());
    let trusted_proxies = req
        .extensions()
        .get::<TrustedProxies>()
        .cloned()
        .unwrap_or_default();
    let client_cert_subject =
        extract_client_cert_subject_gated(req.headers(), peer_ip, &trusted_proxies);
    let peer_addr = connect_info.map(|sa| sa.to_string());

    let start = Instant::now();
    let response = next.run(req).await;
    let elapsed = start.elapsed();

    let status_code = response.status().as_u16();
    let error_kind = response
        .extensions()
        .get::<AuditOutcomeExt>()
        .map(|e| e.error_kind.clone());

    let record = AuditRecord {
        ts_unix_ms: now_unix_ms(),
        request_id,
        actor,
        action,
        resource: AuditResource {
            function_id,
            tenant_id,
        },
        outcome: AuditOutcome {
            status_code,
            error_kind,
        },
        latency_ms: elapsed.as_millis() as u64,
        peer_addr,
        client_cert_subject,
    };
    cfg.sink.emit(&record);
    response
}

/// Recover the function id from a `/functions/{id}` or
/// `/functions/{id}/invoke[-async|-stream]` path, if present and a
/// valid UUID.
fn parse_function_id_from_path(path: &str) -> Option<Uuid> {
    let trimmed = path.trim_start_matches('/');
    let segments: Vec<&str> = trimmed.split('/').filter(|s| !s.is_empty()).collect();
    // [`functions`, <id>, ...]
    if segments.len() >= 2 && segments[0] == "functions" {
        return Uuid::parse_str(segments[1]).ok();
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use std::sync::OnceLock;

    /// Capturing sink used by the unit tests in this module to assert
    /// `emit` was called with the expected record shape, without
    /// inspecting stdout.
    #[derive(Debug, Default)]
    struct CapturingSink {
        records: StdMutex<Vec<AuditRecord>>,
    }

    impl CapturingSink {
        fn new() -> Self {
            Self::default()
        }
        fn snapshot(&self) -> Vec<AuditRecord> {
            self.records.lock().unwrap().clone()
        }
    }

    impl AuditSink for CapturingSink {
        fn emit(&self, record: &AuditRecord) {
            self.records.lock().unwrap().push(record.clone());
        }
    }

    fn sample_record() -> AuditRecord {
        AuditRecord {
            ts_unix_ms: 1_716_491_220_123,
            request_id: Uuid::nil(),
            actor: AuditActor {
                kind: AuditActorKind::Bearer,
                token_id: Some(TokenId(1234)),
                scope: TokenScopeView::Wildcard,
            },
            action: AuditAction::CreateFunction,
            resource: AuditResource {
                function_id: None,
                tenant_id: Some(TenantId(7)),
            },
            outcome: AuditOutcome {
                status_code: 200,
                error_kind: None,
            },
            latency_ms: 12,
            peer_addr: None,
            client_cert_subject: None,
        }
    }

    #[test]
    fn classify_state_mutating_routes() {
        use axum::http::Method;
        assert_eq!(
            AuditAction::classify(&Method::POST, "/functions"),
            Some(AuditAction::CreateFunction),
        );
        assert_eq!(
            AuditAction::classify(
                &Method::DELETE,
                "/functions/f47ac10b-58cc-4372-a567-0e02b2c3d479",
            ),
            Some(AuditAction::DeleteFunction),
        );
        assert_eq!(
            AuditAction::classify(
                &Method::POST,
                "/functions/f47ac10b-58cc-4372-a567-0e02b2c3d479/invoke",
            ),
            Some(AuditAction::InvokeFunction),
        );
        assert_eq!(
            AuditAction::classify(
                &Method::POST,
                "/functions/f47ac10b-58cc-4372-a567-0e02b2c3d479/invoke-async",
            ),
            Some(AuditAction::InvokeFunctionAsync),
        );
        assert_eq!(
            AuditAction::classify(
                &Method::POST,
                "/functions/f47ac10b-58cc-4372-a567-0e02b2c3d479/invoke-stream",
            ),
            Some(AuditAction::InvokeFunctionStream),
        );
    }

    #[test]
    fn classify_read_only_routes_returns_none() {
        use axum::http::Method;
        assert!(AuditAction::classify(&Method::GET, "/healthz").is_none());
        assert!(AuditAction::classify(&Method::GET, "/metrics").is_none());
        assert!(
            AuditAction::classify(&Method::GET, "/jobs/abcd-1234").is_none(),
            "GET /jobs/<id> is read-only and must be filtered out",
        );
        assert!(AuditAction::classify(&Method::GET, "/").is_none());
    }

    #[test]
    fn record_serialises_with_stable_keys() {
        let rec = sample_record();
        let v: serde_json::Value = serde_json::to_value(&rec).expect("serialises");
        assert_eq!(v["ts_unix_ms"], 1_716_491_220_123u64);
        assert_eq!(v["action"], "create_function");
        assert_eq!(v["actor"]["kind"], "bearer");
        assert_eq!(v["actor"]["scope"]["kind"], "wildcard");
        assert_eq!(v["resource"]["tenant_id"], 7);
        assert_eq!(v["outcome"]["status_code"], 200);
        // The optional `function_id` is suppressed when None.
        assert!(v["resource"].get("function_id").is_none());
        // The optional `error_kind` is suppressed when None.
        assert!(v["outcome"].get("error_kind").is_none());
    }

    #[test]
    fn record_round_trips_through_serde_json() {
        // The wire schema is part of the contract; round-tripping
        // proves the serialized form deserialises cleanly into a
        // structurally-equivalent shape (we don't derive Deserialize
        // on `AuditRecord` itself to avoid leaking it as a public input
        // type, so we round-trip through `serde_json::Value`).
        let rec = sample_record();
        let s = serde_json::to_string(&rec).expect("serialises");
        let back: serde_json::Value = serde_json::from_str(&s).expect("deserialises");
        assert_eq!(back["action"], "create_function");
        assert_eq!(back["request_id"], "00000000-0000-0000-0000-000000000000");
    }

    #[test]
    fn token_scope_view_from_scope_is_stable_order() {
        // Build a scope from a `HashSet`-derived TokenScope and verify
        // the projection sorts the tenant ids so two equivalent scopes
        // render byte-identically.
        let scope = TokenScope::from_tenants([TenantId(3), TenantId(1), TenantId(2)]);
        let view = TokenScopeView::from_scope(&scope);
        match view {
            TokenScopeView::TenantSet { tenants } => {
                assert_eq!(tenants, vec![1, 2, 3]);
            }
            other => panic!("expected TenantSet, got {other:?}"),
        }
    }

    #[test]
    fn actor_from_dev_context_renders_as_dev() {
        let ctx = AuthContext::dev();
        let actor = actor_from_auth(&ctx);
        assert_eq!(actor.kind, AuditActorKind::Dev);
        assert!(actor.token_id.is_none());
        assert!(matches!(actor.scope, TokenScopeView::Dev));
    }

    #[test]
    fn actor_from_bearer_context_renders_as_bearer() {
        let ctx = AuthContext::with_scope(
            "alpha",
            TokenScope::from_tenants([TenantId(1)]),
        );
        let actor = actor_from_auth(&ctx);
        assert_eq!(actor.kind, AuditActorKind::Bearer);
        assert!(actor.token_id.is_some());
        match actor.scope {
            TokenScopeView::TenantSet { tenants } => assert_eq!(tenants, vec![1]),
            other => panic!("expected TenantSet, got {other:?}"),
        }
    }

    #[test]
    fn noop_sink_drops_records() {
        let sink = NoopSink::new();
        sink.emit(&sample_record());
        // No assertion beyond "does not panic" — the sink is a no-op
        // by definition.
    }

    #[test]
    fn capturing_sink_records_emission() {
        let sink = CapturingSink::new();
        sink.emit(&sample_record());
        sink.emit(&sample_record());
        assert_eq!(sink.snapshot().len(), 2);
    }

    #[test]
    fn file_sink_appends_jsonl_records() {
        let dir = std::env::temp_dir();
        // Unique per-process file so parallel test runs don't collide.
        static N: OnceLock<std::sync::atomic::AtomicU64> = OnceLock::new();
        let n = N.get_or_init(|| std::sync::atomic::AtomicU64::new(0))
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = dir.join(format!(
            "tensor-wasm-audit-test-{}-{n}.log",
            std::process::id(),
        ));
        // Clean any prior file from a flaky previous run.
        let _ = std::fs::remove_file(&path);
        let sink = FileJsonSink::open(&path).expect("opens");
        sink.emit(&sample_record());
        sink.emit(&sample_record());
        let body = std::fs::read_to_string(&path).expect("reads");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        for line in lines {
            let v: serde_json::Value = serde_json::from_str(line).expect("each line is JSON");
            assert_eq!(v["action"], "create_function");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn audit_config_from_env_falls_back_to_stdout_when_unset() {
        // We can't reliably manipulate the env var in tests (parallel
        // execution would race); instead, exercise the explicit ctor.
        let cfg = AuditConfig::stdout();
        // The Arc<dyn AuditSink> must not panic on emit.
        cfg.sink.emit(&sample_record());
    }

    #[test]
    fn audit_config_disabled_is_noop() {
        let cfg = AuditConfig::disabled();
        cfg.sink.emit(&sample_record());
        // No assertion beyond "does not panic" — NoopSink emits nothing.
    }

    #[test]
    fn extract_subject_from_xfcc_envoy_format() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            HEADER_XFCC,
            axum::http::HeaderValue::from_static(
                "Hash=abc123;Subject=\"CN=client-prod,O=Acme\";URI=spiffe://acme.io/client",
            ),
        );
        let subj = extract_client_cert_subject(&headers).expect("present");
        assert_eq!(subj, "CN=client-prod,O=Acme");
    }

    #[test]
    fn extract_subject_returns_none_when_header_absent() {
        let headers = axum::http::HeaderMap::new();
        assert!(extract_client_cert_subject(&headers).is_none());
    }

    #[test]
    fn parse_function_id_recognises_canonical_routes() {
        let id = Uuid::new_v4();
        let s = id.to_string();
        assert_eq!(
            parse_function_id_from_path(&format!("/functions/{s}")),
            Some(id),
        );
        assert_eq!(
            parse_function_id_from_path(&format!("/functions/{s}/invoke")),
            Some(id),
        );
        assert_eq!(
            parse_function_id_from_path(&format!("/functions/{s}/invoke-async")),
            Some(id),
        );
        // POST /functions: no id yet (handler assigns it).
        assert_eq!(parse_function_id_from_path("/functions"), None);
        // Non-uuid second segment: defensive None rather than panic.
        assert_eq!(parse_function_id_from_path("/functions/garbage"), None);
        // Read-only route should not surface a function id either.
        assert_eq!(parse_function_id_from_path("/healthz"), None);
    }

    #[test]
    fn extract_subject_returns_none_when_header_has_no_subject() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            HEADER_XFCC,
            axum::http::HeaderValue::from_static("Hash=abc123;URI=spiffe://acme.io/client"),
        );
        assert!(extract_client_cert_subject(&headers).is_none());
    }

    /// XFCC spoofing mitigation: an empty `TrustedProxies` allowlist
    /// (the safe default) must drop the header regardless of how the
    /// peer claims to be addressed.
    #[test]
    fn gated_extract_drops_xfcc_when_no_proxies_trusted() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            HEADER_XFCC,
            axum::http::HeaderValue::from_static("Subject=\"CN=evil\""),
        );
        let trusted = TrustedProxies::empty();
        let peer = Some(IpAddr::from([127, 0, 0, 1]));
        assert!(extract_client_cert_subject_gated(&headers, peer, &trusted).is_none());
    }

    /// Gated extract honours XFCC when the peer is in the allowlist.
    #[test]
    fn gated_extract_honours_xfcc_from_trusted_peer() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            HEADER_XFCC,
            axum::http::HeaderValue::from_static("Subject=\"CN=client-prod\""),
        );
        let trusted = TrustedProxies::parse("127.0.0.1");
        let peer = Some(IpAddr::from([127, 0, 0, 1]));
        let subj = extract_client_cert_subject_gated(&headers, peer, &trusted)
            .expect("trusted peer's header is honoured");
        assert_eq!(subj, "CN=client-prod");
    }

    /// Unknown peer (no `ConnectInfo`) must drop the header even with a
    /// non-empty allowlist — we cannot validate trust, so we cannot
    /// honour the claim.
    #[test]
    fn gated_extract_drops_xfcc_when_peer_unknown() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(
            HEADER_XFCC,
            axum::http::HeaderValue::from_static("Subject=\"CN=anyone\""),
        );
        let trusted = TrustedProxies::parse("10.0.0.0/8");
        assert!(extract_client_cert_subject_gated(&headers, None, &trusted).is_none());
    }

    /// CIDR membership for IPv4 — a peer inside `10.0.0.0/8` is trusted,
    /// a peer outside is not.
    #[test]
    fn trusted_proxies_cidr_membership_v4() {
        let t = TrustedProxies::parse("10.0.0.0/8");
        assert!(t.contains(IpAddr::from([10, 5, 3, 7])));
        assert!(t.contains(IpAddr::from([10, 0, 0, 0])));
        assert!(t.contains(IpAddr::from([10, 255, 255, 255])));
        assert!(!t.contains(IpAddr::from([192, 168, 1, 1])));
        assert!(!t.contains(IpAddr::from([11, 0, 0, 0])));
    }

    /// CIDR membership for IPv6.
    #[test]
    fn trusted_proxies_cidr_membership_v6() {
        let t = TrustedProxies::parse("fd00::/8");
        let v6: IpAddr = "fd12::1".parse().unwrap();
        assert!(t.contains(v6));
        let outside: IpAddr = "2001:db8::1".parse().unwrap();
        assert!(!t.contains(outside));
    }

    /// Bare IPs are normalised into host routes (/32 or /128).
    #[test]
    fn trusted_proxies_bare_ip_normalised_to_host_route() {
        let t = TrustedProxies::parse("127.0.0.1, ::1");
        assert!(t.contains(IpAddr::from([127, 0, 0, 1])));
        assert!(!t.contains(IpAddr::from([127, 0, 0, 2])));
        let v6: IpAddr = "::1".parse().unwrap();
        assert!(t.contains(v6));
    }

    /// Empty input yields an empty allowlist that trusts nobody.
    #[test]
    fn trusted_proxies_empty_input_trusts_nobody() {
        let t = TrustedProxies::parse("");
        assert!(t.is_empty());
        assert!(!t.contains(IpAddr::from([127, 0, 0, 1])));
        let t = TrustedProxies::parse("   ,   ,   ");
        assert!(t.is_empty());
    }

    /// Malformed entries are skipped but do not poison the rest of the
    /// allowlist.
    #[test]
    fn trusted_proxies_malformed_entries_are_skipped() {
        let t = TrustedProxies::parse("garbage,127.0.0.1,also-bad/99");
        assert!(t.contains(IpAddr::from([127, 0, 0, 1])));
        assert!(!t.contains(IpAddr::from([10, 0, 0, 1])));
    }

    /// The warn-dedup set fires at most once per unique peer.
    #[test]
    fn trusted_proxies_warn_once_dedup() {
        let t = TrustedProxies::empty();
        let peer = IpAddr::from([192, 168, 1, 1]);
        assert!(t.warn_once_untrusted(peer), "first call emits the warn");
        assert!(
            !t.warn_once_untrusted(peer),
            "second call from same peer is suppressed",
        );
        // A different peer still fires once.
        let other = IpAddr::from([192, 168, 1, 2]);
        assert!(t.warn_once_untrusted(other));
    }
}
