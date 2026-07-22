// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Project-wide error types.
//!
//! [`TensorWasmError`] is the single, unified error returned by every public API in
//! the TensorWasm workspace. Inner error sources are preserved via `#[source]` chains;
//! `std::io::Error` is wired via `#[from]`. Wasmtime, cust, and serialisation
//! errors are converted at their crate boundaries and surface here as
//! `WasmTrap`, `WasmCompile`, `CudaError`, and `Serialization` with string
//! contexts.
//!
//! String-carrying variants store their message as `Box<str>` rather than
//! `String` — error values are typically constructed once and then propagated
//! up the call stack untouched, so we trade away the spare capacity an owned
//! `String` carries (and shrink each variant's footprint by one pointer-sized
//! word). Callers that have a `String` should pass it via `.into()`; callers
//! with a `&str` should use `.into()` likewise.
//!
//! # Display sanitisation
//!
//! The `CudaError`, `WasmTrap`, `WasmCompile`, and `Serialization` variants
//! carry inner strings produced by third-party crates (wasmtime, cust, serde,
//! ...). Those messages can leak host filesystem paths, raw pointer
//! addresses, or fragments of tenant-supplied source bytes — none of which is
//! safe to render into a response body or end-user log line. The
//! `Display` impls for those four variants therefore emit a stable, opaque
//! label only (e.g. `"cuda driver call failed"`). The inner string is still
//! reachable through `Debug` formatting for operator-side diagnostic logs and
//! through the `inner()` accessor for callers that need the raw text in a
//! trusted context. Variants whose fields are structured (`MemoryExhausted`,
//! `KernelTimeout`, `TenantIsolationViolation`) format normally — those
//! fields are integers and tenant-controlled identifiers, not opaque strings
//! from a vendor crate.

use std::io;

use thiserror::Error;

use crate::types::TenantId;

/// Returns a short, stable name for an [`std::io::ErrorKind`] suitable for
/// the sanitised `Display` of [`TensorWasmError::Io`].
///
/// Surfaces the variant identifier (`"NotFound"`, `"PermissionDenied"`,
/// ...) rather than the prose `Display` form (`"entity not found"`, ...) —
/// the variant name is what dashboards, alert rules, and operators grep
/// for. Returns a `&'static str` so the (very hot) error `Display` path
/// does not allocate; an explicit `match` replaces the prior
/// `format!("{:?}", ...)`. `std::io::ErrorKind` is `#[non_exhaustive]`, so
/// kinds not enumerated here (and any future additions) fall back to the
/// static `"Other"`.
fn io_kind_name(err: &io::Error) -> &'static str {
    match err.kind() {
        io::ErrorKind::NotFound => "NotFound",
        io::ErrorKind::PermissionDenied => "PermissionDenied",
        io::ErrorKind::ConnectionRefused => "ConnectionRefused",
        io::ErrorKind::ConnectionReset => "ConnectionReset",
        io::ErrorKind::ConnectionAborted => "ConnectionAborted",
        io::ErrorKind::NotConnected => "NotConnected",
        io::ErrorKind::AddrInUse => "AddrInUse",
        io::ErrorKind::AddrNotAvailable => "AddrNotAvailable",
        io::ErrorKind::BrokenPipe => "BrokenPipe",
        io::ErrorKind::AlreadyExists => "AlreadyExists",
        io::ErrorKind::WouldBlock => "WouldBlock",
        io::ErrorKind::InvalidInput => "InvalidInput",
        io::ErrorKind::InvalidData => "InvalidData",
        io::ErrorKind::TimedOut => "TimedOut",
        io::ErrorKind::WriteZero => "WriteZero",
        io::ErrorKind::Interrupted => "Interrupted",
        io::ErrorKind::Unsupported => "Unsupported",
        io::ErrorKind::UnexpectedEof => "UnexpectedEof",
        io::ErrorKind::OutOfMemory => "OutOfMemory",
        // `io::ErrorKind` is `#[non_exhaustive]`; map the unenumerated
        // remainder (including future additions) to a stable static label.
        _ => "Other",
    }
}

/// Returns `true` if an [`std::io::ErrorKind`] is plausibly transient and
/// the caller should retry the I/O operation. Used by
/// [`TensorWasmError::is_retryable`] to differentiate `Io(_)` errors that
/// should map to `503 Service Unavailable` (retryable) from those that
/// should map to `500 Internal Server Error` or a tenant-side 4xx (hard
/// miss).
///
/// The retryable set is intentionally narrow — every kind not listed
/// here defaults to non-retryable so a future addition to
/// `std::io::ErrorKind` does not silently flip a hard error into a
/// `503`. Kinds in the retryable set:
///
/// * [`io::ErrorKind::WouldBlock`] — non-blocking socket / fd would
///   block; the canonical retry-now kind.
/// * [`io::ErrorKind::TimedOut`] — operation deadline expired; the
///   underlying resource may recover.
/// * [`io::ErrorKind::Interrupted`] — signal-interrupted syscall
///   (`EINTR`); restarting almost always succeeds.
/// * [`io::ErrorKind::WriteZero`] — short write; a follow-up write may
///   drain the rest.
/// * [`io::ErrorKind::ConnectionReset`], [`io::ErrorKind::ConnectionAborted`],
///   [`io::ErrorKind::BrokenPipe`] — peer dropped the connection;
///   re-establishing and retrying may succeed.
fn is_retryable_io_kind(kind: io::ErrorKind) -> bool {
    matches!(
        kind,
        io::ErrorKind::WouldBlock
            | io::ErrorKind::TimedOut
            | io::ErrorKind::Interrupted
            | io::ErrorKind::WriteZero
            // The connection-reset family is INTENTIONALLY retryable: in this
            // workspace `Io(_)` reaches a peer-dropped connection only on the
            // snapshot-fetch / collector-export client paths, where the peer
            // (object store, OTLP collector) commonly recycles idle keep-alive
            // sockets and a fresh connection succeeds. It is deliberately NOT
            // treated as a hard server fault. If a future caller surfaces a
            // *client-initiated* reset (where retrying would loop), split that
            // path before relying on this classification.
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::BrokenPipe
    )
}

/// Number of consecutive bare (un-prefixed) hex digits that trips the
/// long-hex redaction rule used by [`redact`]. Sixteen digits is a 64-bit
/// value printed without a `0x` prefix; the floor is high enough that short
/// bare-hex / decimal literals embedded in ordinary diagnostics (a `byte 42`
/// offset, a small error code) are left intact.
const LONG_HEX_THRESHOLD: usize = 16;

/// Returns `true` if `b` is an ASCII hex digit (`0-9`, `a-f`, `A-F`).
fn is_hex_digit(b: u8) -> bool {
    b.is_ascii_digit() || matches!(b, b'a'..=b'f' | b'A'..=b'F')
}

/// Returns `true` if `c` is a token-boundary punctuation symbol that
/// [`redact`] treats as a delimiter (in addition to ASCII whitespace).
///
/// Sensitive shapes routinely arrive wrapped in or joined by punctuation —
/// `(/dev/shm/x)`, `addr=0x7ffe`, the comma-joined `a,/dev/shm/x`. Splitting
/// the scan on these characters (and copying them through verbatim) means each
/// path / pointer / hex run is examined on its own rather than buried inside a
/// larger token whose leading byte is ordinary text, which previously let the
/// sensitive fragment slip through unmasked.
fn is_token_separator(c: u8) -> bool {
    matches!(c, b'[' | b']' | b'(' | b')' | b'{' | b'}' | b'=' | b',' | b';')
}

/// Returns `true` if `c` is trailing/leading punctuation that
/// [`redact_token`] peels off a candidate token before applying the shape
/// rules. Mirrors the trailing-trim set so a sensitive token wrapped in
/// punctuation on *either* side (`'/dev/shm/x'`, `"0x7ffe"`) is still masked.
fn is_peelable_punct(c: char) -> bool {
    matches!(
        c,
        ':' | ',' | ';' | '.' | '(' | ')' | '[' | ']' | '{' | '}' | '\'' | '"' | '`'
    )
}

/// Masks pointer-, path-, and long-hex-shaped tokens in `s`, returning a new
/// `String`. See [`TensorWasmError::redacted_inner`] for the exact rules; this
/// is the hand-rolled, dependency-free scanner backing that accessor.
fn redact(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        // Tokens are delimited by ASCII whitespace OR a token-separator
        // punctuation symbol (`[](){}=,;`); copy any such delimiter through
        // verbatim so the redacted text keeps the original layout and so a
        // sensitive run joined to ordinary text by punctuation (`addr=0x7ffe`,
        // `(/dev/shm/x)`, `a,/dev/shm/x`) is examined on its own.
        if bytes[i].is_ascii_whitespace() || is_token_separator(bytes[i]) {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }
        // Find the end of the current token (run of non-delimiter bytes).
        let start = i;
        while i < bytes.len()
            && !bytes[i].is_ascii_whitespace()
            && !is_token_separator(bytes[i])
        {
            i += 1;
        }
        let token = &s[start..i];
        out.push_str(&redact_token(token));
    }
    out
}

/// Redacts a single token, preserving the ASCII punctuation that wraps it on
/// *either* side (`:`, `,`, `;`, `.`, brackets/braces/parens, quotes) so a
/// masked token embedded in prose keeps its surrounding punctuation. The
/// leading peel mirrors the trailing peel so a sensitive run wrapped on the
/// left (`'/dev/shm/x`, `"0x7ffe`) is still recognised by the shape rules.
fn redact_token(token: &str) -> String {
    const MASK: &str = "<redacted>";

    // Peel trailing punctuation off so e.g. `0x1234,` masks the address but
    // keeps the comma.
    let after_trailing = token.trim_end_matches(is_peelable_punct);
    let suffix = &token[after_trailing.len()..];
    // Peel leading punctuation off symmetrically so e.g. `'/dev/shm/x` masks
    // the path but keeps the opening quote.
    let core = after_trailing.trim_start_matches(is_peelable_punct);
    let prefix = &after_trailing[..after_trailing.len() - core.len()];

    if core.is_empty() {
        return token.to_string();
    }

    let bytes = core.as_bytes();
    // Mask the token if it matches any address/path shape below. Folded into a
    // single `||` chain (rather than an if/else ladder of `true` arms) so each
    // shape is one predicate — clippy flags the ladder as `if_same_then_else`.
    let masked =
        // Pointer-shaped: `0x` / `0X` followed by >=1 hex digit.
        (core.len() > 2
            && bytes[0] == b'0'
            && (bytes[1] == b'x' || bytes[1] == b'X')
            && bytes[2..].iter().all(|&b| is_hex_digit(b)))
        // Unix-style path: starts with `/` and contains another `/`.
        || (bytes[0] == b'/' && core[1..].contains('/'))
        // Windows drive path: drive letter + `:` + a separator (`\` or `/`)
        // + body. Accepts both the backslash form (`C:\Users\...`) and the
        // forward-slash form (`C:/Users/...`) that Rust / many tools emit.
        || (core.len() > 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && (bytes[2] == b'\\' || bytes[2] == b'/'))
        // Windows UNC path: leading `\\` followed by a server/share body
        // (`\\server\share\...`). Requires at least one byte after the `\\`
        // so a bare `\\` is left intact.
        || (core.len() > 2 && bytes[0] == b'\\' && bytes[1] == b'\\')
        // Long bare hex run (>= threshold digits, no `0x` prefix).
        || (core.len() >= LONG_HEX_THRESHOLD && bytes.iter().all(|&b| is_hex_digit(b)));

    if masked {
        format!("{prefix}{MASK}{suffix}")
    } else {
        token.to_string()
    }
}

/// 64-bit FNV-1a offset basis. Used by [`fnv1a`] to seed the digest backing
/// [`ErrorId`]. FNV-1a is chosen over `DefaultHasher` because it is
/// deterministic and process-independent — the same error content must mint
/// the same [`ErrorId`] across every node so a tenant-facing 4xx body can be
/// pivoted to operator logs on a *different* host.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// 64-bit FNV-1a prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Deterministic, process-independent 64-bit FNV-1a digest of `bytes`.
///
/// Deliberately *not* `std::collections::hash_map::DefaultHasher` — that hasher
/// is randomly seeded per process (and its algorithm is unspecified), so it
/// cannot mint a stable id that two different hosts agree on. FNV-1a is a tiny,
/// well-defined, dependency-free function whose output depends only on the
/// input bytes, which is exactly what cross-node correlation requires.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// An opaque, correlation-friendly identifier for a [`TensorWasmError`].
///
/// Rendered as a fixed-width lowercase-hex token (e.g. `err_1a2b3c4d5e6f7a8b`).
/// The id is **deterministically derived** from the error's
/// [`kind`](TensorWasmError::kind) and its inner diagnostic content via a
/// process-independent FNV-1a digest — it is *not* a random per-instance nonce
/// and carries no timestamp or counter. Two errors with identical kind and
/// content therefore share an id, by design: the id exists so a tenant-facing
/// 4xx response body (which embeds the id via
/// [`TensorWasmError::display_with_id`]) can be pivoted to the matching
/// operator log line (which records the same id alongside the unredacted
/// detail) — possibly on a different node — *without* the 4xx body ever
/// exposing the sensitive inner string.
///
/// The id space is opaque: callers MUST treat the value as a correlation token
/// only and MUST NOT parse it, reverse it, or infer the underlying error
/// content from it. It is safe to surface in tenant-facing output — it reveals
/// nothing beyond "two responses came from the same error shape".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ErrorId(u64);

impl ErrorId {
    /// The raw 64-bit digest behind this id. Exposed for callers that want to
    /// key a map or metric on the id without going through its hex `Display`.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for ErrorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Fixed-width 16-hex-digit rendering with an `err_` sigil so the token
        // is recognisable in a log grep and never collides with a bare hex
        // address. Width is pinned so the token length is stable across ids.
        write!(f, "err_{:016x}", self.0)
    }
}

/// A [`Display`](std::fmt::Display) wrapper that renders a [`TensorWasmError`]'s
/// sanitised message followed by its [`ErrorId`], for use in tenant-facing 4xx
/// response bodies. Produced by [`TensorWasmError::display_with_id`].
///
/// The wrapped error's own `Display` is left untouched (so existing callers and
/// the `error_display_does_not_leak` regression test keep their exact output);
/// this wrapper is the opt-in, id-bearing render.
#[derive(Debug, Clone, Copy)]
pub struct DisplayWithId<'a>(&'a TensorWasmError);

impl std::fmt::Display for DisplayWithId<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `{}` on the inner error uses its sanitised, leak-safe `Display`; the
        // appended id is opaque and safe to expose. Together they let a tenant
        // quote the id to support and an operator pivot it to the full detail.
        write!(f, "{} [{}]", self.0, self.0.error_id())
    }
}

/// The unified error type for every TensorWasm crate.
///
/// Variants are deliberately broad — host-level code matches on the variant to
/// classify failures into tenant-facing vs operator-facing responses. Inner
/// error sources are preserved via `#[source]` chains.
///
/// **Non-exhaustive**: callers MUST use `..` in `match` arms so new variants
/// added in a future minor release do not break downstream code. The enum has
/// no `Default` impl, so the `Self { .., ..Default::default() }` pattern does
/// not apply here — construct variants explicitly.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TensorWasmError {
    /// A call into the CUDA driver or runtime failed.
    ///
    /// The inner string is the underlying vendor message (cust / cuda-oxide /
    /// cudarc); it can contain raw pointer addresses and host paths, so it is
    /// deliberately omitted from `Display` and only surfaced via `Debug`.
    #[error("cuda driver call failed")]
    CudaError(Box<str>),

    /// A Wasm trap was triggered during execution (divide-by-zero, OOB access, ...).
    ///
    /// The inner string is wasmtime's trap message, which may include host
    /// instruction-pointer addresses; it is omitted from `Display` and only
    /// surfaced via `Debug`.
    #[error("wasm trap")]
    WasmTrap(Box<str>),

    /// Compiling Wasm bytes to native code failed.
    ///
    /// The inner string is wasmtime's compiler diagnostic, which can echo
    /// fragments of the tenant-supplied module bytes; it is omitted from
    /// `Display` and only surfaced via `Debug`.
    #[error("wasm compile failed")]
    WasmCompile(Box<str>),

    /// The instance exceeded its memory quota.
    #[error("memory exhausted: requested {requested} bytes, limit {limit}")]
    MemoryExhausted {
        /// Bytes the instance attempted to allocate.
        requested: u64,
        /// Bytes the tenant is allowed.
        limit: u64,
    },

    /// The tenant exceeded its per-tenant GPU memory cap.
    ///
    /// Distinct from [`Self::MemoryExhausted`] (the host-side / CPU
    /// quota) so dashboards, alerts, and tenant-facing error responses
    /// can distinguish "host RAM cap tripped" from "GPU memory cap
    /// tripped". The cap is the value passed to
    /// `TenantContextBuilder::with_gpu_memory_bytes_cap` (and recorded
    /// on `TenantContext::gpu_memory_bytes_cap`); `current` is the
    /// tenant's `gpu_bytes_in_use` *before* the rejected allocation
    /// would have been added.
    ///
    /// v0.3.7 enforcement is in-process — the allocator path
    /// (`tensor-wasm-mem::TensorWasmMemoryCreator::with_tenant_context`)
    /// consults the counter on every `UnifiedBuffer::new_on` and refuses
    /// to allocate when the cap would be exceeded. v0.4 will additionally
    /// pin a CUDA driver-level cap via
    /// `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, ...)`
    /// (CUDA 11.2+).
    #[error("gpu memory exhausted: requested {requested} bytes, limit {limit}, current {current}")]
    GpuMemoryExhausted {
        /// Bytes the tenant attempted to allocate on the GPU.
        requested: u64,
        /// Per-tenant GPU memory cap in bytes
        /// (`TenantContext::gpu_memory_bytes_cap.unwrap()`).
        limit: u64,
        /// Bytes the tenant had already reserved on the GPU at the time
        /// the allocation was rejected. The would-be new total is
        /// `current + requested`.
        current: u64,
    },

    /// A GPU kernel exceeded its deadline.
    #[error("kernel timeout after {elapsed_ms} ms (deadline {deadline_ms} ms)")]
    KernelTimeout {
        /// Time elapsed before timeout was enforced.
        elapsed_ms: u64,
        /// Configured per-kernel deadline in milliseconds.
        deadline_ms: u64,
    },

    /// An instance accessed memory or resources belonging to another tenant.
    ///
    /// **`Display` is intentionally sanitised.** The HTTP layer surfaces error
    /// `Display` to tenants as part of 4xx response bodies; including
    /// `tenant_id` or `resource` here would let tenant A learn tenant B's id
    /// and on-disk path layout. The structured fields remain on the variant
    /// so server-side `tracing::error!(?err, ...)` (which uses `Debug`) still
    /// gets the full context for operator triage.
    #[error("tenant isolation violation")]
    TenantIsolationViolation {
        /// Offending tenant identifier. Logged server-side via `Debug`; never
        /// included in `Display`.
        tenant_id: TenantId,
        /// Free-form description of the resource that was accessed out of
        /// scope. Logged server-side via `Debug`; never included in `Display`.
        resource: Box<str>,
    },

    /// An I/O error from the host OS.
    ///
    /// **`Display` is intentionally sanitised.** `std::io::Error::Display`
    /// echoes the underlying OS error string and — when the error was built
    /// from a filesystem operation — frequently includes the offending path.
    /// Both are unsafe to render into a tenant-facing 4xx body. The
    /// sanitised `Display` here surfaces only the `ErrorKind` discriminant
    /// (e.g. `"NotFound"`, `"PermissionDenied"`) which is a closed,
    /// non-sensitive enum. The inner `io::Error` remains reachable through
    /// `Debug` and via [`std::error::Error::source`] for server-side
    /// operator logs.
    #[error("I/O error (kind: {})", io_kind_name(.0))]
    Io(#[from] io::Error),

    /// A (de)serialisation error.
    ///
    /// The inner string is the serde / format-crate message and may quote
    /// untrusted input bytes verbatim; it is omitted from `Display` and only
    /// surfaced via `Debug`.
    #[error("serialization error")]
    Serialization(Box<str>),

    /// A snapshot was rejected by the reader's freshness check
    /// ([`max_age`](../tensor_wasm_snapshot/reader/struct.SnapshotReader.html#method.with_max_age)).
    ///
    /// Distinct from [`Self::Serialization`] so dashboards, alerts, and
    /// audit logs can pin replay-attempt rejections separately from
    /// generic format errors. All three fields are millisecond
    /// timestamps (or a millisecond duration); the variant carries
    /// enough context for an operator to triage a clock-skew false
    /// positive without consulting any other field on the snapshot.
    ///
    /// Introduced in T9 alongside
    /// `tensor_wasm_snapshot::reader::SnapshotReader::with_max_age`.
    /// The freshness check is **opt-in** — operators must set
    /// `max_age` on their reader to receive this error; the default
    /// reader continues to accept arbitrarily old snapshots for
    /// backward compatibility with v0.3.x captures.
    #[error(
        "snapshot too old: created {created_unix_ms} ms, now {now_unix_ms} ms, \
         max age {max_age_ms} ms"
    )]
    SnapshotTooOld {
        /// Wall-clock time the snapshot was captured, in milliseconds
        /// since the Unix epoch (from
        /// `SnapshotMetadata::created_unix_ms`).
        created_unix_ms: u64,
        /// Reader-side wall-clock time at the moment freshness was
        /// checked, in milliseconds since the Unix epoch.
        now_unix_ms: u64,
        /// Maximum age the reader was configured to accept, in
        /// milliseconds (`max_age` as passed to
        /// `SnapshotReader::with_max_age`).
        max_age_ms: u64,
    },
}

impl TensorWasmError {
    /// Returns `true` if the error is plausibly transient and retrying may succeed
    /// (timeouts, certain I/O conditions). Used by the API layer to decide
    /// between `503 Service Unavailable` (retryable) and `500 Internal Server Error`.
    ///
    /// `WasmCompile` and `TenantIsolationViolation` are *not* retryable —
    /// recompiling identical bytes will fail identically, and an isolation
    /// breach is a hard policy decision rather than a transient condition.
    ///
    /// `GpuMemoryExhausted` is likewise *not* retryable: it signals the
    /// per-tenant GPU memory cap (`TenantContext::gpu_memory_bytes_cap`)
    /// was hit — a fixed policy ceiling, not a transient resource
    /// shortage. Retrying the same allocation against the same cap fails
    /// identically, so it maps to a hard `4xx`-class rejection rather than
    /// `503`. `MemoryExhausted` (the host/CPU quota) stays retryable: that
    /// quota gates against transient host-RAM pressure that can recover as
    /// other instances release memory.
    ///
    /// `Io(_)` is classified by inspecting [`std::io::Error::kind`]: transient
    /// kinds (`WouldBlock`, `TimedOut`, `Interrupted`, `WriteZero`, the
    /// connection-reset family) flag as retryable, while hard-miss kinds
    /// (`NotFound`, `PermissionDenied`, `AlreadyExists`, ...) do not. The
    /// per-kind classifier is more useful than a blanket `Io(_) -> true`
    /// because the API layer otherwise returns `503` for hard 404-class
    /// failures and the CLI's retry loop spins on doomed requests.
    pub fn is_retryable(&self) -> bool {
        match self {
            TensorWasmError::KernelTimeout { .. } | TensorWasmError::MemoryExhausted { .. } => true,
            TensorWasmError::Io(err) => is_retryable_io_kind(err.kind()),
            _ => false,
        }
    }

    /// Returns the inner diagnostic string for the four variants that wrap a
    /// vendor message (`CudaError`, `WasmTrap`, `WasmCompile`,
    /// `Serialization`). For every other variant — and for `Io`, whose
    /// `std::io::Error` source is already reachable via [`std::error::Error::source`]
    /// — this returns `None`.
    ///
    /// This accessor exists so server-side operator logs can record the full
    /// vendor message even though `Display` deliberately omits it. **Never
    /// expose the returned string to end-users / response bodies**: that is
    /// precisely the leak surface the sanitised `Display` impls protect against.
    pub fn inner(&self) -> Option<&str> {
        match self {
            TensorWasmError::CudaError(s)
            | TensorWasmError::WasmTrap(s)
            | TensorWasmError::WasmCompile(s)
            | TensorWasmError::Serialization(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the inner diagnostic string with the most obviously sensitive
    /// token shapes masked, as an operator-log middle tier between the fully
    /// opaque [`Display`](std::fmt::Display) and the raw [`inner()`](Self::inner)
    /// / [`Debug`](std::fmt::Debug) text.
    ///
    /// For the four vendor-string variants (`CudaError`, `WasmTrap`,
    /// `WasmCompile`, `Serialization`) this returns `Some` with a redacted
    /// copy of the inner string; for every other variant (including `Io`,
    /// whose source is reachable via [`std::error::Error::source`]) it returns
    /// `None`, mirroring [`inner()`](Self::inner).
    ///
    /// # Redaction rules
    ///
    /// Scanning is hand-rolled (no `regex` dependency) and replaces each of the
    /// following token shapes with the literal `<redacted>`:
    ///
    /// * **Pointer-shaped tokens** — a `0x` / `0X` prefix followed by one or
    ///   more hex digits (`0x7ffe0000`). Catches raw addresses and instruction
    ///   pointers echoed by wasmtime / cust.
    /// * **Unix-style paths** — a run beginning with `/` that contains at least
    ///   one more `/` (`/dev/shm/tenant`, `/tmp/mod.wasm`). A lone `/` (e.g. a
    ///   division operator in a diagnostic) is left intact.
    /// * **Windows-style paths** — a drive letter followed by `:` and a
    ///   separator (`\` *or* `/`) and a path body (`C:\Users\...`,
    ///   `C:/Users/...`), or a UNC path with a leading `\\` and a
    ///   server/share body (`\\server\share\...`).
    /// * **Long bare hex tokens** — a run of 16 or more hex digits *without* a
    ///   `0x` prefix (e.g. a 64-bit address printed bare, or a hash). The
    ///   16-digit floor keeps short bare-hex / decimal literals that appear in
    ///   ordinary diagnostics (`byte 42`, a small error code) from being
    ///   swallowed. Note that any `0x`-prefixed run, however short, is masked
    ///   by the pointer rule above.
    ///
    /// A token is delimited by ASCII whitespace *or* a token-separator
    /// punctuation symbol (`[](){}=,;`), so a sensitive run wrapped in or
    /// joined to ordinary text by punctuation (`(/dev/shm/x)`, `addr=0x7ffe`,
    /// `a,/dev/shm/x`) is examined on its own rather than buried inside a
    /// larger token. Punctuation immediately adjacent to a token on either
    /// side (a leading or trailing `:` `,` quote / bracket) is peeled off
    /// before the shape rules run and re-emitted around the mask. The masking is
    /// deliberately conservative — it is a defence-in-depth aid for operator
    /// logs, **not** a guarantee that the result is safe for tenant-facing
    /// output. Anything shown to end-users must still go through the opaque
    /// [`Display`](std::fmt::Display).
    pub fn redacted_inner(&self) -> Option<String> {
        self.inner().map(redact)
    }

    /// Returns a stable, machine-readable variant name (used in metrics labels).
    pub fn kind(&self) -> &'static str {
        match self {
            TensorWasmError::CudaError(_) => "cuda",
            TensorWasmError::WasmTrap(_) => "wasm_trap",
            TensorWasmError::WasmCompile(_) => "wasm_compile",
            TensorWasmError::MemoryExhausted { .. } => "memory_exhausted",
            TensorWasmError::GpuMemoryExhausted { .. } => "gpu_memory_exhausted",
            TensorWasmError::KernelTimeout { .. } => "kernel_timeout",
            TensorWasmError::TenantIsolationViolation { .. } => "tenant_isolation",
            TensorWasmError::Io(_) => "io",
            TensorWasmError::Serialization(_) => "serialization",
            TensorWasmError::SnapshotTooOld { .. } => "snapshot_too_old",
        }
    }

    /// Returns an opaque, correlation-friendly [`ErrorId`] for this error.
    ///
    /// The id is **deterministically derived** from [`kind`](Self::kind) and
    /// the error's content (the inner vendor string for the string-carrying
    /// variants, the structured fields for the rest) via the
    /// process-independent [`fnv1a`] digest — it is not a random nonce. Two
    /// errors with the same kind and content share an id by design.
    ///
    /// The id is surfaced to tenants in the 4xx body via
    /// [`display_with_id`](Self::display_with_id) and to operators alongside
    /// the unredacted detail via [`redacted_inner_with_id`](Self::redacted_inner_with_id)
    /// (and may simply be re-derived by calling this method on the same error
    /// in a `tracing` span), so a 4xx response can be pivoted to the matching
    /// operator log line — possibly on a different node — without the 4xx body
    /// ever leaking the inner string.
    ///
    /// The digest is seeded with `kind()` and a single separator byte so two
    /// variants that happen to carry the same field bytes (e.g. a `CudaError`
    /// and a `WasmTrap` with identical inner text) still mint distinct ids.
    pub fn error_id(&self) -> ErrorId {
        // Build a small, allocation-light content rendering, then digest
        // `kind() || 0x1f || content`. The `0x1f` (ASCII unit separator) byte
        // cannot appear in `kind()` (a fixed `[a-z_]` set), so it unambiguously
        // delimits the kind from the content and prevents cross-variant
        // collisions.
        use std::fmt::Write as _;
        let mut buf = String::new();
        buf.push_str(self.kind());
        buf.push('\u{1f}');
        match self {
            TensorWasmError::CudaError(s)
            | TensorWasmError::WasmTrap(s)
            | TensorWasmError::WasmCompile(s)
            | TensorWasmError::Serialization(s) => buf.push_str(s),
            TensorWasmError::MemoryExhausted { requested, limit } => {
                let _ = write!(buf, "{requested}/{limit}");
            }
            TensorWasmError::GpuMemoryExhausted {
                requested,
                limit,
                current,
            } => {
                let _ = write!(buf, "{requested}/{limit}/{current}");
            }
            TensorWasmError::KernelTimeout {
                elapsed_ms,
                deadline_ms,
            } => {
                let _ = write!(buf, "{elapsed_ms}/{deadline_ms}");
            }
            TensorWasmError::TenantIsolationViolation {
                tenant_id,
                resource,
            } => {
                let _ = write!(buf, "{}/{resource}", tenant_id.get());
            }
            TensorWasmError::Io(err) => {
                buf.push_str(io_kind_name(err));
            }
            TensorWasmError::SnapshotTooOld {
                created_unix_ms,
                now_unix_ms,
                max_age_ms,
            } => {
                let _ = write!(buf, "{created_unix_ms}/{now_unix_ms}/{max_age_ms}");
            }
        }
        ErrorId(fnv1a(buf.as_bytes()))
    }

    /// Returns a [`Display`](std::fmt::Display) adapter that renders the
    /// error's sanitised message followed by its [`ErrorId`], suitable for a
    /// tenant-facing 4xx response body (e.g. `"wasm trap [err_1a2b3c4d5e6f7a8b]"`).
    ///
    /// The bare [`Display`](std::fmt::Display) impl is intentionally left
    /// unchanged (so existing callers and the leak regression tests keep their
    /// exact output); this adapter is the opt-in render that embeds the
    /// pivotable id. The id portion is opaque and safe to expose — only the
    /// inner vendor string is sensitive, and that is never included here.
    pub fn display_with_id(&self) -> DisplayWithId<'_> {
        DisplayWithId(self)
    }

    /// Like [`redacted_inner`](Self::redacted_inner) but pairs the redacted
    /// string with this error's [`ErrorId`], so an operator log line carries
    /// both the (defence-in-depth masked) detail and the same id a tenant
    /// received via [`display_with_id`](Self::display_with_id).
    ///
    /// Returns `None` for the variants without an inner vendor string, exactly
    /// mirroring [`redacted_inner`](Self::redacted_inner) / [`inner`](Self::inner).
    /// Callers that want the id for a *structured* variant (which has no inner
    /// string to redact) should call [`error_id`](Self::error_id) directly.
    pub fn redacted_inner_with_id(&self) -> Option<(ErrorId, String)> {
        self.redacted_inner().map(|r| (self.error_id(), r))
    }
}

/// Validate a GPU allocation against a per-tenant byte cap.
///
/// Returns `Ok(())` if adding `requested` bytes to the tenant's `current`
/// reservation stays within `cap`, and
/// `Err(TensorWasmError::GpuMemoryExhausted { .. })` otherwise. The returned
/// error carries `requested`, `limit` (`= cap`), and `current` so the rejection
/// is fully self-describing for dashboards and tenant-facing responses.
///
/// The check is overflow-safe: the prospective new total `current + requested`
/// is computed with [`u64::checked_add`], and an arithmetic overflow (which can
/// only happen for absurd inputs near [`u64::MAX`]) is itself treated as
/// exceeding the cap rather than wrapping to a small value and spuriously
/// admitting the allocation.
///
/// This is the single shared quota predicate the allocator path
/// (`tensor-wasm-mem`) and the tenant accounting layer (`tensor-wasm-tenant`)
/// can both call, so the "would this allocation breach the cap?" decision —
/// and the error it produces on rejection — lives in exactly one place.
///
/// ```
/// use tensor_wasm_core::error::{check_allocation, TensorWasmError};
/// // 100 already used, asking for 50 more, cap 200 -> fits.
/// assert!(check_allocation(100, 50, 200).is_ok());
/// // 100 used + 150 requested = 250 > 200 -> rejected.
/// assert!(matches!(
///     check_allocation(100, 150, 200),
///     Err(TensorWasmError::GpuMemoryExhausted { .. })
/// ));
/// ```
pub fn check_allocation(current: u64, requested: u64, cap: u64) -> Result<(), TensorWasmError> {
    let fits = current
        .checked_add(requested)
        .is_some_and(|new_total| new_total <= cap);
    if fits {
        Ok(())
    } else {
        Err(TensorWasmError::GpuMemoryExhausted {
            requested,
            limit: cap,
            current,
        })
    }
}

/// Convenience alias used throughout the workspace.
pub type Result<T, E = TensorWasmError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_cuda_is_opaque() {
        // Display must NOT echo the inner vendor string — it can carry pointer
        // addresses and host paths. The inner string is reachable via inner()
        // and Debug for operator-side logging.
        let e = TensorWasmError::CudaError("ctx not current at 0x7ffe0000".into());
        assert_eq!(format!("{e}"), "cuda driver call failed");
        assert_eq!(e.inner(), Some("ctx not current at 0x7ffe0000"));
    }

    #[test]
    fn display_memory_exhausted_fields() {
        let e = TensorWasmError::MemoryExhausted {
            requested: 1024,
            limit: 512,
        };
        let s = e.to_string();
        assert!(s.contains("1024"));
        assert!(s.contains("512"));
    }

    #[test]
    fn display_kernel_timeout_fields() {
        let e = TensorWasmError::KernelTimeout {
            elapsed_ms: 1500,
            deadline_ms: 1000,
        };
        let s = e.to_string();
        assert!(s.contains("1500"));
        assert!(s.contains("1000"));
    }

    #[test]
    fn display_isolation_violation_is_sanitised() {
        // Display is surfaced to tenants via 4xx response bodies; it must NOT
        // leak the offending tenant id or the resource path (the latter often
        // encodes another tenant's on-disk layout). See the regression test in
        // `tests/error_display_does_not_leak.rs` for the full leak-shaped
        // assertions; this inline test pins the exact Display string.
        let e = TensorWasmError::TenantIsolationViolation {
            tenant_id: crate::types::TenantId(42),
            resource: "/dev/shm/other-tenant".into(),
        };
        assert_eq!(e.to_string(), "tenant isolation violation");
    }

    #[test]
    fn debug_isolation_violation_still_carries_fields() {
        // Debug IS used for server-side `tracing::error!(?err)` and must
        // continue to expose the structured fields for operator triage —
        // only Display is sanitised.
        let e = TensorWasmError::TenantIsolationViolation {
            tenant_id: crate::types::TenantId(42),
            resource: "/dev/shm/other-tenant".into(),
        };
        let dbg = format!("{e:?}");
        assert!(dbg.contains("42"), "Debug should expose tenant_id: {dbg}");
        assert!(
            dbg.contains("/dev/shm/other-tenant"),
            "Debug should expose resource: {dbg}",
        );
    }

    #[test]
    fn display_wasm_trap_is_opaque() {
        let e = TensorWasmError::WasmTrap("unreachable at ip 0x401234".into());
        assert_eq!(format!("{e}"), "wasm trap");
        assert_eq!(e.inner(), Some("unreachable at ip 0x401234"));
    }

    #[test]
    fn display_wasm_compile_is_opaque() {
        let e = TensorWasmError::WasmCompile("bad opcode 0xfe in /tmp/mod.wasm".into());
        assert_eq!(format!("{e}"), "wasm compile failed");
        assert_eq!(e.inner(), Some("bad opcode 0xfe in /tmp/mod.wasm"));
    }

    #[test]
    fn display_io() {
        // The sanitised `Display` surfaces the `ErrorKind` variant name but
        // MUST NOT echo the inner OS string (which routinely carries
        // filesystem paths and host-specific diagnostic text). See the
        // regression test in `tests/error_display_does_not_leak.rs` for
        // the leak-shaped assertions; this inline test pins the exact
        // shape of the sanitised render.
        let e = TensorWasmError::Io(io::Error::new(io::ErrorKind::NotFound, "missing"));
        let s = format!("{e}");
        assert!(s.contains("I/O error"));
        assert!(s.contains("NotFound"));
        assert!(
            !s.contains("missing"),
            "Display must not leak the inner OS string, got: {s}",
        );
    }

    #[test]
    fn display_serialization_is_opaque() {
        let e = TensorWasmError::Serialization("bad json at byte 42: 'secret-tenant-key'".into());
        assert_eq!(format!("{e}"), "serialization error");
        assert_eq!(e.inner(), Some("bad json at byte 42: 'secret-tenant-key'"),);
    }

    #[test]
    fn inner_returns_none_for_structured_variants() {
        // Variants without an inner vendor string return None — there's
        // nothing to hand to the operator log beyond the Display output.
        let mem = TensorWasmError::MemoryExhausted {
            requested: 1,
            limit: 1,
        };
        let kt = TensorWasmError::KernelTimeout {
            elapsed_ms: 1,
            deadline_ms: 1,
        };
        let iso = TensorWasmError::TenantIsolationViolation {
            tenant_id: crate::types::TenantId(1),
            resource: "x".into(),
        };
        let io = TensorWasmError::Io(io::Error::other("x"));
        assert!(mem.inner().is_none());
        assert!(kt.inner().is_none());
        assert!(iso.inner().is_none());
        assert!(io.inner().is_none());
    }

    #[test]
    fn io_from_conversion() {
        let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "nope");
        let e: TensorWasmError = io_err.into();
        assert!(matches!(e, TensorWasmError::Io(_)));
        assert_eq!(e.kind(), "io");
    }

    #[test]
    fn kind_stable_names() {
        assert_eq!(TensorWasmError::CudaError("x".into()).kind(), "cuda");
        assert_eq!(TensorWasmError::WasmTrap("x".into()).kind(), "wasm_trap");
        assert_eq!(
            TensorWasmError::WasmCompile("x".into()).kind(),
            "wasm_compile"
        );
        assert_eq!(
            TensorWasmError::Serialization("x".into()).kind(),
            "serialization"
        );
    }

    #[test]
    fn retryable_classification() {
        assert!(TensorWasmError::KernelTimeout {
            elapsed_ms: 1,
            deadline_ms: 1
        }
        .is_retryable());
        assert!(TensorWasmError::MemoryExhausted {
            requested: 1,
            limit: 1
        }
        .is_retryable());
        assert!(!TensorWasmError::WasmTrap("x".into()).is_retryable());
        assert!(!TensorWasmError::CudaError("x".into()).is_retryable());
    }

    #[test]
    fn gpu_memory_exhausted_is_not_retryable() {
        // The per-tenant GPU memory cap is a fixed policy ceiling, not a
        // transient shortage: retrying the same allocation against the same
        // cap fails identically, so it must map to a hard 4xx-class
        // rejection rather than a retryable `503`. Contrast with
        // `MemoryExhausted` (the host/CPU quota), which stays retryable
        // because host RAM pressure can recover as peers release memory.
        let gpu = TensorWasmError::GpuMemoryExhausted {
            requested: 1024,
            limit: 512,
            current: 256,
        };
        assert!(
            !gpu.is_retryable(),
            "GpuMemoryExhausted must NOT be flagged as retryable",
        );
        // Sibling CPU-quota variant remains retryable.
        assert!(TensorWasmError::MemoryExhausted {
            requested: 1024,
            limit: 512,
        }
        .is_retryable());
    }

    #[test]
    fn io_kind_name_maps_common_kinds() {
        // `io_kind_name` returns a zero-alloc `&'static str` for the common
        // kinds; spot-check a few of the mappings and the non-exhaustive
        // fallback. The strings are what dashboards/alert rules grep for, so
        // they must stay stable.
        assert_eq!(
            io_kind_name(&io::Error::from(io::ErrorKind::NotFound)),
            "NotFound"
        );
        assert_eq!(
            io_kind_name(&io::Error::from(io::ErrorKind::PermissionDenied)),
            "PermissionDenied"
        );
        assert_eq!(
            io_kind_name(&io::Error::from(io::ErrorKind::WouldBlock)),
            "WouldBlock"
        );
        assert_eq!(
            io_kind_name(&io::Error::from(io::ErrorKind::TimedOut)),
            "TimedOut"
        );
        assert_eq!(
            io_kind_name(&io::Error::from(io::ErrorKind::AlreadyExists)),
            "AlreadyExists"
        );
        // `io::Error::other` produces `ErrorKind::Other`, which is not
        // enumerated explicitly and must hit the static fallback.
        assert_eq!(io_kind_name(&io::Error::other("x")), "Other");
    }

    #[test]
    fn wasm_compile_is_not_retryable() {
        // Recompiling the same Wasm bytes will fail identically — never retry.
        let e = TensorWasmError::WasmCompile("invalid opcode 0xfe".into());
        assert!(
            !e.is_retryable(),
            "WasmCompile must not be flagged as retryable",
        );
    }

    #[test]
    fn tenant_isolation_violation_is_not_retryable() {
        // An isolation breach is a hard policy decision — retrying is a security
        // bug, not a recovery strategy.
        let e = TensorWasmError::TenantIsolationViolation {
            tenant_id: crate::types::TenantId(1),
            resource: "/dev/shm/foreign".into(),
        };
        assert!(
            !e.is_retryable(),
            "TenantIsolationViolation must not be flagged as retryable",
        );
    }

    #[test]
    fn io_error_is_retryable_wouldblock_true() {
        // `WouldBlock` is the canonical retry-now I/O kind — pending a
        // socket send buffer or a non-blocking file descriptor. The
        // `is_retryable()` classifier must surface it as retryable so
        // the API layer renders `503 Service Unavailable` rather than
        // `500 Internal Server Error` (and so the CLI's retry loop
        // engages instead of bailing out).
        let e = TensorWasmError::Io(io::Error::from(io::ErrorKind::WouldBlock));
        assert!(
            e.is_retryable(),
            "Io(WouldBlock) must be classified as retryable",
        );
    }

    #[test]
    fn is_retryable_io_kind_pins_retryable_family() {
        // Pin the exact retryable I/O set so a future edit that adds/removes a
        // kind is a conscious, reviewed change. These map to `503` (the API
        // layer retries) rather than a hard `500`/`4xx`.
        for kind in [
            io::ErrorKind::WouldBlock,
            io::ErrorKind::TimedOut,
            io::ErrorKind::Interrupted,
            io::ErrorKind::WriteZero,
            io::ErrorKind::ConnectionReset,
            io::ErrorKind::ConnectionAborted,
            io::ErrorKind::BrokenPipe,
        ] {
            assert!(
                is_retryable_io_kind(kind),
                "{kind:?} must be classified as retryable",
            );
        }
    }

    #[test]
    fn is_retryable_io_kind_pins_hard_miss_kinds() {
        // Hard-miss kinds must NOT be retryable: retrying fails identically and
        // would spin the CLI's retry loop / return `503` for a permanent error.
        for kind in [
            io::ErrorKind::NotFound,
            io::ErrorKind::PermissionDenied,
            io::ErrorKind::AlreadyExists,
            io::ErrorKind::InvalidInput,
            io::ErrorKind::InvalidData,
            io::ErrorKind::UnexpectedEof,
        ] {
            assert!(
                !is_retryable_io_kind(kind),
                "{kind:?} must NOT be classified as retryable",
            );
        }
    }

    #[test]
    fn io_error_is_retryable_notfound_false() {
        // `NotFound` is a hard miss — retrying the same path will fail
        // identically. The classifier must NOT flag it as retryable;
        // otherwise the API layer would return `503 Service
        // Unavailable` for what should be a permanent `404`-class
        // failure and the CLI's retry loop would spin on a doomed
        // request.
        let e = TensorWasmError::Io(io::Error::from(io::ErrorKind::NotFound));
        assert!(
            !e.is_retryable(),
            "Io(NotFound) must NOT be classified as retryable",
        );
    }

    #[test]
    fn string_construction_via_into() {
        // String, &str, and Box<str> should all convert into the inner Box<str>
        // via the standard library `From` impls — exercise each path so future
        // refactors that break ergonomics are caught here. Display is opaque
        // (see `display_cuda_is_opaque`), so we verify the inner string was
        // captured via `inner()` rather than re-asserting the format string.
        let from_string: Box<str> = String::from("hello").into();
        let from_str: Box<str> = "hello".into();
        let from_box: Box<str> = Box::<str>::from("hello");
        let e1 = TensorWasmError::CudaError(from_string);
        let e2 = TensorWasmError::CudaError(from_str);
        let e3 = TensorWasmError::CudaError(from_box);
        assert_eq!(e1.inner(), Some("hello"));
        assert_eq!(e2.inner(), Some("hello"));
        assert_eq!(e3.inner(), Some("hello"));
    }

    #[test]
    fn redacted_inner_masks_pointer() {
        // A `0x`-prefixed address must be masked; the surrounding prose
        // survives.
        let e = TensorWasmError::CudaError("ctx not current at 0x7ffe0000".into());
        let r = e.redacted_inner().expect("vendor variant yields Some");
        assert!(
            !r.contains("0x7ffe0000"),
            "pointer must be masked, got: {r}",
        );
        assert!(r.contains("<redacted>"), "expected mask token, got: {r}");
        assert!(r.contains("ctx not current at"), "prose must survive: {r}");
    }

    #[test]
    fn redacted_inner_masks_unix_path() {
        let e = TensorWasmError::WasmCompile("bad module at /dev/shm/tenant-7/mod.wasm".into());
        let r = e.redacted_inner().expect("vendor variant yields Some");
        assert!(
            !r.contains("/dev/shm/tenant-7"),
            "unix path must be masked, got: {r}",
        );
        assert!(r.contains("<redacted>"), "expected mask token, got: {r}");
        assert!(r.contains("bad module at"), "prose must survive: {r}");
    }

    #[test]
    fn redacted_inner_masks_windows_path() {
        let e = TensorWasmError::Serialization("read failed: C:\\Users\\op\\secret.json".into());
        let r = e.redacted_inner().expect("vendor variant yields Some");
        assert!(
            !r.contains("C:\\Users"),
            "windows path must be masked, got: {r}",
        );
        assert!(r.contains("<redacted>"), "expected mask token, got: {r}");
    }

    #[test]
    fn redacted_inner_masks_long_hex() {
        // A 64-bit address printed without a `0x` prefix (16 hex digits) trips
        // the long-hex rule.
        let e = TensorWasmError::WasmTrap("trap at deadbeefcafebabe".into());
        let r = e.redacted_inner().expect("vendor variant yields Some");
        assert!(
            !r.contains("deadbeefcafebabe"),
            "long hex token must be masked, got: {r}",
        );
        assert!(r.contains("<redacted>"), "expected mask token, got: {r}");
    }

    #[test]
    fn redacted_inner_preserves_ordinary_text() {
        // Ordinary diagnostic prose — words, a lone `/` (division, not a
        // path), and small integers — must pass through untouched, so the
        // redaction does not destroy operator-useful detail. (A `0x`-prefixed
        // literal is intentionally NOT used here: the documented
        // pointer-shaped rule masks any `0x` + hex run, including short ones.)
        let e = TensorWasmError::WasmCompile("invalid opcode at index 3 / 4 byte 42".into());
        let r = e.redacted_inner().expect("vendor variant yields Some");
        assert_eq!(
            r, "invalid opcode at index 3 / 4 byte 42",
            "ordinary text must be preserved verbatim",
        );
    }

    #[test]
    fn redacted_inner_none_for_structured_variants() {
        // Mirrors `inner()`: variants without a vendor string have nothing to
        // redact and return `None`.
        let mem = TensorWasmError::MemoryExhausted {
            requested: 1,
            limit: 1,
        };
        let io = TensorWasmError::Io(io::Error::other("/secret/path"));
        assert!(mem.redacted_inner().is_none());
        assert!(io.redacted_inner().is_none());
    }

    #[test]
    fn error_id_is_deterministic_and_content_sensitive() {
        // Same kind + same content -> same id (so a 4xx body and an operator
        // log line, possibly on different nodes, agree).
        let a = TensorWasmError::CudaError("ctx lost".into());
        let b = TensorWasmError::CudaError("ctx lost".into());
        assert_eq!(a.error_id(), b.error_id());
        // Different content -> different id.
        let c = TensorWasmError::CudaError("ctx not current".into());
        assert_ne!(a.error_id(), c.error_id());
    }

    #[test]
    fn error_id_distinguishes_variants_with_identical_content() {
        // The `kind() || 0x1f || content` seeding means a CudaError and a
        // WasmTrap carrying byte-identical inner text still mint distinct ids.
        let cuda = TensorWasmError::CudaError("boom".into());
        let trap = TensorWasmError::WasmTrap("boom".into());
        assert_ne!(cuda.error_id(), trap.error_id());
    }

    #[test]
    fn error_id_display_is_fixed_width_opaque_token() {
        let e = TensorWasmError::WasmTrap("x".into());
        let id = e.error_id().to_string();
        // `err_` sigil + 16 lowercase hex digits.
        assert!(id.starts_with("err_"), "expected err_ sigil, got: {id}");
        assert_eq!(id.len(), "err_".len() + 16, "id must be fixed width: {id}");
        assert!(
            id["err_".len()..].bytes().all(|b| b.is_ascii_hexdigit()),
            "id body must be hex: {id}",
        );
    }

    #[test]
    fn display_with_id_appends_id_without_leaking_inner() {
        // The id-bearing tenant render must carry the sanitised message and the
        // opaque id, but NEVER the sensitive inner string.
        let e = TensorWasmError::CudaError("ctx not current at 0x7ffe0000".into());
        let rendered = e.display_with_id().to_string();
        assert!(
            rendered.starts_with("cuda driver call failed"),
            "must lead with the sanitised Display: {rendered}",
        );
        assert!(
            rendered.contains(&e.error_id().to_string()),
            "must embed the error id: {rendered}",
        );
        assert!(
            !rendered.contains("0x7ffe0000"),
            "must NOT leak the inner vendor string: {rendered}",
        );
        // The bare Display impl stays unchanged (backward compatible).
        assert_eq!(format!("{e}"), "cuda driver call failed");
    }

    #[test]
    fn redacted_inner_with_id_pairs_id_and_redaction() {
        let e = TensorWasmError::WasmCompile("bad module at /dev/shm/tenant-7/mod.wasm".into());
        let (id, redacted) = e
            .redacted_inner_with_id()
            .expect("vendor variant yields Some");
        assert_eq!(id, e.error_id());
        assert!(redacted.contains("<redacted>"), "expected mask: {redacted}");
        assert!(
            !redacted.contains("/dev/shm/tenant-7"),
            "path must be masked: {redacted}",
        );
        // Structured variants have no inner string to redact -> None.
        let mem = TensorWasmError::MemoryExhausted {
            requested: 1,
            limit: 1,
        };
        assert!(mem.redacted_inner_with_id().is_none());
    }

    #[test]
    fn check_allocation_admits_within_cap() {
        assert!(check_allocation(100, 50, 200).is_ok());
        // Exactly at the cap is admitted (the new total equals the cap).
        assert!(check_allocation(150, 50, 200).is_ok());
        // Zero request always fits when already within cap.
        assert!(check_allocation(200, 0, 200).is_ok());
    }

    #[test]
    fn check_allocation_rejects_over_cap_with_fields() {
        let err = check_allocation(100, 150, 200)
            .expect_err("100 + 150 = 250 > 200 must be rejected");
        match err {
            TensorWasmError::GpuMemoryExhausted {
                requested,
                limit,
                current,
            } => {
                assert_eq!(requested, 150);
                assert_eq!(limit, 200);
                assert_eq!(current, 100);
            }
            other => panic!("expected GpuMemoryExhausted, got: {other:?}"),
        }
    }

    #[test]
    fn check_allocation_treats_overflow_as_over_cap() {
        // `current + requested` overflows u64; the checked add must reject
        // rather than wrap to a small value and spuriously admit.
        let err = check_allocation(u64::MAX, 1, u64::MAX);
        assert!(
            matches!(err, Err(TensorWasmError::GpuMemoryExhausted { .. })),
            "arithmetic overflow must be treated as exceeding the cap",
        );
    }
}
