// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Per-tenant scoped bearer tokens.
//!
//! Implements PATH-TO-V1 v0.4 exit-criterion *Scoped tokens*: each accepted
//! bearer token carries a [`TokenScope`] describing which tenants it may
//! address. Routes that bind to a tenant call
//! [`crate::rate_limit::AuthContext::authorize_tenant`] before doing any
//! per-tenant work and return `403 Forbidden { kind: "tenant_scope_denied" }`
//! when the caller's token does not cover the requested tenant.
//!
//! ## Wire format for `TENSOR_WASM_API_TOKENS`
//!
//! Two entry shapes are accepted in the comma-separated allowlist:
//!
//! 1. `token1:tenant=1,2,3` — scoped: the token grants access to tenants
//!    `1`, `2`, and `3` only.
//! 2. `token1:tenant=*` — wildcard: the token grants access to every tenant.
//! 3. `token1` — bare (legacy): treated as the wildcard form with a one-time
//!    deprecation warning emitted at startup. Scheduled for removal in v1.0.
//!
//! Forms can be mixed in the same env var; the parser splits on commas at
//! the *top level only*. Because tenant lists are themselves comma-separated
//! and live inside an entry, the parser groups all comma-separated values
//! that follow a `:tenant=` clause back into a single entry.
//!
//! ## Parser strategy
//!
//! The grammar is small enough that hand-rolling a one-pass scanner is the
//! lowest-overhead choice. We split the input on commas to obtain *raw
//! pieces*, then walk them left-to-right, joining pieces back together when
//! we observe a piece that is itself a continuation of a `:tenant=` clause
//! (i.e. it starts with neither a token name nor a colon — a numeric or `*`
//! continuation list). The walker accepts the same trailing whitespace the
//! pre-existing parser does (`split(',').map(str::trim)`).

use std::collections::{HashMap, HashSet};
use std::fmt;

use tensor_wasm_core::types::TenantId;

/// Tenant scope attached to a single bearer token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TenantScope {
    /// Wildcard — the token grants access to every tenant. Produced by
    /// `token:tenant=*` and by the legacy bare-token shape (with deprecation
    /// warning).
    All,
    /// Explicit allowlist — the token grants access only to the listed
    /// tenant ids.
    Set(HashSet<TenantId>),
}

impl TenantScope {
    /// `true` if `tenant` is covered by this scope.
    pub fn allows(&self, tenant: TenantId) -> bool {
        match self {
            TenantScope::All => true,
            TenantScope::Set(s) => s.contains(&tenant),
        }
    }

    /// `true` if this scope is the wildcard form. Used by callers (and
    /// integration tests) that want to assert legacy semantics held.
    pub fn is_all(&self) -> bool {
        matches!(self, TenantScope::All)
    }
}

/// Authorization metadata derived from a single bearer-token entry.
///
/// Stored as a value in [`ParsedTokens::token_scopes`]; the bearer string
/// itself is the map key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenScope {
    /// Tenants this token may address.
    pub tenants: TenantScope,
}

impl TokenScope {
    /// Wildcard scope — covers every tenant.
    pub fn all() -> Self {
        Self {
            tenants: TenantScope::All,
        }
    }

    /// Construct a scope from an explicit set of tenants.
    pub fn from_tenants<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = TenantId>,
    {
        Self {
            tenants: TenantScope::Set(iter.into_iter().collect()),
        }
    }

    /// Convenience accessor: `true` if the scope admits `tenant`.
    pub fn allows(&self, tenant: TenantId) -> bool {
        self.tenants.allows(tenant)
    }
}

/// Alias for the raw bearer-token string used as the map key in
/// [`ParsedTokens`]. Kept as a type alias rather than a newtype because the
/// existing [`crate::middleware::AuthConfig`] also stores the raw string.
pub type BearerString = String;

/// Output of [`parse_tokens_env`].
#[derive(Debug, Default, Clone)]
pub struct ParsedTokens {
    /// All accepted bearer tokens with their scopes.
    pub token_scopes: HashMap<BearerString, TokenScope>,
    /// Count of entries that used the legacy bare-token shape and were
    /// silently coerced to the wildcard scope. The server emits a single
    /// `tracing::warn!` at startup if this is nonzero.
    pub deprecated_count: usize,
}

/// Errors produced when parsing a single token entry. The wrapper around
/// [`parse_tokens_env`] turns each into a startup-time `tracing::warn!`
/// rather than aborting the process — the gateway prefers to keep the
/// remaining valid entries.
#[derive(Debug, PartialEq, Eq)]
pub enum ScopeParseError {
    /// The entry was empty after trimming.
    EmptyEntry,
    /// The entry was `:tenant=...` with no bearer token before the colon.
    MissingBearer,
    /// The entry contained a `:tenant=` clause with no value after `=`.
    EmptyTenantList,
    /// The tenant list contained a token that was neither `*` nor a valid
    /// `u64`. Carries the offending fragment for diagnostics.
    InvalidTenant(String),
    /// The entry mixed `*` with explicit tenant ids in the same clause.
    /// Either use `tenant=*` exactly or list ids; we refuse to silently
    /// promote `tenant=1,*` to wildcard because the user's intent is
    /// ambiguous.
    WildcardMixedWithIds,
    /// The entry used an unrecognised key after the colon (only `tenant=` is
    /// supported in v0.4). Carries the offending key.
    UnknownKey(String),
}

impl fmt::Display for ScopeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScopeParseError::EmptyEntry => f.write_str("empty token entry"),
            ScopeParseError::MissingBearer => {
                f.write_str("token entry has scope clause but no bearer token")
            }
            ScopeParseError::EmptyTenantList => {
                f.write_str("tenant= clause has no value")
            }
            ScopeParseError::InvalidTenant(s) => {
                write!(f, "tenant id {s:?} is neither '*' nor a u64")
            }
            ScopeParseError::WildcardMixedWithIds => {
                f.write_str("tenant=* cannot be mixed with explicit tenant ids")
            }
            ScopeParseError::UnknownKey(k) => {
                write!(f, "unknown scope key {k:?} (only 'tenant=' is supported)")
            }
        }
    }
}

impl std::error::Error for ScopeParseError {}

/// Parse a single entry of the `TENSOR_WASM_API_TOKENS` allowlist.
///
/// Accepted shapes (after trimming surrounding whitespace):
///
/// * `token` — legacy bare bearer. Returns [`TokenScope::all`].
/// * `token:tenant=*` — explicit wildcard.
/// * `token:tenant=1,2,3` — explicit allowlist.
///
/// The parser is intentionally strict on the scoped form: an empty tenant
/// list, an unknown key, or a tenant id that does not parse as a `u64`
/// returns an [`Err`]. The caller decides whether to ignore (skip the
/// entry) or surface the error — see [`parse_tokens_env`] for the policy
/// applied at startup.
pub fn parse_token_entry(s: &str) -> Result<(BearerString, TokenScope), ScopeParseError> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err(ScopeParseError::EmptyEntry);
    }
    match trimmed.split_once(':') {
        None => {
            // Bare-token form. Coerce to wildcard; the caller stamps the
            // deprecation warning.
            Ok((trimmed.to_owned(), TokenScope::all()))
        }
        Some((bearer, rest)) => {
            let bearer = bearer.trim();
            if bearer.is_empty() {
                return Err(ScopeParseError::MissingBearer);
            }
            let rest = rest.trim();
            // The v0.4 grammar only knows the `tenant=` key. Future keys
            // (e.g. `qps=`) can extend this match without breaking the
            // existing wire format.
            let value = match rest.split_once('=') {
                Some(("tenant", v)) => v.trim(),
                Some((other, _)) => {
                    return Err(ScopeParseError::UnknownKey(other.to_owned()));
                }
                None => {
                    return Err(ScopeParseError::UnknownKey(rest.to_owned()));
                }
            };
            if value.is_empty() {
                return Err(ScopeParseError::EmptyTenantList);
            }
            // Split the comma-separated tenant list, sniffing for wildcard.
            let mut saw_wildcard = false;
            let mut ids: HashSet<TenantId> = HashSet::new();
            for raw in value.split(',') {
                let frag = raw.trim();
                if frag.is_empty() {
                    return Err(ScopeParseError::InvalidTenant(frag.to_owned()));
                }
                if frag == "*" {
                    saw_wildcard = true;
                } else {
                    match frag.parse::<u64>() {
                        Ok(v) => {
                            ids.insert(TenantId(v));
                        }
                        Err(_) => return Err(ScopeParseError::InvalidTenant(frag.to_owned())),
                    }
                }
            }
            if saw_wildcard && !ids.is_empty() {
                return Err(ScopeParseError::WildcardMixedWithIds);
            }
            let scope = if saw_wildcard {
                TokenScope::all()
            } else {
                TokenScope {
                    tenants: TenantScope::Set(ids),
                }
            };
            Ok((bearer.to_owned(), scope))
        }
    }
}

/// Parse the `TENSOR_WASM_API_TOKENS` env-var value into a map of bearer →
/// scope, plus a count of legacy (bare) entries that were coerced to the
/// wildcard scope.
///
/// The input may mix the two entry shapes freely. Splitting is two-pass:
///
/// 1. Split on commas to obtain *raw fragments*.
/// 2. Walk fragments left to right, re-joining any fragment that does not
///    contain a `:` and follows an entry whose scope clause was `tenant=`
///    (a continuation of the tenant list).
///
/// Entries that fail [`parse_token_entry`] are dropped with a
/// `tracing::warn!` so a malformed entry never causes the gateway to refuse
/// to start. The remaining valid entries are returned in [`ParsedTokens`].
pub fn parse_tokens_env(env_value: &str) -> ParsedTokens {
    let mut out = ParsedTokens::default();
    if env_value.trim().is_empty() {
        return out;
    }

    // Re-glue fragments that belong to a single `tenant=...` list which was
    // itself comma-separated. We accumulate fragments into `current`; a
    // fragment that starts a new entry is one that either looks like
    // `token` or `token:tenant=...`. A continuation fragment is one that
    // does not contain `:` AND we are currently mid-tenant-list.
    let raw_fragments: Vec<&str> = env_value.split(',').collect();
    let mut entries: Vec<String> = Vec::with_capacity(raw_fragments.len());
    let mut current = String::new();
    let mut in_tenant_list = false;

    for frag in raw_fragments {
        let trimmed = frag.trim();
        if trimmed.is_empty() {
            // Empty fragments (`",,"`) flush whatever we have so the
            // subsequent fragment starts a fresh entry. Without this,
            // `foo:tenant=,1` would be silently glued into `foo:tenant=1`.
            if !current.is_empty() {
                entries.push(std::mem::take(&mut current));
            }
            in_tenant_list = false;
            continue;
        }
        let has_colon = trimmed.contains(':');
        // A continuation fragment must be a *tenant-list-shaped* token: the
        // wildcard `*` or a `u64`. Anything else (e.g. a bare bearer named
        // `foo`) is treated as a new entry — otherwise `a:tenant=*,foo` would
        // silently glue `foo` onto the wildcard list and fail with
        // `WildcardMixedWithIds`, rather than parsing as two entries.
        let looks_like_tenant_continuation =
            trimmed == "*" || trimmed.parse::<u64>().is_ok();
        if in_tenant_list && !has_colon && looks_like_tenant_continuation {
            // Continuation of the previous entry's tenant list.
            current.push(',');
            current.push_str(trimmed);
        } else {
            // Start a new entry.
            if !current.is_empty() {
                entries.push(std::mem::take(&mut current));
            }
            current.push_str(trimmed);
            in_tenant_list = has_colon
                && trimmed
                    .split_once(':')
                    .is_some_and(|(_, rest)| rest.trim().starts_with("tenant="));
        }
    }
    if !current.is_empty() {
        entries.push(current);
    }

    for entry in entries {
        let is_bare = !entry.contains(':');
        match parse_token_entry(&entry) {
            Ok((bearer, scope)) => {
                if is_bare {
                    out.deprecated_count += 1;
                }
                out.token_scopes.insert(bearer, scope);
            }
            Err(e) => {
                // We deliberately do not fail startup on a malformed entry —
                // the operator can fix it without losing all the other
                // tokens in the allowlist. The warning carries enough
                // context to find it.
                tracing::warn!(
                    target: "tensor_wasm_api::token_scope",
                    error = %e,
                    entry = %entry,
                    "ignoring malformed TENSOR_WASM_API_TOKENS entry",
                );
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(it: impl IntoIterator<Item = u64>) -> HashSet<TenantId> {
        it.into_iter().map(TenantId).collect()
    }

    #[test]
    fn parse_bare_entry_yields_wildcard() {
        let (bearer, scope) = parse_token_entry("alpha").expect("parses");
        assert_eq!(bearer, "alpha");
        assert!(scope.tenants.is_all());
    }

    #[test]
    fn parse_bare_entry_trims_whitespace() {
        let (bearer, _) = parse_token_entry("  alpha  ").expect("parses");
        assert_eq!(bearer, "alpha");
    }

    #[test]
    fn parse_wildcard_entry_yields_wildcard() {
        let (bearer, scope) = parse_token_entry("alpha:tenant=*").expect("parses");
        assert_eq!(bearer, "alpha");
        assert!(scope.tenants.is_all());
    }

    #[test]
    fn parse_single_tenant_entry() {
        let (bearer, scope) = parse_token_entry("alpha:tenant=7").expect("parses");
        assert_eq!(bearer, "alpha");
        assert_eq!(scope.tenants, TenantScope::Set(ids([7])));
    }

    #[test]
    fn parse_multi_tenant_entry() {
        let (bearer, scope) = parse_token_entry("alpha:tenant=1,2,3").expect("parses");
        assert_eq!(bearer, "alpha");
        assert_eq!(scope.tenants, TenantScope::Set(ids([1, 2, 3])));
        assert!(scope.allows(TenantId(2)));
        assert!(!scope.allows(TenantId(4)));
    }

    #[test]
    fn parse_multi_tenant_entry_with_whitespace() {
        let (bearer, scope) = parse_token_entry("alpha:tenant= 1 , 2 , 3 ").expect("parses");
        assert_eq!(bearer, "alpha");
        assert_eq!(scope.tenants, TenantScope::Set(ids([1, 2, 3])));
    }

    #[test]
    fn parse_empty_entry_errors() {
        assert_eq!(parse_token_entry("").unwrap_err(), ScopeParseError::EmptyEntry);
        assert_eq!(parse_token_entry("   ").unwrap_err(), ScopeParseError::EmptyEntry);
    }

    #[test]
    fn parse_missing_bearer_errors() {
        assert_eq!(
            parse_token_entry(":tenant=1").unwrap_err(),
            ScopeParseError::MissingBearer
        );
    }

    #[test]
    fn parse_empty_tenant_list_errors() {
        assert_eq!(
            parse_token_entry("alpha:tenant=").unwrap_err(),
            ScopeParseError::EmptyTenantList
        );
    }

    #[test]
    fn parse_invalid_tenant_errors() {
        let err = parse_token_entry("alpha:tenant=not-a-number").unwrap_err();
        assert_eq!(err, ScopeParseError::InvalidTenant("not-a-number".into()));
    }

    #[test]
    fn parse_unknown_key_errors() {
        // We accept only `tenant=` in v0.4.
        let err = parse_token_entry("alpha:qps=100").unwrap_err();
        assert_eq!(err, ScopeParseError::UnknownKey("qps".into()));
    }

    #[test]
    fn parse_missing_eq_errors() {
        let err = parse_token_entry("alpha:tenant").unwrap_err();
        assert_eq!(err, ScopeParseError::UnknownKey("tenant".into()));
    }

    #[test]
    fn parse_wildcard_mixed_with_ids_errors() {
        let err = parse_token_entry("alpha:tenant=1,*,2").unwrap_err();
        assert_eq!(err, ScopeParseError::WildcardMixedWithIds);
    }

    #[test]
    fn parse_tenant_list_with_trailing_comma_is_invalid() {
        // `alpha:tenant=1,` → the second fragment is empty, which we treat
        // as an invalid tenant rather than silently accepting it.
        let err = parse_token_entry("alpha:tenant=1,").unwrap_err();
        assert_eq!(err, ScopeParseError::InvalidTenant(String::new()));
    }

    #[test]
    fn parse_env_empty_returns_default() {
        let parsed = parse_tokens_env("");
        assert!(parsed.token_scopes.is_empty());
        assert_eq!(parsed.deprecated_count, 0);

        let parsed = parse_tokens_env("   ");
        assert!(parsed.token_scopes.is_empty());
        assert_eq!(parsed.deprecated_count, 0);
    }

    #[test]
    fn parse_env_single_bare_token() {
        let parsed = parse_tokens_env("alpha");
        assert_eq!(parsed.deprecated_count, 1);
        assert_eq!(parsed.token_scopes.len(), 1);
        assert!(parsed.token_scopes.get("alpha").unwrap().tenants.is_all());
    }

    #[test]
    fn parse_env_three_bare_tokens() {
        let parsed = parse_tokens_env("alpha,beta,gamma");
        assert_eq!(parsed.deprecated_count, 3);
        assert_eq!(parsed.token_scopes.len(), 3);
        for name in ["alpha", "beta", "gamma"] {
            assert!(
                parsed.token_scopes.get(name).unwrap().tenants.is_all(),
                "{name} not wildcard",
            );
        }
    }

    #[test]
    fn parse_env_mixed_bare_and_scoped() {
        let parsed = parse_tokens_env("alpha,beta:tenant=1,2,gamma:tenant=*");
        assert_eq!(parsed.deprecated_count, 1, "only `alpha` is bare");
        assert_eq!(parsed.token_scopes.len(), 3);
        assert!(parsed.token_scopes["alpha"].tenants.is_all());
        assert_eq!(
            parsed.token_scopes["beta"].tenants,
            TenantScope::Set(ids([1, 2]))
        );
        assert!(parsed.token_scopes["gamma"].tenants.is_all());
    }

    #[test]
    fn parse_env_scoped_token_with_long_tenant_list() {
        let parsed = parse_tokens_env("alpha:tenant=1,2,3,4,5,6,7");
        assert_eq!(parsed.deprecated_count, 0);
        assert_eq!(parsed.token_scopes.len(), 1);
        let scope = &parsed.token_scopes["alpha"];
        for t in 1..=7u64 {
            assert!(scope.allows(TenantId(t)));
        }
        assert!(!scope.allows(TenantId(8)));
    }

    #[test]
    fn parse_env_handles_whitespace_around_entries() {
        let parsed = parse_tokens_env("  alpha , beta:tenant=1 , gamma:tenant=*  ");
        assert_eq!(parsed.token_scopes.len(), 3);
        assert!(parsed.token_scopes.contains_key("alpha"));
        assert!(parsed.token_scopes.contains_key("beta"));
        assert!(parsed.token_scopes.contains_key("gamma"));
    }

    #[test]
    fn parse_env_drops_malformed_entry_but_keeps_others() {
        // The middle entry has no bearer prefix → dropped. The bookend
        // entries survive.
        let parsed = parse_tokens_env("alpha,:tenant=1,gamma:tenant=*");
        assert_eq!(parsed.token_scopes.len(), 2);
        assert!(parsed.token_scopes.contains_key("alpha"));
        assert!(parsed.token_scopes.contains_key("gamma"));
        assert_eq!(parsed.deprecated_count, 1, "only `alpha` was bare");
    }

    #[test]
    fn parse_env_two_scoped_tokens_with_continuation_lists() {
        // The tricky case: two scoped tokens, each with a comma-separated
        // tenant list. The continuation logic must group `5,6` back with
        // `beta:tenant=4` and `1,2,3` with `alpha:tenant=`.
        let parsed = parse_tokens_env("alpha:tenant=1,2,3,beta:tenant=4,5,6");
        assert_eq!(parsed.deprecated_count, 0);
        assert_eq!(parsed.token_scopes.len(), 2);
        assert_eq!(
            parsed.token_scopes["alpha"].tenants,
            TenantScope::Set(ids([1, 2, 3]))
        );
        assert_eq!(
            parsed.token_scopes["beta"].tenants,
            TenantScope::Set(ids([4, 5, 6]))
        );
    }

    #[test]
    fn parse_env_bare_token_after_scoped_token() {
        // Tricky-case the continuation logic: `a:tenant=*,bare` must parse
        // as two entries, not one. The naive "no colon → continuation"
        // rule would glue `bare` onto the wildcard list and produce
        // WildcardMixedWithIds; the continuation-must-be-numeric rule
        // correctly splits.
        let parsed = parse_tokens_env("a:tenant=*,bare,b:tenant=4,5");
        assert_eq!(parsed.token_scopes.len(), 3);
        assert!(parsed.token_scopes["a"].tenants.is_all());
        assert!(parsed.token_scopes["bare"].tenants.is_all());
        assert_eq!(parsed.deprecated_count, 1, "`bare` is bare");
        assert_eq!(
            parsed.token_scopes["b"].tenants,
            TenantScope::Set(ids([4, 5]))
        );
    }

    #[test]
    fn tenant_scope_allows_in_set() {
        let s = TokenScope::from_tenants([TenantId(1), TenantId(2)]);
        assert!(s.allows(TenantId(1)));
        assert!(s.allows(TenantId(2)));
        assert!(!s.allows(TenantId(3)));
    }

    #[test]
    fn tenant_scope_wildcard_admits_arbitrary() {
        let s = TokenScope::all();
        assert!(s.allows(TenantId(0)));
        assert!(s.allows(TenantId(u64::MAX)));
    }
}
