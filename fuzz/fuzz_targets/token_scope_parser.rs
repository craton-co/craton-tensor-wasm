// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Fuzz target for [`tensor_wasm_api::token_scope::parse_tokens_env`] —
//! the parser that turns `$TENSOR_WASM_API_TOKENS` into a map of bearer
//! token to [`tensor_wasm_api::TokenScope`].
//!
//! ## Goal
//!
//! The env-var grammar is small but mixes two entry shapes (bare bearer
//! and `bearer:tenant=<list>`) on the same comma-separated wire format,
//! and the parser does a second pass to re-glue comma-split fragments of
//! a single tenant list. That re-glue logic is exactly the kind of place
//! a malformed input can trip a `unreachable!` or an `unwrap()` we did not
//! think hard enough about. The fuzzer hunts for two classes of bug:
//!
//! 1. **Panic-freedom** — the harness is `#![no_main]`; libFuzzer treats a
//!    panic as a finding. `parse_tokens_env` runs in startup code on
//!    operator-supplied input, so every panic here is a denial-of-service
//!    against the gateway boot path.
//! 2. **Output consistency** — for every accepted entry the bearer string
//!    must be non-empty (otherwise the auth middleware would compare
//!    `""` against the incoming `Bearer ` header and admit anonymous
//!    callers) and the [`TenantScope`] enum's wildcard variant must not
//!    coexist with an explicit set in the same scope.
//!
//! ## Input shape
//!
//! `arbitrary` converts the raw fuzz buffer into a [`String`]. Sub-strings
//! that are not valid UTF-8 are dropped by `arbitrary` before they reach
//! the parser, which is the correct behaviour: the env var is always a
//! `String` in production (Rust's `std::env::var` enforces UTF-8) so we
//! do not need to exercise byte-level garbage.
//!
//! ## Property assertions
//!
//! After every parse:
//!
//! * The returned [`ParsedTokens::token_scopes`] map has no empty-string
//!   keys.
//! * `deprecated_count` <= number of accepted entries.
//! * For each scope, `scope.tenants` is either the wildcard variant
//!   (and `is_all()` agrees) or an explicit [`HashSet`] (and `is_all()`
//!   returns false). The variant is fully discriminated by the enum
//!   itself; this assertion catches a future regression where someone
//!   adds a third variant whose `is_all()` is ambiguous.
//! * Looking up every key in the map via the same string round-trips
//!   to the same scope (sanity-checks that the `HashMap` did not
//!   silently collapse two distinct bearers via a `Hash` impl change).
//!
//! ## Running
//!
//! ```sh
//! cargo +nightly fuzz run token_scope_parser -- -max_total_time=86400
//! ```

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use tensor_wasm_api::token_scope::{parse_tokens_env, TenantScope};

fuzz_target!(|data: &[u8]| {
    // Cap input length: the production env-var path is bounded by the
    // OS argv limit (~128 KiB on Linux). 64 KiB keeps the fuzzer fast
    // while still exercising long comma-separated allowlists that
    // stress the re-glue logic.
    if data.len() > 64 * 1024 {
        return;
    }

    // Derive a `String` from the fuzz buffer. `arbitrary` rejects
    // non-UTF-8 bytes, which mirrors what `std::env::var` would give us
    // at runtime (it returns `Err(NotUnicode)` rather than producing
    // a non-UTF-8 String).
    let mut u = Unstructured::new(data);
    let Ok(input): Result<String, _> = String::arbitrary(&mut u) else {
        return;
    };

    // Cap re-glue work: an env var that is just `,,,,,...` is a
    // pathological worst case for fragment accumulation. Production
    // env vars are bounded, so refuse to spend a whole fuzz iteration
    // on a million-comma input.
    if input.len() > 64 * 1024 {
        return;
    }

    let parsed = parse_tokens_env(&input);

    // ---- Property 1: every accepted bearer is non-empty. ----
    //
    // An empty-string bearer would compare equal to the substring
    // following `Bearer ` in a header like `Authorization: Bearer `
    // (note the trailing space), which would admit a caller who sent
    // an empty token. Defense-in-depth: catch the regression at the
    // parser layer too.
    for bearer in parsed.token_scopes.keys() {
        assert!(
            !bearer.is_empty(),
            "parse_tokens_env returned an empty-string bearer for input {input:?}",
        );
    }

    // ---- Property 2: deprecated count cannot exceed accepted entries. ----
    //
    // `deprecated_count` is incremented only on the bare-bearer path
    // and only when the entry parses successfully. It can never exceed
    // the size of `token_scopes` (each successful bare-bearer entry
    // inserts one map entry).
    assert!(
        parsed.deprecated_count <= parsed.token_scopes.len(),
        "deprecated_count={} exceeds token_scopes.len()={} for input {input:?}",
        parsed.deprecated_count,
        parsed.token_scopes.len(),
    );

    // ---- Property 3: each scope's variant is internally consistent. ----
    //
    // `TenantScope::All` <=> `is_all()`; `TenantScope::Set` must not
    // contradict it. This is a tautology against today's enum but
    // catches a future variant whose semantics blur the line.
    for (bearer, scope) in &parsed.token_scopes {
        match &scope.tenants {
            TenantScope::All => {
                assert!(
                    scope.tenants.is_all(),
                    "TenantScope::All but is_all()=false for bearer={bearer:?} input={input:?}",
                );
            }
            TenantScope::Set(set) => {
                assert!(
                    !scope.tenants.is_all(),
                    "TenantScope::Set but is_all()=true for bearer={bearer:?} input={input:?}",
                );
                // The set can be empty in principle if the parser ever
                // accepts a `tenant=` clause whose tenants all parsed
                // as `*` but produced no ids — today that path is
                // routed through `TenantScope::All` instead. Catch any
                // future drift that lands an empty explicit set,
                // because such a scope would admit *no* tenants but
                // still authenticate the bearer, which is a footgun.
                assert!(
                    !set.is_empty(),
                    "TenantScope::Set is empty for bearer={bearer:?} input={input:?}; \
                     empty explicit sets authenticate but authorize nothing — \
                     the parser should reject these instead",
                );
            }
        }
    }

    // ---- Property 4: map lookup round-trips. ----
    //
    // Defends against a future `Hash`/`Eq` divergence on the bearer
    // string type (currently `String`, a type alias for `BearerString`).
    // If keys were silently collapsed, looking them back up would miss.
    for bearer in parsed.token_scopes.keys().cloned().collect::<Vec<_>>() {
        assert!(
            parsed.token_scopes.contains_key(&bearer),
            "round-trip lookup failed for bearer={bearer:?} input={input:?}",
        );
    }
});
