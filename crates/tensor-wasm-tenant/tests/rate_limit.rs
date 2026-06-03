// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! Time-windowed rate quota (`TenantContextBuilder::with_rate_limit` +
//! `TenantContext::try_acquire_op`): the token bucket admits up to `burst`
//! operations back-to-back, rejects once drained, and refills over wall-clock
//! time. Complements the deterministic injected-clock unit tests in
//! `src/context.rs` with an end-to-end check through the public builder API,
//! including a real `sleep` to confirm the monotonic-clock refill fires
//! without any clock injection.

use std::thread::sleep;
use std::time::Duration;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_tenant::{RateLimited, TenantContext};

#[test]
fn admits_up_to_burst_then_rejects() {
    // A generous steady-state rate with a small burst: the first `burst`
    // launches go through immediately, the next is rate-limited. The window
    // here is tight enough (5 ops, 100/s steady) that no meaningful refill
    // occurs between the back-to-back calls.
    let ctx = TenantContext::builder(TenantId(1))
        .with_rate_limit(/*ops_per_sec*/ 100, /*burst*/ 5)
        .build();
    assert!(ctx.has_rate_limit());

    for i in 0..5 {
        ctx.try_acquire_op()
            .unwrap_or_else(|_| panic!("op {i} within burst must be admitted"));
    }
    let err: RateLimited = ctx
        .try_acquire_op()
        .expect_err("op past burst must be rejected");
    assert_eq!(err.requested, 1);
    assert_eq!(err.burst, 5);
    assert_eq!(err.ops_per_sec, 100);
}

#[test]
fn refills_over_time() {
    // Fast refill (200 ops/s → one token every 5 ms) with a burst of 2.
    // Drain the burst, sleep long enough to accrue several tokens (capped at
    // burst), then confirm we can acquire again. Uses a real sleep so this
    // exercises the production `Instant::now()` refill path end-to-end.
    let ctx = TenantContext::builder(TenantId(2))
        .with_rate_limit(/*ops_per_sec*/ 200, /*burst*/ 2)
        .build();

    ctx.try_acquire_op().unwrap();
    ctx.try_acquire_op().unwrap();
    assert!(
        ctx.try_acquire_op().is_err(),
        "bucket should be drained after the initial burst"
    );

    // 50 ms at 200/s would refill ~10 tokens, capped at the burst of 2.
    sleep(Duration::from_millis(50));
    ctx.try_acquire_op()
        .expect("a token must have refilled after the sleep");
    ctx.try_acquire_op()
        .expect("burst worth of tokens refilled after the sleep");
    assert!(
        ctx.try_acquire_op().is_err(),
        "refill is capped at the burst depth, not the elapsed-time credit"
    );
}

#[test]
fn no_rate_limit_by_default_admits_unconditionally() {
    // Omitting `with_rate_limit` preserves the historical pure byte-cap
    // behaviour: every acquire is an unconditional Ok.
    let ctx = TenantContext::builder(TenantId(3)).build();
    assert!(!ctx.has_rate_limit());
    for _ in 0..100_000 {
        ctx.try_acquire_op()
            .expect("no limiter configured must always admit");
    }
}

#[test]
fn bytes_per_sec_budget_via_try_acquire_ops() {
    // The same bucket expresses a bytes/sec budget by acquiring N tokens at a
    // time. burst = 1 KiB admits a 768 + 256 pair, then rejects.
    let ctx = TenantContext::builder(TenantId(4))
        .with_rate_limit(/*bytes_per_sec*/ 1024, /*burst*/ 1024)
        .build();
    ctx.try_acquire_ops(768).unwrap();
    ctx.try_acquire_ops(256).unwrap();
    let err = ctx
        .try_acquire_ops(1)
        .expect_err("byte budget exhausted for this window");
    assert_eq!(err.available, 0);
}
