// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! HTTP-label-validation bench: measures
//! [`HttpRequestLabels::try_new`] against a 100-route allow-list.
//!
//! Motivation. The HTTP metrics middleware in `tensor-wasm-api` calls
//! `HttpRequestLabels::try_new` once per request to validate the
//! `(route, method, status)` tuple before attaching it to a labelled
//! `Family<...>`. The validator's hot path is the route lookup; this
//! bench pins it after the `Vec<&'static str>` -> `HashSet<&'static str>`
//! migration so a future regression that drags the lookup back to
//! `O(n)` is caught here rather than by a flame graph in production.
//!
//! What's measured. Three groups, all with a 100-route allow-list:
//!
//! * `metrics_label_validation/try_new/first` — the first registered
//!   route (used to be the best case under the linear scan; with the
//!   hash index it is identical to the average case, which is the
//!   whole point of the migration).
//! * `metrics_label_validation/try_new/last` — the last registered
//!   route (used to be the worst case under the linear scan).
//! * `metrics_label_validation/try_new/miss` — a route not present in
//!   the allow-list, exercising the `Err(LabelError::UnknownRoute)`
//!   path (which still walks the full hash bucket chain on a miss).
//!
//! All three should now sit on top of each other inside Criterion's
//! noise band. They are not on the CI regression-gate path — this is a
//! diagnostic bench for the metrics hot path; see
//! [`bench-results/README.md`](../../bench-results/README.md).
//!
//! Sample sites. The bench feeds `try_new_with_allowlist` rather than
//! `try_new` so it never touches the process-global `OnceLock` slot
//! (set-once semantics make repeated bench invocations on the same
//! process flaky otherwise). The two paths share the validator's hot
//! code, so the numbers are interchangeable for the
//! `RouteAllowlist::lookup` measurement at hand.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tensor_wasm_core::metrics::{HttpRequestLabels, RouteAllowlist};

/// Build a stable 100-route allow-list. `Box::leak` is fine here: the
/// bench process exits at the end of the run and the leaked memory is
/// bounded at 100 short strings. The list is built once at the start of
/// the bench (outside any `iter` closure) so the lookups timed below
/// are pure validator work, not allow-list construction.
fn build_100_route_allowlist() -> std::sync::Arc<RouteAllowlist> {
    let routes: Vec<&'static str> = (0..100)
        .map(|i| -> &'static str { Box::leak(format!("/route_{i}").into_boxed_str()) })
        .collect();
    // The leaked `Vec` is small; the `Arc<RouteAllowlist>` holds the
    // index for the lifetime of the bench.
    let leaked_slice: &'static [&'static str] = Box::leak(routes.into_boxed_slice());
    RouteAllowlist::new(leaked_slice)
}

fn bench_try_new(c: &mut Criterion) {
    let allowlist = build_100_route_allowlist();
    let mut group = c.benchmark_group("metrics_label_validation/try_new");
    group.measurement_time(Duration::from_secs(3));

    // Three representative samples:
    //   first: index 0 in declaration order.
    //   last:  index 99 in declaration order.
    //   miss:  not present at all.
    //
    // Under the old `Vec` linear scan `first` was sub-100ns and `last`
    // was ~100x slower; with the `HashSet` index all three should land
    // in the same noise band, which is exactly the regression we want
    // the bench to catch if it ever returns.
    let cases: [(&str, &str); 3] = [
        ("first", "/route_0"),
        ("last", "/route_99"),
        ("miss", "/not-a-route"),
    ];
    for (label, route) in cases {
        // Criterion's `bench_with_input` passes the closure a `&I`; here
        // `I = &str` so the closure receives `&&str` and must deref
        // once to recover the `&str` the validator wants.
        group.bench_with_input(BenchmarkId::from_parameter(label), &route, |b, route| {
            b.iter(|| {
                // The `Result` is intentionally swallowed, but it (and the
                // loop-invariant inputs) are wrapped in `black_box` so LLVM
                // can't hoist or elide the call — the constant args would
                // otherwise let it prove the result unused. Matches every
                // other bench in the crate.
                let out = HttpRequestLabels::try_new_with_allowlist(
                    black_box(*route),
                    black_box("GET"),
                    black_box(200),
                    Some(&allowlist),
                );
                black_box(out.is_ok());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_try_new);
criterion_main!(benches);
