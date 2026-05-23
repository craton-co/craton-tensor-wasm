# Project Bali — Performance

This document describes how Bali measures performance, what the current
reference numbers look like, and how the CI regression gate works. Two
reference points matter: (a) the **host-only path** that a developer laptop
and the local CI runner exercise (no CUDA libraries, the back-pressure and
snapshot machinery exercised against host memory only), and (b) the
**CUDA-host path** that a CUDA-equipped self-hosted runner will measure once
the deployment work in S22 lands. Until then, GPU-side numbers in this doc
are modeled estimates, clearly marked.

## How we measure

Every bench lives in [`crates/bali-bench/benches/`](../crates/bali-bench/benches/)
and is a [Criterion](https://bheisler.github.io/criterion.rs/book/) bench
declared with `harness = false` in `Cargo.toml`. The defaults we rely on:

- A warm-up phase before any sample is recorded (Criterion default, 3 s
  locally, 1 s in CI).
- At least **30 samples** per metric, so the reported P50 is meaningfully
  stable.
- Criterion's built-in outlier detection and coefficient-of-variation
  reporting. We aim for **CV < 5%**; benches that exceed that on the
  reference machines are flagged as noisy and either tightened (more
  samples, longer measurement window) or excluded from the regression gate.
- All numbers below are **P50 from local runs** unless noted otherwise.
  P95/P99 are visible in the Criterion HTML reports under
  `target/criterion/` after a run.

## Bench inventory

| Bench | File | What it measures | Throughput unit |
|---|---|---|---|
| kernel_dispatch/serial | kernel_dispatch.rs | Per-dispatch overhead (back-pressure permit acquire+release) | dispatches/sec |
| kernel_dispatch/concurrent_cap64 | kernel_dispatch.rs | Throughput under concurrency cap | dispatches/sec |
| cold_start/capture | cold_start.rs | Snapshot capture (bincode + zstd encode) | bytes/sec |
| cold_start/restore | cold_start.rs | Snapshot restore (zstd decode + bincode decode) | bytes/sec |
| cold_start/disk_round_trip | cold_start.rs | capture + fs::write + fs::read + restore | bytes/sec |
| memory_bandwidth/sequential | memory_bandwidth.rs | Host-side `copy_from_slice` | bytes/sec |
| memory_bandwidth/random_stride | memory_bandwidth.rs | Strided 64-byte copies | bytes/sec |
| jit_compile/emit_text | jit_compile.rs | PTX text-emit latency | iters/sec |
| jit_compile/fingerprint | jit_compile.rs | Blueprint hash latency | iters/sec |
| e2e/healthz | e2e_inference.rs | Full router roundtrip latency | requests/sec |
| e2e/create_function | e2e_inference.rs | POST /functions latency | requests/sec |
| e2e/invoke_not_found | e2e_inference.rs | Error-path latency | requests/sec |

`kernel_dispatch` was added in S9; the remaining four bench files
(`cold_start`, `memory_bandwidth`, `jit_compile`, `e2e_inference`) were
introduced in S19 alongside this document.

## Reference numbers (host-only, modeled)

The figures below are **placeholders pending the S22 self-hosted CI runner**
and come from quick spot-checks on a developer laptop. Treat them as
ballpark, not contractual — see *Regression policy* for how the gate handles
drift.

| Bench | Input | P50 |
|---|---|---|
| kernel_dispatch/serial | 1000 dispatches | ~150 µs total (~150 ns / dispatch) |
| kernel_dispatch/concurrent_cap64 | 1000 dispatches, 4 worker threads | ~80 µs total |
| cold_start/capture | 1 MiB snapshot | ~5 ms |
| cold_start/capture | 16 MiB snapshot | ~75 ms |
| cold_start/capture | 128 MiB snapshot | ~600 ms |
| cold_start/restore | 1 MiB | ~3 ms |
| cold_start/restore | 16 MiB | ~50 ms |
| cold_start/restore | 128 MiB | ~400 ms |
| memory_bandwidth/sequential | 16 MiB | ~3 ms (5+ GB/s host RAM) |
| jit_compile/emit_text | matmul[16x16x16] | ~5-20 µs |
| e2e/healthz | — | ~30-60 µs |
| e2e/create_function | 9 byte payload | ~40-80 µs |
| e2e/invoke_not_found | — | ~30-60 µs |

The host-only `kernel_dispatch` number is essentially a Tokio semaphore
acquire+release round-trip — there is no GPU work in the loop, so the
number reflects scheduler overhead, not real launch latency.

## CUDA-host path (deferred to S22)

When the self-hosted runner with a real GPU is online, the numbers above
will shift in well-understood ways. Rough expectations:

- **`cold_start/restore`** picks up an additional 20-200 ms of UVM
  page-migration cost on first touch, depending on PCIe bandwidth. PCIe 4.0
  x16 is ~32 GB/s on paper, less in practice once you account for
  small-transfer overhead and contention with other workloads on the host.
- **`kernel_dispatch`** overhead drops to roughly **5-20 µs per dispatch**
  once the immediate-resolve stub is replaced with a CUDA Event-based sync
  in the runtime. The semaphore cost stays the same; what changes is that
  each permit is now backed by a real launch + event record.
- **`memory_bandwidth`** for device-resident buffers is dominated by HBM2
  or HBM3 bandwidth — 500-3000 GB/s for large sequential transfers,
  depending on the SKU. Strided patterns fall off the same way they do on
  host RAM, just at much higher absolute throughput.
- **`jit_compile`** is mostly host-side text emission today; once the
  nvrtc-backed path lands, expect a one-time ~10-100 ms hit per unique
  blueprint, amortized by the fingerprint cache.

This section will be replaced with measured ranges (not estimates) when
S22 completes.

## Regression policy

The `bench` workflow in `.github/workflows/bench.yml` runs the full bench
suite on pull requests that touch `crates/bali-bench/**` or `crates/*/src/**`,
and compares the result against a committed baseline at
`bench-results/baseline.json`. **Any metric that regresses by more than 10%
fails the build.** New benches are added to the baseline in a separate
commit, after a clean run on the reference machine, so that adding a bench
never lands together with a code change in the same PR.

Re-baseline procedure:

1. On a clean `main`, run `make ci-bench` (defined in the project Makefile,
   matches the flags the workflow uses).
2. Inspect the diff between `target/criterion/*` and
   `bench-results/baseline.json`. The Criterion HTML reports are the
   easiest way to see what moved and why.
3. Commit the new baseline **only** once you've reviewed each metric's
   change and confirmed it's intentional. A re-baseline PR should explain
   what caused the shift (faster code, slower code, noisier host, etc.).

If a regression is real and expected (e.g. a feature trade-off), the
re-baseline commit and the feature commit should land back-to-back, with
the re-baseline commit message linking to the feature PR.

## How to run locally

```sh
# Full suite (slow — uses Criterion defaults):
cargo bench --workspace

# A single bench file:
cargo bench -p bali-bench --bench cold_start

# Compile-only — CI step zero, useful as a fast sanity check:
cargo bench --workspace --no-run

# Match the CI flags exactly (shorter warm-up + measurement windows):
make ci-bench
```

After a run, open `target/criterion/report/index.html` for the full
Criterion report, including P95/P99, histograms, and regression plots
against the previous local run.

See [BUILD.md](BUILD.md) for the wider build-and-test workflow, and
[`crates/bali-bench/benches/`](../crates/bali-bench/benches/) for the
bench sources.

---

_Status: S19 scaffold. Numbers re-baseline once the S22 self-hosted CUDA runner is online._
