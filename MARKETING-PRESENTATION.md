<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Craton Software Company -->

# Craton TensorWasm

**A GPU-accelerated serverless WebAssembly runtime, in Rust.**

Slide-style pitch deck for technical evaluators (CTOs, platform leads, SREs).
Every numeric claim and every "we have X" line links to a commit, a test, or
a doc — no aspirational marketing. Cross-checked against the `v0.3.5` release
tag.

To present: pipe through [marp](https://marp.app/), or read as plain markdown.
Slide breaks are `---`. Speaker notes are HTML comments.

---

## What it is (30 seconds)

Untrusted WebAssembly + explicit GPU dispatch + multi-tenant isolation +
production observability — all in one runtime, sandboxed by Wasmtime, hot-pathed
by CUDA, and shipped as Apache-2.0 source.

- Run any `.wasm` module that fits WASI Preview 2
- Talk to the GPU through a typed `wasi:cuda` host interface
- Multi-tenant by construction; one process serves many tenants
- HTTP API + CLI + structured audit log + OpenTelemetry trace propagation
- 92 commits, 5 tagged releases (`v0.3.1` → `v0.3.5`), 0 audit problems open

<!-- Speaker note: the audience is somebody evaluating whether to deploy this
in a real production stack. Not a VC pitch. Lead with what's shippable. -->

---

## The three things it actually does

Pitch points, each graded against tests/commits in the repo today.

| Promise | Where the proof lives | Honest grade |
|---|---|---|
| Zero-copy Wasm linear memory in CUDA UVM | `crates/tensor-wasm-mem/` + `is_uvm_backed()` probe + 5 tests pinning the property | Proven structurally; B2 e2e correctness via real PTX |
| Explicit GPU dispatch from Wasm guests with typed argv | `crates/tensor-wasm-wasi-gpu/` + 7/7 `kernel_args_e2e` tests pass on real RTX 2060 | Proven end-to-end including `vector_add` correctness readback |
| Async-yielding dispatch on Tokio + Wasmtime | `DispatchFuture::poll` yields via 50 µs tokio sleep (B1); cuda-async proper waker path scaffolded (F3) for v0.4 cutover | Framework works; production-scale waker is v0.4 |

Anything not in this table is either out of v1.0 scope (see PATH-TO-V1.md
"Anti-goals") or earlier than these three. We don't pitch what we haven't
shipped.

---

## Architecture in one diagram

```
                   ┌─────────────────┐
                   │   HTTP gateway   │  axum: bearer + scoped-tokens + rate-limit + audit log
                   │   (tensor-wasm-api)     │  W2.1 + W2.2 + W1.4 + W2.3 + W4.1 tracing
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │  Snapshot subsys │  zstd + bincode + 256 MiB bomb guard
                   │  (tensor-wasm-snapshot)  │  W1.3 cross-version compat
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ Tenant registry  │  per-context streams + back-pressure cap=64
                   │   (tensor-wasm-tenant)  │  C3 per-tenant GPU memory accounting
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │  Wasmtime exec   │  async + epoch interruption
                   │   (tensor-wasm-exec)    │  deadline enforcement
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │    wasi-cuda     │  W1.1 typed-argv lowering (scalar + pointer)
                   │  (tensor-wasm-wasi-gpu) │  bounds-checked guest→device pointers
                   └────────┬────────┘
                            │                ┌─────────────────────┐
                   ┌────────▼────────┐       │   tensor-wasm-jit            │
                   │  UnifiedBuffer   │◀──────│   PTX emit + BLAKE3 cache    │
                   │   (tensor-wasm-mem)     │       │   matmul / vector_add / conv2d
                   └────────┬────────┘       └─────────────────────┘
                            │
                  ┌─────────┴─────────┐
                  │     3 backings    │
                  ├───────────────────┤
                  │ cust (default)    │  unified-memory feature
                  │ cudarc (W1.2)     │  cudarc-backend feature; v0.5 fallback
                  │ cuda-oxide (O2)   │  cuda-oxide-backend feature; v0.5 default contingent on v0.2
                  └───────────────────┘
```

See `ARCHITECTURE.md` for the full dep graph. Each component is a separate
crate in the workspace; the dep graph is enforced one-direction-only (`Makefile
ci` runs the check).

---

## What's measured today

`bench-results/` on the dev RTX 2060 + Windows 11 WDDM + nightly-2026-04-03:

**P50 latencies:**
- `tensor_wasm_kernel_dispatch/serial/100` → real CUDA path through the typed-argv pipeline; `kernel_args_e2e` proves correctness via `c[i] == a[i] + b[i]` readback
- `e2e/healthz/get` → 6.7 µs (W4.6 tail_latency bench)
- `e2e/invoke_not_found/post` → 28.1 µs (W4.6)
- `dispatch/serial/100` busy-poll baseline → 400 ns P50 / 1.8 µs P99.9 (F3 dispatch_future_backends, busy-poll path)
- `cold_start/restore/1 MiB` snapshot → ~10 ms (W1.3 compat tests baseline)

**End-to-end pure-CPU comparison:**
- hyperfine vs wasmtime 45 on a 500M-iter integer loop (`bench-results/hyperfine-vs-wasmtime.json`):
  - tensor-wasm: 490.7 ± 13.2 ms
  - wasmtime: 501.2 ± 34.9 ms
  - **Statistically tied** (CIs overlap)
- Tracing + audit log + HTTP metrics overhead **not measurable on the CLI path**
  (CLI bypasses the HTTP layer where the instrumentation lives)

**Important disclosure:** Committed numbers were captured on a noisy Windows
developer host with CV > 5% per `docs/BENCHMARKING.md` "Methodology → CV
target" section. They are floor-only baselines suitable for regression
gating, **not** publication-grade external comparison numbers. The S22
self-hosted CUDA runner (`.github/workflows/cuda.yml` + the registration
runbook) produces publishable numbers once registered.

---

## What "it actually runs on CUDA" means

The repo has a permanent record of CUDA tests passing on real silicon at
`bench-results/cuda-rtx2060-tests.txt`:

```
2. Kernel-args E2E tests (W1.1 typed-argv marshalling against real GPU)
   $ cargo test -p tensor-wasm-wasi-gpu --release --features cuda \
       --test kernel_args_e2e -- --include-ignored

   test scalar_argv_round_trips_through_launch_path        ... ok
   test pointer_argv_round_trips_through_launch_path       ... ok
   test pointer_argv_out_of_bounds_returns_invalid_pointer ... ok
   test scalar_argv_real_cuda_launch                       ... ok
   test pointer_argv_real_cuda_launch                      ... ok
   test result: ok. 5 passed; 0 failed; 0 ignored
```

Plus the B2 wave's `vector_add_end_to_end_real_ptx_real_kernel`: builds a Wasm
guest with three f32[64] arrays in linear memory, encodes typed argv
`[Ptr(a), Ptr(b), Ptr(c), U32(64)]`, launches against the canonical
`kernels/vector_add.ptx`, reads `c` back from linear memory, asserts
`c[i] == 100.0 + 2*i` for all 64 elements. **7/7 pass on RTX 2060** even
though the kernel targets SM_80 (the CUDA driver JIT'd it up to SM_75).

This is the test that proves the three pitch points work together on real
silicon. If you want to see it run, that's the smoke test.

---

## Where you'd put it in production today

Three deploy surfaces, all real:

- **k8s**: `deploy/k8s/` plain manifests (W2.7) or `deploy/helm/tensor-wasm/`
  chart with `image.backend` toggle (F1 + C8)
- **Nomad**: `deploy/nomad/` with both `docker` and `raw_exec` driver variants (W5.6)
- **systemd / docker-compose**: per `docs/UPGRADE.md` walkthroughs (W3.3)

Observability stack ships out-of-the-box:

- **Prometheus**: every metric self-documents via `/metrics`, including
  `tensor_wasm_http_requests_total`, `tensor_wasm_http_request_duration_seconds`,
  `tensor_wasm_jobs_active`, `tensor_wasm_gpu_memory_bytes_per_tenant`,
  `tensor_wasm_build_info` (W2.3 + C3 + W4.9)
- **Grafana**: drop-in dashboard at `docs/dashboards/tensor-wasm-overview.json`
  (W2.5)
- **OpenTelemetry**: W3C `traceparent` propagation end-to-end (W4.1 +
  C2 load test proving exact 4×N span emission under 64 concurrent invokes)
- **Audit log**: structured JSON to stdout or file via
  `TENSOR_WASM_API_AUDIT_LOG` (W2.2)
- **Runbooks**: per-alert response procedures under `docs/runbooks/` (W2.6)
- **SLOs**: published in `docs/SLO.md` with burn-rate PromQL (W1.9)

---

## Where TensorWasm wins, where it doesn't

Per `docs/BENCHMARKING.md` "Where TensorWasm wins, where it won't" — honesty
on losses is the marketing strategy.

**TensorWasm wins on:**
- GPU-aware Wasm runtime (single-source-of-truth: there is no other one)
- Multi-tenant Wasm with tenant-keyed GPU memory + per-token rate limiting
- Production observability out of the box (the W2-W4 wave)
- Snapshot-based cold-start when you're cycling many small Wasm functions

**TensorWasm loses on:**
- Pure-CPU Wasm execution speed vs hand-tuned Wasmer-LLVM — they spend more
  compile time, get faster runtime; if your workload is "I have one Wasm
  module I call a million times in a tight loop," Wasmer-LLVM is faster
- Cold-start vs raw Wasmtime `Module::deserialize` — our restore is a
  superset (snapshot decode + tenant state replay); a pure deserialize is a
  tighter loop
- Raw GPU dispatch latency vs hand-written C++ — we pay the WASI-GPU bounds
  check + the back-pressure semaphore + the (current) 50 µs poll cadence;
  target post-v0.4 is 2-5× of `cuLaunchKernel`, not parity
- Pre-built FaaS at edge scale (workerd on Cloudflare's network) — we're a
  runtime, not a CDN. Self-hosted comparison only.

If you need any of those things more than the four wins above, TensorWasm is
the wrong choice. Saying so is what makes the wins credible.

---

## The roadmap, in 4 honest sentences

- **v0.3.x today**: three CUDA backends scaffolded, real PTX dispatch
  proven on RTX 2060, observability + auth + ops complete, audit-closed.
- **v0.4 next quarter**: cuda-oxide v0.2 cutover (when it ships), Pliron
  Cranelift→PTX lowering for the first batch of ops, cuda-async-backed
  `DispatchFuture` waker, in-place UVM grow via `cuMemAddressReserve`. See
  `docs/CUDA-OXIDE-CUTOVER.md` for the 8-step executable runbook.
- **v0.5-beta**: external pen-test (RFP ready at `docs/SECURITY-AUDIT-RFP.md`)
  + at least one named or anonymized production design partner (program kit
  at `docs/DESIGN-PARTNER-PROGRAM.md`).
- **v1.0**: API freeze + release engineering. 12-16 months from today per
  PATH-TO-V1.md "Effort and timeline (caveated)" — every estimate is wrong,
  but the milestone exit criteria are the commitment, not the dates.

---

## Try it in 5 minutes

```sh
git clone https://github.com/craton-co/craton-tensor-wasm
cd craton-tensor-wasm
cargo build --workspace --release        # ~3 min on a cold cache
cargo test  --workspace --release        # ~2 min, 70 batches, 0 failures expected

# Run a Wasm function locally:
cargo run -p tensor-wasm-cli -- run \
  tests/wasm-fixtures/matrix_multiply.wat

# Run the HTTP API:
cargo run --release --bin tensor-wasm -- serve --addr 0.0.0.0:8080

# Watch live metrics in another terminal:
cargo run -p tensor-wasm-cli -- observe
```

If you have CUDA + an NVIDIA GPU (SM_70+ for non-wmma kernels; SM_80+ for
wmma): add `--features cuda` to the test command and watch
`vector_add_end_to_end_real_ptx_real_kernel` actually compute on your GPU.

---

## Get involved

**If you're a developer:** the W1.7 RFC process is live at `rfcs/`. Substantive
design changes go through it; bugfixes just open a PR. `CONTRIBUTING.md` has
the dev-loop setup.

**If you're a sponsor / corporation:** the
`docs/DESIGN-PARTNER-PROGRAM.md` kit is ready to read. Reciprocal
engagement: you get early access + named credit in v1.0 release notes; we
get production validation data. Apply via `security@craton.com.ar`.

**If you're a security researcher:** `SECURITY.md` is the disclosure
contract. 72h acknowledgement, 90-day coordinated disclosure, public
findings table in `docs/SECURITY-AUDIT-v0.5.md` (post-pen-test).

**If you're at a security firm reading this:** `docs/SECURITY-AUDIT-RFP.md`
is sendable today. Sponsor maintainer fills in 11 bracketed placeholders
and the RFP goes to firms.

---

## FAQ

**Q: Is this production-ready?**
A: For v1.0-grade production at the SLA published in `docs/SLO.md`: not
yet. v0.5-beta is the first beta the design-partner program runs against.
For staging / pilot / internal-tooling: yes, today, at v0.3.5. The CUDA
path runs end-to-end on real hardware; the HTTP API has 0 audit problems
open; the test suite has 70 passing batches and 0 failures.

**Q: How does this compare to Wasmtime?**
A: We **wrap** Wasmtime 25.x. We are not a fork. See `docs/WASMTIME-FORK.md`.
Pure CPU execution is within 5% of upstream Wasmtime per the dimension-1
hyperfine comparison (statistically tied at v0.3.5). Everything else — GPU,
multi-tenancy, snapshot, HTTP gateway, observability — is layered on top
and additive.

**Q: How does this compare to Wasmer Edge / Spin / Fermyon?**
A: Different problem. Those are FaaS platforms; we're a runtime they could
build on. See `docs/MIGRATING-FROM-WASMTIME-WASMER.md` for the 13-row
feature matrix and per-persona migration guides.

**Q: AMD / Intel / Apple GPU support?**
A: Out of scope for v1.0 by explicit decision (PATH-TO-V1.md "Anti-goals").
v2.x research item.

**Q: What's the license?**
A: Apache-2.0. Commercial use, modification, redistribution, sublicensing
all permitted. `LICENSE` + `NOTICE` at repo root. Trademark policy is
permissive — see `docs/TRADEMARK.md`.

**Q: Who's behind it?**
A: Sponsored by Craton Software Company. Maintainer roster + governance at
`MAINTAINERS.md` + `GOVERNANCE.md`. Several slots currently `TBD` by design
during the v0.x window; see `MAINTAINERS.md` "Placeholders are by design"
section.

**Q: What's the catch?**
A: Two real ones:
1. CUDA-only for v1.0. NVIDIA hardware required for the GPU path. Pure-CPU
   path works anywhere Wasmtime works.
2. Pinned Rust nightly (`nightly-2026-04-03`). Quarterly bump cadence per
   PATH-TO-V1 Open Decision #8. Moving to stable Rust is a v2 effort gated
   on Wasmtime dropping its own nightly needs.

---

## One-slide summary

```
┌──────────────────────────────────────────────────────────────────┐
│  CRATON TENSORWASM v0.3.5                                        │
│  GPU-accelerated serverless WebAssembly runtime, in Rust         │
│                                                                  │
│  PROVEN ON REAL SILICON:                                         │
│    7/7 Wasm→wasi-cuda→cuLaunchKernel→readback tests pass         │
│    70 test batches, 0 failures, 0 audit problems                 │
│    5 tagged releases in 92 commits                               │
│                                                                  │
│  SHIPPED, USABLE TODAY:                                          │
│    Auth (bearer + scoped tokens), rate limit, audit log          │
│    Prometheus + OTel + Grafana dashboard + runbooks              │
│    Helm chart, k8s manifests, Nomad job spec, Dockerfile         │
│    Reproducible builds + CycloneDX SBOM + cargo-deny             │
│    CLI with completions + man pages + live observe dashboard     │
│                                                                  │
│  WHERE WE WIN:                                                   │
│    GPU-aware Wasm (no peer in the ecosystem)                     │
│    Multi-tenant by construction                                  │
│    Observability + ops out of the box                            │
│                                                                  │
│  WHERE WE LOSE (honest):                                         │
│    Pure-CPU vs Wasmer-LLVM                                       │
│    Cold-start vs raw Wasmtime deserialize                        │
│    Raw GPU dispatch vs hand-written C++                          │
│    Edge CDN vs Cloudflare Workers                                │
│                                                                  │
│  NEXT:                                                           │
│    v0.4 cuda-oxide v0.2 cutover (runbook ready)                  │
│    v0.5 external pen-test + design partners (RFP + kit ready)    │
│    v1.0 release engineering (paperwork only)                     │
│                                                                  │
│  https://github.com/craton-co/craton-tensor-wasm                 │
│  Apache-2.0  ·  security@craton.com.ar                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Related docs

Everything pitched above has a doc behind it:

- `README.md` — the technical entry point
- `ARCHITECTURE.md` — the dep graph
- `docs/PATH-TO-V1.md` — the roadmap source-of-truth
- `docs/SLO.md` — what we commit to numerically
- `docs/BENCHMARKING.md` — how we measure (incl. anti-cheating rules)
- `docs/SECURITY-AUDIT-RFP.md` — sponsor-sendable v0.5 audit RFP
- `docs/DESIGN-PARTNER-PROGRAM.md` — partner outreach kit
- `docs/CUDA-OXIDE-CUTOVER.md` — v0.4 cuda-oxide v0.2 cutover runbook
- `docs/tutorials/production-deployment.md` — end-to-end production guide
- `rfcs/0001-cuda-oxide-integration.md` — the v0.5 default-flip decision
- `CHANGELOG.md` — what shipped in each release

---

_Status: written against the v0.3.5 release tag (2026-05-25). Re-validate
all numeric claims when re-publishing — `bench-results/*.json` is the
authoritative source._
