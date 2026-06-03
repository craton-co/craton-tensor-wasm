# Cooperative deadlines via `wasi:scheduler/host`

*Roadmap feature #4 — see [PATH-TO-V1.md § Post-v0.3.6 strategic features
item 4](PATH-TO-V1.md#4-cooperative-deadlines-via-wasi-yield).*

TensorWasm enforces per-instance deadlines via two complementary
mechanisms:

1. **Epoch interruption (hard floor).** The wasmtime engine's epoch
   counter advances every `epoch_tick` (default 10 ms). Each store has
   a per-call epoch deadline derived from the configured
   `SpawnConfig::deadline`; once the counter passes the deadline the
   guest is preempted at the next safe point regardless of what it is
   doing. This applies to every guest, including adversarial ones,
   and is the security-relevant guarantee.
2. **Cooperative yield (this surface).** A new
   `wasi:scheduler/host@0.1.0` interface gives well-behaved guests a
   way to offer suspend points at safe boundaries (per matrix tile,
   between kernel dispatches, etc.). The host returns a non-zero code
   when the per-instance deadline is approaching and the guest is
   expected to wind down voluntarily.

The cooperative path **does not replace** the epoch interrupt — it
just trims the wait between "deadline approaching" and "guest
actually stops" from up to one epoch tick down to one yield-loop
iteration. The improvement is most visible on P99 latency under MPS
contention, where the difference between 10 ms and one yield-loop
iteration directly determines how long tail requests block other
tenants.

## Interface

The WIT lives in
`crates/tensor-wasm-wasi-gpu/wit/wasi-scheduler.wit`:

```wit
package wasi:scheduler@0.1.0;

interface host {
    yield: func() -> u32;
    deadline-remaining-ms: func() -> u32;
}
```

`yield` returns one of:

| Code | Name                          | Meaning                                                                  |
|------|-------------------------------|--------------------------------------------------------------------------|
| 0    | `YIELD_CODE_CONTINUE`         | Continue execution. Either no deadline or comfortable budget remaining.  |
| 1    | `YIELD_CODE_DEADLINE_APPROACHING` | Remaining budget < ~10 ms. SHOULD wind down — finish current tile.   |
| 2    | `YIELD_CODE_STOP`             | Deadline elapsed. MUST stop. Epoch interrupt will trip at the next tick. |

Guests SHOULD treat any non-zero value as "wind down and return"
to remain forward-compatible with future codes.

`deadline-remaining-ms` returns milliseconds of budget remaining
or `u32::MAX` (the sentinel) when the instance has no configured
deadline. The sentinel value is intentionally indistinguishable from
"very large finite budget" so guests cannot fingerprint whether the
host applied a deadline or not.

## When should guests yield?

The runtime cost of `yield` is a single host-function call: an atomic
increment plus an `Instant::elapsed()` comparison. Cheap, but not
free. The right cadence is "at every safe boundary in the inner
loop":

- **Per tile** in a matmul kernel — between processing each
  `(tile_row, tile_col)` block.
- **Between kernel dispatches** in a multi-kernel pipeline — between
  calls to `wasi_cuda_launch`.
- **Per iteration** of a long-running loop that doesn't naturally
  contain a `wasi_cuda_*` call (which would already give the
  scheduler a host-call hook).

Calling `yield` thousands of times per millisecond is wasteful;
calling it less than once per millisecond defeats the purpose. A
useful rule of thumb is "at least once per 1 ms of compute, at most
once per 10 µs."

## Security notes

- **This is COOPERATIVE.** A malicious guest that never calls `yield`
  pays no penalty — the epoch interrupt still trips at the configured
  deadline. The cooperative path is a P99 optimisation for the
  well-behaved majority.
- **No capability gate.** Unlike `wasi:cuda/host`, the scheduler
  surface has no `enable_scheduler` flag. A guest that imports the
  interface always observes a working surface: with no deadline
  configured (the unbounded case), `yield` is a no-op CONTINUE and
  `deadline-remaining-ms` returns `u32::MAX`. There are no resources
  to gate.
- **No state leak.** The yield counter (`SchedulerContext::yield_count`)
  is host-private; the guest cannot read it. The deadline-remaining
  value the guest can read is derived only from the configured
  deadline plus the elapsed wall-clock, both of which the host
  controls; it does not reflect any cross-instance state.

## Wiring (embedder reference)

```rust
use tensor_wasm_wasi_gpu::scheduler::{add_scheduler_to_linker, SchedulerContext};

struct MyStore { scheduler: SchedulerContext, /* ... */ }

let mut linker: wasmtime::Linker<MyStore> = wasmtime::Linker::new(&engine);
add_scheduler_to_linker(&mut linker, |s: &MyStore| &s.scheduler)?;
```

`tensor-wasm-exec` wires this automatically: `InstanceState` carries
a `SchedulerContext` whose budget mirrors `SpawnConfig::deadline`,
and `TensorWasmExecutor::call_export` re-arms it at the start of each
call alongside the wasmtime epoch deadline.

## See also

- `crates/tensor-wasm-wasi-gpu/src/scheduler.rs` — host-side
  implementation, unit tests, and the `SchedulerContext` API.
- `crates/tensor-wasm-wasi-gpu/tests/scheduler_yield.rs` — end-to-end
  integration tests through a wasmtime `Linker`.
- `docs/PATH-TO-V1.md#post-v036-strategic-features` — strategic
  rationale and risk assessment.
