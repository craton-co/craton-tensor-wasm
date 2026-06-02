# Pre-instantiated instance pool

Roadmap feature #5. Closes the `Instance::new_async` cost on the warm
invocation path by pre-spawning N instances per `(tenant, module-hash)`
tuple under MPS and drawing from a channel on each `invoke`.

## Motivation

Cold-start of a TensorWasm instance today walks the full Wasmtime
`Instance::new_async` path on every `spawn_instance` call: module
lookup in the per-engine LRU cache, store construction, resource
limiter wiring, epoch deadline arming, and the `Instance::new_async`
await itself (which runs the guest's `start` function inside the
async runtime). For short-lived RPC-shaped guests — the typical
shape of an embedded-AI workload — this overhead dominates the
useful work.

The pre-instantiated pool keeps a small queue of fully-instantiated,
ready-to-call instances per `(tenant, module-hash)` tuple. On
`acquire`, the executor draws from the queue (O(1), wait-free in
the common case via `crossbeam-channel`) instead of spawning. On
`drop` (return-to-pool), the instance is reset and put back on
the queue.

This is complementary to the pooling allocator + MPK memory backend
(`EngineConfig::backend = MemoryBackend::PoolingMpk`): that one
addresses linear-memory allocation cost; the instance pool addresses
the rest of the instantiation path.

## Status: Wired (T37)

As of v0.3.7 (T37) the warm pool is wired through the invoke path.
The public API surface — `InstancePool`, `InstancePoolConfig`,
`PooledInstance`, `TensorWasmExecutor::with_instance_pool` — is stable,
and `InstancePool::acquire` now draws a pre-spawned instance from the
per-`(tenant, module-hash)` channel (with reset-on-return) instead of
falling through to `spawn_instance`. See
[`FEATURE-STATUS.md`](FEATURE-STATUS.md) for the canonical status.

Embedders opt in by constructing a pool and attaching it via the
builder.

## Configuration

```rust
use tensor_wasm_exec::instance_pool::{InstancePool, InstancePoolConfig};

let pool = InstancePool::new(InstancePoolConfig {
    // Pre-spawn N instances per (tenant, module-hash) tuple.
    // 0 disables pooling (the v0.3.6 default).
    warm_instances_per_tuple: 4,
    // Global cap across all tuples. 0 = unlimited (within the
    // executor's max_instances).
    max_total_warm: 256,
});
```

## Reset-on-return contract (v0.4)

When a `PooledInstance` is dropped, v0.4 will:

1. **Reset linear memory** to the post-`start`-function snapshot. The
   pooling-allocator path can do this by remapping the underlying
   pages; the unified-buffer path zeroes the writable region.
2. **Reset table state.** Function-table entries return to their
   declared initial values.
3. **Reset globals.** Mutable globals return to their `start`-time
   values; immutable globals are not touched.
4. **Cancel any in-flight epoch deadline.** The store's epoch
   counter is rearmed to `u64::MAX` until the next `acquire`.
5. **Return the instance to the warm channel.** If the channel is
   full (because `max_total_warm` is reached), the instance is
   terminated instead.

If any step fails, the instance is terminated rather than returned —
a stuck reset must never poison the pool. Failed resets increment
a Prometheus counter (`tensor_wasm_pool_reset_failed_total`,
labelled by tenant) so operators see the problem.

## v0.4 implementation plan

1. Replace the `HashMap<PoolKey, ()>` placeholder in `InstancePool`
   with `HashMap<PoolKey, crossbeam_channel::Receiver<TensorWasmInstance>>`.
   Each receiver is paired with a sender held by the pool itself
   for return-to-pool delivery.
2. Add a `prewarm(executor, wasm, cfg, n)` API that compiles the
   module once and spawns `n` instances into the channel for that
   tuple, respecting `max_total_warm`.
3. Implement `Drop` on `PooledInstance` to run the reset sequence
   above and send the instance back through the channel.
4. Add metrics: `tensor_wasm_pool_warm_instances` gauge,
   `tensor_wasm_pool_draws_total` counter (labelled `hit | miss`),
   `tensor_wasm_pool_reset_failed_total` counter.
5. Wire the API layer's `invoke` handler to call `acquire` instead
   of `spawn_instance` when a pool is attached, falling back to the
   bare spawn for one-shot calls.

## Cross-references

- `EngineConfig::PoolingMpk` — the pooling-allocator + MPK linear
  memory backend that this pool sits on top of.
- `EngineConfig::max_instances` — the per-executor live-instance
  cap; the pool counts warm instances against this cap.
- `PATH-TO-V1.md` — feature #5 in the post-v0.3.6 strategic
  features list.
