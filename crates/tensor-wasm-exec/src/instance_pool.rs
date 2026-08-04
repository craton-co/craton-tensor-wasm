// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Pre-instantiated instance pool (roadmap feature #5, T37 implementation).
//!
//! Pre-spawns N instances per `(tenant_id, module_hash)` tuple and draws
//! from a per-tuple [`crossbeam_channel`] on `acquire`. Closes the
//! `Instance::new_async` cost on the warm invocation path. The pooling
//! allocator + MPK backend already exists (see `EngineConfig::PoolingMpk`);
//! this module sits on top of it, addressing the instantiation-path cost
//! that the memory backend does not.
//!
//! ## Channel layout
//!
//! Each entry in `InstancePool::pools` is a `(Sender, Receiver)` pair
//! bounded at `warm_instances_per_tuple` (floored at 1 — a capacity-0
//! `crossbeam_channel` is rendezvous-shaped and would deadlock the
//! release-to-channel path). The first [`InstancePool::acquire`] call
//! for a `(tenant, module_hash)` key spawns `warm_instances_per_tuple`
//! detached instances and seeds the channel with them. Subsequent
//! `acquire` calls drain the channel via `try_recv` (wait-free, O(1));
//! on empty they fall through to [`TensorWasmExecutor::spawn_instance`]
//! up to the executor's existing `max_instances` ceiling.
//!
//! Tenant + module isolation is enforced by the key: instance A drawn
//! by tenant T1 from module M never lands in tenant T2's channel, nor
//! in T1's channel for module M'. Both halves of the key are part of
//! the `PoolKey` tuple.
//!
//! ## Reset trade-off
//!
//! On [`InstancePool::release`], the spent instance is **discarded** and
//! a fresh one is spawned from the cached [`wasmtime::Module`] to take
//! its place in the channel. We pay for the wasmtime instantiate
//! (`Instance::new_async`) but skip the Cranelift compile entirely
//! (the module is already in the executor's per-module LRU cache).
//! Rationale:
//!
//! - Linear memory must return to its post-`start` snapshot, with no
//!   guest-written bytes leaking across tenants or invocations. The
//!   cheapest way to guarantee this without a per-module bytecode-level
//!   snapshot is to drop the [`wasmtime::Store`] and re-instantiate.
//! - Mutable globals, table state, and any allocated WASI fds beyond
//!   stdio are scrubbed for free by the same drop.
//! - The compile step (`Module::from_binary` → Cranelift) is by far the
//!   expensive part of the spawn path on multi-MiB modules; the
//!   `Instance::new_async` cost on a cached module is small but
//!   non-zero (store construction, limiter wiring, `start` execution).
//!
//! If the replacement reset exceeds ~10 ms (the deadline-aware drop
//! threshold), the new instance is dropped rather than enqueued —
//! cheaper to let the next `acquire` spawn fresh than to block invoke
//! behind a slow reset.
//!
//! ## Streaming spawns are not recycled
//!
//! When [`crate::executor::SpawnConfig::streaming`] is set (T34 SSE
//! `/invoke-stream`), the streaming context carries a one-shot
//! `mpsc::Sender` whose receiver is drained for the duration of the
//! response and then dropped. There is no way to "reset" a streaming
//! context for a subsequent invoke; the gateway constructs a fresh
//! channel for each call. [`InstancePool::release`] therefore drops
//! streaming instances unconditionally — they are never returned to
//! the warm channel.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use dashmap::DashMap;
use tensor_wasm_core::types::{InstanceId, TenantId};
use tracing::warn;

use crate::executor::{ExecError, SpawnConfig, TensorWasmExecutor};
use crate::instance::TensorWasmInstance;

/// BLAKE3 digest of a Wasm module's bytes. Re-exported so embedders that
/// hold module hashes outside the pool (e.g. for diagnostic logging)
/// share the same shape.
pub type ModuleHash = [u8; 32];

/// Hard cap on how long a single reset+enqueue may take before the
/// replacement instance is dropped instead of being placed in the
/// channel. Chosen as a deadline-aware "fast enough or skip" threshold:
/// at the bench loop's hot path, blocking invoke for more than 10 ms on
/// a pool replenishment defeats the latency win the pool exists for.
///
/// Pool drops that exceed this budget log a `warn!` so operators see
/// the regression — a sustained miss usually indicates Cranelift cache
/// thrash or pathological `start`-function work.
const POOL_RESET_DEADLINE: Duration = Duration::from_millis(10);

/// Pool configuration. One pool per executor.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct InstancePoolConfig {
    /// Pre-spawn N instances per (tenant, module-hash) tuple. Default 0
    /// = pool disabled (every `acquire` falls through to
    /// [`TensorWasmExecutor::spawn_instance`]).
    ///
    /// The pool's channel capacity is `max(warm_instances_per_tuple, 1)`
    /// — a rendezvous channel (capacity 0) would deadlock the release
    /// path, so we floor the channel size at 1 even when pre-warming is
    /// disabled. With a 0 setting the warm channel is empty at startup
    /// and the first `release` populates it; this gives the second
    /// `acquire` of the same tuple a warm draw at the cost of paying
    /// cold-start on the first.
    pub warm_instances_per_tuple: usize,

    /// Maximum number of pre-spawned instances across all tuples. Pools
    /// honour this cap before spawning a new tuple. Default 0 = unlimited
    /// (within the executor's `max_instances`).
    pub max_total_warm: usize,
}

impl InstancePoolConfig {
    /// Construct a pool config. Kept available for foreign-crate callers
    /// even though [`InstancePoolConfig`] is `#[non_exhaustive]`.
    pub fn new(warm_instances_per_tuple: usize, max_total_warm: usize) -> Self {
        Self {
            warm_instances_per_tuple,
            max_total_warm,
        }
    }
}

/// Key for the per-tuple warm-pool map. Both halves are load-bearing:
/// the tenant id enforces tenant isolation (T2 must never observe an
/// instance previously used by T1), and the module hash enforces module
/// isolation (a tenant's module A invoke must never observe an instance
/// compiled from module B).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PoolKey {
    tenant_id: TenantId,
    module_hash: ModuleHash,
}

/// Per-tuple entry: a bounded sender/receiver pair plus a cached
/// [`wasmtime::Module`] so the reset path skips re-compilation.
struct PoolEntry {
    sender: Sender<TensorWasmInstance>,
    receiver: Receiver<TensorWasmInstance>,
    module: wasmtime::Module,
    /// Capacity the channel was constructed with (mirrors
    /// `channel_capacity()`, itself `warm_instances_per_tuple + 1` floored).
    ///
    /// DEAD CODE, retained deliberately (DOCS finding): it is written at
    /// `build_entry` time but never read — the runtime release path keys off
    /// `try_send` returning `TrySendError::Full`, not this field, and
    /// `crossbeam_channel` does not expose the bounded capacity back to us, so
    /// stashing it here is the only record of the per-entry bound. Kept for
    /// observability / future diagnostics (a `capacity()` accessor would light
    /// it up the moment a consumer needs it). The `#[allow(dead_code)]` is
    /// load-bearing: without it `-D warnings` CI fails on the unread field.
    #[allow(dead_code)]
    capacity: usize,
}

/// Pre-instantiated instance pool. Backed by a per-tuple
/// `crossbeam_channel` of pre-spawned instances; see module docs for the
/// channel layout, reset trade-off, and streaming-spawn carve-out.
pub struct InstancePool {
    cfg: InstancePoolConfig,
    /// Per-(tenant, module-hash) channel registry. [`DashMap`] gives
    /// wait-free reads on the hot acquire path: `try_recv` is a single
    /// atomic operation against the per-entry channel.
    pools: Arc<DashMap<PoolKey, Arc<PoolEntry>>>,
    /// Per-key single-flight guard for the first-build pre-spawn loop.
    /// Concurrent first-`acquire`s for the same `(tenant, module_hash)`
    /// share one [`tokio::sync::OnceCell`]: exactly one runs the
    /// (`await`-heavy) pre-spawn loop while the losers park on
    /// `get_or_try_init` and observe the winner's [`PoolEntry`] instead
    /// of redundantly Cranelift-compiling/instantiating their own
    /// (thundering-herd fix). The shard write-guard on this map is only
    /// held long enough to clone the `Arc<OnceCell>` — never across the
    /// `await` inside the initializer — so it cannot deadlock the build.
    inflight: Arc<DashMap<PoolKey, Arc<tokio::sync::OnceCell<Arc<PoolEntry>>>>>,
    /// Total warm-and-in-channel count across all entries. Counts UP on
    /// pre-spawn / release-enqueue and DOWN on acquire-from-channel /
    /// reset-failure-drop. Operators observe this via
    /// [`Self::warm_count`].
    warm_total: Arc<AtomicUsize>,
    /// Total successful pool draws (hits + misses). Inc'd on every
    /// [`Self::acquire`] regardless of outcome; the
    /// [`Self::hit_count`] / [`Self::miss_count`] split below records
    /// which path was taken.
    draws_total: Arc<AtomicUsize>,
    /// Subset of `draws_total` that hit a warm instance in the channel.
    hit_count: Arc<AtomicUsize>,
    /// Subset of `draws_total` that fell through to a fresh spawn.
    miss_count: Arc<AtomicUsize>,
    /// Number of releases that resulted in the instance being dropped
    /// (channel full, reset failed, streaming carve-out, or deadline
    /// exceeded). Distinct from the executor's
    /// `instance_terminations_total`: terminations come from the
    /// registry path, drops come from the pool path.
    drops_total: Arc<AtomicUsize>,
}

impl InstancePool {
    /// Construct a new pool with the given config.
    pub fn new(cfg: InstancePoolConfig) -> Self {
        Self {
            cfg,
            pools: Arc::new(DashMap::new()),
            inflight: Arc::new(DashMap::new()),
            warm_total: Arc::new(AtomicUsize::new(0)),
            draws_total: Arc::new(AtomicUsize::new(0)),
            hit_count: Arc::new(AtomicUsize::new(0)),
            miss_count: Arc::new(AtomicUsize::new(0)),
            drops_total: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Effective channel capacity per tuple. Floored at 1 so the bounded
    /// channel does not degenerate to a rendezvous shape (which would
    /// deadlock the release path).
    fn channel_capacity(&self) -> usize {
        self.cfg.warm_instances_per_tuple.max(1)
    }

    /// Acquire a pre-spawned instance or fall through to a fresh spawn.
    ///
    /// Algorithm (T37):
    /// 1. Compute the `(tenant, module_hash)` key — building the entry
    ///    on first observation by compiling the module (via the
    ///    executor's cached path), allocating a bounded channel of the
    ///    configured size, and pre-spawning `warm_instances_per_tuple`
    ///    detached instances into it.
    /// 2. `try_recv` on the per-tuple channel. On hit, register the
    ///    drawn instance with the executor and return a
    ///    [`PooledInstance`] handle.
    /// 3. On miss, fall through to [`TensorWasmExecutor::spawn_instance`]
    ///    so the call still proceeds (admission control still applies —
    ///    the executor's `max_instances` cap fires here if saturated).
    ///
    /// Streaming spawns ([`SpawnConfig::streaming`] set) bypass the
    /// warm-channel try_recv and always fall through to a fresh spawn:
    /// the streaming context is one-shot (the gateway-held receiver is
    /// drained per call), so the warm channel never holds streaming
    /// instances.
    pub async fn acquire(
        &self,
        executor: &TensorWasmExecutor,
        wasm: &[u8],
        cfg: SpawnConfig,
    ) -> Result<PooledInstance, ExecError> {
        self.draws_total.fetch_add(1, Ordering::Relaxed);

        // Streaming carve-out: never serve a streaming spawn from the
        // warm channel. The gateway built a fresh channel for THIS
        // invoke; reusing a stale streaming-state instance would
        // either drop the new emit-chunk calls on the floor or leak
        // the gateway's receiver across requests. Fall straight through
        // to spawn_instance.
        if cfg.streaming.is_some() {
            self.miss_count.fetch_add(1, Ordering::Relaxed);
            let id = executor.spawn_instance(cfg, wasm).await?;
            return Ok(PooledInstance {
                inner: Some(id),
                origin: None,
            });
        }

        // Input carve-out (M-1): never serve an input-bearing spawn from
        // the warm channel. Warm instances bake the FIRST caller's
        // `SpawnConfig::input` into their `InstanceState` at build time;
        // `register_pooled_instance` resets only the `instance_id`, never
        // re-staging this caller's `input`. Reusing a warm instance for a
        // spawn that carries its own input would let request N+1 observe
        // request N's staged input bytes via the guest input channel
        // (same-tenant input bleed). Fall straight through to
        // `spawn_instance` so this caller's `input` is staged onto a
        // fresh instance, and stamp `origin: None` so `release` drops it
        // rather than recycling its input-tainted state — mirroring the
        // streaming carve-out above.
        if !cfg.input.is_empty() {
            self.miss_count.fetch_add(1, Ordering::Relaxed);
            let id = executor.spawn_instance(cfg, wasm).await?;
            return Ok(PooledInstance {
                inner: Some(id),
                origin: None,
            });
        }

        // Compute the digest now so we can stamp the origin on the
        // returned handle. `blake3::hash` is SIMD-fast; the executor's
        // module cache uses the same key shape.
        let module_hash: ModuleHash = *blake3::hash(wasm).as_bytes();

        // Ensure the per-tuple entry exists (and is pre-warmed). This
        // returns the cached Module alongside the channel handles so
        // the reset-on-release path can re-instantiate without paying
        // for another compile.
        let entry = self.ensure_entry(executor, wasm, &cfg, module_hash).await?;

        // Try the warm channel first. `try_recv` is wait-free.
        match entry.receiver.try_recv() {
            Ok(inst) => {
                self.warm_total.fetch_sub(1, Ordering::Relaxed);
                self.hit_count.fetch_add(1, Ordering::Relaxed);
                let id = executor.register_pooled_instance(inst)?;
                Ok(PooledInstance {
                    inner: Some(id),
                    origin: Some(module_hash),
                })
            }
            Err(_empty) => {
                // Warm pool exhausted (or never seeded with
                // `warm_instances_per_tuple = 0`). Fall through to a
                // fresh spawn — the executor's `max_instances`
                // admission control still applies here.
                self.miss_count.fetch_add(1, Ordering::Relaxed);
                let id = executor.spawn_instance(cfg, wasm).await?;
                Ok(PooledInstance {
                    inner: Some(id),
                    origin: Some(module_hash),
                })
            }
        }
    }

    /// Release an instance back to the pool.
    ///
    /// The instance is detached from the executor's registry (the slot
    /// stays charged for the moment) and dropped — its linear memory
    /// could be holding sensitive guest state, so "reset" here means
    /// "re-instantiate", never "rewind". A fresh replacement is then
    /// spawned from the cached [`wasmtime::Module`] and `try_send`'d to
    /// the channel; on full / send-error / streaming carve-out / reset
    /// deadline exceeded, the replacement is dropped and the slot is
    /// released.
    ///
    /// `pooled` carries the originating module hash so the release path
    /// dispatches to the correct per-tuple channel without re-hashing
    /// (the release path no longer has the wasm bytes on hand).
    pub async fn release(
        &self,
        executor: &TensorWasmExecutor,
        pooled: PooledInstance,
        cfg: &SpawnConfig,
    ) {
        let id = pooled.id();
        let origin = pooled.origin();
        // Defuse the handle so its `Drop` is a no-op — we own the
        // lifecycle from here.
        let _ = pooled.into_inner();

        // Detach the spent instance from the registry. The slot stays
        // charged — we'll release it explicitly below after deciding
        // whether to enqueue a replacement.
        let spent = match executor.detach_pooled_instance(id).await {
            Some(inst) => inst,
            None => {
                // Already gone — nothing to release. This is a
                // double-release race (e.g. the auto-terminate guard
                // fired). Skip silently; the slot accounting is the
                // executor's responsibility on the terminate path.
                return;
            }
        };

        // Drop the spent instance BEFORE we attempt to spawn a
        // replacement so its slot is freed first — under tight
        // `max_instances`, charging the replacement before releasing
        // the spent one would trip CapacityExhausted.
        drop(spent);
        executor.release_instance_slot(cfg.tenant_id);

        // Streaming spawns are never recycled — count the drop and
        // stop here. See module docs.
        if cfg.streaming.is_some() || origin.is_none() {
            self.drops_total.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let module_hash = origin.expect("origin is Some by the check above");

        let key = PoolKey {
            tenant_id: cfg.tenant_id,
            module_hash,
        };
        let entry = match self.pools.get(&key) {
            Some(e) => e.value().clone(),
            None => {
                // Entry vanished (shutdown / config reload race). The
                // spent slot is already released; nothing to do.
                self.drops_total.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };

        // Global cap check: don't replenish past `max_total_warm`.
        if self.cfg.max_total_warm > 0
            && self.warm_total.load(Ordering::Relaxed) >= self.cfg.max_total_warm
        {
            self.drops_total.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Spawn the replacement under the reset-deadline budget.
        // Re-instantiating from a cached module skips the Cranelift
        // compile entirely; only the `Instance::new_async` cost
        // remains. If it overruns the budget we drop the replacement
        // and let the next acquire spawn fresh on its own time.
        let start = Instant::now();
        let replacement = match executor
            .rebuild_pooled_from_module(cfg, &entry.module)
            .await
        {
            Ok(inst) => inst,
            Err(err) => {
                // Reset failed — defensible failure modes are
                // `CapacityExhausted` (we lost the race to a peer
                // acquire), `EpochTickerNotRunning` (the ticker died
                // mid-flight), or a wasmtime trap inside `start`.
                warn!(
                    target: "tensor_wasm_exec::instance_pool",
                    error = %err,
                    "pool reset failed; replacement instance dropped",
                );
                self.drops_total.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        let elapsed = start.elapsed();
        if elapsed > POOL_RESET_DEADLINE {
            warn!(
                target: "tensor_wasm_exec::instance_pool",
                elapsed_ms = elapsed.as_millis() as u64,
                budget_ms = POOL_RESET_DEADLINE.as_millis() as u64,
                "pool reset exceeded deadline; replacement instance dropped to keep invoke latency bounded",
            );
            drop(replacement);
            executor.release_instance_slot(cfg.tenant_id);
            self.drops_total.fetch_add(1, Ordering::Relaxed);
            return;
        }

        // Enqueue the fresh instance. On `Full` the channel is at
        // capacity — drop the replacement (the channel cannot hold
        // more than `warm_instances_per_tuple`).
        match entry.sender.try_send(replacement) {
            Ok(()) => {
                self.warm_total.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Full(dropped)) | Err(TrySendError::Disconnected(dropped)) => {
                drop(dropped);
                executor.release_instance_slot(cfg.tenant_id);
                self.drops_total.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Ensure the per-tuple entry exists, creating it and pre-warming
    /// it on the first observation. The cached [`wasmtime::Module`] is
    /// produced as a side-effect (via the executor's per-module LRU
    /// cache) so subsequent reset calls don't re-run Cranelift.
    async fn ensure_entry(
        &self,
        executor: &TensorWasmExecutor,
        wasm: &[u8],
        cfg: &SpawnConfig,
        module_hash: ModuleHash,
    ) -> Result<Arc<PoolEntry>, ExecError> {
        // On the hot path the entry already exists and we return early
        // without touching the module compiler.
        let key = PoolKey {
            tenant_id: cfg.tenant_id,
            module_hash,
        };
        if let Some(entry) = self.pools.get(&key) {
            return Ok(entry.value().clone());
        }

        // First observation for this tuple — pre-warm under a per-key
        // single-flight guard. Two concurrent first-`acquire`s both miss
        // the `pools.get` fast path above; without this guard they would
        // each run the full pre-spawn loop (Cranelift-compiling and
        // instantiating `warm_n` instances apiece) and the loser would
        // then have to drain + refund its wasted work. Instead we share
        // one `OnceCell` per key so only the winner runs the build; the
        // losers park on `get_or_try_init` and observe the winner's
        // `PoolEntry`.
        //
        // The shard write-guard on `inflight` is dropped immediately
        // after we clone the `Arc<OnceCell>` (the `entry`/`or_insert`
        // call below borrows the shard only for that statement), so it is
        // never held across the `await` inside the initializer — no
        // shard-guard-across-await deadlock.
        let cell = self
            .inflight
            .entry(key.clone())
            .or_insert_with(|| Arc::new(tokio::sync::OnceCell::new()))
            .value()
            .clone();

        // `get_or_try_init` runs the initializer on exactly one task per
        // `OnceCell`; concurrent callers await its completion and clone
        // the resulting `Arc<PoolEntry>`. On initializer error every
        // waiter observes the same `Err` and no `PoolEntry` is cached, so
        // a later `acquire` retries the build cleanly.
        let entry = cell
            .get_or_try_init(|| self.build_entry(executor, wasm, cfg, key.clone()))
            .await?
            .clone();

        // The entry now lives in `pools` (the fast path will find it) and
        // in the cell. Drop the single-flight cell so `inflight` does not
        // grow unbounded; a straggler that already cloned this `Arc` is
        // unaffected, and a fresh caller hits the `pools` fast path.
        self.inflight.remove(&key);

        Ok(entry)
    }

    /// Build and register the per-tuple [`PoolEntry`]: allocate the
    /// bounded channel, run the pre-spawn loop, and insert into `pools`.
    /// Invoked at most once per key via the `inflight` `OnceCell`, so the
    /// slot/`warm_total` accounting here is driven by a single builder —
    /// there is no lost-race refund to perform. The previous
    /// double-build defence (drain-local-channel + refund slots on a lost
    /// `DashEntry` race) is therefore gone: a second builder can no
    /// longer exist for the same key.
    async fn build_entry(
        &self,
        executor: &TensorWasmExecutor,
        wasm: &[u8],
        cfg: &SpawnConfig,
        key: PoolKey,
    ) -> Result<Arc<PoolEntry>, ExecError> {
        // We use `build_pooled_instance` for the first instance so the
        // compiled module + hash are returned, then
        // `rebuild_pooled_from_module` for the rest (no need to
        // re-compute the hash or re-check the cache).
        let warm_n = self.cfg.warm_instances_per_tuple;
        // Effective channel capacity must accommodate `warm_n` plus at
        // least the slot we'll fill on a single release.
        let cap = self.channel_capacity();
        let (sender, receiver) = bounded::<TensorWasmInstance>(cap);

        let mut module_opt: Option<wasmtime::Module> = None;
        for i in 0..warm_n {
            // Global cap check: respect `max_total_warm`.
            if self.cfg.max_total_warm > 0
                && self.warm_total.load(Ordering::Relaxed) >= self.cfg.max_total_warm
            {
                break;
            }
            if i == 0 {
                // PERF finding 6: pass the digest the pool already computed
                // (`key.module_hash`) so the build does not re-hash the wasm.
                let (inst, module, _) = executor
                    .build_pooled_instance_with_hash(cfg, wasm, key.module_hash)
                    .await?;
                // Single builder per key + a channel sized at `cap >=
                // warm_n` means `try_send` cannot fail here for capacity
                // reasons; treat any error defensively by releasing the
                // slot we charged so admission control stays balanced.
                if sender.try_send(inst).is_err() {
                    executor.release_instance_slot(cfg.tenant_id);
                } else {
                    self.warm_total.fetch_add(1, Ordering::Relaxed);
                }
                module_opt = Some(module);
            } else {
                let module = module_opt.as_ref().expect("module captured on i==0");
                let inst = executor.rebuild_pooled_from_module(cfg, module).await?;
                if sender.try_send(inst).is_err() {
                    executor.release_instance_slot(cfg.tenant_id);
                } else {
                    self.warm_total.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        // If `warm_n == 0` we never compiled — do it now so the cached
        // Module is available for the reset path.
        let module = match module_opt {
            Some(m) => m,
            None => {
                // `build_pooled_instance` charges a slot we don't
                // want here (we're only after the module). Use the
                // module-cache path directly instead.
                //
                // Note: this branch only runs when
                // `warm_instances_per_tuple == 0`. We still need a
                // cached module to fuel the reset path on the first
                // release after a miss-spawn.
                //
                // The executor's `compile_module_cached` is async
                // because Cranelift compile is dispatched to a
                // blocking pool; we call through `build_pooled_instance`
                // here too and immediately release the slot + drop the
                // instance — simplest path that keeps every internal
                // invariant honest.
                // PERF finding 6: reuse the pool's precomputed digest.
                let (inst, module, _) = executor
                    .build_pooled_instance_with_hash(cfg, wasm, key.module_hash)
                    .await?;
                drop(inst);
                executor.release_instance_slot(cfg.tenant_id);
                module
            }
        };
        let entry = Arc::new(PoolEntry {
            sender,
            receiver,
            module,
            capacity: cap,
        });
        // Single-flight: this is the only builder for `key`, so the
        // insert is uncontended. Use `entry()` purely to insert; a
        // pre-existing value would be a logic error (the `OnceCell`
        // already serialises us), so overwrite-by-insert is fine.
        self.pools.insert(key, entry.clone());
        Ok(entry)
    }

    /// Number of currently warm (in-channel) instances across all tuples.
    pub fn warm_count(&self) -> usize {
        self.warm_total.load(Ordering::Relaxed)
    }

    /// Total successful pool draws (hits + misses). Monotonic counter.
    pub fn draws_total(&self) -> usize {
        self.draws_total.load(Ordering::Relaxed)
    }

    /// Draws that hit a warm instance in the channel. Monotonic counter.
    pub fn hit_count(&self) -> usize {
        self.hit_count.load(Ordering::Relaxed)
    }

    /// Draws that fell through to a fresh spawn. Monotonic counter.
    pub fn miss_count(&self) -> usize {
        self.miss_count.load(Ordering::Relaxed)
    }

    /// Number of pool drops (channel full, reset failure, streaming
    /// carve-out, or reset deadline exceeded). Monotonic counter.
    pub fn drops_total(&self) -> usize {
        self.drops_total.load(Ordering::Relaxed)
    }

    /// Borrow the pool's configuration.
    pub fn config(&self) -> &InstancePoolConfig {
        &self.cfg
    }

    /// Drain every warm channel and release its instances. Called by
    /// embedders that want to free pool-held wasmtime resources without
    /// dropping the executor (e.g. on a config reload).
    ///
    /// Idempotent — calling on an already-empty pool is a no-op. Slot
    /// accounting on the executor is restored: every instance pulled
    /// out is dropped, and the corresponding slot is released.
    pub fn shutdown(&self, executor: &TensorWasmExecutor) {
        for entry in self.pools.iter() {
            let tenant = entry.key().tenant_id;
            while let Ok(inst) = entry.value().receiver.try_recv() {
                drop(inst);
                executor.release_instance_slot(tenant);
                self.warm_total.fetch_sub(1, Ordering::Relaxed);
                self.drops_total.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// RAII wrapper around a pool-drawn instance.
///
/// On the success path the caller's `invoke` loop calls
/// [`InstancePool::release`] explicitly before this handle drops, which
/// is what returns the instance (via reset + fresh-replacement) to the
/// warm channel. If the handle is dropped *without* an explicit
/// release — caller panic, future cancellation, etc. — the instance
/// remains registered with the executor and the existing auto-terminate
/// drop-guard (api S-20) in `call_export_with_args_then_terminate`
/// catches it. A bare-drop without an outer terminate guard leaks the
/// registry entry; T37 keeps that case out of scope.
///
/// The handle carries the originating module hash so
/// [`InstancePool::release`] can dispatch to the correct per-tuple
/// channel without re-hashing the wasm bytes (which the release path
/// no longer has on hand).
pub struct PooledInstance {
    inner: Option<InstanceId>,
    /// Module hash for the (tenant, module) tuple this instance was
    /// drawn from. `None` for the streaming carve-out path, which
    /// never returns to a channel — release simply drops these.
    origin: Option<ModuleHash>,
}

impl PooledInstance {
    /// Borrow the underlying [`InstanceId`].
    pub fn id(&self) -> InstanceId {
        self.inner.expect("PooledInstance was already returned")
    }

    /// Take ownership of the underlying [`InstanceId`] (caller becomes
    /// responsible for the lifecycle).
    pub fn into_inner(mut self) -> InstanceId {
        self.inner
            .take()
            .expect("PooledInstance was already returned")
    }

    /// Module hash this handle was drawn from (when known). Streaming
    /// spawns return `None` — they bypass the warm channel.
    pub(crate) fn origin(&self) -> Option<ModuleHash> {
        self.origin
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_disabled() {
        let cfg = InstancePoolConfig::default();
        assert_eq!(cfg.warm_instances_per_tuple, 0);
        assert_eq!(cfg.max_total_warm, 0);
    }

    #[test]
    fn warm_count_starts_at_zero() {
        let pool = InstancePool::new(InstancePoolConfig::default());
        assert_eq!(pool.warm_count(), 0);
    }

    #[test]
    fn config_round_trips() {
        let pool = InstancePool::new(InstancePoolConfig {
            warm_instances_per_tuple: 4,
            max_total_warm: 32,
        });
        assert_eq!(pool.config().warm_instances_per_tuple, 4);
        assert_eq!(pool.config().max_total_warm, 32);
    }

    #[test]
    fn channel_capacity_floor_at_one() {
        // A `warm_instances_per_tuple = 0` config must still allocate a
        // capacity-1 channel — a rendezvous channel would deadlock the
        // release path.
        let pool = InstancePool::new(InstancePoolConfig::default());
        assert_eq!(pool.channel_capacity(), 1);
    }
}
