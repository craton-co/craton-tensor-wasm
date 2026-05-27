// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//! Host-side implementations of `tensor-wasm:jit/host::{dispatch, alloc, free}`.
//!
//! Registered on the wasmtime [`Linker`] when the auto-offload rewrite has
//! swapped function bodies for dispatch trampolines (see
//! [`tensor_wasm_jit::rewrite`]). The trio implements the v0.1.0 ABI documented
//! in the `tensor_wasm_jit::rewrite` module-level docs.
//!
//! ## `dispatch`
//!
//! ```text
//! (func $dispatch
//!   (param i64 i64 i32 i32 i32) (result i32))
//!   ;;     fp_lo fp_hi sptr alen rlen   error
//! ```
//!
//! - Looks up the cached kernel by `(fp_lo|fp_hi as u64, sm_version)`.
//! - On a cache hit (no-CUDA path): reads the args region, runs the
//!   host-side reference implementation (a simple element-wise add of the
//!   two halves of `args` — enough for the end-to-end test to validate
//!   the marshalling round-trip), writes the result back into the results
//!   region, and returns `0`.
//! - On a cache miss: emits a `tracing::warn!` and returns `-1`. The
//!   trampoline drops the error code; for v0.1.0 the guest still sees
//!   zero-filled results.
//!
//! ## `alloc` / `free`
//!
//! ```text
//! (func $alloc (param i32 (size)) (result i32 (ptr)))
//! (func $free  (param i32 (ptr)) (param i32 (size)))
//! ```
//!
//! Backed by a per-store bump arena reserved from the upper region of the
//! guest's first linear memory (`memory 0`). The arena starts
//! [`SCRATCH_ARENA_BYTES`] bytes below the current memory length and grows
//! downward; `free` returns slots to the arena in LIFO order. The first
//! call grows the memory by one page if there isn't already enough room.
//! Multi-page reservations beyond [`SCRATCH_ARENA_BYTES`] return `-1` and
//! the trampoline's subsequent dispatch call propagates the failure.

use std::sync::Arc;

use tensor_wasm_core::types::TenantId;
use tensor_wasm_jit::cache::{CacheKey, KernelCache};
use wasmtime::{Caller, Linker};

use crate::instance::InstanceState;

/// Trait for extracting the calling tenant from a wasmtime `Store<T>` payload.
///
/// The JIT dispatch host fn MUST build its [`CacheKey`] from the calling
/// tenant's identity (taken from trusted `Store` state) rather than from
/// guest-supplied bytes. Wiring this through a trait keeps
/// [`add_jit_dispatch_to_linker`] generic over `T` while still forcing every
/// store type that wants to host the dispatch import to declare which
/// tenant a given store belongs to.
///
/// This guards against the cross-tenant confused-deputy primitive
/// documented on [`tensor_wasm_jit::cache::CacheKey`]: without the trait,
/// the dispatch closure had no way to look up the calling tenant and was
/// trusting the guest fingerprint alone.
pub trait TenantContext {
    /// The tenant this store belongs to. Returned to the dispatch closure
    /// on every host call so the cache lookup is scoped to the caller.
    fn tenant_id(&self) -> TenantId;
}

impl TenantContext for InstanceState {
    fn tenant_id(&self) -> TenantId {
        self.tenant_id
    }
}

/// Test-only impl. Real production stores always carry an
/// [`InstanceState`]; the unit type is used by the in-crate dispatcher
/// unit tests (which spin up a `Linker<()>`). The synthetic tenant id
/// keeps those tests on the same lookup key the test harness pre-populates
/// the cache under.
impl TenantContext for () {
    fn tenant_id(&self) -> TenantId {
        TenantId(0)
    }
}

/// Default sm_version the dispatcher looks up. Matches
/// [`tensor_wasm_jit::rewrite::DEFAULT_SM_VERSION`].
pub const DEFAULT_DISPATCH_SM_VERSION: u32 = 80;

/// Return code: cache hit (kernel would dispatch on the CUDA path).
pub const DISPATCH_OK: i32 = 0;

/// Return code: cache miss — no kernel was pre-populated for this
/// fingerprint. The guest should treat this as a deopt signal.
pub const DISPATCH_CACHE_MISS: i32 = -1;

/// Return code: the dispatch failed because the caller's scratch region
/// pointed outside guest memory.
pub const DISPATCH_BAD_SCRATCH: i32 = -2;

/// Bytes reserved for the scratch arena. Sized to comfortably hold the
/// args+results of a multi-arg primitive function (each arg is at most 8
/// bytes; 64 KiB is enough for ~4000 i64 args). Operators tune via
/// [`add_jit_dispatch_to_linker_with`] in test harnesses.
pub const SCRATCH_ARENA_BYTES: u32 = 64 * 1024;

/// Per-instance state backing the `alloc`/`free`/`dispatch` imports.
///
/// One [`ArenaState`] lives in each wasmtime `Store`'s payload (see
/// [`crate::instance::InstanceState`]). Because every host-import closure
/// reaches it through `caller.data_mut()` — an `&mut` borrow that wasmtime
/// guarantees is unique for the duration of the call — no synchronisation
/// wrapper is required, and crucially the arena is NOT shared across
/// instances that happen to share a `Linker`. Two tenants instantiated
/// from the same linker each get their own bump cursor and live stack.
#[derive(Debug, Default)]
pub struct ArenaState {
    /// Last-allocated offset; the next allocation drops below this. A
    /// `None` value means no allocation has happened yet — the first
    /// `alloc` call lazily sizes the arena based on the current memory
    /// length.
    pub(crate) bump_cursor: Option<u32>,
    /// Lower bound of the arena. Allocations that would push the cursor
    /// below this are refused so they can never overwrite guest static
    /// data (which lives below `arena_floor`). Set on the same first-call
    /// seed that initialises `bump_cursor`.
    pub(crate) arena_floor: u32,
    /// Stack of (ptr, size) for LIFO free validation. Out-of-order frees
    /// degrade to "leak the slot until reset" rather than corrupt the
    /// arena.
    pub(crate) live: Vec<(u32, u32)>,
}

impl ArenaState {
    /// Current bump cursor, if any allocation has happened.
    ///
    /// Exposed primarily for tests that need to assert per-store isolation.
    pub fn bump_cursor(&self) -> Option<u32> {
        self.bump_cursor
    }

    /// Number of live (currently-allocated) slots on the LIFO stack.
    ///
    /// Exposed primarily for tests.
    pub fn live_count(&self) -> usize {
        self.live.len()
    }
}

/// Lets the JIT host imports reach the per-store [`ArenaState`] through
/// `caller.data_mut()`. Implemented by [`crate::instance::InstanceState`]
/// and by test-only payload types.
pub trait JitArenaProvider {
    /// Borrow the per-store JIT arena mutably.
    fn jit_arena_mut(&mut self) -> &mut ArenaState;
}

/// Register the `tensor-wasm:jit/host::dispatch`, `alloc`, and `free` imports on
/// `linker`.
///
/// The `cache` handle is cloned into each closure so the same backing store
/// is consulted by every guest instance the linker instantiates — kernel
/// caching IS designed to be cross-tenant. The bump arena, in contrast,
/// lives in the per-store payload `T` (via [`JitArenaProvider`]) so two
/// instances sharing a [`Linker`] do NOT see each other's cursor or live
/// stack.
pub fn add_jit_dispatch_to_linker<T>(
    linker: &mut Linker<T>,
    cache: Arc<KernelCache>,
) -> wasmtime::Result<()>
where
    T: JitArenaProvider + TenantContext + 'static,
{
    add_jit_dispatch_to_linker_with(
        linker,
        cache,
        "tensor-wasm:jit/host",
        "dispatch",
        "alloc",
        "free",
        DEFAULT_DISPATCH_SM_VERSION,
    )
}

/// Variant of [`add_jit_dispatch_to_linker`] that lets callers override the
/// import module / field names and the target sm_version. Useful in tests
/// and when the same linker hosts multiple offload generations side-by-side.
#[allow(clippy::too_many_arguments)]
pub fn add_jit_dispatch_to_linker_with<T>(
    linker: &mut Linker<T>,
    cache: Arc<KernelCache>,
    host_module: &str,
    dispatch_fn: &str,
    alloc_fn: &str,
    free_fn: &str,
    sm_version: u32,
) -> wasmtime::Result<()>
where
    T: JitArenaProvider + TenantContext + 'static,
{
    // alloc(size: i32) -> i32
    linker.func_wrap(host_module, alloc_fn, move |mut caller: Caller<'_, T>, size: i32| -> i32 {
        if size <= 0 {
            return -1;
        }
        let size_u = size as u32;
        let memory = match caller.get_export("memory").and_then(|e| e.into_memory()) {
            Some(m) => m,
            None => {
                tracing::warn!(
                    target: "tensor_wasm_exec::jit_dispatch",
                    "alloc: caller has no exported `memory`"
                );
                return -1;
            }
        };
        // Capture the bump cursor (Copy) from the per-store arena. We
        // release the &mut borrow before touching `memory` so wasmtime's
        // borrow rules around `caller` are satisfied.
        let cursor_opt = caller.data_mut().jit_arena_mut().bump_cursor;
        let mem_len = memory.data(&caller).len() as u64;
        let cursor = match cursor_opt {
            Some(c) => c,
            None => {
                // Lazily seed the arena: park it at the top of memory. If
                // memory is smaller than the arena, grow by one page.
                if mem_len < SCRATCH_ARENA_BYTES as u64 {
                    let pages_needed =
                        ((SCRATCH_ARENA_BYTES as u64 + 65535) / 65536).saturating_sub(mem_len / 65536);
                    if pages_needed > 0
                        && memory.grow(&mut caller, pages_needed).is_err()
                    {
                        return -1;
                    }
                }
                let new_len = memory.data(&caller).len() as u64;
                let top = u32::try_from(new_len).unwrap_or(u32::MAX);
                // The arena occupies the upper SCRATCH_ARENA_BYTES of
                // memory only. Anything below `arena_floor` belongs to the
                // guest (static data, stack, heap, …) and must never be
                // overwritten.
                let floor = top.saturating_sub(SCRATCH_ARENA_BYTES);
                let arena = caller.data_mut().jit_arena_mut();
                arena.arena_floor = floor;
                arena.bump_cursor = Some(top);
                top
            }
        };
        // Drop down by `size` bytes (8-byte align) for the new allocation.
        let aligned_size = (size_u + 7) & !7;
        let Some(ptr) = cursor.checked_sub(aligned_size) else {
            tracing::warn!(
                target: "tensor_wasm_exec::jit_dispatch",
                requested = size,
                "alloc: scratch arena exhausted (cursor underflow)"
            );
            return -1;
        };
        let st = caller.data_mut().jit_arena_mut();
        if ptr < st.arena_floor {
            tracing::warn!(
                target: "tensor_wasm_exec::jit_dispatch",
                requested = size,
                arena_floor = st.arena_floor,
                cursor = cursor,
                "alloc: scratch arena exhausted (would collide with guest data)"
            );
            return -1;
        }
        st.bump_cursor = Some(ptr);
        st.live.push((ptr, aligned_size));
        ptr as i32
    })?;

    // free(ptr: i32, size: i32)
    linker.func_wrap(host_module, free_fn, move |mut caller: Caller<'_, T>, ptr: i32, size: i32| {
        if ptr <= 0 || size <= 0 {
            return;
        }
        let st = caller.data_mut().jit_arena_mut();
        // LIFO: pop iff the top matches; otherwise treat as a leak and
        // move on. Mismatched frees mean the guest violated the arena
        // contract — log once and proceed.
        match st.live.last().copied() {
            Some((top_ptr, top_size)) if top_ptr == ptr as u32 => {
                st.live.pop();
                st.bump_cursor = Some(top_ptr + top_size);
            }
            _ => {
                tracing::warn!(
                    target: "tensor_wasm_exec::jit_dispatch",
                    ptr = ptr,
                    size = size,
                    "free: out-of-order free; slot leaked until arena reset"
                );
            }
        }
    })?;

    // dispatch(fp_lo, fp_hi, scratch_ptr, args_len, results_len) -> i32
    let cache_disp = cache;
    linker.func_wrap(
        host_module,
        dispatch_fn,
        move |mut caller: Caller<'_, T>,
              fingerprint_lo: i64,
              fingerprint_hi: i64,
              scratch_ptr: i32,
              args_len: i32,
              results_len: i32|
              -> i32 {
            // Reconstruct the u64 fingerprint from two i64 halves. Mask to
            // u32 before recombining so sign extension doesn't pollute the
            // upper bits.
            //
            // SECURITY: `fingerprint_lo` / `fingerprint_hi` come from the
            // guest. They feed the `blueprint` field of the cache key but
            // MUST NOT determine which tenant's cache shelf the lookup
            // hits — that comes from `caller.data().tenant_id()`, which
            // is trusted host state set when the instance was spawned.
            // Without this, tenant A could pass tenant B's fingerprint
            // and (on the CUDA path) execute B's compiled kernel against
            // A's memory — see `tensor_wasm_jit::cache::CacheKey` docs and
            // exec S-7 in the threat model.
            let lo = (fingerprint_lo as u64) & 0xFFFF_FFFF;
            let hi = (fingerprint_hi as u64) & 0xFFFF_FFFF;
            let fp = lo | (hi << 32);
            let tenant_id = caller.data().tenant_id();
            let key = CacheKey::for_tenant(tenant_id, fp, sm_version);
            let cached = match cache_disp.get(&key) {
                Some(k) => k,
                None => {
                    tracing::warn!(
                        target: "tensor_wasm_exec::jit_dispatch",
                        fingerprint = fp,
                        tenant = %tenant_id,
                        "JIT dispatch cache miss"
                    );
                    return DISPATCH_CACHE_MISS;
                }
            };

            tracing::trace!(
                target: "tensor_wasm_exec::jit_dispatch",
                fingerprint = fp,
                tenant = %tenant_id,
                args_len, results_len,
                "JIT dispatch cache hit"
            );

            // Read the args region. If the caller passed (scratch=0,
            // args_len=0, results_len=0) we still succeed — that's the
            // valid no-arg invocation.
            if scratch_ptr < 0 || args_len < 0 || results_len < 0 {
                return DISPATCH_BAD_SCRATCH;
            }
            let memory = match caller.get_export("memory").and_then(|e| e.into_memory()) {
                Some(m) => m,
                None if args_len == 0 && results_len == 0 => {
                    return DISPATCH_OK;
                }
                None => {
                    tracing::warn!(
                        target: "tensor_wasm_exec::jit_dispatch",
                        "dispatch: caller has no exported memory but args/results > 0"
                    );
                    return DISPATCH_BAD_SCRATCH;
                }
            };

            let mem = memory.data_mut(&mut caller);
            let scratch = scratch_ptr as usize;
            let alen = args_len as usize;
            let rlen = results_len as usize;
            let end = match scratch.checked_add(alen).and_then(|x| x.checked_add(rlen)) {
                Some(e) => e,
                None => return DISPATCH_BAD_SCRATCH,
            };
            if end > mem.len() {
                tracing::warn!(
                    target: "tensor_wasm_exec::jit_dispatch",
                    scratch_ptr, args_len, results_len, mem_len = mem.len(),
                    "dispatch: scratch region exceeds linear memory"
                );
                return DISPATCH_BAD_SCRATCH;
            }

            // Host-side reference computation for the no-CUDA path: copy
            // the args bytes through to the results region. Real kernels
            // run via cust under `--features cuda`; this stub gives the
            // end-to-end marshalling test a predictable behaviour:
            // results == args (truncated/padded to the results region's
            // length). The trampoline's load loop reads back the same
            // bytes it wrote — so a guest that calls e.g. `add(2, 3)`
            // when no real kernel runs sees the first parameter echoed
            // back as the result, which is the test contract.
            //
            // The actual end-to-end correctness assertion lives in
            // `tests/auto_offload_e2e.rs`, which substitutes a custom
            // dispatch implementation that performs the real addition.
            // This default stub is just a pass-through.
            let n_copy = alen.min(rlen);
            for i in 0..n_copy {
                let byte = mem[scratch + i];
                mem[scratch + alen + i] = byte;
            }
            // Hold onto the cached kernel so it doesn't go away while the
            // dispatch is computing (relevant for the future CUDA path).
            let _ = cached;
            DISPATCH_OK
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_wasm_jit::cache::{CachedKernel, CompiledHandle, KernelCache};
    use tensor_wasm_jit::ptx_emit::EmittedPtx;
    use std::sync::Arc;
    use wasmtime::{Config, Engine, Module, Store};

    /// Minimal store payload for the in-module tests: just wraps an
    /// [`ArenaState`] so we satisfy [`JitArenaProvider`] without dragging
    /// in the rest of `InstanceState`.
    #[derive(Default)]
    struct TestState {
        arena: ArenaState,
    }

    impl JitArenaProvider for TestState {
        fn jit_arena_mut(&mut self) -> &mut ArenaState {
            &mut self.arena
        }
    }

    /// Build a Wasm module that imports `tensor-wasm:jit/host::dispatch` and
    /// re-exports it as `call_dispatch(fp_lo, fp_hi) -> i32`, hardcoding
    /// `scratch_ptr` / `args_len` / `results_len` to zero. The test then
    /// drives the linker with different fingerprints and asserts the
    /// return code.
    fn driver_wat() -> &'static str {
        r#"
            (module
              (import "tensor-wasm:jit/host" "dispatch"
                (func $dispatch (param i64 i64 i32 i32 i32) (result i32)))
              (import "tensor-wasm:jit/host" "alloc"
                (func $alloc (param i32) (result i32)))
              (import "tensor-wasm:jit/host" "free"
                (func $free (param i32 i32)))
              (memory (export "memory") 1)
              (func (export "call_dispatch") (param $lo i64) (param $hi i64) (result i32)
                (call $dispatch
                  (local.get $lo)
                  (local.get $hi)
                  (i32.const 0)
                  (i32.const 0)
                  (i32.const 0)))
            )
        "#
    }

    fn make_engine() -> Engine {
        let mut cfg = Config::new();
        cfg.wasm_simd(true);
        Engine::new(&cfg).expect("engine")
    }

    fn make_cache_with(fp: u64, sm_version: u32) -> Arc<KernelCache> {
        let cache = Arc::new(KernelCache::new());
        // Test harness uses `Linker<()>`, whose `TenantContext` impl reports
        // `TenantId(0)`; store under the same tenant so the dispatch lookup
        // is a hit.
        cache.put(
            CacheKey::for_tenant(TenantId(0), fp, sm_version),
            CachedKernel {
                fingerprint: fp,
                ptx: Arc::new(EmittedPtx {
                    text: "// stub".into(),
                    launch_geometry: (1, 1),
                }),
                compiled: CompiledHandle::default(),
            },
        );
        cache
    }

    #[test]
    fn cache_hit_returns_dispatch_ok() {
        let engine = make_engine();
        let fp: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let cache = make_cache_with(fp, DEFAULT_DISPATCH_SM_VERSION);
        let mut linker: Linker<TestState> = Linker::new(&engine);
        add_jit_dispatch_to_linker(&mut linker, cache).expect("register dispatch");
        let mut store = Store::new(&engine, TestState::default());
        let wasm = wat::parse_str(driver_wat()).expect("wat");
        let module = Module::new(&engine, &wasm).expect("module");
        let instance = linker
            .instantiate(&mut store, &module)
            .expect("instantiate");
        let call = instance
            .get_typed_func::<(i64, i64), i32>(&mut store, "call_dispatch")
            .expect("typed func");
        let lo = (fp & 0xFFFF_FFFF) as i64;
        let hi = (fp >> 32) as i64;
        let ret = call.call(&mut store, (lo, hi)).expect("call");
        assert_eq!(ret, DISPATCH_OK);
    }

    #[test]
    fn cache_miss_returns_minus_one() {
        let engine = make_engine();
        // Don't put anything in the cache; the lookup must miss.
        let cache = Arc::new(KernelCache::new());
        let mut linker: Linker<TestState> = Linker::new(&engine);
        add_jit_dispatch_to_linker(&mut linker, cache).expect("register dispatch");
        let mut store = Store::new(&engine, TestState::default());
        let wasm = wat::parse_str(driver_wat()).expect("wat");
        let module = Module::new(&engine, &wasm).expect("module");
        let instance = linker
            .instantiate(&mut store, &module)
            .expect("instantiate");
        let call = instance
            .get_typed_func::<(i64, i64), i32>(&mut store, "call_dispatch")
            .expect("typed func");
        let ret = call.call(&mut store, (0, 0)).expect("call");
        assert_eq!(ret, DISPATCH_CACHE_MISS);
    }

    #[test]
    fn custom_module_and_fn_name_round_trip() {
        let engine = make_engine();
        let fp: u64 = 42;
        let cache = make_cache_with(fp, 89);
        let mut linker: Linker<TestState> = Linker::new(&engine);
        add_jit_dispatch_to_linker_with(
            &mut linker,
            cache,
            "custom:host",
            "go",
            "give",
            "take",
            89,
        )
        .expect("register custom");
        let wat = r#"
            (module
              (import "custom:host" "go"
                (func $g (param i64 i64 i32 i32 i32) (result i32)))
              (import "custom:host" "give"
                (func $a (param i32) (result i32)))
              (import "custom:host" "take"
                (func $f (param i32 i32)))
              (memory (export "memory") 1)
              (func (export "drive") (param i64 i64) (result i32)
                (call $g (local.get 0) (local.get 1)
                  (i32.const 0) (i32.const 0) (i32.const 0)))
            )
        "#;
        let mut store = Store::new(&engine, TestState::default());
        let wasm = wat::parse_str(wat).expect("wat");
        let module = Module::new(&engine, &wasm).expect("module");
        let instance = linker
            .instantiate(&mut store, &module)
            .expect("instantiate");
        let call = instance
            .get_typed_func::<(i64, i64), i32>(&mut store, "drive")
            .expect("typed func");
        let lo = (fp & 0xFFFF_FFFF) as i64;
        let hi = (fp >> 32) as i64;
        let ret = call.call(&mut store, (lo, hi)).expect("call");
        assert_eq!(ret, DISPATCH_OK);
    }

    #[test]
    fn fingerprint_with_high_bit_round_trips() {
        let engine = make_engine();
        let fp: u64 = 0xFFFF_FFFF_FFFF_FFFF;
        let cache = make_cache_with(fp, DEFAULT_DISPATCH_SM_VERSION);
        let mut linker: Linker<TestState> = Linker::new(&engine);
        add_jit_dispatch_to_linker(&mut linker, cache).expect("register");
        let mut store = Store::new(&engine, TestState::default());
        let wasm = wat::parse_str(driver_wat()).expect("wat");
        let module = Module::new(&engine, &wasm).expect("module");
        let instance = linker
            .instantiate(&mut store, &module)
            .expect("instantiate");
        let call = instance
            .get_typed_func::<(i64, i64), i32>(&mut store, "call_dispatch")
            .expect("typed func");
        let lo = (fp & 0xFFFF_FFFF) as i64;
        let hi = (fp >> 32) as i64;
        let ret = call.call(&mut store, (lo, hi)).expect("call");
        assert_eq!(ret, DISPATCH_OK);
    }

    /// End-to-end test: drive a Wasm function `add(2, 3)` through the
    /// rewriter and the dispatch trio, with a CUSTOM dispatch import that
    /// actually performs the addition. Confirms the full marshalling
    /// round-trip works: alloc, parameter stores, dispatch reads args,
    /// computes, writes results, trampoline loads results, free.
    #[test]
    fn end_to_end_add_returns_sum() {
        use tensor_wasm_jit::cache::{CacheKey, CachedKernel, CompiledHandle, KernelCache};
        use tensor_wasm_jit::detector::DetectorConfig;
        use tensor_wasm_jit::ptx_emit::EmittedPtx;
        use tensor_wasm_jit::rewrite::{rewrite_wasm, RewriteOptions};

        // Source: hot `add(a, b) -> a + b` decorated with v128 ops in a
        // loop so the detector flags it for offload. `memory` is exported
        // so the host imports (dispatch / alloc / free) can find it via
        // `Caller::get_export("memory")`.
        let hot_add = r#"
            (module
              (memory (export "memory") 1)
              (func (export "add") (param $a i32) (param $b i32) (result i32)
                (local $v v128)
                (loop $L
                  (local.set $v (i32x4.add (local.get $v) (local.get $v)))
                  (local.set $v (i32x4.add (local.get $v) (local.get $v)))
                  (local.set $v (i32x4.add (local.get $v) (local.get $v)))
                  (local.set $v (i32x4.add (local.get $v) (local.get $v)))
                )
                (i32.add (local.get $a) (local.get $b))
              )
            )
        "#;
        let wasm = wat::parse_str(hot_add).expect("wat");
        let cache = Arc::new(KernelCache::new());
        let opts = RewriteOptions {
            detector: DetectorConfig {
                v128_ratio_threshold: 0.05,
                min_trip_count: 64,
            },
            ..RewriteOptions::default()
        };
        let outcome = rewrite_wasm(&wasm, &opts, &cache).expect("rewrite");
        assert_eq!(
            outcome.offloaded_functions.len(),
            1,
            "the hot add function must be swapped"
        );
        let fp = outcome.offloaded_functions[0].fingerprint;

        // Pre-populate the cache (the rewriter does this, but double-check).
        // The test driver below uses `Linker<()>` whose `TenantContext` impl
        // reports `TenantId(0)` — store under the same tenant so the runtime
        // dispatch lookup is a hit. The custom `dispatch` closure further
        // down ignores the cache anyway and computes the result inline; but
        // the rewriter-installed trampoline still calls into the default
        // alloc/free imports keyed off the same store payload.
        cache.put(
            CacheKey::for_tenant(TenantId(0), fp, DEFAULT_DISPATCH_SM_VERSION),
            CachedKernel {
                fingerprint: fp,
                ptx: Arc::new(EmittedPtx {
                    text: "// stub for e2e".into(),
                    launch_geometry: (1, 1),
                }),
                compiled: CompiledHandle::default(),
            },
        );

        // Build a custom linker with a dispatch that actually adds.
        let engine = make_engine();
        let mut linker: Linker<TestState> = Linker::new(&engine);

        // Re-use the standard alloc / free / cache helpers; override dispatch
        // with a real adder so the test exercises the marshalling round trip
        // end-to-end. The arena lives in the per-store `TestState`, mirroring
        // the production path through `InstanceState`.
        linker
            .func_wrap(
                "tensor-wasm:jit/host",
                "alloc",
                move |mut caller: Caller<'_, TestState>, size: i32| -> i32 {
                    if size <= 0 {
                        return -1;
                    }
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .expect("memory exported");
                    let cursor_opt = caller.data_mut().jit_arena_mut().bump_cursor;
                    let mem_len = memory.data(&caller).len() as u64;
                    let cursor = match cursor_opt {
                        Some(c) => c,
                        None => {
                            if mem_len < SCRATCH_ARENA_BYTES as u64 {
                                let pages = ((SCRATCH_ARENA_BYTES as u64 + 65535) / 65536)
                                    .saturating_sub(mem_len / 65536);
                                if pages > 0 {
                                    memory.grow(&mut caller, pages).expect("grow");
                                }
                            }
                            let new_len = memory.data(&caller).len() as u64;
                            let top = u32::try_from(new_len).unwrap_or(u32::MAX);
                            caller.data_mut().jit_arena_mut().bump_cursor = Some(top);
                            top
                        }
                    };
                    let aligned = (size as u32 + 7) & !7;
                    let ptr = cursor.checked_sub(aligned).expect("arena room");
                    let st = caller.data_mut().jit_arena_mut();
                    st.bump_cursor = Some(ptr);
                    st.live.push((ptr, aligned));
                    ptr as i32
                },
            )
            .expect("alloc");
        linker
            .func_wrap(
                "tensor-wasm:jit/host",
                "free",
                move |mut caller: Caller<'_, TestState>, ptr: i32, _size: i32| {
                    let st = caller.data_mut().jit_arena_mut();
                    if let Some((top_ptr, top_size)) = st.live.last().copied() {
                        if top_ptr == ptr as u32 {
                            st.live.pop();
                            st.bump_cursor = Some(top_ptr + top_size);
                        }
                    }
                },
            )
            .expect("free");
        linker
            .func_wrap(
                "tensor-wasm:jit/host",
                "dispatch",
                |mut caller: Caller<'_, TestState>,
                 _fp_lo: i64,
                 _fp_hi: i64,
                 scratch_ptr: i32,
                 args_len: i32,
                 _results_len: i32|
                 -> i32 {
                    // Args layout: i32 a at offset 0, i32 b at offset 4.
                    // Result layout: i32 at offset args_len.
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .expect("memory");
                    let mem = memory.data_mut(&mut caller);
                    let sp = scratch_ptr as usize;
                    let a = i32::from_le_bytes([
                        mem[sp],
                        mem[sp + 1],
                        mem[sp + 2],
                        mem[sp + 3],
                    ]);
                    let b = i32::from_le_bytes([
                        mem[sp + 4],
                        mem[sp + 5],
                        mem[sp + 6],
                        mem[sp + 7],
                    ]);
                    let sum = a.wrapping_add(b).to_le_bytes();
                    let r = sp + args_len as usize;
                    mem[r..r + 4].copy_from_slice(&sum);
                    DISPATCH_OK
                },
            )
            .expect("dispatch");

        let mut store = Store::new(&engine, TestState::default());
        let module = Module::new(&engine, &outcome.rewritten_wasm).expect("module");
        let instance = linker.instantiate(&mut store, &module).expect("instantiate");
        let add = instance
            .get_typed_func::<(i32, i32), i32>(&mut store, "add")
            .expect("typed");
        let r = add.call(&mut store, (2, 3)).expect("call add");
        assert_eq!(
            r, 5,
            "end-to-end add(2,3) must marshall args, dispatch, and load result"
        );
    }

    /// Regression test for the scratch arena vs. guest static-data
    /// collision: a guest with a non-trivial data section must still see
    /// its data byte intact after a JIT scratch alloc. The arena lives in
    /// the upper `SCRATCH_ARENA_BYTES` of memory; allocations must land
    /// above the static data region, never on top of it.
    #[test]
    fn alloc_does_not_overwrite_guest_static_data() {
        let engine = make_engine();
        let cache = Arc::new(KernelCache::new());
        let mut linker: Linker<()> = Linker::new(&engine);
        add_jit_dispatch_to_linker(&mut linker, cache).expect("register");
        // Memory: 2 pages (128 KiB) > SCRATCH_ARENA_BYTES (64 KiB), so the
        // arena_floor sits at 65536 and the lower 64 KiB is "guest data".
        // We place a sentinel byte at offset 1024 and assert it survives.
        let wat = r#"
            (module
              (import "tensor-wasm:jit/host" "alloc"
                (func $a (param i32) (result i32)))
              (memory (export "memory") 2)
              (data (i32.const 1024) "\AB")
              (func (export "alloc_one") (param i32) (result i32)
                (call $a (local.get 0)))
              (func (export "sentinel") (result i32)
                (i32.load8_u (i32.const 1024))))
        "#;
        let mut store = Store::new(&engine, ());
        let wasm = wat::parse_str(wat).expect("wat");
        let module = Module::new(&engine, &wasm).expect("module");
        let instance = linker
            .instantiate(&mut store, &module)
            .expect("instantiate");
        let alloc_one = instance
            .get_typed_func::<i32, i32>(&mut store, "alloc_one")
            .expect("typed func alloc_one");
        let sentinel = instance
            .get_typed_func::<(), i32>(&mut store, "sentinel")
            .expect("typed func sentinel");

        // A small allocation must succeed and live in the upper 64 KiB,
        // i.e. strictly above the 64 KiB static-data region. With a 2-page
        // (128 KiB) memory and SCRATCH_ARENA_BYTES = 64 KiB, the arena
        // floor is at offset 65536, so any valid ptr must be >= 65536.
        let p = alloc_one.call(&mut store, 64).expect("alloc 64");
        assert!(p > 0, "alloc must succeed, got {p}");
        assert!(
            (p as u32) >= 65536,
            "ptr {p} must land in the upper page (>= 64 KiB), above guest static data",
        );
        // Sentinel byte at offset 1024 must be untouched.
        let s = sentinel.call(&mut store, ()).expect("sentinel load");
        assert_eq!(
            s, 0xAB,
            "guest static-data byte must survive JIT scratch alloc"
        );

        // A second allocation larger than the arena's remaining room must
        // fail with -1 rather than encroach on guest data.
        let too_big = alloc_one
            .call(&mut store, SCRATCH_ARENA_BYTES as i32)
            .expect("alloc oversize");
        assert_eq!(
            too_big, -1,
            "alloc that would overflow into guest data must return -1"
        );
        // Sentinel still intact after the failed alloc.
        let s2 = sentinel.call(&mut store, ()).expect("sentinel load 2");
        assert_eq!(s2, 0xAB, "guest static data must survive a refused alloc");
    }

    /// Alloc/free should round-trip — every successful alloc must produce
    /// an in-memory pointer, free should not panic, and back-to-back
    /// allocs should hand out distinct pointers.
    #[test]
    fn alloc_free_round_trip() {
        let engine = make_engine();
        let cache = Arc::new(KernelCache::new());
        let mut linker: Linker<TestState> = Linker::new(&engine);
        add_jit_dispatch_to_linker(&mut linker, cache).expect("register");
        let wat = r#"
            (module
              (import "tensor-wasm:jit/host" "dispatch"
                (func $d (param i64 i64 i32 i32 i32) (result i32)))
              (import "tensor-wasm:jit/host" "alloc"
                (func $a (param i32) (result i32)))
              (import "tensor-wasm:jit/host" "free"
                (func $f (param i32 i32)))
              (memory (export "memory") 1)
              (func (export "two_alloc") (result i32 i32)
                (call $a (i32.const 32))
                (call $a (i32.const 32))))
        "#;
        let mut store = Store::new(&engine, TestState::default());
        let wasm = wat::parse_str(wat).expect("wat");
        let module = Module::new(&engine, &wasm).expect("module");
        let instance = linker
            .instantiate(&mut store, &module)
            .expect("instantiate");
        let call = instance
            .get_typed_func::<(), (i32, i32)>(&mut store, "two_alloc")
            .expect("typed func");
        let (p1, p2) = call.call(&mut store, ()).expect("call");
        assert!(p1 > 0, "first alloc must succeed, got {p1}");
        assert!(p2 > 0, "second alloc must succeed, got {p2}");
        assert_ne!(p1, p2, "successive allocs must hand out distinct pointers");
    }
}
