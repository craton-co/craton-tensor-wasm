//! Host-side `bali:jit/host::dispatch` implementation.
//!
//! Registered on the wasmtime [`Linker`] when the auto-offload rewrite has
//! swapped function bodies for dispatch trampolines (see
//! [`bali_jit::rewrite`]). Looks up the kernel by fingerprint in the shared
//! [`KernelCache`]; runs the cached PTX (CUDA path, lands once the `cuda`
//! feature is on) or returns a sentinel (no-CUDA stub).
//!
//! ABI (matches `bali_jit::rewrite::build_trampoline`):
//!
//! ```text
//! (func $dispatch (param i64 i64 i32 i32) (result i32))
//!         ^^^      ^^^       ^^^^^^^^^^^^^
//!     fp_lo, fp_hi, args_ptr,  args_len
//! ```
//!
//! Return values:
//!
//! - `0`  — kernel cache hit (the rewrite pre-populated this fingerprint).
//!   For the no-CUDA stub this just signals "would have dispatched".
//! - `-1` — kernel cache miss. A `tracing` warn event is emitted so
//!   operators can spot deopt regressions.
//!
//! `args_ptr` / `args_len` are reserved for the future per-call argument
//! marshalling described in the plan; the v0.1.0 trampoline always passes
//! `(0, 0)` and kernels read state through shared memory / globals.

use std::sync::Arc;

use bali_jit::cache::{CacheKey, KernelCache};
use wasmtime::{Caller, Linker};

/// Default sm_version the dispatcher looks up. Matches
/// [`bali_jit::rewrite::DEFAULT_SM_VERSION`].
pub const DEFAULT_DISPATCH_SM_VERSION: u32 = 80;

/// Return code: cache hit (kernel would dispatch on the CUDA path).
pub const DISPATCH_OK: i32 = 0;

/// Return code: cache miss — no kernel was pre-populated for this
/// fingerprint. The guest should treat this as a deopt signal.
pub const DISPATCH_CACHE_MISS: i32 = -1;

/// Register the `bali:jit/host::dispatch` import on `linker`.
///
/// The `cache` handle is cloned into the closure so the same backing store
/// is consulted by every guest instance the linker instantiates.
pub fn add_jit_dispatch_to_linker<T>(
    linker: &mut Linker<T>,
    cache: Arc<KernelCache>,
) -> wasmtime::Result<()>
where
    T: 'static,
{
    add_jit_dispatch_to_linker_with(
        linker,
        cache,
        "bali:jit/host",
        "dispatch",
        DEFAULT_DISPATCH_SM_VERSION,
    )
}

/// Variant of [`add_jit_dispatch_to_linker`] that lets callers override the
/// import module / field names and the target sm_version. Useful in tests
/// and when the same linker hosts multiple offload generations side-by-side.
pub fn add_jit_dispatch_to_linker_with<T>(
    linker: &mut Linker<T>,
    cache: Arc<KernelCache>,
    host_module: &str,
    host_fn: &str,
    sm_version: u32,
) -> wasmtime::Result<()>
where
    T: 'static,
{
    linker.func_wrap(
        host_module,
        host_fn,
        move |_caller: Caller<'_, T>,
              fingerprint_lo: i64,
              fingerprint_hi: i64,
              _args_ptr: i32,
              _args_len: i32|
              -> i32 {
            // Reconstruct the u64 fingerprint from two i64 halves. The
            // rewriter packs the low 32 bits of the u64 into `lo` and the
            // high 32 bits into `hi`; both are sign-extended to i64 when
            // they cross the Wasm boundary. Mask to u32 before recombining
            // so the sign extension doesn't pollute the upper bits.
            let lo = (fingerprint_lo as u64) & 0xFFFF_FFFF;
            let hi = (fingerprint_hi as u64) & 0xFFFF_FFFF;
            let fp = lo | (hi << 32);
            let key = CacheKey {
                blueprint: fp,
                sm_version,
            };
            match cache.get(&key) {
                Some(_kernel) => {
                    // CUDA path: launch the kernel with cust here. No-CUDA
                    // stub: signal "would dispatch" with a success return
                    // so the guest trampoline finishes its synthesised
                    // zero-result return.
                    tracing::trace!(
                        target: "bali_exec::jit_dispatch",
                        fingerprint = fp,
                        "JIT dispatch cache hit"
                    );
                    DISPATCH_OK
                }
                None => {
                    tracing::warn!(
                        target: "bali_exec::jit_dispatch",
                        fingerprint = fp,
                        "JIT dispatch cache miss"
                    );
                    DISPATCH_CACHE_MISS
                }
            }
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bali_jit::cache::{CachedKernel, CompiledHandle, KernelCache};
    use bali_jit::ptx_emit::EmittedPtx;
    use std::sync::Arc;
    use wasmtime::{Config, Engine, Module, Store};

    /// Build a Wasm module that imports `bali:jit/host::dispatch` and
    /// re-exports it as `call_dispatch(fp_lo, fp_hi) -> i32`, hardcoding
    /// `args_ptr` / `args_len` to zero. The test then drives the linker
    /// with different fingerprints and asserts the return code.
    fn driver_wat() -> &'static str {
        r#"
            (module
              (import "bali:jit/host" "dispatch"
                (func $dispatch (param i64 i64 i32 i32) (result i32)))
              (func (export "call_dispatch") (param $lo i64) (param $hi i64) (result i32)
                (call $dispatch
                  (local.get $lo)
                  (local.get $hi)
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
        cache.put(
            CacheKey {
                blueprint: fp,
                sm_version,
            },
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
        let mut linker: Linker<()> = Linker::new(&engine);
        add_jit_dispatch_to_linker(&mut linker, cache).expect("register dispatch");
        let mut store = Store::new(&engine, ());
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
        let mut linker: Linker<()> = Linker::new(&engine);
        add_jit_dispatch_to_linker(&mut linker, cache).expect("register dispatch");
        let mut store = Store::new(&engine, ());
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
        let mut linker: Linker<()> = Linker::new(&engine);
        add_jit_dispatch_to_linker_with(&mut linker, cache, "custom:host", "go", 89)
            .expect("register custom");
        let wat = r#"
            (module
              (import "custom:host" "go"
                (func $g (param i64 i64 i32 i32) (result i32)))
              (func (export "drive") (param i64 i64) (result i32)
                (call $g (local.get 0) (local.get 1) (i32.const 0) (i32.const 0)))
            )
        "#;
        let mut store = Store::new(&engine, ());
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
        // A fingerprint with bit 63 set must survive the i64 <-> u64
        // round trip without sign-extension corruption.
        let engine = make_engine();
        let fp: u64 = 0xFFFF_FFFF_FFFF_FFFF;
        let cache = make_cache_with(fp, DEFAULT_DISPATCH_SM_VERSION);
        let mut linker: Linker<()> = Linker::new(&engine);
        add_jit_dispatch_to_linker(&mut linker, cache).expect("register");
        let mut store = Store::new(&engine, ());
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
}
