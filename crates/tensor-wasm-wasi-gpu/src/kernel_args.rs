// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Typed argv lowering for the `wasi:cuda` `launch` host function.
//!
//! Guests writing GPU kernels through TensorWasm's `wasi-cuda` ABI pass kernel
//! parameters as a single opaque byte buffer (`args_ptr`, `args_len`). To
//! turn that buffer into the `void**`-style array CUDA's `cuLaunchKernel`
//! expects, the host needs a frozen per-argument packing format.
//!
//! ## Wire format
//!
//! The buffer is a sequence of `(tag, value)` records. Each record starts
//! with a single tag byte ([`ArgTag`]) followed by the value bytes in
//! little-endian order. There is **no inter-argument padding** — the host
//! reads exactly `tag.size()` bytes after each tag byte, which matches how
//! a guest authored against this ABI would `memcpy` values into a `&mut
//! [u8]`. (CUDA's `cuLaunchKernel` itself doesn't care about host-side
//! padding because each arg pointer is independently addressed.)
//!
//! | Tag | Type | Value bytes | Wire size (incl. tag) |
//! |-----|------|-------------|------------------------|
//! | 0x01 | `i32` | 4 | 5 |
//! | 0x02 | `i64` | 8 | 9 |
//! | 0x03 | `f32` | 4 | 5 |
//! | 0x04 | `f64` | 8 | 9 |
//! | 0x05 | `u32` | 4 | 5 |
//! | 0x06 | `u64` | 8 | 9 |
//! | 0x07 | `ptr` | 4 (guest ptr) + 4 (guest byte len) | 9 |
//!
//! Pointer arguments carry both the guest-memory offset and the byte length
//! the kernel will access. The host bounds-checks the `[ptr, ptr+len)`
//! window against the caller's linear memory before producing any device
//! pointer — the same posture as the rest of the wasi-cuda host fns. On
//! CUDA builds the resolved host pointer feeds directly into the
//! `cuLaunchKernel` argv (CUDA Unified Memory makes managed host pointers
//! valid as device pointers); on no-CUDA builds the resolved pointer is
//! captured for inspection but never dereferenced as a device address.
//!
//! ## Sanity caps
//!
//! - [`MAX_KERNEL_ARGS`] caps the number of records per launch.
//! - [`MAX_KERNEL_ARGS_BYTES`] caps the total wire-format buffer size.
//!
//! Both caps exist so a malformed-but-in-bounds buffer cannot pin
//! arbitrary host memory inside the parser. Buffers that exceed either
//! cap surface as [`AbiError::KernelArgsUnsupported`], reusing the
//! existing code for "the host won't accept this shape" — distinct from
//! [`AbiError::InvalidArgs`] (the tag bytes are malformed) and from
//! [`AbiError::InvalidPointer`] (the guest pointer is OOB).
//!
//! See `wit/wasi-cuda.wit` and `docs/RISKS.md` for the public contract.

use crate::abi::AbiError;

/// Hard cap on the number of argv records the host will accept per launch.
///
/// CUDA itself allows up to 4096 kernel parameters; we cap lower here so a
/// malicious guest can't force the host to allocate a multi-megabyte
/// metadata array before the launch is rejected. Most kernels take ≤ 16
/// args; raising this is cheap if a real workload needs it.
pub const MAX_KERNEL_ARGS: usize = 128;

/// Hard cap on the total wire-format buffer length the host will parse.
///
/// At [`MAX_KERNEL_ARGS`] = 128 records of the widest type (9 bytes each)
/// the buffer is at most 1152 bytes; we round up to 4 KiB so a guest
/// emitting a slightly-extended format (future extra-tag bytes) still
/// fits. Buffers above this cap return
/// [`AbiError::KernelArgsUnsupported`].
pub const MAX_KERNEL_ARGS_BYTES: usize = 4 * 1024;

/// Tag byte identifying a single kernel-argument record's type.
///
/// The numeric values are part of the ABI: bumping a value is a breaking
/// change. New tags are appended (next free: `0x08`).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgTag {
    /// 32-bit signed integer. Value bytes: `[i32 LE; 1]`.
    I32 = 0x01,
    /// 64-bit signed integer. Value bytes: `[i64 LE; 1]`.
    I64 = 0x02,
    /// 32-bit IEEE-754 float. Value bytes: `[f32 LE; 1]`.
    F32 = 0x03,
    /// 64-bit IEEE-754 float. Value bytes: `[f64 LE; 1]`.
    F64 = 0x04,
    /// 32-bit unsigned integer. Value bytes: `[u32 LE; 1]`.
    U32 = 0x05,
    /// 64-bit unsigned integer. Value bytes: `[u64 LE; 1]`.
    U64 = 0x06,
    /// Guest pointer + byte length. Value bytes: `[u32 LE; 2]` — the first
    /// `u32` is the guest-memory offset, the second is the number of
    /// bytes the kernel will touch starting at that offset.
    Ptr = 0x07,
}

impl ArgTag {
    /// Parse a single tag byte. Returns [`AbiError::InvalidArgs`] for any
    /// byte value not currently assigned.
    pub fn from_byte(b: u8) -> Result<Self, AbiError> {
        Ok(match b {
            0x01 => ArgTag::I32,
            0x02 => ArgTag::I64,
            0x03 => ArgTag::F32,
            0x04 => ArgTag::F64,
            0x05 => ArgTag::U32,
            0x06 => ArgTag::U64,
            0x07 => ArgTag::Ptr,
            _ => return Err(AbiError::InvalidArgs),
        })
    }

    /// Number of value bytes that follow the tag in the wire format.
    pub const fn value_bytes(self) -> usize {
        match self {
            ArgTag::I32 | ArgTag::F32 | ArgTag::U32 => 4,
            ArgTag::I64 | ArgTag::F64 | ArgTag::U64 => 8,
            ArgTag::Ptr => 8, // u32 ptr + u32 len
        }
    }
}

/// A single fully-parsed and bounds-checked kernel argument.
///
/// `LoweredArg` is the host-side representation that flows into
/// `cuLaunchKernel` (via [`build_kernel_param_storage`]) on CUDA builds
/// and is recorded for inspection on no-CUDA builds.
#[derive(Debug, Clone, PartialEq)]
pub enum LoweredArg {
    /// 32-bit signed scalar.
    I32(i32),
    /// 64-bit signed scalar.
    I64(i64),
    /// 32-bit float scalar.
    F32(f32),
    /// 64-bit float scalar.
    F64(f64),
    /// 32-bit unsigned scalar.
    U32(u32),
    /// 64-bit unsigned scalar.
    U64(u64),
    /// Pointer argument: a fully bounds-checked host pointer plus the
    /// declared byte length and the original guest-memory offset.
    ///
    /// `host_ptr` is a raw pointer into the caller's linear memory; under
    /// CUDA Unified Memory it is also a valid device address. The
    /// `guest_offset` field is preserved for log lines / tests so a
    /// pointer arg round-trips losslessly through parsing.
    Ptr {
        /// Host-side raw pointer into the caller's linear memory.
        ///
        /// Treat as opaque — dereferencing is the caller's responsibility
        /// and is unsafe in general (the underlying linear memory may
        /// move on a `memory.grow`). The CUDA launch site captures the
        /// pointer into a parameter slot and immediately hands it to
        /// `cuLaunchKernel`, before any guest code can run.
        host_ptr: *const u8,
        /// Byte length the kernel will access.
        len: u32,
        /// Original guest-memory offset (for logging / tests).
        guest_offset: u32,
    },
}

// SAFETY: `LoweredArg::Ptr::host_ptr` is a raw pointer into Wasm linear
// memory. We never share it across threads — the launch path moves it
// straight into a `cuLaunchKernel` parameter slot, awaits the
// `spawn_blocking(synchronize)`, then drops the `Vec<LoweredArg>` before
// returning to the wasmtime fiber. Recording it under `Send` is
// nevertheless safe because no concurrent reader exists by construction.
unsafe impl Send for LoweredArg {}
unsafe impl Sync for LoweredArg {}

/// Parse a tagged-argv byte buffer into a host-side `Vec<LoweredArg>`.
///
/// `mem` is the caller's linear-memory snapshot (as exposed by
/// `wasmtime::Memory::data`); pointer arguments are bounds-checked
/// against it.
///
/// The parser is strict: it rejects any byte left over after the last
/// record, refuses unknown tag bytes, and refuses an over-long buffer or
/// too-many-args count. The intent is that a malformed buffer is detected
/// here, before any CUDA call.
///
/// Errors:
/// - [`AbiError::KernelArgsUnsupported`] — buffer exceeds the sanity caps
///   ([`MAX_KERNEL_ARGS`] / [`MAX_KERNEL_ARGS_BYTES`]).
/// - [`AbiError::InvalidArgs`] — buffer is malformed (unknown tag,
///   trailing bytes, value bytes truncated against a tag).
/// - [`AbiError::InvalidPointer`] — a pointer arg's `[ptr, ptr+len)`
///   window lies outside `mem`.
pub fn parse_argv(buf: &[u8], mem: &[u8]) -> Result<Vec<LoweredArg>, AbiError> {
    if buf.len() > MAX_KERNEL_ARGS_BYTES {
        return Err(AbiError::KernelArgsUnsupported);
    }
    let mut out: Vec<LoweredArg> = Vec::new();
    let mut i = 0usize;
    while i < buf.len() {
        if out.len() >= MAX_KERNEL_ARGS {
            return Err(AbiError::KernelArgsUnsupported);
        }
        let tag = ArgTag::from_byte(buf[i])?;
        i += 1;
        let need = tag.value_bytes();
        if i.checked_add(need).map_or(true, |end| end > buf.len()) {
            // Tag promised `need` bytes but the buffer is shorter — the
            // record is truncated, the guest's packing is malformed.
            return Err(AbiError::InvalidArgs);
        }
        let val = &buf[i..i + need];
        i += need;
        let arg = match tag {
            ArgTag::I32 => {
                let mut b = [0u8; 4];
                b.copy_from_slice(val);
                LoweredArg::I32(i32::from_le_bytes(b))
            }
            ArgTag::I64 => {
                let mut b = [0u8; 8];
                b.copy_from_slice(val);
                LoweredArg::I64(i64::from_le_bytes(b))
            }
            ArgTag::F32 => {
                let mut b = [0u8; 4];
                b.copy_from_slice(val);
                LoweredArg::F32(f32::from_le_bytes(b))
            }
            ArgTag::F64 => {
                let mut b = [0u8; 8];
                b.copy_from_slice(val);
                LoweredArg::F64(f64::from_le_bytes(b))
            }
            ArgTag::U32 => {
                let mut b = [0u8; 4];
                b.copy_from_slice(val);
                LoweredArg::U32(u32::from_le_bytes(b))
            }
            ArgTag::U64 => {
                let mut b = [0u8; 8];
                b.copy_from_slice(val);
                LoweredArg::U64(u64::from_le_bytes(b))
            }
            ArgTag::Ptr => {
                let mut p = [0u8; 4];
                p.copy_from_slice(&val[0..4]);
                let mut l = [0u8; 4];
                l.copy_from_slice(&val[4..8]);
                let guest_offset = u32::from_le_bytes(p);
                let len = u32::from_le_bytes(l);
                let start = guest_offset as usize;
                let end = start
                    .checked_add(len as usize)
                    .ok_or(AbiError::InvalidPointer)?;
                if end > mem.len() {
                    return Err(AbiError::InvalidPointer);
                }
                // SAFETY: `start` is in-bounds for `mem` (checked above);
                // for a zero-length pointer the pointer is still a valid
                // single-byte offset into the slice as long as
                // `start <= mem.len()`. We use `as_ptr().add(start)` which
                // requires that bound.
                let host_ptr = unsafe { mem.as_ptr().add(start) };
                LoweredArg::Ptr {
                    host_ptr,
                    len,
                    guest_offset,
                }
            }
        };
        out.push(arg);
    }
    Ok(out)
}

/// Storage backing the `void**` array `cuLaunchKernel` expects.
///
/// CUDA's `cuLaunchKernel` takes a `*mut *mut c_void` whose elements each
/// point to the value bytes of a single argument. The lifetimes are
/// awkward: the value bytes must outlive the launch call, and so must
/// the pointer array. [`KernelParamStorage`] owns both so the launch
/// site can take a `*mut *mut c_void` into it and pass that to
/// `cuLaunchKernel` without leaking.
///
/// Currently only used on CUDA builds; the no-CUDA host path records the
/// parsed [`Vec<LoweredArg>`] directly. The struct is still defined
/// non-conditionally so the type checker can prove the no-CUDA build
/// keeps the lowering free of CUDA-specific types.
pub struct KernelParamStorage {
    /// Per-arg value storage: each `Box<[u8]>` holds the raw bytes
    /// `cuLaunchKernel` reads through the corresponding pointer slot.
    /// Allocated as a `Box<[u8]>` (rather than `Vec<u8>`) so the
    /// backing memory cannot be moved by a push — the pointer in
    /// `slots` would otherwise dangle.
    _values: Vec<Box<[u8]>>,
    /// Pointer slots in the order CUDA reads them. Each entry points
    /// into the matching `_values[i]` slice — for scalar args that
    /// slice holds the value bytes; for `Ptr` args it holds the
    /// little-endian bytes of the resolved host pointer (CUDA's argv
    /// is "address-of pointer", not "pointer").
    slots: Vec<*mut std::ffi::c_void>,
}

// SAFETY: the pointers inside `slots` reference either heap memory
// owned by `_values` or live wasm linear-memory bytes whose lifetime
// the launch site enforces externally (the `Vec<LoweredArg>` it
// originated from is held in the same stack frame). Like
// `LoweredArg`, we never share these across threads.
unsafe impl Send for KernelParamStorage {}

impl KernelParamStorage {
    /// Borrow the pointer slots as a `*mut *mut c_void` suitable for
    /// `cuLaunchKernel`.
    pub fn as_ptr(&mut self) -> *mut *mut std::ffi::c_void {
        self.slots.as_mut_ptr()
    }

    /// Number of arguments in this storage.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// True when the storage carries no arguments.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// Build a [`KernelParamStorage`] from a previously-parsed `Vec<LoweredArg>`.
///
/// On CUDA builds the resulting storage is handed directly to
/// `cuLaunchKernel`; on no-CUDA builds it is only useful as a sanity
/// check that the layout compiles cleanly across both feature
/// configurations.
pub fn build_kernel_param_storage(args: &[LoweredArg]) -> KernelParamStorage {
    let mut values: Vec<Box<[u8]>> = Vec::with_capacity(args.len());
    let mut slots: Vec<*mut std::ffi::c_void> = Vec::with_capacity(args.len());

    // Inline boxing per arg. Splitting this into a closure that captures
    // `&mut values, &mut slots` is tempting but the closure binding
    // would hold the mut-borrows until end-of-fn, preventing the
    // subsequent move into `KernelParamStorage`. Manual repetition is
    // the cheapest workaround.
    //
    // Note on CUDA semantics for `Ptr`: CUDA expects the *address of*
    // the device pointer in each argv slot. We stash the pointer's
    // bytes in a little box and point the slot at *that*. Under CUDA
    // Unified Memory `host_ptr` doubles as a device address; for
    // non-UVM allocations the guest would have to call `cuMemAlloc`
    // separately (out of scope for v0.2 — see `docs/RISKS.md`).
    for a in args {
        let bytes: Box<[u8]> = match a {
            LoweredArg::I32(v) => Vec::from(v.to_ne_bytes()).into_boxed_slice(),
            LoweredArg::I64(v) => Vec::from(v.to_ne_bytes()).into_boxed_slice(),
            LoweredArg::F32(v) => Vec::from(v.to_ne_bytes()).into_boxed_slice(),
            LoweredArg::F64(v) => Vec::from(v.to_ne_bytes()).into_boxed_slice(),
            LoweredArg::U32(v) => Vec::from(v.to_ne_bytes()).into_boxed_slice(),
            LoweredArg::U64(v) => Vec::from(v.to_ne_bytes()).into_boxed_slice(),
            LoweredArg::Ptr { host_ptr, .. } => {
                Vec::from((*host_ptr as usize).to_ne_bytes()).into_boxed_slice()
            }
        };
        let mut boxed = bytes;
        slots.push(boxed.as_mut_ptr() as *mut std::ffi::c_void);
        values.push(boxed);
    }

    KernelParamStorage {
        _values: values,
        slots,
    }
}

/// Helper for tests and host-stub paths: encode a sequence of
/// [`LoweredArg`] back into the wire format.
///
/// Inverse of [`parse_argv`] for scalar values; for `Ptr` it re-emits the
/// `guest_offset` / `len` pair (the resolved `host_ptr` is host-side
/// state and does not round-trip). This helper makes integration tests
/// readable — they can author argv as `vec![LoweredArg::I32(1), ...]`,
/// encode it, and feed the bytes through the launch path.
pub fn encode_argv(args: &[LoweredArg]) -> Vec<u8> {
    let mut out = Vec::new();
    for a in args {
        match a {
            LoweredArg::I32(v) => {
                out.push(ArgTag::I32 as u8);
                out.extend_from_slice(&v.to_le_bytes());
            }
            LoweredArg::I64(v) => {
                out.push(ArgTag::I64 as u8);
                out.extend_from_slice(&v.to_le_bytes());
            }
            LoweredArg::F32(v) => {
                out.push(ArgTag::F32 as u8);
                out.extend_from_slice(&v.to_le_bytes());
            }
            LoweredArg::F64(v) => {
                out.push(ArgTag::F64 as u8);
                out.extend_from_slice(&v.to_le_bytes());
            }
            LoweredArg::U32(v) => {
                out.push(ArgTag::U32 as u8);
                out.extend_from_slice(&v.to_le_bytes());
            }
            LoweredArg::U64(v) => {
                out.push(ArgTag::U64 as u8);
                out.extend_from_slice(&v.to_le_bytes());
            }
            LoweredArg::Ptr {
                guest_offset, len, ..
            } => {
                out.push(ArgTag::Ptr as u8);
                out.extend_from_slice(&guest_offset.to_le_bytes());
                out.extend_from_slice(&len.to_le_bytes());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Backing slice big enough that any in-test pointer arg has room.
    /// We use 4 KiB which exceeds the per-buffer cap but is fine for `mem`.
    fn fake_mem() -> Vec<u8> {
        vec![0u8; 4096]
    }

    #[test]
    fn empty_buffer_parses_to_empty_vec() {
        let mem = fake_mem();
        let out = parse_argv(&[], &mem).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn scalar_argv_round_trips() {
        let mem = fake_mem();
        let args = vec![
            LoweredArg::I32(-7),
            LoweredArg::U32(0xDEAD_BEEF),
            LoweredArg::I64(-1_000_000_000_000),
            LoweredArg::U64(0xC0FFEE_BABE_F00D),
            LoweredArg::F32(core::f32::consts::PI),
            LoweredArg::F64(core::f64::consts::E),
        ];
        let buf = encode_argv(&args);
        let parsed = parse_argv(&buf, &mem).unwrap();
        assert_eq!(parsed.len(), args.len());
        assert_eq!(parsed, args);
    }

    #[test]
    fn pointer_argv_resolves_host_ptr_and_records_offset() {
        let mem = fake_mem();
        let args = vec![LoweredArg::Ptr {
            host_ptr: std::ptr::null(),
            len: 128,
            guest_offset: 64,
        }];
        let buf = encode_argv(&args);
        let parsed = parse_argv(&buf, &mem).unwrap();
        assert_eq!(parsed.len(), 1);
        match &parsed[0] {
            LoweredArg::Ptr {
                host_ptr,
                len,
                guest_offset,
            } => {
                assert_eq!(*len, 128);
                assert_eq!(*guest_offset, 64);
                // Resolved host pointer is mem.as_ptr() + 64.
                let expected = unsafe { mem.as_ptr().add(64) };
                assert_eq!(*host_ptr, expected);
            }
            other => panic!("expected Ptr, got {other:?}"),
        }
    }

    #[test]
    fn mixed_argv_preserves_order() {
        let mem = fake_mem();
        let args = vec![
            LoweredArg::I32(11),
            LoweredArg::Ptr {
                host_ptr: std::ptr::null(),
                len: 16,
                guest_offset: 32,
            },
            LoweredArg::F64(1.5),
            LoweredArg::Ptr {
                host_ptr: std::ptr::null(),
                len: 8,
                guest_offset: 1024,
            },
            LoweredArg::U64(42),
        ];
        let buf = encode_argv(&args);
        let parsed = parse_argv(&buf, &mem).unwrap();
        assert_eq!(parsed.len(), 5);
        // Spot-check non-pointer fields directly; verify pointers by
        // their preserved guest_offset.
        assert!(matches!(parsed[0], LoweredArg::I32(11)));
        assert!(matches!(parsed[2], LoweredArg::F64(v) if v == 1.5));
        assert!(matches!(parsed[4], LoweredArg::U64(42)));
        match &parsed[1] {
            LoweredArg::Ptr { guest_offset, len, .. } => {
                assert_eq!(*guest_offset, 32);
                assert_eq!(*len, 16);
            }
            _ => panic!("idx 1 not Ptr"),
        }
        match &parsed[3] {
            LoweredArg::Ptr { guest_offset, len, .. } => {
                assert_eq!(*guest_offset, 1024);
                assert_eq!(*len, 8);
            }
            _ => panic!("idx 3 not Ptr"),
        }
    }

    #[test]
    fn unknown_tag_byte_returns_invalid_args() {
        let mem = fake_mem();
        // 0xFF is not assigned. Expect InvalidArgs, NOT KernelArgsUnsupported.
        let err = parse_argv(&[0xFF, 0, 0, 0, 0], &mem).unwrap_err();
        assert_eq!(err, AbiError::InvalidArgs);
    }

    #[test]
    fn truncated_record_returns_invalid_args() {
        let mem = fake_mem();
        // Tag promises 8 bytes (I64) but only 3 follow.
        let buf = [ArgTag::I64 as u8, 1, 2, 3];
        let err = parse_argv(&buf, &mem).unwrap_err();
        assert_eq!(err, AbiError::InvalidArgs);
    }

    #[test]
    fn oversized_buffer_returns_kernel_args_unsupported() {
        let mem = fake_mem();
        let buf = vec![0u8; MAX_KERNEL_ARGS_BYTES + 1];
        let err = parse_argv(&buf, &mem).unwrap_err();
        assert_eq!(err, AbiError::KernelArgsUnsupported);
    }

    #[test]
    fn too_many_args_returns_kernel_args_unsupported() {
        // Pack `MAX_KERNEL_ARGS + 1` I32 records; each takes 5 bytes, so
        // the total stays under MAX_KERNEL_ARGS_BYTES (= 4096) for the
        // currently-defined caps (128 * 5 = 640 bytes). The parser must
        // refuse the (cap+1)-th record explicitly before it lowers.
        let mem = fake_mem();
        let one = vec![ArgTag::I32 as u8, 0, 0, 0, 0];
        let mut buf = Vec::new();
        for _ in 0..(MAX_KERNEL_ARGS + 1) {
            buf.extend_from_slice(&one);
        }
        // Guard the assumption above: stay under the byte cap.
        assert!(buf.len() <= MAX_KERNEL_ARGS_BYTES);
        let err = parse_argv(&buf, &mem).unwrap_err();
        assert_eq!(err, AbiError::KernelArgsUnsupported);
    }

    #[test]
    fn pointer_out_of_bounds_returns_invalid_pointer() {
        let mem = vec![0u8; 32];
        let args = vec![LoweredArg::Ptr {
            host_ptr: std::ptr::null(),
            len: 16,
            guest_offset: 24, // [24, 40) ⊄ [0, 32)
        }];
        let buf = encode_argv(&args);
        let err = parse_argv(&buf, &mem).unwrap_err();
        assert_eq!(err, AbiError::InvalidPointer);
    }

    #[test]
    fn pointer_offset_overflow_returns_invalid_pointer() {
        let mem = vec![0u8; 32];
        // guest_offset = u32::MAX, len = 1024 — start + len overflows
        // u32 → caught by the checked_add in parse_argv.
        let args = vec![LoweredArg::Ptr {
            host_ptr: std::ptr::null(),
            len: 1024,
            guest_offset: u32::MAX,
        }];
        let buf = encode_argv(&args);
        let err = parse_argv(&buf, &mem).unwrap_err();
        assert_eq!(err, AbiError::InvalidPointer);
    }

    #[test]
    fn zero_length_pointer_at_memory_end_is_allowed() {
        // A common safe pattern: zero-length pointer pointing at one-past-end.
        let mem = vec![0u8; 32];
        let args = vec![LoweredArg::Ptr {
            host_ptr: std::ptr::null(),
            len: 0,
            guest_offset: 32,
        }];
        let buf = encode_argv(&args);
        let parsed = parse_argv(&buf, &mem).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn build_kernel_param_storage_slot_count_matches_args() {
        let args = vec![
            LoweredArg::I32(1),
            LoweredArg::F64(2.0),
            LoweredArg::Ptr {
                host_ptr: std::ptr::null(),
                len: 8,
                guest_offset: 0,
            },
        ];
        let storage = build_kernel_param_storage(&args);
        assert_eq!(storage.len(), 3);
        assert!(!storage.is_empty());
    }

    #[test]
    fn arg_tag_value_bytes_pinned() {
        // Wire-format guard: bumping any of these is a breaking change.
        assert_eq!(ArgTag::I32.value_bytes(), 4);
        assert_eq!(ArgTag::I64.value_bytes(), 8);
        assert_eq!(ArgTag::F32.value_bytes(), 4);
        assert_eq!(ArgTag::F64.value_bytes(), 8);
        assert_eq!(ArgTag::U32.value_bytes(), 4);
        assert_eq!(ArgTag::U64.value_bytes(), 8);
        assert_eq!(ArgTag::Ptr.value_bytes(), 8);
    }

    #[test]
    fn arg_tag_byte_values_pinned() {
        // ABI guard: these constants are part of the on-the-wire format.
        assert_eq!(ArgTag::I32 as u8, 0x01);
        assert_eq!(ArgTag::I64 as u8, 0x02);
        assert_eq!(ArgTag::F32 as u8, 0x03);
        assert_eq!(ArgTag::F64 as u8, 0x04);
        assert_eq!(ArgTag::U32 as u8, 0x05);
        assert_eq!(ArgTag::U64 as u8, 0x06);
        assert_eq!(ArgTag::Ptr as u8, 0x07);
    }
}
