// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Strongly-typed newtype identifiers used across the TensorWasm workspace.
//!
//! Each identifier is a transparent wrapper around an unsigned integer. The
//! newtype wrapper prevents accidentally passing a `TenantId` where an
//! `InstanceId` was expected — a class of bugs that would otherwise be silent.

use std::fmt;

use serde::{Deserialize, Serialize};

macro_rules! id_newtype {
    (
        $(#[$attr:meta])*
        $name:ident($inner:ty), prefix = $prefix:literal
    ) => {
        $(#[$attr])*
        #[derive(
            Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(
            /// Raw inner integer.
            ///
            /// **`#[doc(hidden)]` is intentional.** The field stays
            /// `pub` so existing tuple-construction call sites (and
            /// `serde(transparent)` deserialisation, which round-trips
            /// the bare integer) continue to compile, but the field is
            /// excluded from the rustdoc surface so new code is steered
            /// toward [`Self::new`] / [`Self::get`]. The v0.4 strict-mode
            /// follow-up is expected to flip this to a private field
            /// once the `TenantId(...)` literal sites in the workspace
            /// have migrated.
            #[doc(hidden)]
            pub $inner,
        );

        impl $name {
            /// Construct from the raw inner integer.
            ///
            /// Prefer this over the `Self(inner)` tuple-construction
            /// path for any new code — it leaves room for a future
            /// validating constructor (e.g. rejecting reserved sentinel
            /// ids) without breaking the workspace.
            pub const fn new(inner: $inner) -> Self {
                Self(inner)
            }

            /// Borrow the underlying integer.
            pub const fn get(self) -> $inner {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}{}", $prefix, self.0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                // Debug uses the same compact form as Display so traces stay readable.
                write!(f, "{}{}", $prefix, self.0)
            }
        }

        impl From<$inner> for $name {
            fn from(v: $inner) -> Self {
                Self(v)
            }
        }
    };
}

id_newtype! {
    /// Per-tenant identifier. Stable across instance lifetimes.
    ///
    /// Allocated by the API layer (`tensor-wasm-api`) when a tenant first appears; never
    /// recycled within the lifetime of a node.
    ///
    /// # Validity
    ///
    /// [`TenantId::new`] and `From<u64>` accept **any** `u64` and perform no
    /// validation — including `0`, which is the legitimate default/anonymous
    /// tenant used workspace-wide (e.g. OpenAI clients that send no
    /// `X-TensorWasm-Tenant` header resolve to `TenantId(0)`). `0` is **not**
    /// invalid and is never rejected.
    ///
    /// The single reserved sentinel is [`u64::MAX`]; see
    /// [`TenantId::try_new`] for the validating constructor that rejects it.
    TenantId(u64), prefix = "T#"
}

/// The reserved sentinel tenant id rejected by [`TenantId::try_new`].
///
/// `u64::MAX` is set aside so future code can use it as an in-band "no tenant"
/// / poison marker without colliding with a real allocation. Note that `0` is
/// **not** reserved — it is the default/anonymous tenant.
const RESERVED_TENANT_ID: u64 = u64::MAX;

/// Error returned by [`TenantId::try_new`] when the supplied id is reserved.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TenantIdError {
    /// The rejected raw id (always [`u64::MAX`] today).
    id: u64,
}

impl fmt::Display for TenantIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tenant id {} is reserved", self.id)
    }
}

impl fmt::Debug for TenantIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TenantIdError({})", self.id)
    }
}

impl std::error::Error for TenantIdError {}

impl TenantId {
    /// Validating constructor: build a `TenantId`, rejecting the reserved
    /// [`u64::MAX`] sentinel.
    ///
    /// Unlike [`TenantId::new`] / `From<u64>` (which accept any `u64` and never
    /// fail), this returns [`TenantIdError`] for the reserved id. **`0` is
    /// accepted** — it is the default/anonymous tenant, not a reserved value.
    ///
    /// ```
    /// use tensor_wasm_core::types::TenantId;
    /// assert!(TenantId::try_new(0).is_ok());      // default/anonymous tenant
    /// assert!(TenantId::try_new(42).is_ok());
    /// assert!(TenantId::try_new(u64::MAX).is_err()); // reserved sentinel
    /// ```
    pub const fn try_new(id: u64) -> Result<Self, TenantIdError> {
        if id == RESERVED_TENANT_ID {
            Err(TenantIdError { id })
        } else {
            Ok(Self(id))
        }
    }

    /// Whether this id equals the reserved [`u64::MAX`] sentinel.
    ///
    /// `TenantId(0)` is **not** reserved.
    pub const fn is_reserved(self) -> bool {
        self.0 == RESERVED_TENANT_ID
    }

    /// Whether this id would be accepted by [`TenantId::try_new`] — i.e. it is
    /// not the reserved sentinel. The inverse of [`TenantId::is_reserved`].
    pub const fn is_valid(self) -> bool {
        !self.is_reserved()
    }
}

id_newtype! {
    /// Per-instance identifier. Unique across the workspace; safe to use as a
    /// cache key, span attribute, or log field.
    ///
    /// 128 bits because instance churn on a busy serverless node can exceed
    /// 10⁶ instances/second — a 64-bit counter would wrap inside a deployment
    /// lifetime under that load.
    InstanceId(u128), prefix = "I#"
}

id_newtype! {
    /// Compiled-kernel identifier returned by `wasi_cuda_load_ptx`.
    ///
    /// Scoped to the parent `InstanceId`; reusing a `KernelId` across instances
    /// is a programming error and the host will return `CudaError`.
    KernelId(u64), prefix = "K#"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tenant_id_display() {
        assert_eq!(TenantId(42).to_string(), "T#42");
    }

    #[test]
    fn instance_id_display() {
        assert_eq!(InstanceId(7).to_string(), "I#7");
    }

    #[test]
    fn kernel_id_display() {
        assert_eq!(KernelId(99).to_string(), "K#99");
    }

    #[test]
    fn debug_matches_display() {
        assert_eq!(format!("{:?}", TenantId(1)), "T#1");
        assert_eq!(format!("{:?}", InstanceId(1)), "I#1");
        assert_eq!(format!("{:?}", KernelId(1)), "K#1");
    }

    #[test]
    fn from_inner_round_trip() {
        let t: TenantId = 1234u64.into();
        assert_eq!(t.get(), 1234);
    }

    #[test]
    fn ord_and_hash_work() {
        use std::collections::BTreeSet;
        let mut s: BTreeSet<KernelId> = BTreeSet::new();
        s.insert(KernelId(3));
        s.insert(KernelId(1));
        s.insert(KernelId(2));
        let v: Vec<_> = s.into_iter().collect();
        assert_eq!(v, vec![KernelId(1), KernelId(2), KernelId(3)]);
    }

    #[test]
    fn serde_round_trip_tenant() {
        let t = TenantId(0xDEAD_BEEF);
        let s = serde_json::to_string(&t).unwrap();
        // transparent serde — should be a bare integer literal, not a struct
        assert_eq!(s, "3735928559");
        let back: TenantId = serde_json::from_str(&s).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn serde_round_trip_instance_large() {
        let i = InstanceId(u128::MAX);
        let s = serde_json::to_string(&i).unwrap();
        let back: InstanceId = serde_json::from_str(&s).unwrap();
        assert_eq!(back, i);
    }

    #[test]
    fn newtypes_are_not_interchangeable() {
        // This is a compile-time guarantee, not a runtime check. The block below
        // would fail to compile if uncommented; we keep it as a static reminder.
        //
        //   let t = TenantId(1);
        //   let _: InstanceId = t;  // mismatched types
    }

    // --- new/get round-trips -----------------------------------------------
    //
    // The `id_newtype!` macro generates `const fn new(inner) -> Self` and
    // `const fn get(self) -> inner` on every newtype. The macro itself isn't
    // tested anywhere, so a refactor that accidentally drops one method or
    // flips the constness would slip through. These tests pin the contract
    // per type for the two non-`TenantId` newtypes (TenantId is already
    // exercised by `from_inner_round_trip`).

    #[test]
    fn kernel_id_new_get_round_trip() {
        let raw: u64 = 0xCAFE_BABE;
        let k = KernelId::new(raw);
        assert_eq!(k.get(), raw);
        // `new` is `const`: verify it works in a const context too.
        const K: KernelId = KernelId::new(7);
        assert_eq!(K.get(), 7);
    }

    #[test]
    fn instance_id_new_get_round_trip() {
        // Cover the full 128-bit width and the boundary values so a future
        // change to the inner type (e.g. accidentally narrowing to u64) is
        // caught here.
        for raw in [0u128, 1, u128::MAX / 2, u128::MAX - 1, u128::MAX] {
            let i = InstanceId::new(raw);
            assert_eq!(i.get(), raw, "round-trip failed for raw={raw}");
        }
        const I: InstanceId = InstanceId::new(u128::MAX);
        assert_eq!(I.get(), u128::MAX);
    }

    #[test]
    fn kernel_id_new_matches_tuple_construction() {
        // The macro defines both `Self(inner)` and `new(inner)`; they must be
        // observationally identical.
        assert_eq!(KernelId::new(42), KernelId(42));
        assert_eq!(KernelId::new(0), KernelId(0));
    }

    #[test]
    fn instance_id_new_matches_tuple_construction() {
        assert_eq!(InstanceId::new(42), InstanceId(42));
        assert_eq!(InstanceId::new(u128::MAX), InstanceId(u128::MAX));
    }

    // --- TenantId::try_new validation --------------------------------------
    //
    // L-7: `new`/`From<u64>` stay non-validating; `try_new` is the fallible
    // path that rejects only the reserved `u64::MAX` sentinel. Tenant `0` is
    // the default/anonymous tenant and must remain valid.

    #[test]
    fn try_new_accepts_default_tenant_zero() {
        let t = TenantId::try_new(0).expect("tenant 0 is the default/anonymous tenant");
        assert_eq!(t, TenantId(0));
        assert!(t.is_valid());
        assert!(!t.is_reserved());
    }

    #[test]
    fn try_new_accepts_normal_id() {
        let t = TenantId::try_new(42).expect("normal id is valid");
        assert_eq!(t.get(), 42);
        assert!(t.is_valid());
    }

    #[test]
    fn try_new_rejects_reserved_sentinel() {
        let err = TenantId::try_new(u64::MAX).expect_err("u64::MAX is reserved");
        assert_eq!(
            err.to_string(),
            format!("tenant id {} is reserved", u64::MAX)
        );
        // The non-validating paths still accept it, unchanged.
        assert!(TenantId::new(u64::MAX).is_reserved());
        assert!(TenantId::from(u64::MAX).is_reserved());
    }

    #[test]
    fn try_new_is_const() {
        const OK: Result<TenantId, TenantIdError> = TenantId::try_new(7);
        const BAD: Result<TenantId, TenantIdError> = TenantId::try_new(u64::MAX);
        assert_eq!(OK, Ok(TenantId(7)));
        assert!(BAD.is_err());
    }
}
