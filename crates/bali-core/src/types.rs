//! Strongly-typed newtype identifiers used across the Bali workspace.
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
        pub struct $name(pub $inner);

        impl $name {
            /// Construct from the raw inner integer.
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
    /// Allocated by the API layer (`bali-api`) when a tenant first appears; never
    /// recycled within the lifetime of a node.
    TenantId(u64), prefix = "T#"
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
}
