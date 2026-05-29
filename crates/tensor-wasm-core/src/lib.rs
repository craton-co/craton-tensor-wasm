// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company

//! Foundational types, errors, metrics, and telemetry shared by every TensorWasm crate.
#![deny(missing_docs)]
// Enable the unstable `doc_cfg` attribute on the docs.rs (nightly) build only.
// `Cargo.toml` passes `--cfg docsrs` via `package.metadata.docs.rs.rustdoc-args`,
// so the `feature(doc_cfg)` opt-in (and the `#[doc(cfg(...))]` annotations it
// unlocks) is compiled exclusively there and never affects the stable build.
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod error;
pub mod mem_pool;
pub mod metrics;
pub mod telemetry;
pub mod types;
