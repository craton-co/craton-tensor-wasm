# Wasmtime Fork Assessment

## Decision

**TensorWasm v0.1.0 does NOT fork Wasmtime.**

S10 of the plan asked us to either configure Cranelift's existing flags or
fork `wasmtime-cranelift` to install a compilation hook that flags GPU-offload
candidates. After investigation we chose a third path: walk a **simplified
intermediate representation** (see [`tensor_wasm_jit::detector::BlockIR`]) that the
TensorWasm front-end populates from the Wasm bytes via `wasmparser`.

This avoids:

- A long-lived patch against `wasmtime-cranelift` that drifts with every
  upstream minor release.
- A maintenance burden that diverges from upstream JIT improvements.
- The risk of subtle correctness bugs at the CLIF-rewrite boundary.

## Cost

The trade-off is that TensorWasm's detector cannot see Cranelift's downstream
optimisation results (register pressure, constant folding, loop unrolling).
In practice this matters less than expected because:

- The detector triggers based on *structural* features (v128 op ratio,
  static loop trip count) which `wasmparser` can extract directly from Wasm.
- Cranelift's post-pass optimisations rarely change the v128 ratio of a basic
  block by more than a few percent.
- When the detector misclassifies a candidate, `DeoptGuard` (S13) catches
  the error at runtime and re-executes on CPU.

## When we would fork

If empirical evidence shows that Cranelift's downstream IR contains
information we can't derive from `wasmparser` (most likely: post-inlining
trip count refinement), we revisit the decision. The risk register
([`docs/RISKS.md`](./RISKS.md)) tracks this entry alongside the rest of
v0.1's open architectural risks.

## Upstream contributions

We are tracking
[bytecodealliance/wasmtime#9876](https://github.com/bytecodealliance/wasmtime/issues/9876)
(hypothetical issue placeholder), which would expose `cranelift::Module`'s
CLIF passes as a public extension point — once that lands the simplified IR
becomes optional and the project can opt into richer Cranelift integration
without a fork.

---

_Status: S10 of the plan. Re-assess at end of Phase 3 (S14)._
