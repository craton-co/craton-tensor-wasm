// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//
// kernels/vector_add.cu
//
// Element-wise float32 vector addition: c[i] = a[i] + b[i].
//
// SOURCE OF TRUTH for the committed PTX fixtures:
//   * kernels/vector_add.ptx                                 (sm_80, canonical)
//   * kernels/vector_add_sm75.ptx                            (sm_75, Turing dev box)
//   * crates/tensor-wasm-wasi-gpu/tests/fixtures/vector_add.ptx
//   * crates/tensor-wasm-wasi-gpu/tests/fixtures/vector_add_sm75.ptx
//
// ## Why this file exists (roadmap fix #7)
//
// The committed `.ptx` fixtures were hand-authored. On the RTX 2060 dev box
// (CUDA 13.x driver) BOTH the sm_80 fixture AND a hand-written sm_75 variant
// were rejected by the driver JIT with `CUDA_ERROR_INVALID_PTX` — see
// `docs/GPU-VALIDATION-2026-05-30.md`. Hand-authored PTX is brittle: the JIT
// in newer CUDA toolkits is stricter than the assembler that produced the
// original ISA-8.0 text. The fix is to stop hand-writing PTX and emit it from
// `nvcc`, which produces canonical, JIT-loadable output.
//
// ## Regenerate the fixtures (run on a CUDA host with nvcc)
//
//   make ptx            # regenerates both sm_75 and sm_80 fixtures + copies
//
// or directly:
//
//   nvcc -ptx -arch=sm_75 -o kernels/vector_add_sm75.ptx kernels/vector_add.cu
//   nvcc -ptx -arch=sm_80 -o kernels/vector_add.ptx      kernels/vector_add.cu
//   cp kernels/vector_add_sm75.ptx \
//      crates/tensor-wasm-wasi-gpu/tests/fixtures/vector_add_sm75.ptx
//   cp kernels/vector_add.ptx \
//      crates/tensor-wasm-wasi-gpu/tests/fixtures/vector_add.ptx
//
// `-arch=sm_75` matches the dev box (compute capability 7.5); sm_75 PTX JITs
// forward onto any newer device (Ampere/Hopper/Blackwell), so it is the
// fixture the end-to-end launch test prefers on sub-Ampere hardware. The sm_80
// fixture is kept as the canonical/Ampere target. Bump the arch list when the
// CI GPU baseline's MINIMUM compute capability moves (see docs/CUDA-SETUP.md).
//
// `extern "C"` suppresses C++ name mangling so the exported entry symbol is
// exactly `vector_add` — the literal name the host looks up via
// `Module::get_function("vector_add")`. The parameter order/types
// (const float* a, const float* b, float* c, int n) match the tagged argv the
// wasi-cuda launch tests marshal: [Ptr(a), Ptr(b), Ptr(c), U32(n)].

extern "C" __global__ void vector_add(const float *a, const float *b,
                                      float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
