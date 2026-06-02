// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Craton Software Company
//
// Dimension-3 (kernel dispatch overhead) competitor reference harness.
//
// Measures the *raw* CUDA driver-API `cuLaunchKernel` dispatch latency for the
// same `vector_add` kernel TensorWasm's wasi-cuda launch path drives, so the
// ratio `tensor_wasm_dispatch_ns / raw_cuda_dispatch_ns` (docs/BENCHMARKING.md
// dimension 3) can be computed against a real lower bound.
//
// This is the "no sandbox, no WASM, no back-pressure" floor: a tight loop of
// driver-API launches on a single stream, same kernel / arg layout / grid as
// the TensorWasm path. It deliberately uses the DRIVER API (cuLaunchKernel),
// not the runtime API (<<<>>>), to match exactly what tensor-wasm-wasi-gpu's
// host::launch calls.
//
// Build (RTX 2060 / SM_75; nvcc 13.2):
//   nvcc -O3 -arch=sm_75 comparison-harness/cuda/raw_launch.cu -o raw_launch.exe -lcuda
//
// Run:
//   raw_launch.exe [iters] [n_elems]
//   defaults: iters=100000, n_elems=65536
//
// Output: one JSON line on stdout (prefix RAW_CUDA) with per-launch ns
// percentiles over `iters` samples (launch-only; the synchronize is amortized
// out by timing the launch enqueue, then a final stream sync for correctness).

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <string>

#define CU_CHECK(call)                                                          \
    do {                                                                        \
        CUresult _r = (call);                                                   \
        if (_r != CUDA_SUCCESS) {                                               \
            const char* _msg = nullptr;                                         \
            cuGetErrorString(_r, &_msg);                                        \
            std::fprintf(stderr, "CUDA error %d (%s) at %s:%d\n", (int)_r,      \
                         _msg ? _msg : "?", __FILE__, __LINE__);                \
            std::exit(2);                                                       \
        }                                                                       \
    } while (0)

static long long pct(std::vector<long long>& s, double p) {
    // nearest-rank, matching tensor-wasm-bench/benches/tail_latency.rs
    size_t n = s.size();
    size_t rank = (size_t)((p * (double)n) + 0.999999);
    if (rank == 0) rank = 1;
    if (rank > n) rank = n;
    return s[rank - 1];
}

int main(int argc, char** argv) {
    long long iters = (argc > 1) ? atoll(argv[1]) : 100000;
    int n = (argc > 2) ? atoi(argv[2]) : 65536;

    CU_CHECK(cuInit(0));
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CU_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
    CU_CHECK(cuCtxSetCurrent(ctx));

    char name[256] = {0};
    cuDeviceGetName(name, sizeof(name) - 1, dev);

    // Load the arch-matched PTX (same fixture family TensorWasm loads).
    CUmodule mod;
    CUresult lr = cuModuleLoad(&mod, "kernels/vector_add_sm75.ptx");
    if (lr != CUDA_SUCCESS) {
        // Fall back to the sm_80 fixture if running on Ampere+.
        CU_CHECK(cuModuleLoad(&mod, "kernels/vector_add.ptx"));
    }
    CUfunction fn;
    CU_CHECK(cuModuleGetFunction(&fn, mod, "vector_add"));

    size_t bytes = (size_t)n * sizeof(float);
    CUdeviceptr a, b, c;
    CU_CHECK(cuMemAlloc(&a, bytes));
    CU_CHECK(cuMemAlloc(&b, bytes));
    CU_CHECK(cuMemAlloc(&c, bytes));

    CUstream stream;
    CU_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

    unsigned block_x = 256;
    unsigned grid_x = (unsigned)((n + (int)block_x - 1) / (int)block_x);

    void* args[] = {&a, &b, &c, &n};

    // Warm-up: 1000 launches + sync (JIT cache, context warm, clocks up).
    for (int i = 0; i < 1000; ++i) {
        CU_CHECK(cuLaunchKernel(fn, grid_x, 1, 1, block_x, 1, 1, 0, stream,
                                args, nullptr));
    }
    CU_CHECK(cuStreamSynchronize(stream));

    // Measure per-launch enqueue latency (the dispatch overhead), matching
    // the TensorWasm dispatch bench which times the launch path, not the GPU
    // compute completion.
    std::vector<long long> samples;
    samples.reserve(iters);
    for (long long i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        CU_CHECK(cuLaunchKernel(fn, grid_x, 1, 1, block_x, 1, 1, 0, stream,
                                args, nullptr));
        auto t1 = std::chrono::high_resolution_clock::now();
        samples.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    CU_CHECK(cuStreamSynchronize(stream));

    std::sort(samples.begin(), samples.end());
    long long p50 = pct(samples, 0.50);
    long long p95 = pct(samples, 0.95);
    long long p99 = pct(samples, 0.99);
    long long p999 = pct(samples, 0.999);
    long long mx = samples.back();
    long long mn = samples.front();
    double sum = 0;
    for (auto v : samples) sum += (double)v;
    double mean = sum / (double)samples.size();

    std::printf(
        "RAW_CUDA {\"harness\":\"raw_cuLaunchKernel\",\"device\":\"%s\","
        "\"kernel\":\"vector_add\",\"n_elems\":%d,\"grid_x\":%u,\"block_x\":%u,"
        "\"samples\":%lld,\"mean_ns\":%.1f,\"min_ns\":%lld,\"p50_ns\":%lld,"
        "\"p95_ns\":%lld,\"p99_ns\":%lld,\"p99_9_ns\":%lld,\"max_ns\":%lld}\n",
        name, n, grid_x, block_x, (long long)samples.size(), mean, mn, p50, p95,
        p99, p999, mx);

    cuStreamDestroy(stream);
    cuMemFree(a);
    cuMemFree(b);
    cuMemFree(c);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return 0;
}
