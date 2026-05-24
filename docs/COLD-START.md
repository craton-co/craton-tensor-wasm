# Cold Start

## Why cold start matters

Serverless GPU workloads live and die by their cold-start latency. A function that ships its weights through a 200 MB container image, JIT-compiles a few PTX modules, and warms the CUDA driver on every invocation cannot meet the millisecond-scale tail-latency targets that interactive inference, RAG retrieval, and online feature stores demand. Craton TensorWasm's `tensor-wasm-snapshot` crate exists to short-circuit that path: instead of re-executing the cold initialisation sequence, the host restores a pre-captured `Snapshot` containing the Wasm linear memory, GPU device memory, and the JIT register file, then resumes execution from where the previous instance left off.

Honest framing matters here. A handful of academic papers and vendor blog posts have claimed "sub-millisecond" GPU cold starts; the marketing has outrun the physics. Real hosts pay for the bincode decode, the zstd decompression, the page-fault-driven UVM migration into device memory, and the first kernel launch before the function can do useful work. This document explains what each of those costs is, gives a calibrated estimate of what we expect to measure once S19's benchmarks land, and lists the levers operators have today to keep cold starts inside their SLO.

## Latency model

A cold restore from a snapshot blob breaks down into five additive components:

1. **Snapshot read** — pulling the compressed blob off local NVMe (or a memory-mapped page cache). At 5–7 GB/s sequential read on a Gen 4 NVMe drive, a 16 MiB blob is ~3 ms; the variance is dominated by whether the page cache is warm.
2. **bincode decode** — zero-copy in spirit, but the deserialiser still walks the buffer and allocates `Vec<u8>` for each memory region. Empirically about 1 GB/s on a single core, so a 16 MiB decompressed body takes ~16 ms.
3. **zstd decompress** — single-threaded at level 3, zstd manages 500–800 MB/s on modern x86. Compression ratios on mostly-random GPU memory are weak (1.05–1.3×), so the decompressed body is typically only ~10–25 % smaller than the source weights.
4. **UVM warmup** — the snapshot lands in host memory; the first kernel launch faults pages over PCIe into device memory. On PCIe Gen 4 ×16, the practical migration ceiling is ~25 GB/s, but the page-fault-driven migration path adds 20–200 ms of latency for large working sets before the first kernel can run uncontended. Hopper's H100 and Blackwell GB200 partially hide this with copy engines, but the cost does not disappear.
5. **First kernel launch** — driver-side overhead is 5–10 µs per launch on warm contexts; the *first* launch after a fresh context creation is closer to 1–3 ms because the driver materialises module-level state lazily.

The total cold-start latency for restoring an instance is therefore approximately

```
T_cold ≈ T_read + T_zstd + T_bincode + T_uvm + T_launch
```

with the UVM term dominating for any snapshot larger than a few MiB.

## Estimated latency table

The table below is a **modelling estimate, not a measurement**. The S19 benchmark suite (see `docs/PERFORMANCE.md` — pending) will replace these numbers with empirical P50/P95/P99s captured on the reference rig (Ryzen 9 7950X3D, RTX 4090, Gen 4 NVMe). Treat the figures as order-of-magnitude guidance only.

| Snapshot size | P50      | P95      | P99      | Dominant term |
| ------------- | -------- | -------- | -------- | ------------- |
| 1 MiB         | ~5 ms    | ~9 ms    | ~14 ms   | first launch + zstd |
| 16 MiB        | ~22 ms   | ~38 ms   | ~55 ms   | bincode + UVM |
| 128 MiB       | ~65 ms   | ~120 ms  | ~180 ms  | UVM migration |
| 512 MiB       | ~150 ms  | ~280 ms  | ~420 ms  | UVM migration |

The shape of the curve is the takeaway: at the 1 MiB end, fixed driver overhead and JIT-cache lookup dominate; from 128 MiB up, the UVM page-fault path dominates and any "sub-millisecond" claim is physically impossible on PCIe Gen 4 hardware.

## Recommendations

- **Keep snapshots small.** A 512 MiB snapshot will never restore in under ~100 ms on commodity hardware. If you can afford to lazily reload weights from a shared model store instead of bundling them into the snapshot, do so. The snapshot path is for hot, latency-sensitive state, not the entire model weights.
- **Pre-warm common kernels.** The first launch after a fresh context creation costs 1–3 ms of driver-resident work. The host can pay this once at process startup by running a trivial no-op kernel against every PTX module the tenant is expected to use; subsequent launches drop to 5–10 µs.
- **Use `cudaMemAdvise(SetReadMostly)` for weight buffers.** For read-only weight regions, advising the driver of the access pattern reduces page-fault traffic on the UVM warmup path. The `tensor-wasm-mem` advise wrappers expose this directly.
- **Prefer the page cache.** Repeated restores of the same snapshot blob will be served from the OS page cache after the first read; the `T_read` term effectively disappears. Co-locate the snapshot store with the worker so this caching actually applies.
- **Avoid zstd levels above 3.** Higher levels marginally improve compression ratios on already-incompressible GPU memory while inflating CPU cost on the hot restore path. The default level 3 is intentional.
- **Measure end-to-end.** The metrics in `tensor-wasm-core::metrics::TensorWasmMetrics` expose per-phase histograms for restore latency; alert on the P99 of the UVM phase, not the total, because the UVM term has the worst tail behaviour.

## Cross references

- `docs/PERFORMANCE.md` — S19 benchmark methodology and the measured numbers that will replace this table.
- `docs/CUDA-SETUP.md` — toolkit and driver versions Craton TensorWasm validates against; UVM behaviour differs meaningfully between CUDA 12.0 and 12.6.
- `docs/AUTO-OFFLOAD.md` — S14's auto-offload heuristic governs which snapshot regions live on device vs. host, and therefore how much of the restore cost is paid up front vs. on demand.
