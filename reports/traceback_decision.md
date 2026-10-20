# Traceback Placement Decision Report (`P1-17` .. `P1-21`, `V140-04`, `V140-05`)

> **Report ID:** `reports/traceback_decision.md`  
> **Time Tag:** `2026-10-20T20:57:38+02:00`  
> **Target Profile:** `lunar_lake_140v_258v_32gb` (Intel Core Ultra 7 258V / Arc 140V `0x64A0`, 32 GB LPDDR5x UMA)  
> **Evidence Artifact:** [`artifacts/benchmarks/phase1_encoder_bench.json`](../artifacts/benchmarks/phase1_encoder_bench.json)  
> **Decision:** **Select GPU-resident traceback (`gpu_resident` / `gpu_resident_split`) as the production default** on Intel Arc 140V.

---

## 1. Evaluated Candidates

All four traceback placements were implemented in `csrc/quantize_xpu/` and `exl3_quant/xpu_quantizer.py` and benchmarked on identical 16-tile Gaussian fixtures (`L = 256`, `cb = CB_MUL1`, `K in {2, 3, 4}`) under the 6.0 GiB `MemoryPolicy` reserve:

1. **`gpu_resident` (Fused single-kernel GPU search + traceback):**
   - Executes `Pass 1 forward` $\rightarrow$ parallel subgroup `argmin_cost` $\rightarrow$ `Pass 1 backward` $\rightarrow$ `Pass 2 forward` $\rightarrow$ `Pass 2 backward` inside a single SYCL kernel per wave (`FusedQuantizeKernel`).
   - Only the final `output_tiles` (`(N, 256)` fp32) and `output_indices` (`(N, 256)` int16) cross back to the host.
2. **`gpu_resident_split` (Split-kernel GPU search + GPU traceback, `P1-17`):**
   - Separates `ForwardOnlyKernel` and `BackwardOnlyKernel` into four ordered submissions per wave on the current XPU stream so forward ACS search and GPU traceback can be timed independently and register pressure in `ForwardOnlyKernel` is minimized.
3. **`d2h_cpu` (Device-edge D2H copy + CPU traceback, `P1-18`):**
   - Runs `ForwardOnlyKernel` on XPU, copies `temp_costs` and the full `temp_edges` tensor (`(B, 256, 65536 >> K)` int16) to a distinct host NumPy buffer after each pass, and executes traceback on the CPU.
4. **`usm_host` (Direct SYCL `malloc_host` USM edge writes + zero-copy CPU traceback, `P1-19`):**
   - Allocates `temp_edges` and `temp_costs` in Level Zero host-accessible USM (`sycl::malloc_host`), allowing the GPU forward kernel to write predecessor edges directly into host-visible memory and the CPU to trace back without an explicit `D2H` memcpy.

---

## 2. Measured Latency, Bandwidth & Memory Pressure (`16` tiles, `L = 256`, `CB_MUL1`)

| `K` | Mode | Total (ms) | Search (ms) | Traceback (ms) | Transfer (ms) | Throughput (tiles/s) | Scratch (`temp_*`) | Distinct Host Staging (`V140-04`) | Bit-Exact vs Ref |
|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| **2** | `gpu_resident` | **28.09** | 27.90 | *(fused)* | **0.16** | 569.7 | 129.0 MiB | 0.016 MiB | **Yes** |
| **2** | `gpu_resident_split` | **25.61** | 25.23 | **0.23** | **0.10** | **624.6** | 129.0 MiB | 0.016 MiB | **Yes** |
| **2** | `d2h_cpu` | 79.20 | 25.25 | 3.05 | 50.85 | 202.0 | 129.0 MiB | **128.016 MiB** | **Yes** |
| **2** | `usm_host` | 41.34 | 25.69 | 7.24 | 0.06 | 387.0 | 129.0 MiB | **128.016 MiB** | **Yes** |
| **3** | `gpu_resident` | **16.05** | 15.96 | *(fused)* | **0.08** | 996.8 | 64.5 MiB | 0.016 MiB | **Yes** |
| **3** | `gpu_resident_split` | **13.77** | 13.39 | **0.24** | **0.10** | **1161.8** | 64.5 MiB | 0.016 MiB | **Yes** |
| **3** | `d2h_cpu` | 28.93 | 13.47 | 2.62 | 12.80 | 553.1 | 64.5 MiB | **64.016 MiB** | **Yes** |
| **3** | `usm_host` | 14.44 | 13.42 | 0.88 | 0.04 | 1107.9 | 64.5 MiB | **64.016 MiB** | **Yes** |
| **4** | `gpu_resident` | **15.85** | 15.63 | *(fused)* | **0.19** | 1009.5 | 32.3 MiB | 0.016 MiB | **Yes** |
| **4** | `gpu_resident_split` | **13.68** | 13.32 | **0.25** | **0.08** | **1169.3** | 32.3 MiB | 0.016 MiB | **Yes** |
| **4** | `d2h_cpu` | 29.64 | 16.31 | 2.98 | 10.31 | 539.8 | 32.3 MiB | **32.016 MiB** | **Yes** |
| **4** | `usm_host` | 22.69 | 13.39 | 7.19 | 0.05 | 705.1 | 32.3 MiB | **32.016 MiB** | **Yes** |

---

## 3. Architectural Analysis & Decision Rationale

1. **Why GPU-resident traceback wins decisively on Arc 140V:**
   - Although each tile's backward pointer walk (`256` dependent loads) runs on a single thread (`tid == 0`) per work-group, a batch of `16` tiles runs **16 backward walks concurrently across the 8 Xe-cores**, completing both Pass 1 (`argmin_cost` + `128` steps) and Pass 2 (`256` steps) in **`0.23–0.25 ms` total** (`~15 µs/tile`), or `< 1.8%` of total encoding time.
   - By contrast, CPU traceback (`d2h_cpu` and `usm_host`) executes serially on the host and takes `0.88–7.24 ms`.

2. **Why `d2h_cpu` is penalized even on Unified Memory (`V140-04`):**
   - Even though the Arc 140V shares physical LPDDR5x DRAM with the CPU, `c_edges.cpu()` allocates a **distinct host staging buffer** (`128 MiB` for `16` tiles at `K=2`) and copies `2 × 128 MiB = 256 MiB` per wave (`50.85 ms` at `K=2`, making `d2h_cpu` **3.1$\times$ slower** end-to-end and doubling the effective physical memory footprint of `temp_edges`!).
   - `ScratchArena` explicitly tracks `distinct_staging_bytes` alongside XPU device allocations so host staging copies are never omitted (`V140-04`).

3. **Why `usm_host` beats `d2h_cpu` but still trails `gpu_resident`:**
   - Writing `temp_edges` directly to `sycl::malloc_host` USM (`P1-19`) eliminates the `50.85 ms` explicit `D2H` copy at `K=2` (`41.34 ms` vs `79.20 ms`), proving zero-copy USM host access works cleanly on Lunar Lake.
   - However, host-coherent USM writes and uncached/coherent CPU random pointer-chasing across `128 MiB` still add `7.24 ms` at `K=2` compared to `0.23 ms` for GPU-resident traceback.

4. **Why `gpu_resident_split` slightly outperforms fused `gpu_resident`:**
   - As shown by the IGC Xe2 (`lnl-m`) assembly dump (`P1-24`), `ForwardOnlyKernel` compiles into `566–600` instructions with `0–6` flag spills (`LOCAL_FIRST_FIT_BC_RA`), whereas fusing two forward loops + argmin + two backward loops into `FusedQuantizeKernel` (`2020–2380` instructions, `HYBRID_BC_RA`) increases live-range pressure (`14` flag spill stores / `20` loads in `SIMD32`).
   - Both `gpu_resident` (upstream-compatible single C++ call `ext.quantize_tiles`) and `gpu_resident_split` produce identical bit-for-bit outputs and remain supported in `XPUTrellisEncoder`.
