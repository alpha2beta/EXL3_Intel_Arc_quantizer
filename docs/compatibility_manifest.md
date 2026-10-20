# Compatibility Manifest: Intel Core Ultra 7 258V / Arc 140V (Lunar Lake)

> **Document ID:** `docs/compatibility_manifest.md`  
> **Profile:** `lunar_lake_140v_258v_32gb`  
> **Date:** 2026-10-20  
> **Tasks:** P0-05, P0-08, V140-01  
> **Status:** Platform environment pinned and empirically probed; encoder/layer qualification in progress.

---

## 1. Hardware Specification & Queried Device Identity

| Parameter | Specification / Probed Value | Provenance / Query Source |
|---|---|---|
| **CPU Model** | Intel(R) Core(TM) Ultra 7 258V | `lscpu` (8 cores, family 6, model 189, stepping 1) |
| **System Memory** | 32 GB on-package LPDDR5x | `free -h` (30.0 GiB available to Linux kernel, 30.0 GiB swap) |
| **Memory Architecture** | Unified Physical Pool (UMA / shared system memory) | Integrated graphics with shared memory controller |
| **Integrated GPU** | Intel Arc 140V Graphics | PCI `00:02.0` (device ID `0x64A0`, rev 04) |
| **Neural Processing Unit** | Intel Core Ultra 200V NPU | PCI `00:0b.0` (device ID `0x643e`, rev 04) |
| **GPU Compute Units (EUs)** | 64 Vector Engines (XVEs) / 8 Xe-cores (subslices) | `torch.xpu.get_device_properties(0)` (`max_compute_units=64`, `gpu_subslice_count=8`) |
| **Local Memory (SLM)** | 128 KiB per work-group | `torch.xpu.get_device_properties(0).local_mem_size` |
| **Last-Level Cache (L3)** | 8192 KiB (8 MiB) | `torch.xpu.get_device_properties(0).last_level_cache_size` |
| **Supported Subgroups** | SIMD16, SIMD32 (`[16, 32]`) | `torch.xpu.get_device_properties(0).sub_group_sizes` |
| **Max Work-Group Size** | 1024 work-items | `torch.xpu.get_device_properties(0).max_work_group_size` |
| **Native Dtype Flags** | `has_fp16=1`, `has_fp64=1`, `has_atomic64=1` | `torch.xpu.get_device_properties(0)` |
| **Device Classification** | `is_integrated_gpu=1` | `torch.xpu.get_device_properties(0)` |

---

## 2. Software Stack & Toolchain Provenance

| Component | Pinned Version / Path | Build & Package Provenance |
|---|---|---|
| **Operating System** | Linux `7.2.11-1-cachyos-deckify` x86_64 | Arch Linux / CachyOS kernel |
| **Python Runtime** | `3.12.14` | `/home/yanchun/Projects/vllm.xpu/.venv/bin/python` (`uv` CPython build) |
| **Build System** | Ninja `1.13.2` + `torch.utils.cpp_extension` | Bundled `pybind11` headers via PyTorch `2.14.0+xpu` |
| **Intel Compute Runtime** | `26.35.39758.10-1.1` | `local/intel-compute-runtime` (Level Zero & OpenCL NEO driver) |
| **Level Zero Loader** | `1.32.0-1` | `local/level-zero-loader` |
| **Unified Runtime Driver** | `1.17.39758` | Intel Level-Zero V2 driver reported by PyTorch XPU |
| **oneAPI DPC++/C++ Compiler** | `2026.0.0_947-1.1` | `/opt/intel/oneapi/compiler/2026.0/bin/icpx` |
| **oneAPI MKL & SYCL BLAS** | `2026.0.0_908-1.1` | `/opt/intel/oneapi/mkl/2026.0` |
| **PyTorch XPU Runtime** | `2.14.0+xpu` | Verified working on `xpu:0` in `/home/yanchun/Projects/vllm.xpu/.venv` |
| **Triton XPU** | `3.8.0+xpu` | Installed in active Python environment |
| **Validation Decoder Oracle** | `exl3xpu 0.1.0` | Co-located at `/home/yanchun/Projects/exl3xpu` (`exl3xpu.ref`) |

---

## 3. Pinned Upstream Repository Revisions

- **Upstream Repository:** `https://github.com/turboderp-org/exllamav3`
- **Pinned Commit:** `4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f`
- **Audited Modules:**
  - `exllamav3/modules/quant/exl3_lib/quantize.py` (`1B11658F5DE96BFE`)
  - `exllamav3/exllamav3_ext/quant/quantize_tiles_kernel.cuh` (`F8399AA3D281625F`)
  - `exllamav3/exllamav3_ext/quant/codebook.cuh` (`0E3C63B323F8D3CC`)
  - `exllamav3/exllamav3_ext/quant/pack.cu` (`27606EED6650ACC3`)
  - `exllamav3/exllamav3_ext/quant/reconstruct.cu` (`F47F121CDB452CF8`)
  - `exllamav3/util/hadamard.py` (`6884841B61378788`)

---

## 4. Memory Accounting & Admission Model

### 4.1 Physical vs Accounting Boundaries

1. **One Physical Pool:** 32 GB system memory is dynamically partitioned between CPU host processes and the integrated GPU driver. There is no discrete VRAM bank.
2. **Runtime Memory Query:** `torch.xpu.mem_get_info()` reports `(free, total)`. Queried total is `30,692,810,752` bytes (~28.6 GiB); measured idle free is **~8.0–8.4 GiB** (it varies with background workloads) while `/proc/meminfo` reports **~13.0–13.9 GiB** `MemAvailable`. The gap is the driver's own pool, which is not visible as free system memory.
3. **Counters Lag Commitments (measured 2026-10-20):** a touched 1 GiB XPU tensor moved `mem_get_info` free by only **~9 MiB** and `MemAvailable` by **~200 MiB**, while `torch.xpu.memory_allocated` moved by **exactly 1024 MiB**. Admission checks must therefore subtract the framework's live allocation total (`memory_allocated`) explicitly; free counters alone over-admit. See `exl3_quant/memory_policy.py` and `artifacts/environment/scratch_allocations.json`.
4. **No Double Counting:** CPU RSS, GPU allocations, shared mappings, and driver accounting overlap. Do not sum RSS and GPU-allocated bytes as independent physical quantities. Measured device-free and `MemAvailable` deltas track each other (within noise) during allocations, confirming one shared backing.
5. **Guardrail Reserve:** A minimum configurable reserve of **6.0 GiB** system memory must remain untouched by live allocations to prevent OS thrashing, page faults, and swap swapping. Enforced by `MemoryPolicy` (`vram_reserve` configurable per run).
6. **Level Zero Single Allocation Boundary:** Tested allocations must obey Level Zero max single allocation queries (measured `max_mem_alloc_size` = `30,692,810,752` B, i.e. the full reported device total).

---

## 5. Pinned Model Target & Calibration Specification (P0-02, P0-06)

| Parameter | Pinned Specification |
|---|---|
| **Primary Dense Architecture** | Meta Llama 3.1 8B (`meta-llama/Meta-Llama-3.1-8B`) |
| **Source Checkpoint Revision** | `48d6d03f933bc570042267aa51a83355a67738d4` (fp16 / bfloat16 base weights) |
| **Secondary Target Architecture**| Qwen 2.5 7B (`Qwen/Qwen2.5-7B`, commit `d4385dd2a52377042ab34a883a451e505ea6bfda`) |
| **Calibration Corpus** | `wikitext-2-raw-v1` (test split) |
| **Calibration Tokens / Samples** | 128 samples $\times$ 2048 sequence length (262,144 tokens total) |
| **Sequence Handling** | Deterministic non-overlapping chunks, left-truncated, seed `42` |
| **Hessian Covariance Dtype** | float32 accumulate ($H = \sum X^T X$), damped with $\lambda = 0.01 \cdot \text{mean}(\text{diag}(H))$ |

---

## 6. Prototype Quantizer Scope & Target Recipes (P0-03)

- **Initial Scope:** Single dense decoder layer conversion (attention projections + MLP).
- **Codebook:** `mul1` (`cb = 2`), affine pseudorandom byte-sum LUT over 16-bit state (913 distinct levels in `[-3.453125, 3.34765625]`).
- **Bitrate Parameter K:** Integer $K \in \{2, 3, 4\}$.
- **Tile Format:** $16 \times 16$ elements ($L = 256$), pre-permuted into `tensor_core_perm` order.
- **Trellis States:** $S = 65536 \gg K$ (16,384 for $K=2$; 8,192 for $K=3$; 4,096 for $K=4$).
- **Packing Format:** State-history container: 16 spans $\times$ 16 elements, low $K$ bits MSB-first into uint16 words, uint32 little-endian pairs with `SWAP16` half-word swap.
- **Downstream Oracle Compatibility:** Bit-exact decode verified live against `exl3xpu.ref` (`tests/test_oracle_parity.py`).

---

## 7. Numerical Acceptance Criteria & Tolerances (P0-07)

| Acceptance Tier | Scope & Metric | Required Threshold | Justification |
|---|---|---|---|
| **Tier 1: Container Structure** | Tensor names, dtypes, metadata | 100% exact match | Format compatibility with ExLlamaV3 `.safetensors` |
| **Tier 2: Discrete Bitpacking** | State-history round-trip (`pack→decode`) | 0 bit difference | Lossless container round trip for low $K$ bits |
| **Tier 3: Weight Reconstruction** | Reconstructed fp16 weights vs LUT | 0 bit difference | Exact codebook index lookup |
| **Tier 4: Quantizer Numerics** | Viterbi path validity & distortion | 100% valid transitions; $\text{MSE} < 0.15$ on $N(0,1)$ at $K=2$ | Monotonic distortion improvement with increasing $K$ |
| **Tier 5: Model Perplexity** | Full model Wikitext-2 perplexity | $\Delta \text{PPL} \le 0.15$ vs CUDA reference | Quality gate preventing catastrophic degradation |
| **Tier 6: Oracle Runtime Interop** | Live decode through `exl3xpu.ref` | 100% bit-exact across all shapes & $K$ | Prevents runtime decode divergence |

---

## 8. Explicit Exclusions & Out-of-Scope Modes

The following modes are explicitly excluded from the Arc 140V initial prototype:
- **MoE (Mixture of Experts):** Non-dense routing topologies deferred to Phase 4.
- **Vision / Audio / Multimodal:** Projections with specialized tokenizers excluded.
- **Multi-Token Prediction (MTP):** Speculative decoding auxiliary heads deferred.
- **Two-Sided LDLQ:** Output Hessian / column damping modes deferred.
- **Fractional Average Bitrates:** Sub-integer mixed-bitrate scheduling deferred.
- **Discrete Arc / Windows:** Relocated to separate optional deployment tracks.

---

## 9. Arc 140V Live Headroom & Allocation Risk (V140-02, V140-07)

- **Measured Idle Baseline [2026-10-20 20:01:23 +0200]:** `MemAvailable` 13.0–13.9 GiB, `mem_get_info` free 8.0–8.4 GiB, swap free 29.5 GiB (≈1.4 GiB already used by other processes). Sampling: 10 snapshots at 1 s intervals (`artifacts/environment/idle_baseline.json`).
- **Configured System Reserve:** 6.0 GiB (mandatory to avoid kernel swapping and driver thrashing).
- **Usable Live Headroom:** **≈ 2.0–2.4 GiB** = `min(device free, MemAvailable) − 6.0 GiB`, before subtracting live framework allocations. This is lower than the earlier 4.0 GiB estimate because the measured device free is ~8.0–8.4 GiB, not ~10 GiB.
- **Measured Allocations Through the Framework Allocator** (`artifacts/environment/scratch_allocations*.json`, `K=1`, `T=256`):
  - `B=1` (16.1 MiB), `B=16` (258 MiB), `B=64` (1.01 GiB): allocated and touched successfully at the 6 GiB reserve (3–51 ms allocation).
  - `B=128` (2.02 GiB): succeeds only when device free ≥ ~8.0 GiB; with a 4 GiB reserve it completes in 79 ms.
  - `B=192` (3.02 GiB): succeeds at a 4 GiB reserve (1.55 s allocation), denied at 6 GiB.
  - `B=256` (4.03 GiB): denied at both reserves (measured bound `B ≤ 222` at 4 GiB reserve; `B ≈ 94` at 6 GiB on the sampled idle state).
  - Oversized 20 GiB request: refused by policy before any allocation.
  - No swap growth in any case; `torch.xpu.memory_allocated` matched the requested bytes exactly.
- **Policy Directive:** On Arc 140V, clamp scratch batch capacity to what the *live* budget admits (`MemoryPolicy.check`), with `B ≤ 64` for `K=1` (1.0 GiB) during full-layer execution to leave headroom for activations, Hessian accumulation, and PyTorch workspace. Do not hard-code batch sizes from the 32 GB label; the device-visible free pool is the binding constraint.
