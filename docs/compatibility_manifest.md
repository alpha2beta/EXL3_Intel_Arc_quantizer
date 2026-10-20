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
| **Operating System** | Linux x86_64 | Arch Linux / rolling kernel 6.x |
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
2. **Runtime Memory Query:** `torch.xpu.mem_get_info()` reports `(free, total)`. Under idle baseline on this system, queried total is `30,692,810,752` bytes (~28.6 GiB) with ~10.7 GiB free.
3. **No Double Counting:** CPU RSS, GPU allocations, shared mappings, and driver accounting overlap. Do not sum RSS and GPU-allocated bytes as independent physical quantities.
4. **Guardrail Reserve:** A minimum configurable reserve of **6.0 GiB** system memory must remain untouched by live allocations to prevent OS thrashing, page faults, and swap swapping.
5. **Level Zero Single Allocation Boundary:** Tested allocations must obey Level Zero max single allocation queries.

---

## 5. Prototype Quantizer Scope & Target Recipes

- **Initial Architecture:** Dense transformer decoder layer (e.g., Llama-3 / Qwen-2.5 7B–8B).
- **Codebook:** `mul1` (`cb = 2`), affine pseudorandom byte-sum LUT over 16-bit state (913 distinct levels in `[-3.453125, 3.34765625]`).
- **Bitrate Parameter K:** Integer $K \in \{2, 3, 4\}$.
- **Tile Format:** $16 \times 16$ elements ($L = 256$), pre-permuted into `tensor_core_perm` order.
- **Trellis States:** $S = 65536 \gg K$ (e.g. 16384 for $K=2$, 8192 for $K=3$, 4096 for $K=4$).
- **Packing Format:** State-history container: 16 spans $\times$ 16 elements, low $K$ bits MSB-first into uint16 words, uint32 little-endian pairs with `SWAP16` half-word swap.
- **De-quantization Compatibility:** Checkpoints produced must decode identically through `exl3xpu.ref` and upstream ExLlamaV3.
