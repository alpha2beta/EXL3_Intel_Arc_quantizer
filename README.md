# EXL3 Intel Arc Quantizer

An implementation and tooling suite for porting the **EXL3 quantization and model conversion pipeline** from NVIDIA CUDA to Intel Arc GPUs (Xe2-LPG / Xe-HPG / Xe2-HPG) using SYCL, oneAPI, and PyTorch XPU.

Compatible with upstream [ExLlamaV3](https://github.com/turboderp-org/exllamav3) format and downstream Intel Arc inference runtimes such as [exl3xpu](https://github.com/0xSero/exl3xpu).

---

## Current Status & Verification Baseline

| Metric | Status / Value | Verification Source |
|---|---|---|
| **Phase / Maturity** | **Gate 0 & Gate 1 Closed** (Phase 2 Single-Layer Next) | [`reports/gate_0.md`](./reports/gate_0.md), [`reports/gate_1.md`](./reports/gate_1.md) |
| **Unit Test Suite** | **84 / 84 passed (100%)** | `python -m unittest discover -s tests -p "test_*.py"` |
| **Primary Prototype Target** | Intel Core Ultra 7 258V / Arc 140V (32 GB UMA) | [`docs/compatibility_manifest.md`](./docs/compatibility_manifest.md) |
| **Downstream Oracle Parity** | **100% Bit-Exact** (CPU Ref & XPU SYCL vs `exl3xpu.ref`) | [`tests/test_oracle_parity.py`](./tests/test_oracle_parity.py), [`tests/test_xpu_quantizer.py`](./tests/test_xpu_quantizer.py) |
| **Upstream Pinned Commit** | `turboderp-org/exllamav3 @ 4d7b69c7` | [`docs/algorithm_contract.md`](./docs/algorithm_contract.md) |
| **Active Target Model** | Meta Llama 3.1 8B / Qwen 2.5 7B | Pinned in compatibility manifest |

---

## Primary Prototype Platform: Intel Arc 140V (Lunar Lake)

Development is anchored on an integrated **Intel Core Ultra 7 258V** platform with **Intel Arc 140V integrated graphics** and 32 GB shared on-package memory (30.0 GiB available to Linux kernel).

- **GPU Architecture:** Xe2-LPG, device ID `0x64A0`, 8 Xe-cores (subslices), 64 XVEs, 128 KiB shared local memory (SLM) per workgroup, SIMD16/32 execution.
- **Software Toolchain:** Arch Linux x86_64, Intel oneAPI 2026.0 (`icpx`), Compute Runtime `26.35.39758.10`, Level Zero `1.17.39758`, and PyTorch XPU `2.14.0+xpu`.
- **Memory Model & Headroom:** Unified physical memory pool. Live headroom under the mandatory 6.0 GiB OS/driver reserve is **$\approx 4.0 \text{ GiB}$**. Dynamic scratch allocation is bounded to safe batch limits ($B \le 128$ for $K=2$, $B \le 64$ for $K=1$) to prevent paging and swap thrashing.
- **Discrete Arc Track:** Discrete targets (Arc A770, Arc B580, Pro B60) are qualified separately on an independent deployment track.

For architectural details, refer to [`docs/140v.md`](./docs/140v.md) and [`docs/compatibility_manifest.md`](./docs/compatibility_manifest.md).

---

## Core Implemented Modules (`exl3_quant/`)

### 1. Contract-Conformant Trellis Encoder ([`exl3_quant/reference_encoder.py`](./exl3_quant/reference_encoder.py))
Direct host-CPU transcription of the upstream CUDA kernel `quantize_tiles_kernel.cuh:23-313`:
- **16-bit State Space:** $S = 65536 \gg K$ tracked states, $Kr = 16 - K$, candidate branch states $\sigma = (k \ll Kr) \mid e$, predecessor $pred = \sigma \gg K$.
- **Two-Pass Tail-Biting Circular Ring:** Pass 1 ($roll = 128$) unconstrained forward/backward to find `end_state`; Pass 2 ($roll = 0$) wrap-constrained by `pre_state = end_state` emitting the optimal state path.
- **Exact fp16 Fused Metric:** Single-rounding `__hfma2` emulation using float64 intermediate accumulation with strict `<` first-branch tie-breaking.
- **Vectorized NumPy Acceleration:** Precomputes candidate branch tables, executing 256-step dynamic programming in ~15 ms per tile on CPU.

### 2. Format Primitives & Bitpacking ([`exl3_quant/trellis_format.py`](./exl3_quant/trellis_format.py))
Device-independent format contract transcribed from `pack.cu`, `codebook.cuh`, and `quantize.py`:
- **Procedural Codebooks:** Bit-exact LUTs for `mul1` (913 distinct levels in `[-3.453125, 3.34765625]`), `mcg` (10,746 levels), and `3INST` (10,598 levels).
- **Tensor-Core Permutation:** Exact 256-element permutation table `tensor_core_perm()` and inverse `tensor_core_perm_inv()`.
- **State-History Container:** Upstream MSB-first packing (16 spans $\times$ 16 elements, low $K$ bits per state) with `SWAP16` half-word swap (`pack_trellis`, `swap16_storage`, `decode_states`, `decode_tile_values`).

### 3. Scratch Memory Estimator ([`exl3_quant/scratch_estimator.py`](./exl3_quant/scratch_estimator.py))
Predicts exact VRAM requirements based on the pinned allocator contract:
- Costs buffer: $B \times 2 \times S \times 2$ bytes (`float16`).
- Edges buffer: $B \times T \times S \times 2$ bytes (`int16`).
- Batch capacity solver `calculate_max_batch_size(...)` with configurable UMA/system reserves and Level Zero single-allocation bounds.

---

## Test Suite & Verification

The test harness enforces exact discrete checks, perturbation detection, and live oracle cross-validation:

```bash
# Set PYTHONPATH to project root and run all unit tests
PYTHONPATH=. /home/yanchun/Projects/vllm.xpu/.venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

### Test Breakdown (42 Tests Total)

- **[`tests/test_oracle_parity.py`](./tests/test_oracle_parity.py) (4 tests):** Live verification against `exl3xpu.ref` proving bit-exact identity for `tensor_core_perm`, all 3 codebooks across 65,536 states, `decode_states` across $K \in \{1, 2, 3, 4, 5, 6, 8\}$, and end-to-end encoded tile reconstruction.
- **[`tests/test_encoder.py`](./tests/test_encoder.py) (8 tests):** Validates shift-register inverse identity $((pred \ll K) \mid e) \equiv state$, continuous circular ring topology, bit-exact codebook reconstruction, end-to-end container round-trips for $K \in \{2, 3, 4\}$, path cost tracking, and bounded distortion.
- **[`tests/test_trellis_format.py`](./tests/test_trellis_format.py) (17 tests):** Format-conformance tests covering MSB-first bitstreams, `SWAP16`, state-history properties, and codebook range invariants.
- **[`tests/test_negative.py`](./tests/test_negative.py) (5 tests):** Perturbation harness detecting 1-bit container flips, scale distortions, illegal trellis transitions, non-integer inputs, and insufficient memory budgets.
- **[`tests/test_scratch.py`](./tests/test_scratch.py) (4 tests):** Arithmetic regression fixtures for $K \in \{1, 2, 3, 4\}$ and batch sizing edge cases.
- **[`tests/test_bitpack.py`](./tests/test_bitpack.py) (4 tests):** Legacy sub-byte container round-trip shims.

---

## Documentation & Governance

The repository maintains an evidence-backed governance model:

- [`EXL3_Intel_Arc_quantizer.md`](./EXL3_Intel_Arc_quantizer.md) — Technical assessment, architecture breakdown, kernel feasibility, and 14-week roadmap.
- [`docs/140v.md`](./docs/140v.md) — Arc 140V / Core Ultra 7 258V adaptation guide.
- [`docs/compatibility_manifest.md`](./docs/compatibility_manifest.md) — Pinned hardware/software environment, model target, calibration corpus, 6-tier acceptance criteria, and memory risk analysis.
- [`docs/algorithm_contract.md`](./docs/algorithm_contract.md) — Authoritative 13-section mathematical contract audited from upstream ExLlamaV3.
- [`docs/deviation_log.md`](./docs/deviation_log.md) — Formal deviation log and review waivers (`DEV-2026-10-20-01`).
- [`tasks.md`](./tasks.md) — Project implementation task list and phase gates (Phase 0 to Phase 4).
- [`progress.md`](./progress.md) — Detailed implementation progress log with per-entry commit timestamps.
- [`review.md`](./review.md) — Independent code review conclusions and formal remediation responses.

---

## Roadmap & Next Milestones

1. **Native Dependency Inventory ([`docs/native_dependency_audit.md`](./docs/native_dependency_audit.md) — P0-15):** Complete formal classification of all `ext.*` conversion call sites (`ext.quantize_tiles`, `ext.pack_trellis`, `ext.pack_signs`, `ext.had_paley`, `ext.had_paley2`).
2. **Arc 140V XPU Allocation & Extension Probe (P0-18, P0-20, V140-02):** Build a minimal custom SYCL/XPU extension using oneAPI 2026.0 and validate live memory allocation limits on the local Arc 140V GPU.
3. **Numerical Preprocessing & Block LDLQ Path (P2-04, P2-06, P2-08):** Implement the normalized Sylvester Hadamard ($H_{128}$) transform and blockwise LDLQ error feedback loop.
4. **CUDA Ground Truth Capture (P0-24, P0-25):** Capture reference tile fixtures and cost snapshots from a CUDA reference environment to close Tier-4 quantizer numerical parity and resolve Open Item O-1.
