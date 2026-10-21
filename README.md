# EXL3 Intel Arc Quantizer

An implementation and tooling suite for porting the **EXL3 quantization and model conversion pipeline** from NVIDIA CUDA to Intel Arc GPUs (Xe2-LPG / Xe-HPG / Xe2-HPG) using SYCL, oneAPI, and PyTorch XPU.

Compatible with upstream [ExLlamaV3](https://github.com/turboderp-org/exllamav3) format and downstream Intel Arc inference runtimes such as [exl3xpu](https://github.com/0xSero/exl3xpu).

---

## Current Status & Verification Baseline

| Metric | Status / Value | Verification Source |
|---|---|---|
| **Phase / Maturity** | **Gate 0 & Gate 1 Closed** (Phase 2 Single-Layer Conversion Next) | [`reports/gate_0.md`](./reports/gate_0.md), [`reports/gate_1.md`](./reports/gate_1.md) |
| **Unit Test Suite** | **143 / 143 passed (100%)** | `python -m unittest discover -s tests -p "test_*.py"` |
| **Primary Prototype Target** | Intel Core Ultra 7 258V / Arc 140V (32 GB UMA) | [`docs/compatibility_manifest.md`](./docs/compatibility_manifest.md) |
| **Downstream Oracle Parity** | **100% Bit-Exact** (CPU Ref & XPU SYCL vs `exl3xpu.ref`) | [`tests/test_oracle_parity.py`](./tests/test_oracle_parity.py), [`tests/test_xpu_quantizer.py`](./tests/test_xpu_quantizer.py) |
| **Upstream Pinned Commit** | `turboderp-org/exllamav3 @ 4d7b69c7` | [`docs/algorithm_contract.md`](./docs/algorithm_contract.md) |
| **Active Target Models** | Meta Llama 3.1 8B / Qwen 2.5 7B | Pinned in compatibility manifest |
| **XPU Encoder Throughput** | Up to **1,301.5 tiles/s** (43.67 G state transitions/s) | [`artifacts/benchmarks/phase1_encoder_bench.json`](./artifacts/benchmarks/phase1_encoder_bench.json) |

---

## Primary Prototype Platform: Intel Arc 140V (Lunar Lake)

Development is anchored on an integrated **Intel Core Ultra 7 258V** platform with **Intel Arc 140V integrated graphics** and 32 GB shared on-package memory (30.0 GiB available to Linux kernel).

- **GPU Architecture:** Xe2-LPG, device ID `0x64A0`, 8 Xe-cores (subslices), 64 XVEs, 128 KiB shared local memory (SLM) per workgroup, SIMD16/32 execution.
- **Software Toolchain:** Arch Linux x86_64, Intel oneAPI 2026.0 (`icpx`), Compute Runtime `26.35.39758.10`, Level Zero `1.17.39758`, and PyTorch XPU `2.14.0+xpu`.
- **Memory Model & Headroom:** Unified physical memory pool. Live headroom under the mandatory 6.0 GiB OS/driver reserve is **$\approx 4.0 \text{ GiB}$**. Dynamic scratch allocation is bounded to safe batch limits ($B \le 128$ for $K=2$, $B \le 64$ for $K=1$) to prevent paging and swap thrashing.
- **Discrete Arc Track:** Discrete targets (Arc A770, Arc B580, Pro B60) are qualified separately on an independent deployment track.

For architectural details, refer to [`docs/140v.md`](./docs/140v.md) and [`docs/compatibility_manifest.md`](./docs/compatibility_manifest.md).

---

## Core Implemented Architecture (`exl3_quant/` & `csrc/`)

### 1. SYCL XPU Tile Encoder ([`csrc/quantize_xpu/`](./csrc/quantize_xpu/) & [`exl3_quant/xpu_quantizer.py`](./exl3_quant/xpu_quantizer.py))
Native Intel Arc SYCL implementation of the EXL3 forward Trellis Viterbi search and circular traceback:
- **Exact fp16 Dynamic Programming:** Add-Compare-Select (ACS) dynamic programming using exact fp16 `__hsub2` and single-rounded `sycl::fma` accumulation (`O-6` on-device verified), bit-for-bit identical to the CPU reference.
- **Two-Pass Tail-Biting Circular Closure:** Pass 1 ($roll = 128$) unconstrained forward/backward to find `end_state`; Pass 2 ($roll = 0$) wrap-constrained by `pre_state = end_state` emitting the optimal path.
- **Traceback Placement Architectures:**
  - `gpu_resident`: GPU-resident circular traceback baseline with sub-millisecond traceback latency (<0.25 ms per wave). Selected as default (`reports/traceback_decision.md`).
  - `d2h_cpu`: Device-to-host edge transfer followed by CPU traceback.
  - `usm_host`: Direct host-accessible edge writes via SYCL host-allocated USM.
- **Geometry & Cache Sweeps:** Evaluated SIMD16 vs SIMD32 workgroups and SLM vs global cost storage. SLM achieves +49% higher throughput at $K \in \{3, 4\}$; global cost storage excels at $K=2$ by avoiding SLM occupancy bottlenecks.
- **Shape Support:** Fully supports $K \in \{1, 2, 3, 4\}$, all 3 codebooks (`mul1`, `mcg`, `3INST`), standard $L=256$ tiles, and $L=160$ n-gram rows.

### 2. Bounded Scratch Arena & Memory Policy ([`exl3_quant/xpu_quantizer.py`](./exl3_quant/xpu_quantizer.py), [`exl3_quant/memory_policy.py`](./exl3_quant/memory_policy.py))
Adaptive, pressure-aware memory manager replacing upstream's unbounded `get_temp_buffers`:
- **Single-Slot Stream Arena (`ScratchArena`):** Event-aware reuse and eviction synchronized with the current PyTorch XPU stream. Prevents cross-$K$ memory leaks.
- **Dual-View Accounting:** Jointly samples OS `/proc/meminfo` and framework allocator (`torch.xpu.mem_get_info` / `memory_allocated`). Enforces a strict 6.0 GiB OS/driver reserve.
- **Pre-Admission Abort & Adaptive Fallback:** Pre-checks requests against usable headroom; raises `MemoryPressureError` before exhausting memory. Automatically retries with halved batch capacities ($B \to B/2$) without changing tile semantics.

### 3. Block LDLQ Decomposition & Preprocessing ([`exl3_quant/ldlq.py`](./exl3_quant/ldlq.py))
Direct contract transcription of the blockwise Cholesky/LDLQ factorization (`quantize.py:416-490`):
- **Contract-Conformant Inversion:** Inverts block diagonal $L_{bb}^{-1}$ and folds into $M_{:, block} = L_{:, block} L_{bb}^{-1}$.
- **Retry-Only Damping Schedule:** Preserves upstream's adaptive diagonal regularisation ($2 \cdot \sigma_{\text{reg}} \cdot \text{mean}(\text{diag}(H))$ with up to 10 retries) without artificial pre-emptive damping.
- **Operator Attribution Probe:** Probed 17 operator/dtype pairs on Lunar Lake XPU; catalogued explicit CPU fallback boundaries (e.g., `torch.linalg.cholesky` on XPU falls back to CPU).
- **One-Sided LDLQ Error Feedback (`P2-08`):** `ldlq()` reproduces upstream's bottom-up span/block traversal (`quantize.py:493-609`): additive target correction from the strictly-lower `L` slice, tensor-core tile permutation, un-permuted reconstruction, and trailing-span residual feedback. Tile encoding is a pluggable callback with bit-exact reference and Arc XPU adapters that reuse a single `ScratchArena`.

### 4. Contract-Conformant CPU Reference Encoder ([`exl3_quant/reference_encoder.py`](./exl3_quant/reference_encoder.py))
Direct host-CPU transcription of the upstream CUDA kernel `quantize_tiles_kernel.cuh:23-313`:
- **16-bit State Space:** $S = 65536 \gg K$ tracked states, $Kr = 16 - K$, candidate branch states $\sigma = (k \ll Kr) \mid e$, predecessor $pred = \sigma \gg K$.
- **Exact fp16 Fused Metric:** Single-rounding `__hfma2` emulation using float64 intermediate accumulation with strict `<` first-branch tie-breaking.
- **Vectorized NumPy Acceleration:** Precomputes candidate branch tables, executing 256-step dynamic programming in ~15 ms per tile on CPU.

### 5. Format Primitives & Bitpacking ([`exl3_quant/trellis_format.py`](./exl3_quant/trellis_format.py))
Device-independent format contract transcribed from `pack.cu`, `codebook.cuh`, and `quantize.py`:
- **Procedural Codebooks:** Bit-exact LUTs for `mul1` (913 distinct levels in `[-3.453125, 3.34765625]`), `mcg` (10,746 levels), and `3INST` (10,598 levels).
- **Tensor-Core Permutation:** Exact 256-element permutation table `tensor_core_perm()` and inverse `tensor_core_perm_inv()`.
- **State-History Container:** Upstream MSB-first packing (16 spans $\times$ 16 elements, low $K$ bits per state) with `SWAP16` half-word swap (`pack_trellis`, `swap16_storage`, `decode_states`, `decode_tile_values`).

### 6. Scratch Memory Estimator ([`exl3_quant/scratch_estimator.py`](./exl3_quant/scratch_estimator.py))
Predicts exact VRAM requirements based on the pinned allocator contract:
- Costs buffer: $B \times 2 \times S \times 2$ bytes (`float16`).
- Edges buffer: $B \times T \times S \times 2$ bytes (`int16`).
- Batch capacity solver `calculate_max_batch_size(...)` with configurable UMA/system reserves and Level Zero single-allocation bounds.

### 7. Sylvester Hadamard Transform ([`exl3_quant/hadamard.py`](./exl3_quant/hadamard.py))
Contract §8 reference path, transcribed from upstream `util/hadamard.py`:
- **Natural Sylvester `H_128`**, orthonormalized by `1/sqrt(128)`; symmetric and self-inverse.
- **Blockwise application** (`H @ x` per 128 rows, `x @ H` per 128 columns), matching upstream `blockwise_preapply_had_{l,r}_` — not a full Kronecker transform.
- **Decode ordering** `H_l → ×su → H_r → ×sv` (`weight_from_quantized`), verified against `exl3xpu.ref.weight_orig`.

### 8. Weight Regularization ([`exl3_quant/regularize.py`](./exl3_quant/regularize.py))
Contract §9.3, transcribed from `quantize.py:1079-1236`:
- `block_rms` with upstream's 32-slice accumulation order.
- Output-scale skew heuristic (top 2% of `sqrt(H_diag)` vs 15% cutoff), out-channel RMS normalization, `sv`/`su` sign+scale folding, and the zero-channel floor (`0.1`) / `sv = 0` rule.
- Global-scale (`g_scale_gss`) search ([`exl3_quant/gss.py`](./exl3_quant/gss.py), P2-04 complete): wrapped-diagonal tile sampling with highest/lowest-RMS outlier tiles, then the upstream coarse grid (0.1–1.9, strided subsample) → fine grid (0.075 around the coarse minimum, full sample) → parabolic refinement with a `0.01` floor. Test-quantization runs through the pluggable encoder protocol (Arc XPU by default); grid evaluations are batched into as few encoder calls as possible.

---

## Test Suite & Verification

The test harness enforces exact discrete checks, perturbation detection, and live oracle cross-validation:

```bash
# Set PYTHONPATH to project root and run all unit tests
PYTHONPATH=. /home/yanchun/Projects/vllm.xpu/.venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

### Complete Test Breakdown (143 Tests Total)

- **[`tests/test_xpu_quantizer.py`](./tests/test_xpu_quantizer.py) (13 tests):** End-to-end XPU quantizer verification: all 65,536 codebook states, bit-exact parity vs `ReferenceTrellisEncoder` across $K \in \{1, 2, 3, 4\}$ and all 3 codebooks, $L=160$ n-gram rows, live oracle decode via `exl3xpu.ref`, edge cases & ties (zeros, constant, outliers), partial batches & waves, input validation, stream-queue ordering, all 4 traceback placements & kernel variants (`SG16`/`SG32`, SLM/global), `ScratchArena` bounded eviction across $K$, automatic batch halving retry, no double scratch residency during retry (B1), and zero resident scratch leakage on failed acquisition (B2).
- **[`tests/test_xpu_extension.py`](./tests/test_xpu_extension.py) (7 tests):** Custom SYCL XPU extension sanity, stream execution, memory lifetime, device cap queries, and on-device measurement of Open Item `O-6` (exact fp16 single-rounding FMA verified against 65,536 random and 11 adversarial inputs).
- **[`tests/test_hadamard.py`](./tests/test_hadamard.py) (11 tests):** Sylvester values/symmetry/orthogonality, non-power-of-two rejection, blockwise-vs-block-diagonal equivalence, involution, in-place validation, bit-exact `exl3xpu.ref.hadamard128`, and decode-order parity with `weight_orig`.
- **[`tests/test_gss.py`](./tests/test_gss.py) (19 tests):** wrapped-diagonal coverage/value verification against manual extraction, extreme-RMS tile inclusion, stage-by-stage recomputation of the coarse→fine→parabolic pipeline, batching/chunking invariance, MSE-below-`g_scale=1` with the real encoder, determinism, huge-weight scale walk-down, `select_refined_scale` floor/basin tests, and `regularize` wiring (folding `weight *= g; su /= g` + decode round-trip with the `g`-scaled `su`).
- **[`tests/test_ldlq_loop.py`](./tests/test_ldlq_loop.py) (13 tests):** identity-`L` equivalence with independent tile quantization, exact additive target correction, independent dense transcription for `buf_size_k ∈ {16,32,64}`, feedback objective improvement on a correlated Hessian, span invariance, upstream diagonal-zeroing equivalence, encoded layout/dtype, input immutability, validation, and XPU↔reference parity.
- **[`tests/test_regularize.py`](./tests/test_regularize.py) (13 tests):** `block_rms` accuracy and slice-boundary accumulation, full regularize→decode round-trip (identity, forced out-scales off, and global-scale folding), skew heuristic both directions, zero-channel flooring, all-zero `zeros` flag, non-mutation, and input validation.
- **[`tests/test_memory_policy.py`](./tests/test_memory_policy.py) (15 tests):** OS `/proc/meminfo` parsing, reserve budget arithmetic, admission boundary enforcement, outstanding allocation tracking, `MemoryPressureError` context, post-commit reserve-breach detection, allocator-growth verification (M1), and live GPU allocation under reserve constraints.
- **[`tests/test_ldlq.py`](./tests/test_ldlq.py) (10 tests):** Algebraic verification of `block_ldl` ($L L^T = H$, block-diagonal inversion fold, identity block diagonals), adaptive retry-damping schedule on ill-conditioned Hessians, input validation, and XPU vs CPU parity.
- **[`tests/test_oracle_parity.py`](./tests/test_oracle_parity.py) (4 tests):** Live verification against `exl3xpu.ref` proving bit-exact identity for `tensor_core_perm`, all 3 codebooks across 65,536 states, `decode_states` across $K \in \{1, 2, 3, 4, 5, 6, 8\}$, and end-to-end encoded tile reconstruction.
- **[`tests/test_encoder.py`](./tests/test_encoder.py) (8 tests):** Validates shift-register inverse identity $((pred \ll K) \mid e) \equiv state$, continuous circular ring topology, bit-exact codebook reconstruction, end-to-end container round-trips for $K \in \{2, 3, 4\}$, path cost tracking, and bounded distortion.
- **[`tests/test_trellis_format.py`](./tests/test_trellis_format.py) (17 tests):** Format-conformance tests covering MSB-first bitstreams, `SWAP16`, state-history properties, and codebook range invariants.
- **[`tests/test_negative.py`](./tests/test_negative.py) (5 tests):** Perturbation harness detecting 1-bit container flips, scale distortions, illegal trellis transitions, non-integer inputs, and insufficient memory budgets.
- **[`tests/test_scratch.py`](./tests/test_scratch.py) (4 tests):** Arithmetic regression fixtures for $K \in \{1, 2, 3, 4\}$ and batch sizing edge cases.
- **[`tests/test_bitpack.py`](./tests/test_bitpack.py) (4 tests):** Legacy sub-byte container round-trip shims.

---

## Phase Reports & Governance Documents

The project maintains an evidence-backed governance model:

- [`reports/gate_0.md`](./reports/gate_0.md) — Gate 0 audit closure report (operator coverage, `block_ldl` benchmark, conditional GO to Phase 1).
- [`reports/gate_1.md`](./reports/gate_1.md) — Gate 1 encoder readiness report (reconstruction parity, bounded scratch, throughput sweeps, GO to Phase 2).
- [`reports/traceback_decision.md`](./reports/traceback_decision.md) — Comprehensive comparative analysis of `gpu_resident`, `d2h_cpu`, and `usm_host` traceback placements on Lunar Lake UMA.
- [`artifacts/benchmarks/phase1_encoder_bench.json`](./artifacts/benchmarks/phase1_encoder_bench.json) — Complete machine-readable benchmark results across $K \in \{1, 2, 3, 4\}$, codebooks, subgroup sizes, and batch capacities.
- [`EXL3_Intel_Arc_quantizer.md`](./EXL3_Intel_Arc_quantizer.md) — Initial technical assessment, kernel feasibility, and roadmap.
- [`docs/140v.md`](./docs/140v.md) — Arc 140V / Core Ultra 7 258V adaptation guide.
- [`docs/compatibility_manifest.md`](./docs/compatibility_manifest.md) — Pinned hardware/software environment, model target, calibration corpus, 6-tier acceptance criteria, and memory risk analysis.
- [`docs/algorithm_contract.md`](./docs/algorithm_contract.md) — Authoritative 13-section mathematical contract audited from upstream ExLlamaV3.
- [`docs/native_dependency_audit.md`](./docs/native_dependency_audit.md) — Complete classification of all upstream `ext.*` conversion sites and XPU porting strategy.
- [`docs/extension_build.md`](./docs/extension_build.md) — Build and linkage requirements for oneAPI 2026.0 / PyTorch XPU C++ extensions.
- [`docs/deviation_log.md`](./docs/deviation_log.md) — Formal deviation log and review waivers (`DEV-2026-10-20-01`).
- [`tasks.md`](./tasks.md) — Implementation task list and phase gates (Gate 0 through Gate 3; Phase 4 has no gate).
- [`progress.md`](./progress.md) — Detailed implementation progress log with per-entry commit timestamps.
- [`review.md`](./review.md) — Independent code review conclusions and formal remediation responses.

---

## Roadmap & Next Milestones (Phase 2)

With Gate 0 and Gate 1 successfully closed, and core numerics implemented (Sylvester $H_{128}$ in `hadamard.py`, regularization in `regularize.py`, LDLQ error feedback in `ldlq.py`, and global-scale search in `gss.py`), Phase 2 focuses on full single-layer execution:

1. **P2-01 / P2-02 / P2-03 — Layer Execution & Hessian Accumulation:** Run forward pass on `Meta-Llama-3.1-8B` / `Qwen2.5-7B` decoder layer on XPU, harvest covariance statistics $H = \sum X^T X$ on the pinned calibration corpus, and profile memory within the 32 GB shared UMA budget.
2. **P2-11 / P2-12 — Layerwise Workspace & Residual Metrics:** Measure factorization residuals, reconstruction error, and proxy loss across layer projections; verify module-lifetime scratch retention.
3. **P2-13 / P2-14 / P2-15 / P2-16 — Checkpoint Packing & Downstream Load:** Pack quantized layer tensors into EXL3-conformant `.safetensors` containers and verify load & inference compatibility against `exl3xpu`.
4. **Gate 2 — Single-Layer Qualification:** Validate repeatable layer conversions without memory growth under the 6.0 GiB reserve, and publish `reports/gate_2.md`.
5. **P0-24 / P0-25 — CUDA Ground Truth Capture:** Capture reference tile fixtures and cost snapshots from a CUDA reference environment to settle Open Item `O-1` and complete Tier-4 quantizer verification.
