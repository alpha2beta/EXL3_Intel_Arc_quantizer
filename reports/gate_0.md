# Gate 0 Audit Readiness Report (`G0-01` .. `G0-03`)

> **Report ID:** `reports/gate_0.md`  
> **Time Tag:** `2026-10-20T20:18:50+02:00`  
> **Target Profile:** `lunar_lake_140v_258v_32gb` (Intel Core Ultra 7 258V / Arc 140V `0x64A0`, 32 GB UMA)  
> **Upstream Reference:** `turboderp-org/exllamav3 @ 4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f`  
> **Gate Decision:** **CONDITIONAL GO to Phase 1** (SYCL tile encoder & bounded scratch management), with CUDA ground-truth fixture comparison (`P0-24`/`P0-25` and Open Item `O-1`) carried explicitly as a tracked blocker for external parity sign-off.

> **Evidence Class (skip_cuda.md §4):** highest class demonstrated by this report is **E3**
> ("meets upstream's CUDA-derived quality ceiling", plus **E2** bit-exactness vs the independent
> `exl3xpu.ref` oracle and **E1** source transcription). **E4 (path parity with upstream's CUDA
> kernel) is NOT available** while deferral `SKIP-CUDA-2026-10-21-01` / waiver `DEV-2026-10-21-01`
> is active — see [`skip_cuda.md`](../skip_cuda.md) §5–§6.

---

## 1. Gate 0 Checklist Verification

| Gate Item | Status | Evidence |
|---|---|---|
| **G0-01** Compatibility manifest, algorithm contract, dependency inventory, working extension skeleton, and fixture/comparison harness | **PASS (with P0-24 CUDA fixture gap recorded)** | `docs/compatibility_manifest.md`, `docs/algorithm_contract.md`, `docs/native_dependency_audit.md`, `docs/extension_build.md`, `csrc/probe_xpu/`, `exl3_quant/xpu_probe.py`, `tests/test_negative.py`, `tests/test_oracle_parity.py` |
| **G0-02** Factorization/operator coverage or measured, explicit fallback for initial scope | **PASS** | `scripts/probe_operators.py`, `artifacts/environment/operator_coverage.json`, `exl3_quant/ldlq.py`, `tests/test_ldlq.py` |
| **G0-03** Record go/no-go, blockers, owners, and refined Phase 1 plan in `reports/gate_0.md` | **PASS** | This report (`reports/gate_0.md`) |

---

## 2. Key Phase 0 Findings on Intel Arc 140V

1. **fp16 FMA Rounding (`O-6` — Resolved On-Device):**
   - Probed via `csrc/probe_xpu/probe_kernels.sycl` across 65,536 random inputs and 11 constructed boundary cases where IEEE-754 single rounding (`fp64 FMA -> fp16`) and `fp32` intermediate double rounding diverge (`artifacts/environment/extension_probe.json`).
   - Both `sycl::fma(sycl::half)` and plain `a * a + c` compile to native `mad (M1, 16)` in `half` precision and match **single rounding** (`0 / 65,536` and `0 / 11` mismatches; `11 / 11` mismatches against double rounding).
   - Both `ReferenceTrellisEncoder` and the Phase 1 SYCL ACS kernel therefore use exact single-rounded fp16 FMA semantics.

2. **Shared-Memory Accounting & Reserve Policy (`V140-02`, `P0-20`, `V140-03`, `V140-04`):**
   - Sampled 10-point idle baseline (`artifacts/environment/idle_baseline.json`): `MemAvailable` `13.87 GiB` median, `torch.xpu.mem_get_info` free `8.40 GiB` median of `28.58 GiB` total, swap free `29.46 GiB`.
   - Crucial platform finding: committing and touching a `1 GiB` XPU tensor decrements `mem_get_info` free by only `~9 MiB` and `MemAvailable` by `~200 MiB`, while `torch.xpu.memory_allocated` records `1024.0 MiB` (`artifacts/environment/scratch_allocations.json`).
   - `MemoryPolicy` (`exl3_quant/memory_policy.py`) therefore enforces:
     $$\text{usable} = \max\bigl(0,\; \min(\text{device\_free}, \text{system\_available}) - \text{reserve} - \text{outstanding\_framework\_allocations}\bigr)$$
   - Representative scratch allocations (`K=1, T=256`) succeeded for `B=1, 16, 64` (`16.1 MiB .. 1.01 GiB`) under the default `6.0 GiB` reserve and up to `B=192` (`3.02 GiB`) under a `4.0 GiB` reserve, with zero swap growth.

3. **Operator Coverage & `block_ldl` Fallback (`P0-21`, `P0-22`):**
   - Probed 17 operator/dtype combinations (`artifacts/environment/operator_coverage.json`).
   - **Native on XPU:** `matmul` (`fp32`/`fp16`/`bf16`), `einsum` (`fp32`/`fp16`), `linalg.inv` (`fp32`), `linalg.solve` (`fp32`), `linalg.solve_triangular` (`fp32`).
   - **Silent CPU Fallbacks (`fp32`):** `linalg.cholesky` (`aten::linalg_cholesky_ex.L`), `linalg.cholesky_ex`, `linalg.qr`, `linalg.svd`, `linalg.eigvalsh`.
   - **Unsupported:** `fp16`/`bf16` decompositions (`cholesky`, `inv`, `solve`).
   - **LDLQ Benchmark (`n=4096`):** CPU Cholesky `114.1 ms` vs XPU fallback `131.2 ms` (`H2D` `8.8 ms` at `7.6 GB/s`); full `block_ldl` (`exl3_quant/ldlq.py`) runs in `142.2 ms` on CPU vs `231.2 ms` on XPU (`max |XPU - CPU| = 1.91e-6`), preserving the contract §9.1 retry-damping schedule without silent algorithmic changes.

4. **SM Analogue & Queue Policy (`R11`, `P1-08`):**
   - `gpu_subslice_count = 8` (Xe-cores) is pinned as the SM analogue (`P = 8`), never `max_compute_units = 64` (XVEs).
   - All custom SYCL kernels execute on `c10::xpu::getCurrentXPUStream(device).queue()`; no secondary SYCL queues are created.

---

## 3. Explicit Open Blockers & Limitations (`P0-29`)

| ID | Blocker / Open Item | Impact | Owner | Resolution Path |
|---|---|---|---|---|
| **P0-24 / P0-25** | Upstream CUDA ground-truth fixtures (`activations`, `H`, `L`, `quantize_tiles` outputs/snapshots) not yet generated on this Intel-only host | Cannot claim bit-exact or tolerance-checked parity against CUDA execution until external fixtures are imported | Quantizer Port Maintainer | Run a fixture-dump script against `turboderp-org/exllamav3 @ 4d7b69c` on an NVIDIA GPU host and import checksummed artifacts |
| **O-1** | `quantize_tiles_kernel.cuh:121, 134` terminal-edge (`t = 255`) ping-pong indexing (`argmin_cost` reads `L-1` while `costs` ends in `L-2`) | Reconstructed circular paths are self-consistent and decode bit-exactly via `exl3xpu.ref`, but exact CUDA tie/terminal state parity awaits `P0-24` | Quantizer Port Maintainer | Compare `t = 255` edge buffer snapshot from CUDA (`P0-25`) against `ReferenceTrellisEncoder` |
| **V140-04 (staging)** | Host<->device distinct staging copy accounting | Shared-backing allocation accounting is verified; distinct staging copy accounting awaits Phase 1 XPU kernel host<->device transfers | Quantizer Port Maintainer | Measure during Phase 1 (`P1-03`/`P1-12`) when host staging buffers and XPU device tensors coexist |

*Policy (`P0-29`):* No CUDA comparison checkbox is marked as passed without executed CUDA fixture evidence.

---

## 4. Refined Phase 1 Plan & Effort Estimate

- **Estimated Effort:** ~2 person-weeks (reduced from the initial 2–4 person-week allowance because `O-6`, the `icpx` SYCL build pipeline, `block_ldl`, and `MemoryPolicy` are already verified).
- **Execution Sequence (Phase 1 -> Gate 1):**
  1. **`P1-03` / `P1-04` — SYCL Tile Encoder (`quantize_tiles`):** Port the 2-phase circular Viterbi ACS forward search and GPU-resident traceback to SYCL (`csrc/`), using subgroup sizes `{16, 32}`, 128 KiB SLM budget, and exact single-rounded `sycl::half` arithmetic.
  2. **`P1-05` .. `P1-08` — Encoder Verification:** Validate against `ReferenceTrellisEncoder` and `exl3xpu.ref` across `K in {1, 2, 3, 4}`, all 3 codebooks (`3inst`, `mcg`, `mul1`), tie-handling rules (`P1-06`), partial batches/edge shapes (`P1-07`), and single-stream queue ordering (`P1-08`).
  3. **`P1-11` .. `P1-16` / `V140-04` / `V140-05` — Bounded Scratch Arena & B Sweep:** Wire `MemoryPolicy` into `ScratchEstimator`, implement a bounded scratch arena with smaller-batch retry (`P1-12`/`P1-13`), sweep `B in {1, 16, 64, 128, 192}` under live memory logging (`P1-14`..`P1-16`), and publish `reports/gate_1.md` (`G1-04`).
