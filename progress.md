# EXL3 Intel Arc Quantizer: Implementation Progress

> **Last Updated:** 2026-10-20 22:02:00 +02:00  
> **Target Platform:** Intel Core Ultra 7 258V (Lunar Lake), Arc 140V integrated GPU (`0x64A0`), 32 GB shared memory  
> **Current Status:** Gate 0 and Gate 1 complete (GO to Phase 2 recorded in `reports/gate_1.md`), and the Phase 1 hardening review (`review-2026-10-20T21:06:51+02:00`) fully remediated: `ScratchArena` reclamation defects B1/B2 fixed with regression tests, and residuals M2/M3/L2/L5 plus the M1 negative test closed. 87 unit tests passing. Next: Phase 2 numerical preprocessing (`P2-04`/`P2-06`).  
> **Time-tag convention:** every entry below carries a local-time timestamp `[YYYY-MM-DD HH:MM:SS ±ZZZZ]` so that entries from different sessions can be ordered unambiguously. Timestamps are taken from the corresponding git commit date where one exists.

---

## Executive Summary

Steps 1 through 15 of the foundational and Phase 1 engineering work have been implemented, unit-tested, and verified against the audited contract:

- Steps 1–5 established the initial CPU sandbox under review waiver `DEV-2026-10-20-01`.
- Steps 6–7 fetched and audited the pinned upstream EXL3 source, establishing [`docs/algorithm_contract.md`](./docs/algorithm_contract.md) and [`exl3_quant/trellis_format.py`](./exl3_quant/trellis_format.py).
- **Step 8 [2026-10-20 17:10:00 +0200]:** Probed the host hardware and software environment, amended `tasks.md` with the Arc 140V platform profile, and published [`docs/compatibility_manifest.md`](./docs/compatibility_manifest.md) (P0-05, P0-08, V140-01).
- **Step 9 [2026-10-20 17:22:55 +0200]:** Replaced the provisional prototype encoder with the **contract-conformant `ReferenceTrellisEncoder`** directly transcribing `quantize_tiles_kernel.cuh:23-313` with NumPy vectorization (P1-01, P1-02, P1-05).
- **Step 10 [2026-10-20 18:55:00 +0200]:** Added live oracle parity test suite ([`tests/test_oracle_parity.py`](./tests/test_oracle_parity.py)) verifying bit-exact decode against the adjacent downstream runtime (`exl3xpu.ref`), and expanded manifest with model/corpus specifications and 140V memory risk analysis.
- **Step 11 [2026-10-20 19:18:02 +0200]:** Published [`docs/native_dependency_audit.md`](./docs/native_dependency_audit.md), classifying all six `ext.*` entry points and the supporting runtime surface, with the XPU port decisions and blockers (P0-15, P0-16).
- **Step 12 [2026-10-20 19:48:23 +0200]:** Built, verified, and committed the **minimal custom SYCL XPU extension** through `torch.utils.cpp_extension.load` (P0-18, P0-19), and **measured open item O-6 on-device**: the Arc 140V `sycl::fma(half)` performs exact fp16 single rounding — never fp32 double rounding — on 65,536 random and 11 adversarial-constructed inputs.
- **Step 13 [2026-10-20 20:03:00 +0200]:** Established the **idle memory baseline** and the **configurable reserve/abort policy** (V140-02), tested representative scratch allocations through the framework allocator up to 3.02 GiB (P0-20, V140-03), and documented the measured **counter-lag** behaviour of the shared-memory accountant (V140-04 partial).
- **Step 14 [2026-10-20 20:18:50 +0200]:** Implemented the contract §9.1 **`block_ldl`** transcription and equation test suite (P0-22), probed **17 operator/dtype pairs** with per-op XPU->CPU fallback attribution and benchmarked `block_ldl` up to `n=4096` (P0-21, P0-22), and published [`reports/gate_0.md`](./reports/gate_0.md) (G0-01..G0-03).
- **Step 15 [2026-10-20 20:57:38 +0200]:** Implemented the **SYCL XPU tile encoder** (`csrc/quantize_xpu/`, `exl3_quant/xpu_quantizer.py`), bounded **`ScratchArena`** with smaller-batch retry, 4-mode **traceback placement comparison** ([`reports/traceback_decision.md`](./reports/traceback_decision.md)), IGC Xe2 (`lnl-m`) ISA resource analysis, and closed **Gate 1** ([`reports/gate_1.md`](./reports/gate_1.md), P1-03..P1-29, V140-04, V140-05, G1-01..G1-03).

```
+-------------------------------------------------------------------------------------------------------+
| Module                       | Task Mapping             | Status                          | Unit Test Suite       |
+------------------------------+--------------------------+---------------------------------+-----------------------+
| 1. Tasks Checklist           | Scope Governance         | Committed (Amended for 140V)    | tasks.md              |
| 2. Scratch Estimator         | P1-09, P1-10, P1-11      | Passed — CONFIRMED CORRECT      | test_scratch.py (4)   |
| 3. Format Primitives & Packer| P0-13, P0-14, P2-13      | Passed — AUDITED CONTRACT       | test_trellis_format(17|
| 4. Golden Reference Encoder  | P1-01, P1-02, P1-05      | Passed — SOURCE TRANSCRIBED     | test_encoder.py (8)   |
| 5. Oracle Parity Suite       | P0-04, P0-28, P2-16      | Passed — LIVE ORACLE BIT-EXACT  | test_oracle_parity (4)|
| 6. Negative Test Harness     | P0-28                    | Passed (Perturbation Det.)      | test_negative.py (5)  |
| 7. Algorithm Contract (docs) | P0-11, P0-13, P0-14, 16  | Published & Maintained          | docs/algorithm_...    |
| 8. 140V Compatibility Manif. | P0-05, P0-08, V140-01    | Pinned Model/Corpus/Mem Risk    | docs/compatibility... |
| 9. Legacy Bitpacker (Shim)   | Legacy Sandbox           | Superseded (Retained for test)  | test_bitpack.py (4)   |
| 10. Native Dependency Audit  | P0-15, P0-16             | Published — 6 ext + 17 runtime  | docs/native_dep...    |
| 11. XPU Extension Probe      | P0-18, P0-19, O-6        | Built & Verified (icpx SYCL)    | test_xpu_extension (7)|
| 12. Memory Policy & Baseline | V140-02/03/04, P0-20     | Measured — Reserve Enforced     | test_memory_policy(14)|
| 13. Block LDL & Op Coverage  | P0-21, P0-22, G0-01..03  | Gate 0 Closed (Conditional GO)  | test_ldlq.py (10)     |
| 14. XPU SYCL Tile Encoder    | P1-03..P1-29, G1-01..03  | Gate 1 Closed (100% Bit-Exact)  | test_xpu_quantizer(11)|
+------------------------------+--------------------------+---------------------------------+-----------------------+
```

---

## Detailed Breakdown of Completed Steps

### Step 1: Git Tracking of Tasks Checklist
* **Time tag:** [2026-10-20 13:27:16 +0200]
* **Commit:** `491818c` (*docs: add tasks.md implementation task list and phase gates*)
* **File:** [`tasks.md`](./tasks.md)
* **Scope:** 283-line implementation checklist establishing phase gates (Gate 0 through Gate 3; Phase 4 has no gate), acceptance evidence templates, numerical tolerances, and regression fixtures.

---

### Step 2: Scratch Memory Estimator & Sizing Unit Tests
* **Time tag:** [2026-10-20 13:35:01 +0200] (fixes folded in at [2026-10-20 13:55:32 +0200], commit `fc73c07`)
* **Commits:** `3ee68d4`, `fc73c07`
* **Source:** [`exl3_quant/scratch_estimator.py`](./exl3_quant/scratch_estimator.py)
* **Test Suite:** [`tests/test_scratch.py`](./tests/test_scratch.py)
* **Implemented Capabilities:**
  * **Tasks P1-09 & P1-10:** Implemented `ScratchEstimator.estimate_scratch(...)` computing exact cost buffer bytes ($B \times 2 \times S \times \text{bytes}$) and predecessor edge bytes ($B \times T \times S \times \text{bytes}$).
  * **Unit Distinction:** Formally distinguished binary units ($\text{GiB} / \text{MiB}$, $1024^3 / 1024^2$) from decimal units ($\text{GB} / \text{MB}$, $1000^3 / 1000^2$).
  * **Task P1-11:** Implemented `ScratchEstimator.calculate_max_batch_size(...)` solving for maximum safe batch capacity $B$ under constrained device VRAM (e.g. 12 GiB Arc B580 vs. 16 GiB Arc A770).
  * **Edge Case Handling:** Strictly rejects configurations where even minimum batch $B=1$ cannot fit within memory constraints (ValueError).
* **Regression Test Verification:** Verified the 4 exact arithmetic fixtures from `tasks.md` lines 121–129:
  * $K=1 \implies S=32768, \text{ edges}=4.0 \text{ GiB}, \text{ costs}=32.0 \text{ MiB}$
  * $K=2 \implies S=16384, \text{ edges}=2.0 \text{ GiB}, \text{ costs}=16.0 \text{ MiB}$
  * $K=3 \implies S=8192,  \text{ edges}=1.0 \text{ GiB}, \text{ costs}=8.0 \text{ MiB}$
  * $K=4 \implies S=4096,  \text{ edges}=0.5 \text{ GiB}, \text{ costs}=4.0 \text{ MiB}$
* **Audit status [2026-10-20 14:19:27 +0200]:** confirmed against upstream `get_temp_buffers` (`quantize.py:52-63`) — shapes `(B, 2, 65536>>K) float16` and `(B, tile_len, 65536>>K) int16`, batch policy `B = 256` or `max(256, 2*multi_processor_count)` for $K \ge 4$. **No rework required.**

---

### Step 3: Sub-Byte Bitpacker (Provisional Reference — now known to be format-incompatible)
* **Time tag:** [2026-10-20 13:35:01 +0200] (fixes folded in at [2026-10-20 13:55:32 +0200], commit `fc73c07`)
* **Commits:** `3ee68d4`, `fc73c07`
* **Source:** [`exl3_quant/bitpack.py`](./exl3_quant/bitpack.py)
* **Test Suite:** [`tests/test_bitpack.py`](./tests/test_bitpack.py)
* **Implemented Capabilities:**
  * **Provisional Scope:** Generic sub-byte integer packing (2, 3, 4, and 8-bit) into 32-bit unsigned words (`uint32`).
  * **Bitstream Packing:** Handles power-of-two word slicing (2, 4, 8 bits) and 3-bit cross-word boundary packing.
  * **Input Validation:** Enforces strict integer typing (rejecting floating-point inputs with fractional components) and bounds checking $[0, 2^{\text{bits}} - 1]$.
* **Test Verification:**
  * Validated bit-exact round-trip reconstruction (`unpack(pack(x)) == x`) across arbitrary sequence lengths, all-zeros, all-max vectors, and out-of-range rejection.
* **Audit status [2026-10-20 14:19:27 +0200]:** **superseded by Step 7.** The upstream container is a *state-history* encoding, not an independent-symbol encoding — see the warning table above. This module must be **replaced**, not patched. Its round-trip tests are self-consistency only and say nothing about interop.

---

### Step 4: CPU Golden Reference Trellis Encoder (Provisional Prototype — now known to be incorrect)
* **Time tag:** [2026-10-20 13:35:01 +0200]
* **Commits:** `3ee68d4`
* **Source:** [`exl3_quant/reference_encoder.py`](./exl3_quant/reference_encoder.py)
* **Test Suite:** [`tests/test_encoder.py`](./tests/test_encoder.py)
* **Implemented Capabilities:**
  * **Provisional `mul1` Codebook (`Mul1Codebook`):** Algebraic level decoding ($level = \text{scale} \cdot (c - \text{zero\_point})$) with zero memory lookups.
  * **Shift-Register Trellis Topology (`TrellisStateSpace`):** Precomputed state transitions ($s_{\text{next}} = ((s_{\text{prev}} \ll \text{bits}) \mid \text{symbol}) \pmod S$).
  * **Viterbi ACS & Traceback (`ReferenceViterbiEncoder`):** Implemented forward Add-Compare-Select (ACS) dynamic programming, ping-pong cost tracking, backpointer table recording, and backward traceback path recovery.
* **Test Verification:**
  * Verified state connectivity, MSE error bounds on Gaussian synthetic weights, and full self-consistency integration (continuous weights $\to$ Viterbi encode $\to$ sub-byte pack $\to$ unpack $\to$ reconstruct).
* **Audit status [2026-10-20 14:19:27 +0200]:** **superseded by Step 6/7.** The trellis scale, the codebook, the metric precision and the tail closure are all different upstream — see the warning table above. Must be **replaced** by a transcription of `quantize_tiles_kernel.cuh`.

---

### Step 5: Perturbation & Negative Test Harness (P0-28)
* **Time tag:** [2026-10-20 13:55:32 +0200]
* **Commit:** `fc73c07`
* **Source:** [`tests/test_negative.py`](./tests/test_negative.py)
* **Verification:** Proved that the test harness actively detects:
  1. Single-bit corruption in packed integer containers.
  2. Scale or codebook parameter distortions.
  3. Illegal / non-conforming trellis transitions.
  4. Floating-point inputs passed to the integer packer.
  5. Inadequate memory budgets during batch capacity solving.

---

### Step 6: Upstream Native Source Audit & Algorithm Contract (P0-11, P0-13, P0-14, P0-16)
* **Time tag:** [2026-10-20 14:19:27 +0200]
* **Commit:** `7647cd6`
* **Artefact:** [`docs/algorithm_contract.md`](./docs/algorithm_contract.md)
* **Method:** the pinned upstream revision was fetched and read directly — **no CUDA and no Intel Arc hardware required**:
  ```text
  https://github.com/turboderp-org/exllamav3 @ 4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f
    exllamav3/modules/quant/exl3_lib/quantize.py            SHA-256 1B11658F5DE96BFE
    exllamav3/exllamav3_ext/quant/quantize_tiles_kernel.cuh  SHA-256 F8399AA3D281625F
    exllamav3/exllamav3_ext/quant/codebook.cuh               SHA-256 0E3C63B323F8D3CC
    exllamav3/exllamav3_ext/quant/pack.cu                    SHA-256 27606EED6650ACC3
    exllamav3/exllamav3_ext/quant/reconstruct.cu             SHA-256 F47F121CDB452CF8
    exllamav3/util/hadamard.py                               SHA-256 6884841B61378788
    exllamav3/conversion/convert_model.py                    SHA-256 EE576CA6D2D56E55
  https://github.com/sybil-solutions/exl3xpu  ->  exl3xpu/ref.py   (independent CPU decode oracle)
  ```
  Note: the upstream organisation is `turboderp-org`; `turboderp/*` does not resolve.
* **Published content (13 sections):** trellis state space and ACS recurrence · fp16 metric and tie-break rules · two-pass tail-biting closure and traceback · `mul1`/`mcg`/`3INST` codebook arithmetic · scratch-buffer model · container layout and decoder window semantics · Hadamard conventions · one-sided LDLQ and error feedback · reconstruction path · implications for this repository · **6 explicitly blocked open items** · method notes.
* **Still blocked after the audit:** encoder ground truth (P0-24/P0-25, needs CUDA), `argmin_cost()` buffer parity, fp16 FMA behaviour on the XPU backend, `.safetensors` header metadata, native dependency inventory (P0-15).

---

### Step 7: Audited Format Primitives & Conformance Tests (P0-13, P0-14, P0-28)
* **Time tag:** [2026-10-20 14:19:27 +0200]
* **Commit:** `7647cd6`
* **Source:** [`exl3_quant/trellis_format.py`](./exl3_quant/trellis_format.py)
* **Test Suite:** [`tests/test_trellis_format.py`](./tests/test_trellis_format.py) — 17 tests
* **Implemented Capabilities:**
  * `codebook_lut(cb)` — fp16-exact LUTs for `mul1`, `mcg`, `3INST`, transcribed from `codebook.cuh:56-123`; fused multiply-add reproduced by exact float64 intermediate with a single rounding to fp16.
  * `tensor_core_perm()` / `tensor_core_perm_inv()` — the exact 256-entry 16×16 tile permutation (`quantize.py:22-49`).
  * `pack_trellis()` / `swap16_storage()` — the upstream container layout (`pack.cu:9-57`).
  * `decode_states()` / `decode_tile_values()` — the decoder window semantics from the independent XPU oracle (`exl3xpu/ref.py:81-115`).
* **Test Verification:**
  * **Layout identity:** for $K \in \{1,2,3,4,5,6,8\}$, decode→pack of a random bit stream reproduces the stream bit for bit — proving the MSB-first span layout and the `SWAP16` pairing.
  * **State-history property:** arbitrary 16-bit states do **not** round-trip; only the low $K$ bits of each state survive — the direct proof that an independent-symbol packer cannot be made upstream-compatible by reordering bits.
  * **Negative tests:** flipped bit, reversed bit order within a word, and perturbed codebook level are all detected.
  * **Measured codebook facts:** `mul1` → 913 distinct levels in [-3.453125, 3.34765625]; `3INST` → 10598 levels in [-3.957, 3.973]; `mcg` → 10746 levels in [-3.949, 3.949].
* **Scope limit:** these are *format-conformance* tests. Encoder **parity** with upstream's CUDA search is still blocked on P0-24.

---

### Step 8: Arc 140V Platform Adaptation & Compatibility Manifest (P0-05, P0-08, V140-01)
* **Time tag:** [2026-10-20 16:42:52 +0200] (adaptation guide), finalized at [2026-10-20 17:22:55 +0200] (manifest and tasks amendment)
* **Commits:** `ee90d59`, `613943a`
* **Files:** [`docs/140v.md`](./docs/140v.md), [`tasks.md`](./tasks.md), [`docs/compatibility_manifest.md`](./docs/compatibility_manifest.md)
* **Target Hardware Probed:**
  * CPU: Intel Core Ultra 7 258V (8 cores, Lunar Lake)
  * GPU: Intel Arc 140V integrated graphics (device ID `0x64A0`, 8 Xe-cores / 64 XVEs, 128 KiB SLM, SIMD16/32)
  * Memory: 32 GB shared system memory (30.0 GiB available to Linux kernel, 30.0 GiB swap)
  * Software: oneAPI 2026.0 (`icpx`), Compute Runtime `26.35.39758.10`, Level Zero `1.17.39758`, PyTorch XPU `2.14.0+xpu`
* **Tasks checklist:** Amended `tasks.md` with integrated GPU memory model, 6–8 GiB system reserve policy, and added profile tasks `V140-01` through `V140-09`.

---

### Step 9: Contract-Conformant Golden Reference Trellis Encoder (P1-01, P1-02, P1-05)
* **Time tag:** [2026-10-20 17:22:55 +0200]
* **Commit:** `613943a`
* **Source:** [`exl3_quant/reference_encoder.py`](./exl3_quant/reference_encoder.py)
* **Test Suite:** [`tests/test_encoder.py`](./tests/test_encoder.py) — 7 tests
* **Implementation:**
  * Replaced provisional prototype with `ReferenceTrellisEncoder` directly transcribing `quantize_tiles_kernel.cuh:23-313`:
    * 16-bit state space: $S = 65536 \gg K$ tracked edges, $Kr = 16 - K$, candidate branch states $\sigma = (k \ll Kr) \mid e$, predecessor $pred = \sigma \gg K$.
    * Two-pass tail-biting schedule: $roll = L/2$ unconstrained $\to$ `end_state` $\to$ $roll = 0$ constrained by `pre_state = end_state`.
    * fp16 metric with fused accumulation and strict `<` tie-breaking.
    * Integrated with `trellis_format.py` for procedural codebook arithmetic, tensor-core permutation, and bitpacking.
  * Verified mathematical inverse identity $((pred \ll K) \mid e) == state$ across all 65,536 combinations.
  * Verified continuous unbroken circular ring topology, bit-exact codebook reconstruction, and 100% round-trip match through `pack_trellis` $\to$ `swap16_storage` $\to$ `decode_states` and `decode_tile_values`.

### Step 10: Downstream Oracle Parity Suite & Manifest Expansion (P0-02, P0-04, P0-06, P0-07, H4, H5)
* **Time tag:** [2026-10-20 18:55:00 +0200]
* **Files:** [`tests/test_oracle_parity.py`](./tests/test_oracle_parity.py), [`docs/compatibility_manifest.md`](./docs/compatibility_manifest.md)
* **Oracle Verification (exl3xpu.ref):**
  * Executed live cross-checks against the installed downstream decoder oracle:
    * `tensor_core_perm()` matches bit for bit.
    * `codebook_lut(cb)` matches across all 65,536 states for `CB_3INST`, `CB_MCG`, and `CB_MUL1`.
    * `decode_states(packed, K)` matches bit for bit across all $K \in \{1, 2, 3, 4, 5, 6, 8\}$.
    * `ReferenceTrellisEncoder` output decoded through `oracle_ref.decode_states` and `oracle_ref.reconstruct_inner` reproduces the reconstructed tile with 100% bit-exact identity.
* **Manifest Expansion:** Formally pinned `meta-llama/Meta-Llama-3.1-8B` (fp16/bf16 base weights), `wikitext-2-raw-v1` corpus parameters, 6-tier numerical acceptance thresholds, explicit exclusions (MoE, vision, 2-sided LDLQ), and documented the ~4.0 GiB usable headroom constraint on Arc 140V under the 6.0 GiB reserve policy.

---

### Step 11: Native Dependency Audit (P0-15, P0-16)
* **Time tag:** [2026-10-20 19:18:02 +0200]
* **Commit:** `d632227`
* **Artefact:** [`docs/native_dependency_audit.md`](./docs/native_dependency_audit.md)
* **Method:** re-fetched the pinned upstream sources and verified their SHA-256 prefixes against [`docs/algorithm_contract.md`](./docs/algorithm_contract.md) §0 before classifying anything. No CUDA execution and no SYCL build; runtime validation is explicitly **not** claimed.
* **Custom-extension inventory (six entry points):**
  * `ext.quantize_tiles` — the only mandatory custom port (SYCL kernel; 512-thread work-groups, 32-wide subgroup reduction, dynamic SLM, fp16 FMA metric, `K ∈ 1..8`, `L ∈ {160, 256}`).
  * `ext.pack_trellis` — already reimplemented device-independently; CPU reuse recommended for the first layer.
  * `ext.unpack_trellis` — debug-only, call site commented out.
  * `ext.pack_signs` — dormant, call sites commented out.
  * `ext.had_paley` / `ext.had_paley2` — **unreachable for `H_128`** because 128 takes the Sylvester branch.
* **Runtime surface classified (17 rows):** stream/device guard, dynamic SLM, 512-thread occupancy, subgroup shuffle, half2/FMA, `__dp4a`, `lop3`, funnel shift, error checks, SM-count query, multi-GPU pinned/event path, `cuda_sync_active`, memory helpers, `lru_cache` scratch residency, and torch linear-algebra coverage.
* **Concrete findings recorded:**
  * `torch.xpu` device properties expose **no** `multi_processor_count`; `get_temp_buffers()` would fail as written for `K >= 4`. Recommended SM analogue for the wave cap is `gpu_subslice_count` (8), to be confirmed under P1-14/V140-03.
  * SLM at `K=2` is 66,240 B (64.7 KiB) per work-group, so the upstream 2-work-group/SLM occupancy target cannot be met on the 128 KiB Arc 140V; `K=1` already keeps costs in global memory.
  * `torch.xpu` exposes `Stream`, `Event`, `synchronize`, `current_stream`, `mem_get_info`, and `set_per_process_memory_fraction`, so those are renames rather than rewrites; shared-memory accounting semantics remain unverified (V140-04).
* **Blockers carried forward:** CUDA ground truth (P0-24/P0-25), terminal-edge parity O-1, true-fp16-FMA parity O-6, and the absent XPU extension skeleton (P0-18).

---

### Step 12: Minimal Custom SYCL XPU Extension & O-6 Measurement (P0-18, P0-19, O-6, P0-23 partial)
* **Time tag:** [2026-10-20 19:48:23 +0200]
* **Commit:** `40896bb`
* **Sources:** [`csrc/probe_xpu/probe_xpu.cpp`](./csrc/probe_xpu/probe_xpu.cpp) (bindings/validation), [`csrc/probe_xpu/probe_kernels.sycl`](./csrc/probe_xpu/probe_kernels.sycl) (device code), [`exl3_quant/xpu_probe.py`](./exl3_quant/xpu_probe.py) (build+probes), [`scripts/probe_xpu_extension.py`](./scripts/probe_xpu_extension.py) (CLI/artifacts), [`tests/test_xpu_extension.py`](./tests/test_xpu_extension.py) (7 tests)
* **Intended-framework integration confirmed:** `torch.utils.cpp_extension.load(sources=[.cpp, .sycl], extra_sycl_cflags=[...])` compiles the mixed SYCL build automatically with `icpx` (oneAPI DPC++ 2026.0); clean rebuild verified via `rm -rf artifacts/build/probe_xpu` → 9.8 s icpx/ninja build.
* **Build caveats discovered (required for reproducibility):**
  * oneAPI `bin` must be on `PATH` (`icpx` discovery drives `SYCL_HOME` detection).
  * Named kernel classes are required: torch's build adds `-fsycl-host-compiler`, which conflicts with `-fsycl-unnamed-lambda`.
  * `c10::Half` needs an explicit `#include <c10/util/Half.h>` in the `.sycl` TU; torch `types.h` auto-insertion is not sufficient.
* **P0-18 verifications — all pass on `xpu:0` (`artifacts/environment/extension_probe.json`):**
  * **Device selection:** kernel queued via `c10::xpu::getCurrentXPUStream(t.get_device()).queue()` on the tensor's device; torch and SYCL device names agree (`Intel(R) Arc(TM) Graphics`).
  * **Output correctness:** elementwise add is **bit-exact** versus a CPU reference over 65,536 fp32 values.
  * **Queue ordering:** interleaved sequence kernel → torch `fill_` → kernel → torch `mul_` yields the expected `210.0` result, proving in-order submission on the shared current queue (P1-08 restriction: no secondary queues anywhere).
  * **Tensor lifetime:** 64 MiB alloc → free → re-alloc → reuse produces correct results with no stale-data corruption; `mem_get_info` before/after recorded.
* **P0-19 capability capture (`artifacts/environment/device_capabilities.json`):** torch props (64 EUs, 8 subslices, fp16 yes, fp64 yes) + SYCL-side caps (max work-group 1024, max subgroup 32, local mem 128 KiB, max allocation 30,692,810,752 B = full shared memory). SM analogue decision R11 (`gpu_subslice_count` = 8) recorded in the artifact.
* **Open item O-6 measured on-device (adapted `docs/algorithm_contract.md` §O-6):**
  * **Random block:** 65,536 fp16 `(dh, cost)` pairs — device `sycl::fma` and plain `a*a+c` both match exact-fp64-FMA→fp16 rounding with **0 mismatches**; also 0 mismatches for host fp32 emulation on this random set.
  * **Adversarial block:** exhaustive structured grid search (dh mantissa patterns `0x3C00..0x3CFF` × tiny fp16 costs `0x0080..0x03FF`) found **11 pairs** where fp32 double rounding differs from exact single rounding — random sampling over `1<<24` samples found none, so these hand-picked exponent pairs are the only discriminating inputs.
  * **Result on all 11 adversarial pairs:** `fma_matches_exact` = 11, `fma_matches_fp32_emul` = 0 — **the device performs exact fp16 single rounding and never exhibits fp32 double rounding.** Porting consequence: the CPU reference and the SYCL kernel will agree if both use exact-FMA semantics; fp32 double-rounding emulation in the reference would *introduce* error, not match the device.
* **P0-23 (partial `[~]`):** smoke test = `tests/test_xpu_extension.py` (skips cleanly without XPU); clean rebuild verified; a dedicated build-docs page is still pending.

---

### Step 13: Idle Baseline, Reserve Policy & Scratch Allocation Measurements (V140-02, P0-20, V140-03, V140-04 partial, P0-23)
* **Time tag:** [2026-10-20 20:08:00 +0200]
* **Commit:** `1a98582`
* **Sources:** [`exl3_quant/memory_policy.py`](./exl3_quant/memory_policy.py) (snapshot + policy), [`scripts/measure_idle_baseline.py`](./scripts/measure_idle_baseline.py), [`scripts/measure_scratch_allocations.py`](./scripts/measure_scratch_allocations.py), [`tests/test_memory_policy.py`](./tests/test_memory_policy.py) (14 tests), [`docs/extension_build.md`](./docs/extension_build.md)
* **Idle baseline (V140-02), 10 samples @1 s (`artifacts/environment/idle_baseline.json`):**
  * `MemAvailable` min/med/max = 13.87 / 13.87 / 14.24 GiB; `MemFree` ≈ 7.7–9.1 GiB.
  * XPU `mem_get_info` free min/med/max = 8.40 / 8.40 / 8.41 GiB of 28.58 GiB; idle `memory_allocated` = 0.
  * Swap free 29.46 GiB (≈1.4 GiB already used by other background processes).
  * Under the 6 GiB reserve: **usable ≈ 2.4 GiB** at sampled idle.
* **Policy (V140-02):** `MemoryPolicy(reserve_bytes=6 GiB, safety_slack=256 MiB, max_single_alloc=optional)`.
  * Admission: `min(device_free, system_available) − reserve − outstanding_framework_allocations`.
  * Pre-allocation abort (`MemoryPressureError` with need/usable/context/snapshot) and post-commit re-check; the oversize probe (20 GiB) is refused **before** touching memory.
* **Scratch allocations through the real allocator (P0-20 / V140-03), `K=1, T=256`:**
  * 6 GiB reserve: B=1 (16.1 MiB, 3 ms), B=16 (258 MiB, 20 ms), B=64 (1.01 GiB, 51 ms) succeed; B=128 (2.02 GiB) and above denied at sampled idle (usable 1.48 GiB after the driver free drifted to 7.48 GiB).
  * 4 GiB reserve: B=128 (2.02 GiB, 79 ms) and B=192 (3.02 GiB, 1.55 s) succeed; B=256 (4.03 GiB) denied. Measured capacity bounds: **B ≈ 94** at 6 GiB and **B ≤ 222** at 4 GiB on the sampled state.
  * No swap growth in any case; `torch.xpu.memory_allocated` matched the requested bytes exactly; allocations are admitted before they are attempted.
* **Shared-memory accounting finding (V140-04, partial):**
  * A touched 1 GiB tensor moved `mem_get_info` free by only **9 MiB** and `MemAvailable` by **200 MiB**, while `memory_allocated` moved by **1024 MiB**. The driver serves shared allocations from a pool the generic counters do not decrement, so free counters alone would over-admit.
  * In the scripted cases, device-free and `MemAvailable` deltas track each other (same direction, within noise), which is the evidence that the two counters describe **one** backing store rather than additive pools.
  * The policy was therefore hardened to subtract live framework allocations explicitly; `docs/compatibility_manifest.md` §4.1/§9 updated with the measured figures.
  * Still untested: accounting for **distinct** host↔device staging copies — these do not exist until the kernel stage is implemented, so V140-04 remains `[~]`.
* **P0-23 (now `[x]`):** reproducible build/test instructions published in [`docs/extension_build.md`](./docs/extension_build.md), including the named-kernel, `c10/util/Half.h`, and `torch::xpu::device_count` constraints; clean rebuild re-verified.

### Step 14: Operator Coverage Probe, `block_ldl` Reference & Benchmark, and Gate 0 Report (P0-17, P0-21, P0-22, P0-29, G0-01..G0-03)
* **Time tag:** [2026-10-20 20:18:50 +0200]
* **Commit:** `6ab0e7c`
* **Sources:** [`exl3_quant/ldlq.py`](./exl3_quant/ldlq.py), [`tests/test_ldlq.py`](./tests/test_ldlq.py) (10 tests), [`scripts/probe_operators.py`](./scripts/probe_operators.py), [`artifacts/environment/operator_coverage.json`](./artifacts/environment/operator_coverage.json), [`reports/gate_0.md`](./reports/gate_0.md)
* **Operator coverage (P0-21, `n=512`):**
  * **Native XPU:** `matmul` (`fp32`/`fp16`/`bf16`), `einsum` (`fp32`/`fp16`), `linalg.inv` (`fp32`), `linalg.solve` (`fp32`), `linalg.solve_triangular` (`fp32`). All within tolerance against CPU `float64`.
  * **Silent CPU fallbacks (`fp32`):** `linalg.cholesky` and `linalg.cholesky_ex` (`aten::linalg_cholesky_ex.L`), `linalg.qr` (`aten::linalg_qr.out`), `linalg.svd` (`aten::_linalg_svd.U`), and `linalg.eigvalsh` (`aten::_linalg_eigh.eigenvalues`).
  * **Unsupported:** low-precision (`fp16`/`bf16`) decompositions (`cholesky`, `inv`, `solve`).
* **`block_ldl` reference & fallback benchmark (P0-22):**
  * Implemented `block_ldl` (`exl3_quant/ldlq.py`) directly from contract §9.1 (`quantize.py:416-490`), preserving the retry-only diagonal damping schedule (`2 * sigma_reg * mean(diag(H))`, up to 10 retries) without pre-emptive damping.
  * Verified all extracted equations in `tests/test_ldlq.py` (`L Lᵀ = H`, `M[:, block] = L[:, block] @ inv(L_bb)`, identity block diagonal, retry schedule on near-indefinite matrices, and XPU vs CPU parity).
  * Benchmarked at `n in {512, 1024, 2048, 4096}`: at `n=4096` (`64 MiB` per fp32 matrix), CPU Cholesky takes `114.1 ms`, XPU Cholesky (implicit fallback) takes `131.2 ms` (`H2D` `8.8 ms` at `7.6 GB/s`), and full `block_ldl` takes `142.2 ms` on CPU vs `231.2 ms` on XPU (`max |XPU - CPU| = 1.91e-6`, zero swap growth).
* **Gate 0 closed (`reports/gate_0.md`, G0-01..G0-03):**
  * Conditional GO to Phase 1 recorded, with `P0-24`/`P0-25` (CUDA ground-truth fixtures), `O-1` (terminal-edge ping-pong parity), and `V140-04` (distinct staging copy accounting) tracked explicitly as open items per `P0-29`.

---

### Step 15: SYCL XPU Tile Encoder, Bounded `ScratchArena`, Traceback Placement Comparison & Gate 1 Report (P1-03..P1-29, V140-04, V140-05, G1-01..G1-03)
* **Time tag:** [2026-10-20 20:57:38 +0200]
* **Commit:** `56cc74f`
* **Sources:** [`csrc/quantize_xpu/quantize_kernels.sycl`](./csrc/quantize_xpu/quantize_kernels.sycl), [`csrc/quantize_xpu/quantize_xpu.cpp`](./csrc/quantize_xpu/quantize_xpu.cpp), [`exl3_quant/xpu_quantizer.py`](./exl3_quant/xpu_quantizer.py), [`scripts/benchmark_phase1_encoder.py`](./scripts/benchmark_phase1_encoder.py), [`tests/test_xpu_quantizer.py`](./tests/test_xpu_quantizer.py) (11 tests), [`artifacts/benchmarks/phase1_encoder_bench.json`](./artifacts/benchmarks/phase1_encoder_bench.json), [`reports/traceback_decision.md`](./reports/traceback_decision.md), [`reports/gate_1.md`](./reports/gate_1.md)
* **Correctness & parity (P1-03..P1-08, P0-26):**
  * Aligned `ReferenceTrellisEncoder.forward` and the SYCL kernel on the audited contract's exact fp16 subtraction (`dh = val - w` in fp16, matching `__hsub2` in `quantize_tiles_kernel.cuh:97, 128`) followed by single-rounded fp16 FMA (`sycl::fma`).
  * Verified **100% bit-exact parity** on `output_indices`, `reconstructed`, `packed`, `end_state`, and `path_cost` across all `65,536` codebook states (`CB_3INST`, `CB_MCG`, `CB_MUL1`), bitrates `K in {1, 2, 3, 4}`, tile lengths `L in {256, 160}`, edge cases (`zeros`, `constant`, `outliers`, `alternating`), partial final batches (`19` tiles with `B=6`), multi-wave launches, and live `exl3xpu.ref` decoding.
* **Bounded `ScratchArena` & batch sweep (P1-11..P1-16, V140-04):**
  * Replaced upstream's unbounded `@lru_cache get_temp_buffers` (`R16`) with a single-residency, stream-synchronized `ScratchArena` enforcing `MemoryPolicy`.
  * Verified automatic halving retry (`B=16 -> 8 -> 4` under simulated failure; `K=1, B=256 -> 128` under the `6.0 GiB` reserve) and zero-leak eviction across alternating `K in (2, 2, 3, 4, 2, 3, 2)`.
  * Completed `V140-04`: `ScratchArenaLogEntry` explicitly records `distinct_staging_bytes` (`128 MiB` for `d2h_cpu`/`usm_host` at `K=2, N=16` vs `16 KiB` for `gpu_resident`).
* **Traceback comparison & Xe2 ISA profiling (P1-17..P1-29, V140-05):**
  * Benchmarked `gpu_resident` (`28.09 ms` at `K=2`, `16.05 ms` at `K=3`), `gpu_resident_split` (`25.61 ms` at `K=2`, `13.77 ms` at `K=3`, GPU traceback `0.23–0.25 ms`), `d2h_cpu` (`79.20 ms` at `K=2`, `50.85 ms` D2H copy), and `usm_host` (`41.34 ms` at `K=2`, zero D2H copy via `sycl::malloc_host`). Selected GPU-resident traceback in [`reports/traceback_decision.md`](./reports/traceback_decision.md).
  * Analyzed IGC Xe2 (`lnl-m`) assembly dumps (`P1-24`, `P1-27`): `0` bytes of GRF spills in `128-GRF` mode (`-ze-opt-large-register-file` `256-GRF` rejected because it halves Xe-core EU thread capacity with zero spill reduction).
  * Confirmed the SLM occupancy transition (`P1-26`): SLM cost storage is `+49%` faster at `K=3, 4` (`1,181–1,188 tiles/s`, `39.6–39.9 G transitions/s`, 2–4 WGs/Xe-core), whereas global cost storage is `+17.4%` faster at `K=2` (`724.0 tiles/s`) where `66.4 KiB` SLM caps occupancy to 1 WG/Xe-core.
  * Closed **Gate 1** in [`reports/gate_1.md`](./reports/gate_1.md) (**GO to Phase 2**).

---

### Step 16: Phase 1 Hardening — `ScratchArena` Reclamation Fixes & Review Residuals (B1, B2, M1–M3, L1–L7)
* **Time tag:** [2026-10-20 22:02:00 +0200]
* **Commits:** `3006a32` (B1/B2/M1 fixes + regression tests), `fcdedd4` (M2/M3/L2/L5/M1-test residuals)
* **Trigger:** [`review-2026-10-20T21:06:51+02:00`](./review.md) — Phase 1 accepted, arena reclamation blocking for Phase 2.
* **B1 — retry double residency (fixed):** `ScratchArena.acquire` now clears `costs`/`edges` and calls `empty_cache()` inside the `except` before halving `B`. Independently reproduced: live allocation at the second attempt is now `0.0 MiB` (previously `32.2 MiB` for a `K=3, B=8 → 4` retry). Regression test: `test_retry_path_no_double_residency_b1`.
* **B2 — failed-acquisition leak (fixed):** failure now frees the buffers and raises a freshly constructed `MemoryPressureError` (`raise clean_err from None`) detached from the allocation frame. Independently reproduced in a clean process: `0.000 MiB` live while the exception (with traceback) is retained, `resident_bytes == 0`. Regression test: `test_failed_acquisition_frees_all_scratch_b2`.
* **M1 — post-commit enforcement (strengthened):** `MemoryPolicy.check_after_commit` now treats **allocator-visible growth** (`device_allocated` delta ≥ expected) as authoritative and the OS/device free counters as advisory — matching the measured Arc 140V counter lag. Added the previously missing negative test `test_allocator_growth_assertion`.
* **M2 — benchmark provenance (closed):** `scripts/benchmark_phase1_encoder.py --cold-build` now forces a clean rebuild; the artifact records a measured `cold_build_seconds = 17.22 s` plus `cold_build_provenance`, and the ISA section carries explicit `provenance`. The hand-edited `13.78 s` value is gone.
* **M3 — production extension docs (closed):** `docs/extension_build.md` documents `csrc/quantize_xpu/` (kernel inventory, build lifecycle, actual torch-injected flags) and the validated `IGC_ShaderDumpEnable=1 → OCL_asm*.asm` path; duplicate `## 4.` headings and absolute `file://` links were removed, and the non-functional `ocloc disasm ...spv` command was replaced.
* **L-series:** L1 (test-matrix wording), L3 (split-mode wave note), L4 (`postprocess_seconds`), L5 (`distinct_staging_bytes` conservatism documented), L6 (NaN divergence in the contract), L7 (README/R16) closed; L2 resolved by dropping the mislabeled `mean_wave_dispatch_ms` field from the artifact.

---

## Test Suite Execution Evidence

**87 unit tests** ran and passed cleanly as of [2026-10-20 22:02:00 +0200] across all test suites:
- `test_scratch.py`: 4 tests (scratch arithmetic and allocation limits)
- `test_negative.py`: 5 tests (perturbation detection and input validation)
- `test_trellis_format.py`: 17 tests (codebook LUTs, permutations, bitpacking, state decoding)
- `test_encoder.py`: 8 tests (contract-conformant reference encoder, path cost tracking, bounded distortion)
- `test_oracle_parity.py`: 4 tests (live bit-exact parity against `exl3xpu.ref`)
- `test_bitpack.py`: 4 tests (sub-byte packing shim)
- `test_xpu_extension.py`: 7 tests (device selection, correctness, ordering, lifetime, caps, O-6 random + adversarial; skip unless `torch.xpu.is_available()`)
- `test_memory_policy.py`: 15 tests (meminfo parsing, reserve arithmetic, admission boundaries, outstanding commitments, abort path, post-commit breach, allocator-growth assertion, live 32 MiB GPU allocation)
- `test_ldlq.py`: 10 tests (`L Lᵀ = H`, block-inverse fold, identity block diagonal, retry-damping schedule, input validation, XPU vs CPU parity)
- `test_xpu_quantizer.py`: 11 tests (XPU codebook parity, `K=1..4` & 3-codebook parity, `L=160` support, live `exl3xpu.ref` parity, edge cases & tie verification, partial chunks & waves, input rejection, async stream ordering, all 4 traceback modes & SG16/32 & SLM/global, bounded `ScratchArena` eviction/reuse, and smaller-batch retry)

```text
PYTHONPATH=. /home/yanchun/Projects/vllm.xpu/.venv/bin/python -m unittest discover -s tests -p "test_*.py"
Ran 87 tests in 46.733s

OK
```

---

## Git Commit History

```text
fcdedd4 2026-10-20 22:02:00 +0200  docs(test): close Phase 1 hardening residuals (M2/M3/L2/L5/M1-test)
3006a32 2026-10-20 21:45:00 +0200  fix(review): address review-2026-10-20T21:06:51+02:00 findings
56cc74f 2026-10-20 20:57:38 +0200  feat(xpu): SYCL tile encoder, ScratchArena, traceback comparison, and Gate 1 report (Phase 1)
1feefcc 2026-10-20 20:20:09 +0200  docs(progress): record Step 14 (operator coverage, block_ldl, Gate 0 report) [2026-10-20 20:18:50 +02:00]
6ab0e7c 2026-10-20 20:18:50 +0200  feat(ldlq): operator coverage probe, block_ldl reference, and Gate 0 report (P0-17/P0-21/P0-22/P0-29/G0)
1a98582 2026-10-20 20:03:00 +0200  feat(mem): idle baseline + reserve-based memory-pressure policy; close V140-02/P0-20/P0-23
40896bb 2026-10-20 19:48:23 +0200  feat(xpu): minimal SYCL extension probe via torch cpp_extension; close P0-18/P0-19
d632227 2026-10-20 19:18:02 +0200  docs: add native dependency audit and close P0-15/P0-16
229dca2 2026-10-20 19:05:33 +0200  fix(review): address review-2026-10-20T18:30:21+02:00 findings
e0194b1 2026-10-20 17:26:48 +0200  docs(progress): record Steps 8-9 with commit-aligned time tags [2026-10-20 17:22:55 +02:00]
613943a 2026-10-20 17:22:55 +0200  feat: contract-conformant reference trellis encoder and 140V manifest
ee90d59 2026-10-20 16:42:52 +0200  Add files via upload (adaptation for 140v as the primary development platform)
cc576d9 2026-10-20 14:30:39 +0200  docs(review): append review conclusion for Steps 6-7 audit progress [2026-10-20 14:28:01 +02:00]
b4aac24 2026-10-20 14:25:06 +0200  docs(progress): record the upstream source audit (Steps 6-7) with per-entry time tags
cab8503 2026-10-20 14:21:20 +0200  docs: reconcile deviation log and review response with the completed upstream audit
7647cd6 2026-10-20 14:19:27 +0200  docs+feat: audit pinned upstream EXL3 source, add algorithm contract and audited format primitives
3160885 2026-10-20 13:56:58 +0200  docs: add formal review response document for review-2026-10-20T13:50:25+02:00
fc73c07 2026-10-20 13:55:32 +0200  fix(review): address review-2026-10-20T13:50:25+02:00 findings
f95c640 2026-10-20 13:51:00 +0200  docs: add code review of tasks.md and progress.md (review-2026-10-20T13:50:25+02:00)
bc048fe 2026-10-20 13:38:49 +0200  docs: add progress.md detailing completed Steps 1 to 4 and test verification evidence
668b7dc 2026-10-20 13:35:48 +0200  docs(tasks): update checklist to reflect completed reference encoder, scratch estimator, and bitpacker
3ee68d4 2026-10-20 13:35:01 +0200  feat: implement scratch estimator, bitpacker, and golden reference Viterbi encoder with test suite
491818c 2026-10-20 13:27:16 +0200  docs: add tasks.md implementation task list and phase gates
7b6b4a7 2026-10-20 13:17:19 +0200  Initial commit: EXL3 Intel Arc quantizer feasibility specification and repository scaffolding
```

---

## Critical Path Next Steps

1. **P2-04 / P2-05 / P2-06 — Regularization, Scales/Signs & Sylvester Hadamard ($H_{128}$):** Implement upstream Hessian regularization, `su`/`sv` scale/sign extraction, and exact $H_{128}$ Hadamard pre/post transforms (`exl3_quant/`).
2. **P2-08 .. P2-12 — Blockwise One-Sided `ldlq` Integration:** Implement the bottom-up `ldlq` error-feedback traversal (`quantize.py:493-609`) combining `block_ldl` (`exl3_quant/ldlq.py`), `XPUTrellisEncoder` (`exl3_quant/xpu_quantizer.py`), and `ScratchArena`.
3. **P2-01 .. P2-03, P2-13 .. P2-18, V140-06 & Gate 2:** Execute calibration/Hessian capture and full single-layer conversion on Arc 140V, validate packed `.safetensors` output through `exl3xpu.ref`, profile peak memory across repeated layer conversions, and publish `reports/gate_2.md`.
