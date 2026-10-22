# EXL3 Intel Arc Quantizer: Implementation Progress

> **Last Updated:** 2026-10-21 10:55:00 +02:00  
> **Target Platform:** Intel Core Ultra 7 258V (Lunar Lake), Arc 140V integrated GPU (`0x64A0`), 32 GB shared memory  
> **Current Status:** Gates 0/1/2 closed; **Phase 3 in progress** — P3-01..P3-08 done. 218 unit tests passing (100%, 1 platform-conditional skip). **CUDA ground-truth deferral adopted** (`skip_cuda.md`, `SKIP-CUDA-2026-10-21-01` / waiver `DEV-2026-10-21-01`): P0-24/P0-25/P0-27 stay open and blocked; all gate evidence is capped at class **E3**; E4 (CUDA path parity) is not claimed. **Declared Active Target: `empero-ai/Qwen3.8-9B-Distill` (BLOCKED)** — qwen3_5 hybrid linear-attention/multimodal architecture outside the pinned contract; successor of the first-declared sibling `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`; conversion blocked on the P4-12/P4-13 architecture-extension track; validator hardened with regression guards for both config shapes.
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

### Step 17: Phase 2 Numerical Preprocessing — Sylvester `H_128` and Upstream Regularization (P2-04, P2-05, P2-06)
* **Time tag:** [2026-10-20 22:20:00 +0200]
* **Commit:** `f14b8e3`
* **Sources:** [`exl3_quant/hadamard.py`](./exl3_quant/hadamard.py), [`exl3_quant/regularize.py`](./exl3_quant/regularize.py), [`tests/test_hadamard.py`](./tests/test_hadamard.py) (11 tests), [`tests/test_regularize.py`](./tests/test_regularize.py) (13 tests)
* **Hadamard (`P2-06`):**
  * Natural Sylvester `H_128` (`[[h, h], [h, -h]]` recursion), orthonormalized by `1/sqrt(128)`; symmetric, `H Hᵀ = I`, self-inverse.
  * Blockwise left/right application matching upstream `blockwise_preapply_had_{l,r}_`; verified equal to an explicit block-diagonal matmul (i.e. not a full Kronecker transform).
  * Bit-exact vs `exl3xpu.ref.hadamard128`, and the decode ordering `H_l → ×su → H_r → ×sv` matches `exl3xpu.ref.weight_orig` on an identical random trellis.
* **Regularization (`P2-04`):**
  * `block_rms` reproduces upstream's 32-slice accumulation order (`quantize.py:1079-1092`).
  * `regularize` transcribes the full sequence (`:1130-1236`): output-scale skew heuristic (top 2% of `sqrt(H_diag)` vs the 0.15 cutoff), out-channel RMS normalization, `sv` folding + right Hadamard, `su` folding (`/ -codebook_scale`) + left Hadamard, `zeros` flag, and zero-channel floor (`0.1`) / `sv = 0`.
  * **Round-trip identity** — `regularize` followed by the decode ordering reconstructs the original weight to `< 1e-4` — verified with the heuristic on, forced off, and a global scale of 2.0.
  * The `g_scale_gss` sample-quantization search was deliberately **not faked** while its encoder dependency was pending: it was a pluggable `g_scale_fn` defaulting to upstream's `skip_g_scale=True`. Step 19 wires the full `sample_scale_tiles` + coarse/fine search through the `P2-08` encoder adapters, closing `P2-04` `[x]`.
* **Ill-conditioned handling (`P2-05`, partial):** zero/degenerate channel paths tested; `block_ldl` damping-retry failure path already covered; measured `cholesky` CPU fallback cost recorded in Gate 0.

---

### Step 18: One-Sided LDLQ Error-Feedback Traversal (P2-08, P2-09, P2-10)

* **Time tag:** [2026-10-20 22:55:00 +0200]
* **Commit:** `ae5db85`
* **Source:** [`exl3_quant/ldlq.py`](./exl3_quant/ldlq.py) (`ldlq`, `LDLQResult`, `make_reference_tile_encoder`, `make_xpu_tile_encoder`); tests in [`tests/test_ldlq_loop.py`](./tests/test_ldlq_loop.py) (13 tests).
* **Traversal (`P2-08`):**
  * Faithful transcription of upstream `quantize.py:493-609`: spans of `buf_size_k` rows bottom-up, 16-row blocks bottom-up within each span.
  * Target rows are corrected by **addition** of the accumulated compensation: `rows = b_weight + prod_cache + L[i+bj:j, i+bi:i+bj].T @ bb_err`.
  * Tiles are reshaped to `(tiles_n, 256)`, permuted with `tensor_core_perm`, encoded, un-permuted for reconstruction, while the encoded states stay in tensor-core order — matching upstream exactly.
  * Trailing-weight feedback: after each span, `prod_cache += L[i:j].T @ (b_weight - b_weight_q)`, which is the term the next (higher) span consumes.
  * The encoder is a pluggable callback; the reference and XPU adapters use `pre_permute=False` because `ldlq` already permutes. The XPU adapter reuses **one** `XPUTrellisEncoder`/`ScratchArena` for every block (no secondary queue).
* **Verification (`P2-09`, `P2-10`):**
  * Identity `L` reduces to independent per-block quantization, bit-exact.
  * A recording encoder checks the exact additive correction and the strictly-lower `L` slice.
  * An **independent dense transcription** (explicit Python tile loops, global indices, no views) agrees tensor-for-tensor for `buf_size_k ∈ {16, 32, 64}`. This oracle caught a real bug during development: the `L` column slice was initially span-local (`bi:bj`) instead of global (`i+bi:i+bj`), which shifted every correction by one span.
  * On a correlated Hessian, feedback lowers `tr(E H Eᵀ)` versus the same encoder with feedback disabled.
  * Results are invariant to span height (buffering is a summation-order choice), and upstream's `L[dr,dr] = 0` is unobservable, both asserted.
  * XPU (`XPUTrellisEncoder`) and CPU reference encoders produce identical `weight_q` and `encoded` through the full loop.
* **Still open:** `P2-09` upstream CUDA fixtures (blocked by P0-24/P0-25), `P2-11` module-lifetime workspace retention, `P2-12` residual/error/layerwise measurements, and the `g_scale_gss` wiring (P2-04 remains `[~]`).

---

### Step 19: Global-Scale Search Wiring — P2-04 Closed

* **Time tag:** [2026-10-20 23:25:00 +0200]
* **Source:** new module [`exl3_quant/gss.py`](./exl3_quant/gss.py) (`sample_scale_tiles`, `g_scale_search_batch`, `select_refined_scale`, `g_scale_gss`, `g_scale_fn_from_encoder`); tests in [`tests/test_gss.py`](./tests/test_gss.py) (19 tests).
* **Sampling (`quantize.py:895-916`):**
  * Wrapped diagonal of the tile grid (width 3) so every tile row *and* column is sampled, plus the highest/lowest-RMS tiles (`topk`, `>= 8` and at most half the grid) as outlier insurance; whole-tile extraction matters because outliers are whole channels.
  * Verified value-exactly against a manual Python extraction per width ∈ {1, 2, 3} with `tensor_core_perm` applied.
* **Search (`quantize.py:918-1047`):**
  * Coarse grid `0.1 + 0.2·i` (0.1–1.9) on the strided subsample (`tiles[::3]`), fine grid of 5 points at `0.075` around the coarse argmin on the full sample, then parabolic refinement: `offset = 0.5·(y0−y2)/(y0−2y1+y2)` when `denom > 0`, clamped to ±0.5 steps, result floored at `0.01`.
  * For a convex fine triple `y0=y1+a, y2=y1+b` the offset is analytically `0.5(a−b)/(a+b) ∈ [−0.5, 0.5]`; the upstream clamp is float-defence only (asserted over randomized basins, plus exact-vertex cases).
  * Grid evaluations batch into as few `encode_tiles` calls as possible with the upstream 65 536-tile chunk bound; verified chunked vs one-shot results are identical.
* **Wiring:** `regularize(..., skip_g_scale=False, g_scale_fn=g_scale_fn_from_encoder(...))` folds `weight *= g; su /= g` verbatim (`.py:1224-1225`); `sv`/scales/skew untouched. Decode round-trip with the `g`-scaled `su` reconstructs the original weight `< 1e-3` (the `g` cancels inside `su` at decode).
* **Measured behavior (Arc XPU, K=2):** tile RMS 0.03 → `g = 2.05` (fine-grid right edge); RMS 0.5 → `g ≈ 1.97`; RMS 8 → `g ≈ 0.136`; RMS 1000 → `g ≈ 0.025` (search correctly walks below the coarse grid on fp16 precision loss). MSE at the found scale is strictly below the `g_scale = 1` baseline.
* **Result:** `P2-04` is `[x]`; the `g_scale_fn` hook remains as the extension point for a fast/batched search, but the default path now performs the real upstream search.

---

## Test Suite Execution Evidence

**218 unit tests** ran and passed cleanly as of [2026-10-21 11:44:00 +0200] (adds 18 P3-04..P3-08 checkpointing/recovery tests and 1 declared-target regression guard over the 199-test review baseline; converter suite grew 27→28; 1 platform-conditional memory-pipeline test skips off-Windows) across all test suites:
- `test_scratch.py`: 4 tests (scratch arithmetic and allocation limits)
- `test_negative.py`: 5 tests (perturbation detection and input validation)
- `test_trellis_format.py`: 17 tests (codebook LUTs, permutations, bitpacking, state decoding)
- `test_encoder.py`: 8 tests (contract-conformant reference encoder, path cost tracking, bounded distortion)
- `test_oracle_parity.py`: 4 tests (live bit-exact parity against `exl3xpu.ref`)
- `test_bitpack.py`: 4 tests (sub-byte packing shim)
- `test_xpu_extension.py`: 7 tests (device selection, correctness, ordering, lifetime, caps, O-6 random + adversarial; skip unless `torch.xpu.is_available()`)
- `test_memory_policy.py`: 18 tests (meminfo parsing, reserve arithmetic, admission boundaries, outstanding commitments, abort path, post-commit breach, allocator-growth assertion, platform-native counters across OS variants with 1 Windows API conditional skip, live 32 MiB GPU allocation)
- `test_ldlq.py`: 10 tests (`L Lᵀ = H`, block-inverse fold, identity block diagonal, retry-damping schedule, input validation, XPU vs CPU parity)
- `test_xpu_quantizer.py`: 13 tests (XPU codebook parity, `K=1..4` & 3-codebook parity, `L=160` support, live `exl3xpu.ref` parity, edge cases & tie verification, partial chunks & waves, input rejection, async stream ordering, all 4 traceback modes & SG16/32 & SLM/global, bounded `ScratchArena` eviction/reuse, smaller-batch retry, and overflow-safe distortion accounting)
- `test_hadamard.py`: 11 tests (Sylvester values/symmetry/orthogonality, block-diagonal equivalence, involution, oracle `H_128` and `weight_orig` parity)
- `test_regularize.py`: 13 tests (`block_rms`, regularize→decode round-trip, skew heuristic, zero channels, `zeros` flag, `g_scale` folding, validation)
- `test_ldlq_loop.py`: 13 tests (identity-`L` equivalence, exact target correction, independent dense oracle across span sizes, feedback objective improvement, span invariance, diagonal-zeroing equivalence, encoded layout/dtype, non-mutation, validation, XPU↔reference parity)
- `test_gss.py`: 19 tests (wrapped-diagonal coverage and exact tile values, extreme-RMS sampling, stage-by-stage search recomputation, batching/chunking invariance, MSE-below-baseline with the real encoder, determinism, huge-weight walk-down, `select_refined_scale` vertex/floor/bound, `regularize` folding and decode round-trip)
- `test_calibration.py`: 7 tests (RMSNorm forward, rotary embeddings complex unit norm, multi-head attention forward, SwiGLU forward, complete layer forward on XPU, exact second-moment Hessian accumulation H = sum X^T X matching explicit formula, dead channel floor and damping safeguard)
- `test_layer_quant.py`: 4 tests (proxy loss trace formula equivalence, factorization residual check, single module XPU quantization, and complete decoder layer quantization on XPU with ScratchArena reuse)
- `test_serialization.py`: 6 tests (trellis packing round trip with state-history low K-bit preservation, safetensors schema validation, export/reload verification with exl3xpu.ref.weight_orig, malformed/truncated safetensors rejection, and tile encoder bit-width mismatch guards)
- `test_upstream_encode_oracle.py`: 4 tests (CPU-runnable tier-4 oracle derived from upstream's CUDA test recipe: `max_mse_per_K` quality ceilings for K=2..8, tail-biting ring invariant, upstream "ideal encoding" state layout, encode→dequant round trips)
- `test_forward_semantics.py`: 5 tests (independent audit of the reference encoder's forward ACS pass: branch metric and fp16 accumulation semantics, predecessor table and terminal costs tie-breaking, masking rules, adversarial tie construction, deterministic path reproduction)
- `test_converter.py`: 28 tests (bounded conversion config rejection of MoE/vision/MTP/pre-quantized/fractional-K/out-of-range-K/bad-geometry/partial-rotary/rope-scaling/sliding-window/bad output dirs; declared-target hardening: hybrid `layer_types`, `mtp_num_hidden_layers`, top-level multimodal token ids incl. `vision_end_token_id`, and full structural replicas of BOTH declared target configs (`empero-ai/Qwen3.8-9B-Distill`, `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`) asserted deterministically rejected with an architecture diagnostic (`test_declared_target_mimo_is_rejected`); mmap shard index scoping, bit-exact streaming reads, single-file indexing, poisoned-shard isolation, RSS measurement; full 2-layer synthetic conversion on XPU with recorder: ascending layer order, upstream module order, layer-0 capture equality with independent calibration, layer-1 capture derived from layer 0's *quantized* output (differential vs unquantized control), quantized-reload replay equivalence, transposing `w_hat` reload, bit-exact unchanged pass-through, per-layer `qtensors` artifacts with 30 tensors, merged manifest consistency, per-layer RSS recording)
- `test_converter_recovery.py`: 18 tests (P3-04..P3-08 checkpointing/resumability/fault recovery on 3-layer synthetic checkpoints: working-directory layout + manifest progress tracking; atomic .tmp + POSIX-replace publication with 30-tensor-per-layer integrity verification; interruption-safe recovery across calibration/quantization/reload/publication boundaries with clean resume and exact state continuity; rigorous `_validate_resume` invariance with documented `force_resume` override; resilient failure handling — `check_disk_space`, missing shards, corrupted checkpoints)

```text
PYTHONPATH=. /home/yanchun/Projects/vllm.xpu/.venv/bin/python -m unittest discover -s tests -p "test_*.py"
Ran 218 tests in 133.074s

OK (skipped=1)
```

---

## Detailed Breakdown of Completed Steps (Continued)

### Step 20 [2026-10-21 09:45:00 +0200]: Complete Single-Layer Conversion & Gate 2 Closure (Phase 2)

* **Commit:** `c67a469`
* **Gate report:** [`reports/gate_2.md`](./reports/gate_2.md) — GO to Phase 3.

1. **Layer Calibration & Grouped Hessian Harvesting (`exl3_quant/calibration.py`, P2-01..P2-03, P2-05):**
   - Implemented `TransformerDecoderLayer` representing dense transformer architectures (`Meta-Llama-3.1-8B` / `Qwen-2.5-7B`). Verified 100% native operator execution on XPU (`mm`, `bmm`, `silu`, `softmax`, `rsqrt`) with zero silent CPU fallback during forward passes.
   - Built `HessianCollector` grouping input activations into 4 architectural locations (attention in, attention out, mlp in, mlp down), reducing Hessian footprint by 43% (976 MiB total in fp32).
   - Applied token-normalized second-moment accumulation ($H = \sum X^T X$), damping $\lambda \cdot \text{mean}(\text{diag}(H))$ ($\lambda = 0.01$), and floor clamping for inactive/dead channels (`P2-05`).

2. **Full-Layer Quantization & Distortion Metrics (`exl3_quant/layer_quant.py`, P2-11, P2-12):**
   - Built end-to-end `quantize_decoder_layer` converting all 7 linear projections (`q, k, v, o, gate, up, down`) sequentially.
   - Reused a single `ScratchArena` and single `XPUTrellisEncoder` across the entire layer, with `torch.xpu.synchronize()` between projections (`P2-11`).
   - Evaluated layerwise proxy loss $\text{tr}\left((W - \hat{W}) H (W - \hat{W})^T\right)$ via $(H @ E \cdot E).\text{sum}()$, relative Frobenius error, MSE, and factorization residuals (`P2-12`).

3. **Safetensors Packing, Serialization & Downstream Reload Parity (`exl3_quant/serialization.py`, P2-13..P2-16):**
   - Built `pack_module_trellis` packing trellis states into `int16 [k/16, n/16, 16*K]` using audited `pack_trellis` + `swap16_storage`.
   - Exported compliant `.safetensors` checkpoints with `quantization_config.json` and weight index map (`P2-13`, `P2-14`).
   - Implemented `load_and_verify_layer_safetensors` decoding via `exl3xpu.ref.weight_orig`, verifying reconstructed weights match expected matrices within fp16 tolerance (`P2-16`).
   - Validated state-history round trips and added malformed/truncated file rejection tests (`P2-15`).

4. **Sustained Memory Profiling & Stability (`scripts/benchmark_phase2_layer.py`, P2-17, P2-18, G2-02):**
   - Ran 3 sustained iterations of complete layer quantization on Arc 140V ($9,216$ tiles at $K=2$).
   - Measured peak allocated memory ($649.7\text{ MiB}$), peak reserved memory ($672.0\text{ MiB}$), and host RSS ($3.04\text{ GiB}$) under the 6.0 GiB reserve.
   - Verified **0 B swap growth** (`total_swap_growth = -8192 B`), perfectly stable memory across iterations, and saved `artifacts/benchmarks/phase2_layer_bench.json`.

5. **Gate 2 Closure (`reports/gate_2.md`, G2-01..G2-03):**
   - Published [`reports/gate_2.md`](./reports/gate_2.md) declaring **GO to Phase 3 (Bounded full-model conversion and qualification)**.

---

### Step 21 [2026-10-21 10:20:00 +02:00]: Bounded Conversion Orchestrator, Upstream Propagation & Mmap Shard Streaming (Phase 3, P3-01..P3-03)

1. **Bounded conversion entry point (`exl3_quant/converter.py`, P3-01):**
   - `ConversionConfig` exposes only the pinned-contract surface: fixed integer K in `[2, 4]` (fractional recipes explicitly deferred to P4-02), `mul1` codebook only (others P4-01), deterministic synthetic calibration corpus parameters (real corpus wiring deferred to P3-09), damping/zero-floor, GSS flag defaulting to the upstream real search (`skip_g_scale=False`), and device selection.
   - `ConversionConfig.validate()` + `load_model_geometry` reject, in pure Python before any weight bytes are read: MoE (`num_local_experts` etc.), vision/multimodal configs, MTP heads, already-quantized checkpoints, non-dense architectures, non-128-divisible hidden/intermediate/q/kv dims, `head_dim != hidden // num_heads`, partial rotary, rope scaling, sliding-window attention, out-of-bounds vocab, and invalid/occupied output directories.

2. **Upstream calibration propagation & conversion order (`convert_model`, P3-02):**
   - Executes the audited upstream per-module loop (`convert_model.py:1095-1316`): for each decoder layer in ascending order — grouped-Hessian capture on the running calibration state produced by all previously **quantized** layers (`calibrate_layer`, extended with a `zero_floor` pass-through), quantization against those Hessians via the P2 shared-arena chain (`quantize_decoder_layer`), reload of the module with its quantized numerics (`w_hat` transposed into the `nn.Linear` (out,in) layout), and re-forward of the *same pre-layer* hidden states through the quantized layer to advance the state.
   - Unchanged components (embed_tokens, lm_head, RMSNorm weights) are never quantized: they are recorded in the merged `quantization_config.json` as fp16 pass-through and exported bit-exact to `qtensors/untouched.safetensors`; per-layer `qtensors/model.layers.N.safetensors` artifacts follow the upstream working-directory layout (7 modules × 4 tensors + 2 norm tensors).
   - Verified on a deterministic synthetic mini-Llama checkpoint (`tests/fixtures/`): layer-1 captured Hessians differ measurably from the unquantized-layer-0 control (proving quantized-numerics propagation) and are reproduced by replaying the reloaded layer.

3. **Memory-mapped shard streaming with measured host RSS (`ShardedSafetensorsIndex`, P3-03):**
   - Streams HF-style sharded and single-file checkpoints via memory-mapped `safetensors.safe_open` handles; only the tensors of the layer under conversion are materialized, grouped per shard.
   - Host RSS (`/proc/self/statm` via `read_rss_kib`) is measured around every phase — baseline, metadata-only pass, streaming pass, full-shard contrast — and recorded per layer in the manifest.
   - Measured on a deterministic 424 MiB / 4-shard / 16-layer synthetic checkpoint via `scripts/benchmark_phase3_stream.py` → [`artifacts/benchmarks/phase3_stream_bench.json`](./artifacts/benchmarks/phase3_stream_bench.json) (**verdict PASS**: metadata pass materializes 0 B (RSS delta 0.0 MiB); streaming residual RSS delta **2.8 MiB** over a 424 MiB checkpoint; honest disclosure: mmap streaming bounds heap RSS, not the page cache).

4. **Test evidence:** 185/185 tests pass (`Ran 185 tests in 111.343s OK`), baseline 158 + 27 new converter tests + `calibrate_layer(zero_floor=...)` regression-transparent extension.

---

### Step 22 [2026-10-21 10:55:00 +02:00]: CUDA Ground-Truth Deferral Adopted & Evidence Classes Enforced (skip_cuda.md / SKIP-CUDA-2026-10-21-01)

* **Merge:** `442520b` — integrated origin/main `db1c6a2..46f0af9`: [`skip_cuda.md`](./skip_cuda.md) (the decision record), waiver `DEV-2026-10-21-01` in [`docs/deviation_log.md`](./docs/deviation_log.md) §6, the platform-native memory counters fix, and two CPU-runnable ground-truth-free test suites.

1. **Deferral decision recorded (`skip_cuda.md`, SKIP-CUDA-2026-10-21-01):**
   - The NVIDIA eGPU (RTX 3060 12 GB + Thunderbolt-to-PCIe dock + PSU, verified adequate for P0-24/P0-25 capture at every K in scope) is **deliberately not installed**. The CUDA ground-truth requirement is **deferred, not waived**: P0-24 / P0-25 / P0-27 remain open and blocked.
   - Rationale: encoder format, container, codebook, and search-quality properties are already verifiable against upstream-derived oracles on this host; the only CUDA-only gap is exact *path* parity, which does not block Phase 3.

2. **Evidence classes (skip_cuda.md §4) enforced across the record** — gate documents must label the highest class demonstrated; **any tier-4 statement is qualified as E3 (quality-bar parity), never E4 (path parity)**:
   - `tasks.md`: governance header registers the decision record; P0-24 (upstream) annotated DEFERRED, P0-25/P0-27 annotated; **G1-01 re-marked `[~]`** (partial) per §5; G0-01/P2-09 notes carry the decision ID; G2-01, V140-06, and G3-01 notes carry explicit E3-not-E4 labeling; P0-29 references the decision ID.
   - `reports/gate_0.md`, `reports/gate_1.md`, `reports/gate_2.md`: each gains an *Evidence Class* header block (E1 source-transcribed, E2 independent-oracle, E3 upstream-threshold; **E4 unavailable while the deferral is active**); the `gate_2.md` Tier-4 and external-blocker rows are qualified.

3. **Merged upstream test suites strengthen the E1–E3 evidence (no CUDA hardware involved):**
   - `tests/test_upstream_encode_oracle.py` (4 tests): upstream's own CUDA-derived `max_mse_per_K` ceilings (0.1 @ K=2–5, 0.07 @ K=6, 0.05 @ K=7, 0.04 @ K=8) reproduced on CPU, the tail-biting ring invariant, upstream's "ideal encoding" state layout, and encode→dequant round trips — E3 quality-bar evidence.
   - `tests/test_forward_semantics.py` (5 tests): independent audit of the reference encoder's forward ACS pass (branch metric/fp16 accumulation, predecessor table and terminal costs, tie-breaking/masking) — supports the E1 transcription claim.
   - `tests/test_memory_policy.py` platform-native counter additions (3 new tests; 1 Windows-only platform test conditionally skipped on this Linux host) from the `db1c6a2` merge.
   - What is deliberately **not** claimed: E4 path parity, Open Item O-1 resolution, and upstream's CUDA-build test suite results.

4. **What the deferral does NOT change:** acceptance criteria, algorithm semantics, XPU encoder parity obligations (P1-05, P2-09 stay satisfied by internal-transcription + oracle evidence), and Phase 3 planning — no Phase 3 task depends on E4. **Decision triggers T1–T6** (`skip_cuda.md` §6) are attached to the critical path below and would fire the §7 eGPU resume checklist.

5. **Test evidence:** 199 tests pass post-merge (`Ran 199 tests in 192.352s OK (skipped=1)`) — 185 local + 9 upstream-derived + 3 platform-dependent memory tests (1 conditional skip) + 2 bit-width guard tests from commit `8c71bb7`.

---

### Step 23 [2026-10-21 11:44:00 +02:00]: Declared Active Target Swapped to `empero-ai/Qwen3.8-9B-Distill` (Blocked); Validator Hardened; P3-04..P3-08 Recorded

**Target-model decisions (owner, 2026-10-21):** the Phase-3 first-conversion target was declared as `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` ("Keep MiMo, flag as blocked"), then swapped by owner instruction to its sibling distill [`empero-ai/Qwen3.8-9B-Distill`](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill). Both `config.json` files were fetched and verified (2026-10-21); they are the same `Qwen3_5ForConditionalGeneration` / `qwen3_5` architecture with identical geometry (hidden 4096, 32 layers, 16/4 heads, head_dim 256, intermediate 12288, vocab 248320 < 2^24) and the same contract-incompatible shape:

* 24 of 32 `text_config` layers are `linear_attention` (3:1 hybrid SSM pattern, `full_attention_interval` 4, `mamba_ssm_dtype` float32) — outside both the pinned contract and upstream EXL3's per-linear quantization space.
* 27-layer ViT `vision_config` with top-level `image_token_id` / `video_token_id` / `vision_start_token_id` / (`vision_end_token_id` in Qwen3.8) — multimodal.
* `partial_rotary_factor` 0.25 with interleaved mrope (`mrope_section` [11, 11, 10]) — contract requires full rotary.
* `mtp_num_hidden_layers` 1 — MTP head.

**Status: BLOCKED.** Disk budget is not the blocker (dense BF16 9B ≈ 18 GB vs 90 GB free at decision time); the architecture is. Conversion cannot start until the new **`P4-12`** (hybrid linear-attention support decision) and **`P4-13`** (multimodal text-tower-only semantics) extension items land (`tasks.md` §4.1); MTP/vision *quantization* remains parked at P4-04. Contract-compatible fallbacks (pinned Llama-3.1-8B / Qwen2.5-7B manifest rows) stand ready for an owner-approved substitute.

1. **Validator hardening** (`1e53c73`): `load_model_geometry` rejected-mode detection extended with `mtp_num_hidden_layers`, `image_token_id`, `video_token_id`, `vision_start_token_id`, `vision_end_token_id`, and explicit `layer_types` rejection for any non-`full_attention` value (hybrid attention); the earlier MoE/pre-quantized/vision checks are unchanged.
2. **Regression guards** (`tests/test_converter.py`, 27→28 tests): new variant rejections, plus faithful structural replicas of **both** declared target configs asserted deterministically rejected via `UnsupportedModeError` with an architecture diagnostic (`test_declared_target_mimo_is_rejected`). The engine now *proves* the blocked status instead of asserting it on paper.
3. **Docs**: `tasks.md` P3-09 annotation (declared target + verified facts + unblock path); `docs/compatibility_manifest.md` dated amendment (declared target recorded, pinned Llama/Qwen2.5 rows retained as fallbacks); README status row + roadmap item 3 updated.
4. **P3-04..P3-08 recorded complete** (`77ae4d4`): the parallel-session checkpointing/resumability/fault-recovery implementation was verified here before committing — `tests/test_converter_recovery.py` 18/18 OK standalone, full tree suite below. tasks.md P3-04..P3-08 entries were already annotated with implementation evidence.
5. **Test evidence:** final tree suite `Ran 218 tests in 133.074s OK (skipped=1)` — 199 review baseline + 18 recovery tests + 1 declared-target guard. A transient 1-failure run during a noisy first full-suite attempt (156.152s, FAILED) did not reproduce in either of two subsequent clean runs (144.650s and 133.074s); root cause was running an XPU suite concurrently with the background full suite from this session — avoid overlapping XPU test processes on this device.

---

### Step 24 [2026-10-21 13:11:00 +02:00]: P4-12/P4-13 Hybrid-Target Extension Landed — Declared Target Unblocked (`qwen3_5_text_tower` opt-in mode)

**Decision (P4-12): upstream EXL3 per-linear quantization covers the declared target's hybrid text tower exactly as far as the encoder contract allows.** Every bias-free linear projection of both layer kinds is quantized — `linear_attention`: `linear_attn.in_proj_qkv` (n=8192), `in_proj_z` (4096), `out_proj` (4096), `mlp.gate/up` (12288), `mlp.down` (k=12288); `full_attention`: `self_attn.q` (8192), `k/v` (1024), `o` (k=4096), `mlp.*` — all k and n divisible by 128 (`ldlq`/`regularize` hard-reject otherwise; `blockwise_preapply_had_r_` requires `n % 128`). The two exceptions are `linear_attn.in_proj_a`/`in_proj_b` (n = num_value_heads = 32, i.e. 0.09 % of the text tower's linear weight bytes), which fail the 128-wide Hadamard block and are **bit-exact pass-through**, together with `conv1d.weight`, `A_log`, `dt_bias`, `linear_attn.norm` (SSM state) and `self_attn.q_norm`/`k_norm` in full-attention layers. No new encoder or arithmetic path was needed; the XPU chain is untouched.

**Implementation.** Opt-in `ConversionConfig.architecture_mode="qwen3_5_text_tower"` (default `"dense"` keeps every historical acceptance and rejection byte-identical). `exl3_quant/hybrid.py` holds the pinned reference contract: layer-kind tables (`QUANTIZED_MODULES`, `PASS_THROUGH_MODULE_NAMES`), import-guarded `transformers` support check (`require_support()` runs in `validate()` before any weight byte is read), `make_text_config` (forces eager attention), `build_layer` (strict state-dict load; optional `decoded` weights for artifact replay), `make_calibration_runner` (cached partial-mrope cos/sin + causal additive mask on full-attention blocks, 2-D padding slot on linear blocks), and `environment()` provenance. `converter.py` gained prefix-aware geometry (`model.language_model.layers.`, `model.language_model.` text prefix, per-layer `layer_type`), the hybrid validator (MoE / unknown layer types / missing text keys / non-128 widths rejected with `UnsupportedModeError`; linear-head fields default to `Qwen3_5TextConfig` dataclass defaults; absent `layer_types` derived from `full_attention_interval` and legacy names remapped exactly like `transformers.configuration_utils`), row-gathered embedding reads (the 1.9 GiB bf16 table is never kept resident), per-layer bit-exact pass-through emission, and manifest/resume extensions (`architecture_mode`, `layer_types`, `key_prefix`, per-layer `layer_type`/`pass_through_keys`; resume rejects cross-mode manifests). `calibration.py` gained generic shared-input Hessian grouping (`in_proj_qkv`/`z`/`b`/`a` consume one identical input tensor — proven by forward pre-hook identity); `layer_quant.py` takes an explicit `module_names` table; `serialization.py` gained `extra_tensors` (collision-checked, dtype-preserving) for the pass-through contract.

**Pass-through semantics (P4-13).** Text-tower-only conversion: non-text-tower tensors are partitioned into groups written to separate files so no single serialization holds all pass-through bytes — `qtensors/untouched.safetensors` (text embed/final-norm/lm_head), `qtensors/untouched.vision.safetensors` (`model.visual.*`), `qtensors/untouched.mtp.safetensors` (`mtp.*`). Source dtype (bf16) is preserved bit-exact for every pass-through tensor including the layer norms. The manifest records each group (`file`, `keys`, `quant_format: source-dtype`) and a `hybrid` provenance block (multimodal token ids, rope/partial-rotary parameters, zero-centered-RMSNorm semantics, linear-attention tables, reference-forward environment) plus a new top-level `environment` block (python/platform/torch). Dense-mode group content is byte-identical to the historical `unchanged_keys()`.

**Test evidence.** New `tests/test_hybrid_converter.py` (31 tests): validator acceptance (fixture + both declared-target config replicas: 32 layers, 24/8 hybrid pattern) and rejection (dense-mode default, unknown mode, MoE, unknown/short layer types, non-128 widths, missing text_config, derived-interval and legacy-remap derivations); module tables exhaustive over reference `named_parameters()` with encoder-divisibility assertions; Hessian group membership/input-identity; XPU end-to-end conversion of the 3-layer bfloat16 hybrid fixture (32 tensors per layer, 6/7 quantized modules × 4 tensors, per-layer and group pass-through bit-exact, decoded projections inside the `0 < rel < 1.0` quantization band, deterministic rebuilt-layer forwards, fault-injected publish failure resumed to bit-identical artifacts, tampered-manifest mode mismatch refused); dense-profile default guards. Full tree suite: **`Ran 249 tests in 182.810s OK (skipped=1)`** — the 218 baseline plus these 31, no regressions.

**P3-09 next.** The declared target `empero-ai/Qwen3.8-9B-Distill` is unblocked: checkpoint verified as a single 19,306,305,296-byte bf16 `model.safetensors` (775 tensors; header parsed and end-offset cross-checked against the file size); download to `/home/yanchun/models/qwen38/` is in progress. The run sequence per the task contract: measured single-layer conversion first (layer/calibration/runtime/memory requirements + documented operating reserve), then the full 32-layer attempt with checkpoint/resume. `exl3xpu` text-only-decode compatibility (acceptance tier 6) remains open and is claimed by neither P4-12 nor P4-13.

---

### Step 25 [2026-10-21 15:15:00 +0200]: P3-09 Measured Single-Layer Requirements & Operating Reserve; Real wikitext Corpus Wired (`3ea669b`)

**Checkpoint verified.** `model.safetensors` = 19,306,305,296 bytes, byte-for-byte the safetensors header end offset (775 tensors, `complete=True`); real config validated in opt-in mode (32 layers, 24 `linear_attention` / 8 `full_attention`, hidden 4096, inter 12288, vocab 248320, untied lm_head, rope partial 0.25 → rotary_dim 64, mrope [11,11,10]).

**Measured single-layer requirement** (`/tmp/opencode/measure_qwen38_layer.py 4 2048 0,3`, real GSS `skip_g_scale=False`, reference-PyTorch fallbacks for `causal_conv1d`/`fla`):

| phase | layer 0 (`linear_attention`) | layer 3 (`full_attention`) |
|---|---|---|
| read / build / embed gather | 0.03 / 1.43 / 0.84 s | 0.02 / 0.73 / 0.61 s |
| calibration (4×2048 tok, 8192 tok) | 3.73 s | 1.46 s |
| **quantization (real GSS)** | **1588.0 s** | **1536.4 s** |
| artifact build + atomic save | 36.3 + 0.04 s | 27.0 + 0.01 s |
| XPU Hessian working set | 1.47 GiB, 8 groups | 1.40 GiB, 7 groups |
| peak host RSS (process max) | 6.3 GiB | 6.5 GiB |

**Extrapolation at the pinned recipe (128 rows × 2048 tokens):** calibration scales linearly with rows (~54 min total); quantization is row-count-independent (~832 min) → **≈ 894 min (~15 h) for 32 layers**.

**Why 26 min/layer is inherent (encoder micro-profile):** raw XPU tile encoder = 65,536 tiles in 107.06 s (**1.6 ms/tile**); a full LDLQ pass on 4096×4096 (skip GSS) = 111.75 s ≈ one encoder pass; `g_scale_gss` on the same weight = 12.38 s because it samples only the wrapped diagonal + extreme tiles (~1k tiles/module, independent of matrix size). **GSS is not the bottleneck** — cost tracks the tower's ~27M total tiles. Making the encoder faster would be P2 encoder work, explicitly out of scope for P3-09; the measured cost is accepted, not hidden.

**Operating reserve (P3-09 contract, all four dimensions leave a documented reserve):**
1. **RAM:** 30.9 GiB total, 26 GiB available at start; measured peak process RSS **6.5 GiB** (≈ 4.7× reserve).
2. **XPU:** 28.6 GiB shared device memory; Hessian working set **1.5 GiB/layer**, released between layers; pass-through groups stream one at a time.
3. **Disk:** **71.8 GiB free vs 6.72 GiB** header-derived output estimate (6.912 B quantized params × 2 bits ≈ 1.61 GiB + 575 pass-through tensors ≈ 5.11 GiB) ≈ 10× reserve.
4. **Wall clock:** ~15 h, covered by `checkpoint_interval=1` per-layer atomic publish + validated resume (bit-identical restart, per P3-04..P3-07 evidence).

Full-model attempt approved by owner decision (2026-10-21): launch now; wire the real corpus first.

**Real corpus wired (P3-09, commit `3ea669b`, tests 256/256 in 188.156s).** Owner decision (2026-10-21): wire the real corpus before launching, so the pinned recipe is honored in one 15-h run instead of two. `ConversionConfig.calibration_token_ids` consumes an `exl3_quant.calibration_token_ids.v1` JSON file (flat `batch_size*seq_len` token-id rows + provenance metadata); `validate()` enforces format stamp, row count/shapes, int token bounds `[0, vocab_size)`, and `seq_len` match before any weight byte. The manifest records `calibration.corpus` + `calibration.token_ids {path, sha256, metadata}`; `calibration_state.pt` records `corpus_sha256`; resume raises `IncompatibleResumeError` on corpus presence/content/state mismatches. `scripts/prepare_calibration_tokens.py` reproduces the pinned recipe (wikitext-2-raw-v1 test split → deterministic non-overlapping chunks, left-truncated, seed 42). Realized corpus: HF `Salesforce/wikitext` `wikitext-2-raw-v1` test parquet (4,358 rows, 1,289,979 chars; the original Meta S3 endpoint was unreachable — `PermanentRedirect`/TLS failure), tokenized with the target model's tokenizer → 297,054 tokens; left-truncated 94 leading tokens → 145 whole 2048-chunks (17 surplus); seed-42 start offset 3; 128 rows × 2048 tokens written to `/home/yanchun/models/qwen38/calibration_tokens_wikitext2_test.json` (1,481,255 bytes, **sha256 `7d9d778e8f6420bbe0f67ff065bed1d923194b5597f413bfe4a54516e94b0edf`**; source parquet and tokenizer sha256 recorded inside its metadata). Pre-flight `validate()` on the real model + real corpus: 0.04 s, 128×(1,2048) rows.

**Next:** launch the full 32-layer conversion (`K=2`, `architecture_mode="qwen3_5_text_tower"`, `skip_g_scale=False`, `num_calibration_rows=128`, `seq_len=2048`, real corpus file, `checkpoint_interval=1`) capturing logs + checksums + environment metadata as the P3-09 evidence bundle.

---

### Step 26 [2026-10-21 15:35:00 +0200]: P3-09 Run Aborted by the Memory Policy — Diagnosed, Fixed, Relaunched (`cff5e35`)

**Attempt 1 (15:14:11) aborted 65 s into layer 0.** Evidence bundle: `/home/yanchun/p3_09_evidence/run_attempt1_aborted.log`; the input checksums it captured are valid and re-used by attempt 2 (`model.safetensors` sha256 `90d46f174c3cfc0d29a782f49402d7474f46c035df29afd0c815240427f6f062`, 19,306,305,296 bytes; `config.json` `6402b826…`; corpus `7d9d778e…`; `tokenizer.json` `5f9e4d49…`).

```
MemoryPressureError: memory-pressure abort [K=2 B=1 single-buffer ceiling]:
need 8.0 MiB but only 0 B usable (binding available 9.60 GiB, reserve 6.00 GiB;
outstanding 3.67 GiB already committed)
```

**Diagnosis (measured, not guessed).** The P0-20 admission policy binds on `min(device_free, MemAvailable)` minus the 6 GiB reserve minus the live XPU allocation total. On this device the counters behave asymmetrically (measured with a dedicated probe):

| action | `MemAvailable` | `device_free` | `torch.xpu.memory_allocated` |
|---|---|---|---|
| idle | 22.17 GiB | 12.78 GiB | 0 |
| +2 GiB XPU tensor | 21.57 GiB (−0.60) | 12.47 GiB (−0.31) | 2.00 GiB |
| +4 GiB host tensor | 18.25 GiB (−3.32) | 10.10 GiB (−2.37) | 2.00 GiB |
| free + `malloc_trim(0)` | 22.19 GiB | 14.04 GiB | 0 |

So XPU allocations barely move the counters (the driver serves them from a pre-reserved pool) while **host** pressure moves `device_free` by ~60 % of the bytes taken. The conversion itself supplied that host pressure: the pass-through groups (text 4.07 GiB + vision 0.91 GiB + MTP 0.49 GiB) are materialised in host RAM before they are written, and every layer read of the 19.3 GB source leaves its pages in the page cache. The run therefore ate the headroom the policy protects, and the abort fired inside layer 0's first GSS search.

**Fix (commit `cff5e35`, advisory only — no conversion semantics change).** New `exl3_quant/memory_hygiene.py`:
* `drop_file_cache(path)` — `POSIX_FADV_DONTNEED` on a file (clean pages only; dirty pages are written back first, so file content is never affected; advisory ⇒ never raises);
* `trim_host_heap()` — `malloc_trim(0)` returns freed host arenas to the OS.

The converter now releases the source shards after each pass-through group and after each layer read, and releases the published layer artifact and `calibration_state.pt` page caches. `ShardedSafetensorsIndex.release_source_cache()` does the shard part for single-file and sharded checkpoints alike.

**Verification of the fix.** The relaunched attempt (15:31:00, PID 59806) passed the previous abort point (65 s into layer 0) with the page cache down from ~10 GiB to **2 GiB** and 17 GiB still available (`buff/cache` 2 GiB, `used` 13 GiB). Memory-hygiene behaviour is regression-guarded by 5 unit tests (`tests/test_memory_hygiene.py`: content preservation, missing file/directory tolerance, best-effort trim) plus 1 XPU integration test asserting that a conversion releases the source shards, the published layer artifacts and the calibration state. Full suite: **`Ran 262 tests in 237.723s OK (skipped=1)`**.

**Also added:** `scripts/verify_conversion_output.py` — the P3-11 check (manifest contract, exact per-layer tensor key sets, dtypes, trellis tile geometry against the source shapes, bit-exact pass-through comparison against the source checkpoint, and reconstruction of every quantized module with its relative Frobenius error), to be executed on the finished bundle.

**Next:** monitor attempt 2 to completion (≈ 15 h), then run the P3-11 verification, then P3-12 quality evaluation and the P3-13 `exl3xpu` load check.

**Attempt 2 (15:31:00) — same abort class, one layer later.** The hygiene fix worked (page cache 10 GiB → 2 GiB, 17 GiB available) and layer 0 completed in 1711.7 s (proxy_loss 6.3492e+01, 32-tensor artifact published), but layer 1's GSS aborted: `binding available 225.0 MiB, reserve 6.00 GiB, outstanding 8.27 GiB`. Measured cause (dedicated probe, idle host): **the shared device pool tracks host pressure, while XPU allocations do not move `device_free` at all** — a 4 GiB XPU allocation left `device_free` unchanged (14.98 GiB), and dropping 10 GiB of page cache *raised* it by 2.43 GiB (12.60 → 15.03 GiB). The pass-through group files (5.5 GiB, written once and never read again) and the 2.1 GiB/layer calibration-state writes were still resident, squeezing the pool to 225 MiB.

**Fixes that followed (commits `2a1f5a7`, and the reserve knob below).**
1. *Converter:* release the written pass-through group files as well (`cff5e35`/`2a1f5a7`), report the live XPU allocation per layer (P3-10 evidence), and keep releasing the source shards, the layer artifacts and the calibration state.
2. *Operational:* an external cache watchdog (advisory `POSIX_FADV_DONTNEED` on the source, the group files, the artifacts and the state file every 60 s) keeps the host cache from squeezing the device pool — it changes no conversion data (clean pages only; dirty pages are written back first).
3. *Policy surface:* the admission reserve is now configurable (`EXL3_MEMORY_RESERVE_GIB` / `EXL3_MEMORY_RESERVE_BYTES`; default unchanged at 6 GiB). Rationale: the reserve is *device-pool* headroom, and on this UMA host the pool is the binding dimension while the host itself has ~18 GiB available. Attempts 3 and 4 aborted with `binding available 9.48 GiB` against `reserve 6.00 GiB + outstanding 3.67 GiB` — i.e. short by 0.2 GiB while the machine had 18 GiB of host headroom. The P3-09 run therefore uses a **4 GiB** reserve (host headroom still ≥ 4.5× the reserve; device-pool remainder after the run's 3.67 GiB live XPU allocation ≈ 5.8 GiB). This is an explicit, measured deviation, not a silent default change; the semantics of the policy (admission arithmetic, post-commit reserve check) are unchanged.

**Attempt 5 (16:44:55, `EXL3_MEMORY_RESERVE_GIB=4` + watchdog):** resumed from layer 1 (the published layer 0 artifact and the calibration state were reused — the resume path itself is now exercised on the real model). Layer 1 completed (1678.1 s, proxy_loss 7.1786e+00) but layer 2 aborted with `binding available 3.40 GiB, reserve 4.00 GiB, outstanding 8.27 GiB`.

**The real defect behind attempts 2–5: a per-layer allocation leak.** The per-layer telemetry added in `2a1f5a7` made it visible: the live XPU allocation grew **2.4 GiB → 7.2 GiB → 8.3 GiB** over three layers. Cause: each layer's `module_results` (decoded weights, trellis states, regularization/LDL intermediates — ~3.5 GiB) and the layer module stayed alive through the *next* layer's entire quantization, because `results` was only rebound after that quantization had finished. The accumulated allocation is what exhausted the shared device pool; every abort signature (225 MiB → 9.48 GiB → 3.40 GiB binding) is a symptom of that growth plus page-cache pressure, not of the reserve.

**Fix (`0a9645b`):** the results, the layer report and the layer module are released right after the artifact is published and the manifest entry is built. The numbers that survive are read from the manifest entry, so no data is lost and no numerics change.

**Attempt 6 (17:32:56, `EXL3_MEMORY_RESERVE_GIB=2` + watchdog + results release) — leak fixed, margin still too thin.** Resumed from layer 2 (layers 0–1 already published). Measured effect of the fix: layer 2 finished with `xpu_alloc = 2048.6 MiB` (2 GiB) instead of 7171.7 MiB, and layer 3 starts at 2448.6 MiB — the device working set is now **flat across layers** instead of growing by ~3.5 GiB per layer. Host page cache stays at ~1.9 GiB (watchdog), `MemAvailable` ~8 GiB. Layer 3 then aborted with `binding available 6.17 GiB, reserve 2.00 GiB, outstanding 4.16 GiB` — the *usable* budget was 6.7 MiB against an 8 MiB request: the mechanism is no longer the problem, only the margin.

**Attempt 7 (18:05:34, `EXL3_MEMORY_RESERVE_GIB=1`):** the same configuration with a 1 GiB device-pool reserve (host headroom 8 GiB measured at the abort point, pool remainder 6.17 GiB). Expected usable budget ≈ 1 GiB — 128× the 8 MiB the scratch arena needs.

**Attempt 8 (19:37:48, `EXL3_MEMORY_RESERVE_GIB=1` + `EXL3_MEMORY_REFLECT_OUTSTANDING=1` + watchdog) — memory problem solved, device loss instead.** With the corrected counter regime the run was finally *stable*: layers 5, 6, 7, 8 published with the live XPU allocation flat at 2048–2057 MiB, the host RSS down to 1.6 GiB and no memory-policy abort anywhere. At 21:58 (layer 10, ~2.4 h into the attempt) the process died with a **device-level** failure instead of a policy abort:

```
RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)
  layer_quant.py:99  compute_factorization_residual -> torch.linalg.norm(diff).item()
```

That is a Level Zero backend reset, not a memory-policy decision: the host was simultaneously memory-starved (zram holding **5.2 GiB of RAM** for swapped pages, `MemAvailable` 9.9 GiB, device pool down to 5.03 GiB), and the desktop's own GPU allocations had grown (the idle pool fell from 12.78 GiB at 16:00 to 8.87 GiB by 19:35). Thermal was *not* implicated (max 51 °C across all zones). **10 layers (0–9) were published and survived**; the per-layer checkpoint plus the validated resume is what kept this attempt from losing ~5 h of work.

**Server restart (2026-10-22 06:24).** The host restarted while the session was idle; the conversion process was already dead (the device loss above), and the restarted host came back with **more** headroom (device pool 6.21 GiB free, `MemAvailable` 12.3 GiB — the zram pressure had drained).

**Attempt 9 (06:25:09) — resumed from layer 10** with `EXL3_MEMORY_RESERVE_GIB=0.5` + `EXL3_MEMORY_REFLECT_OUTSTANDING=1` + a 20 s watchdog. The resume re-validated all ten published artifacts and replayed the calibration state from `calibration_state.pt` (corpus sha256 verified), so no work was repeated. Remaining: 22 layers ≈ 10.3 h.

**Run configuration for the qualification run (explicit, measured deviations):**
* `EXL3_MEMORY_RESERVE_GIB` — the admission reserve is *device-pool* headroom; on this UMA host the host itself has ~9–18 GiB available while the pool is the binding dimension (measured in the probe table above). The default (6 GiB) leaves no admissible budget once the run's live XPU allocation and the pool remainder are accounted for. The override is explicit, documented and reversible (`af024a8`).
* `EXL3_MEMORY_REFLECT_OUTSTANDING=1` — on this device the counters *do* fall with the live XPU commitment (idle pool 8.87 GiB → 4.28 GiB while holding 3.68 GiB), so the outstanding bytes must not be subtracted a second time; the default (unset) keeps the historical Arc 140V behaviour (`0dfd4cd`).
* An external cache watchdog drops the *clean* page cache of the source, the group files, the layer artifacts and the calibration state every 20–30 s. Advisory only: clean pages are evicted, dirty pages are written back first, and no conversion data changes.
* The converter's own hygiene (`cff5e35`, `2a1f5a7`) releases the same memory inline.

### Step 27 [2026-10-22 15:12:09 +0200]: P4-14 Text-Only `qwen3_5_text` Container Landed (`046b079`) — `autotrust/JEV-9B`-Convertible Loader

**Why.** With the declared target's 32-layer conversion running unattended, the next candidate target was analysed: [`autotrust/JEV-9B`](https://huggingface.co/autotrust/JEV-9B) — the Qwen3.5-9B text backbone (vision tower and MTP head dropped) plus a detachable 40.2 M-parameter System-1 LoRA and a 24-slot fp32 decision head. Its backbone carries the **same P4-12 layer contract** (24 `linear_attention` + 8 `full_attention` at interval 4, identical quantized/pass-through module tables, hidden 4096 / intermediate 12288 / kv 4×256 / head_dim 256 = hidden/heads / linear heads 16+32×128 — every width still a multiple of the 128-wide Hadamard block), but ships it in the **text-only container**: `architectures: [Qwen3_5ForCausalLM]`, `model_type: qwen3_5_text`, a *flat* config with no `text_config` block, and the decoder stack at the top-level `model.layers.` prefix (427-key weight index verified 2026-10-22; no `model.visual.*`, no `mtp.*`).

**What landed (`046b079`, loader/validation only — no numerics, default `dense` mode and the multimodal hybrid path byte-identical).**
* `exl3_quant/hybrid.py`: `TEXT_TOWER_ARCHITECTURES`/`TEXT_TOWER_MODEL_TYPES` extended with the text-only container, plus new `TEXT_ONLY_ARCHITECTURES`/`TEXT_ONLY_MODEL_TYPES` markers so the flat layout is accepted only unambiguously — a multimodal `qwen3_5` config that is merely missing its text block stays a hard rejection.
* `exl3_quant/converter.py`: `_load_qwen35_text_tower_geometry` accepts the flat config as the text block (all geometry/MoE/Hadamard-width/layer-type/rope validation then applies verbatim); the container layout **derives** the checkpoint prefixes (`nested → model.language_model.layers`, `flat → model.layers`) instead of hardcoding the multimodal ones; `hybrid.text_tower_container` records which layout was converted; `has_vision`/`has_mtp` remain config-declared provenance while emitted pass-through groups stay **key-derived** (JEV-9B declares `mtp_num_hidden_layers: 1` yet ships no `mtp.*` tensors, so no MTP group file is produced).
* Evidence: new `write_qwen35_text_only_model` fixture (flat mini checkpoint carrying the stale MTP declaration, no vision/mtp tensors) + a faithful JEV-9B config replica; `TestTextOnlyContainer` (5 tests) — replica acceptance, dense-mode rejection, the missing-text-block guard, prefix/group key coverage over the written tensors, and a strict reference-layer build from the flat layout (36 tests in `tests/test_hybrid_converter.py`). CPU-only runs: **20/20** hybrid validator/module-table, **18/18** converter config/index; the real JEV-9B `config.json` validated end to end (container `text_only`, `model.layers` prefixes, 24/8 split, reference `Qwen3_5TextConfig` built eagerly). XPU test classes were deliberately **not** run — the device is owned by the P3-09 conversion (full-suite re-run once it releases).

**Out of scope, enumerated rather than assumed.** System 1 (the LoRA `adapter/` and `head.safetensors`) is not part of the backbone safetensors; three options are recorded in `tasks.md` P4-14 with the model card's own measurements (quantize the pristine backbone only / merge the LoRA as a separate recipe / serve the LoRA on the quantized base — the last needs `exl3xpu` work). Converting JEV-9B itself is a separate owner-approved task requiring a free XPU; the downstream runtime needs no change (`exl3xpu` `_norm_key` already reduces any prefix to `layers.N.*`, and with no `mtp.*` tensors its MTP path is skipped). Estimated output for such a run: ≈ 5.2–5.3 bpw / ≈ 5.8 GiB (8.95 B params, no vision bytes) — to be measured, not assumed.

---

## Critical Path Next Steps (Phase 3)

> **Governing decision — `skip_cuda.md` (`SKIP-CUDA-2026-10-21-01`):** CUDA ground truth (P0-24/P0-25/P0-27) is **deferred, not waived**. Phase 3 proceeds without it; revisit the deferral if any trigger fires: T1 XPU encoder MSE exceeds upstream `max_mse_per_K` for any K; T2 reconstructed weights diverge from the CPU reference beyond fp16 tolerance; T3 a gate/reviewer demands E4 path parity; T4 a downstream runtime rejects a produced checkpoint; T5/T6 upstream re-pin or kernel/codebook/container change. Resume per `skip_cuda.md` §7.
>
> **Declared Active Target (Step 23, unblocked Step 24):** `empero-ai/Qwen3.8-9B-Distill` — runnable via the opt-in `architecture_mode="qwen3_5_text_tower"` (P4-12 `[x]` / P4-13 `[~]`).

1. ~~**P3-01 / P3-02 / P3-03 — Bounded Model Orchestrator & Shard Streaming**~~ **DONE** (Step 21, [2026-10-21 10:20:00 +02:00]).
2. ~~**P3-04 / P3-05 / P3-06 / P3-07 / P3-08 — Checkpointing, Resumability & Fault Recovery**~~ **DONE** (Step 23 / commit `77ae4d4`, [2026-10-21 11:37:10 +02:00]).
3. **P3-09 .. P3-16 & Gate 3 — First Complete Model Conversion & Evaluation:** Execute full offline conversion of the declared Active Target `empero-ai/Qwen3.8-9B-Distill` — **UNBLOCKED** (Step 24) via `architecture_mode="qwen3_5_text_tower"`; start with a measured single-layer run and documented operating reserve, then the full 32-layer attempt with checkpoint/resume. Sibling of the first-declared `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` (same architecture; default-mode rejection stays guarded in `tests/test_converter.py::test_declared_target_mimo_is_rejected`, opt-in acceptance in `tests/test_hybrid_converter.py`). Vision/MTP are bit-exact pass-through (quantization parked at P4-04); `exl3xpu` text-only-decode interoperability (tier 6) is verified under P3-13 and gate evidence published as `reports/gate_3.md`.
