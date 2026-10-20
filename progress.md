# EXL3 Intel Arc Quantizer: Implementation Progress

> **Last Updated:** 2026-10-20 19:48:23 +02:00  
> **Target Platform:** Intel Core Ultra 7 258V (Lunar Lake), Arc 140V integrated GPU (`0x64A0`), 32 GB shared memory  
> **Current Status:** Phase 0 platform manifest, audited algorithm contract, format primitives, live oracle parity, native dependency audit, and a verified minimal SYCL XPU extension established. Open item O-6 (fp16 FMA semantics) measured on-device: exact single rounding confirmed. 49 unit tests passing. Remaining before Gate 0: V140-02/P0-20 idle baseline & scratch allocation, G0-03 report.  
> **Time-tag convention:** every entry below carries a local-time timestamp `[YYYY-MM-DD HH:MM:SS ±ZZZZ]` so that entries from different sessions can be ordered unambiguously. Timestamps are taken from the corresponding git commit date where one exists.

---

## Executive Summary

Steps 1 through 12 of the foundational engineering work have been implemented, unit-tested, and verified against the audited contract:

- Steps 1–5 established the initial CPU sandbox under review waiver `DEV-2026-10-20-01`.
- Steps 6–7 fetched and audited the pinned upstream EXL3 source, establishing [`docs/algorithm_contract.md`](./docs/algorithm_contract.md) and [`exl3_quant/trellis_format.py`](./exl3_quant/trellis_format.py).
- **Step 8 [2026-10-20 17:10:00 +0200]:** Probed the host hardware and software environment, amended `tasks.md` with the Arc 140V platform profile, and published [`docs/compatibility_manifest.md`](./docs/compatibility_manifest.md) (P0-05, P0-08, V140-01).
- **Step 9 [2026-10-20 17:22:55 +0200]:** Replaced the provisional prototype encoder with the **contract-conformant `ReferenceTrellisEncoder`** directly transcribing `quantize_tiles_kernel.cuh:23-313` with NumPy vectorization (P1-01, P1-02, P1-05).
- **Step 10 [2026-10-20 18:55:00 +0200]:** Added live oracle parity test suite ([`tests/test_oracle_parity.py`](./tests/test_oracle_parity.py)) verifying bit-exact decode against the adjacent downstream runtime (`exl3xpu.ref`), and expanded manifest with model/corpus specifications and 140V memory risk analysis.
- **Step 11 [2026-10-20 19:18:02 +0200]:** Published [`docs/native_dependency_audit.md`](./docs/native_dependency_audit.md), classifying all six `ext.*` entry points and the supporting runtime surface, with the XPU port decisions and blockers (P0-15, P0-16).
- **Step 12 [2026-10-20 19:48:23 +0200]:** Built, verified, and committed the **minimal custom SYCL XPU extension** through `torch.utils.cpp_extension.load` (P0-18, P0-19), and **measured open item O-6 on-device**: the Arc 140V `sycl::fma(half)` performs exact fp16 single rounding — never fp32 double rounding — on 65,536 random and 11 adversarial-constructed inputs.

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

## Test Suite Execution Evidence

**49 unit tests** ran and passed cleanly as of [2026-10-20 19:44:00 +0200] across all test suites:
- `test_scratch.py`: 4 tests (scratch arithmetic and allocation limits)
- `test_negative.py`: 5 tests (perturbation detection and input validation)
- `test_trellis_format.py`: 17 tests (codebook LUTs, permutations, bitpacking, state decoding)
- `test_encoder.py`: 8 tests (contract-conformant reference encoder, path cost tracking, bounded distortion)
- `test_oracle_parity.py`: 4 tests (live bit-exact parity against `exl3xpu.ref`)
- `test_bitpack.py`: 4 tests (sub-byte packing shim)
- `test_xpu_extension.py`: 7 tests (device selection, correctness, ordering, lifetime, caps, O-6 random + adversarial; skip unless `torch.xpu.is_available()`)

```text
PYTHONPATH=. /home/yanchun/Projects/vllm.xpu/.venv/bin/python -m unittest discover -s tests -p "test_*.py"
Ran 49 tests in 12.971s

OK
```

---

## Git Commit History

```text
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

1. **V140-02 / P0-20 / V140-03 — Idle Baseline, Reserve Policy & Scratch Sizing:** Measure the idle profile in `artifacts/environment/`, enforce the 6 GiB reserve, reconcile it with the estimator's 2 GiB default, and test representative scratch allocations through the framework allocator; validate the wave-cap choice (`gpu_subslice_count` vs `max_compute_units`).
2. **G0-03 — `reports/gate_0.md`:** Record go/no-go, owners, and the Phase 1 estimate; carry P0-24/P0-25 (CUDA ground truth) and O-1 (terminal-edge parity) as explicit blockers. Note: **O-6 is now resolved on-device** and no longer a blocker.
3. **P0-21 / P0-22 — Operator Coverage & Factorization Fallback:** Probe `cholesky`, `inv`, matmul/einsum and fp16 rounding on the 140V; measure any CPU fallback.
4. **P1-15 / P1-16 — SYCL ACS Kernel Port:** Implement `quantize_tiles` on XPU following the audited contract, now with confidence in the fp16 FMA semantics (O-6) and the extension build path (P0-18).
5. **P2-04 / P2-06 / P2-08 — Numerical Preprocessing & Block LDLQ Path:** Implement Sylvester Hadamard ($H_{128}$) transform and blockwise LDLQ error feedback (device-independent reference track).
