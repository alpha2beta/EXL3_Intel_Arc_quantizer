# EXL3 Intel Arc Quantizer: Implementation Progress

> **Last Updated:** 2026-10-20 14:23:51 +02:00
> **Repository:** `C:/Users/yi40/EXL3_Intel_Arc_quantizer`
> **Current Status:** Phase 0 audit **partially complete** — the native encoder, codebook, tile-permutation and container-layout audits (P0-11, P0-13, P0-14) were performed from pinned upstream source on the host CPU and are published in [`docs/algorithm_contract.md`](./docs/algorithm_contract.md). The provisional CPU prototypes that those audits invalidate are still present and must be replaced. Phase 0 waiver `DEV-2026-10-20-01` remains open.
> **Time-tag convention:** every entry below carries a local-time timestamp `[YYYY-MM-DD HH:MM:SS ±ZZZZ]` so that entries from different sessions can be ordered unambiguously. Timestamps are taken from the corresponding git commit date where one exists.

---

## Executive Summary

Steps 1 through 7 of the CPU-side foundational engineering work have been implemented, unit-tested, and committed to git.

Steps 1–5 established the host-CPU sandbox (sizing model, provisional packer, provisional encoder, negative-test harness) under the review waiver `DEV-2026-10-20-01`.

**Steps 6–7 [2026-10-20 14:19:27 +0200]** are the substantive change: the **upstream native source was fetched and audited** (no CUDA, no Intel Arc hardware required), producing the formal algorithm contract and the first *contract-conformant* code in the repository. That audit **confirms the provisional prototypes in Steps 3–4 are incorrect**, not merely unverified.

> [!WARNING]
> **Provisional Prototype Status (Review Waiver DEV-2026-10-20-01) — updated 2026-10-20 14:19:27 +0200:**
> The Python implementations of trellis search, codebook decoding, and bitpacking in
> `exl3_quant/reference_encoder.py` and `exl3_quant/bitpack.py` remain **provisional** and are
> now **known to be incorrect** with respect to upstream EXL3. The pending audits P0-11/P0-13/P0-14
> have since been performed; the audited values are:
>
> | Prototype convention | Audited upstream convention |
> |---|---|
> | `mul1` levels = `scale * (c - zero_point)`, uniform grid, half-integer zero point | **Pseudorandom LUT over the 16-bit state** — `x = state*0x83DCD12D`; `sum = dp4a(x, 0x01010101, 0x6400)`; `__hfma(fp16(sum), 0x1EEE, 0xC931)` → 913 distinct levels in [-3.453125, 3.34765625] |
> | generic shift register, `num_states` (default 64) | **16-bit states**, `edges = 65536 >> K`, `Kr = 16-K`, branch state `σ = (k << Kr) | e`, predecessor `pred = σ >> K` |
> | fp64 metric and accumulation | **fp16** metric with **fused** accumulation (`__hfma2`), `H_INF = 0x7c00` |
> | unconstrained terminal `argmin` | **two-pass tail-biting ring** (`roll = L/2` then `0`), `pre_state` masking to `H_INF` |
> | independent-symbol LSB-first packing | **state-history container**: MSB-first, 16 spans × 16 elements, low `K` bits of each 16-bit state, `SWAP16` half-word swap |
>
> `exl3_quant/scratch_estimator.py` is **confirmed correct** against upstream `get_temp_buffers` and needs no rework.
> See [`docs/deviation_log.md`](./docs/deviation_log.md) §4 and [`docs/algorithm_contract.md`](./docs/algorithm_contract.md).

```
+-------------------------------------------------------------------------------------------------------+
| Module                       | Task Mapping             | Status                          | Unit Test Suite       |
+------------------------------+--------------------------+---------------------------------+-----------------------+
| 1. Tasks Checklist           | Scope Governance         | Committed (Gates 0–3)           | tasks.md              |
| 2. Scratch Estimator         | P1-09, P1-10, P1-11      | Passed — CONFIRMED CORRECT      | test_scratch.py       |
| 3. Sub-Byte Bitpacker        | P2-13, P2-15             | Provisional — CONFIRMED WRONG   | test_bitpack.py       |
| 4. Golden Reference Encoder  | P1-01, P1-02, P1-04      | Provisional — CONFIRMED WRONG   | test_encoder.py       |
| 5. Negative Test Harness     | P0-28                    | Passed (Perturbation Det.)      | test_negative.py      |
| 6. Algorithm Contract (docs) | P0-11, P0-13, P0-14, 16  | Published 2026-10-20 14:19      | n/a (document)        |
| 7. Audited Format Primitives | P0-13, P0-14, P0-28      | Passed (Format Conformance)     | test_trellis_format.py|
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

## Test Suite Execution Evidence

**34 unit tests** ran and passed cleanly on the host CPU as of [2026-10-20 14:19:27 +0200] — 12 original + 5 negative (Step 5) + 17 trellis-format (Step 7). **Note:** these establish algorithmic and format self-consistency; comparison against an upstream CUDA reference (tier-4 acceptance) is still blocked on P0-24/P0-25.

```text
PS C:\Users\yi40\EXL3_Intel_Arc_quantizer> python -m unittest discover -s tests -p "test_*.py"
..................................
----------------------------------------------------------------------
Ran 34 tests in 0.170s

OK
```

Earlier evidence, superseded:
```text
[2026-10-20 13:38:49 +0200]  Ran 12 tests   (Steps 1–4)
[2026-10-20 13:55:32 +0200]  Ran 17 tests   (Steps 1–5)
```

---

## Git Commit History

```text
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

## Critical Path Next Steps (re-prioritized after the Step 6 audit)

1. **P0-08 — Compatibility Manifest.** Author `docs/compatibility_manifest.md` (upstream commit, model/checkpoint revisions, codebook + K, calibration corpus, tolerances, device targets, exclusions). The last Phase 0 deliverable that needs no hardware.
2. **P1-01 / P1-02 — Replace the provisional encoder.** Transcribe `quantize_tiles_kernel.cuh` into a CPU reference encoder: 16-bit states, `edges = 65536 >> K`, fp16 metric with fused accumulation, two-pass tail-biting ring, strict-`<` tie-break. Reuse `trellis_format.codebook_lut` and `tensor_core_perm`.
3. **P0-16 — Native dependency inventory.** Classify every `ext.*` call site in `convert_model.py` (reusable / XPU-native / port-required / CPU fallback / unsupported) into `docs/native_dependency_audit.md`.
4. **P2-04 / P2-06 / P2-08 — CPU numerical path.** Implement the audited LDLQ error feedback (`quantize.py:566-604`), the normalized Sylvester `H128` transform, and the block traversal with the corrected target weights.
5. **P2-13 / P2-14 — Replace the provisional packer.** Reuse `trellis_format.pack_trellis`; then validate output keys, dtypes, `suh`/`svh` sign-scale vectors and `.safetensors` metadata (P0-14 remainder).
6. **P0-24 / P0-25 — CUDA ground truth.** First hardware task: capture encoder fixtures and cost/edge snapshots from the pinned CUDA revision to close tier-4 acceptance.
