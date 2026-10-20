# Project Deviation Log & Waivers

> **Log ID:** `DEV-2026-10-20-01`  
> **Date:** 2026-10-20  
> **Trigger:** Review `review-2026-10-20T13:50:25+02:00` (Blocking finding B1)

---

## 1. Deviation: Out-of-Order Execution Prior to Gate 0

* **Declared tasks.md Contract:** Phase 1 tasks depend strictly on passing **Gate 0** (`reports/gate_0.md`), following completion of Phase 0 audits (`docs/compatibility_manifest.md`, `docs/native_dependency_audit.md`, `docs/algorithm_contract.md`).
* **Actual Action Taken:** Implemented prototype CPU modules in `exl3_quant/` (`scratch_estimator.py`, `bitpack.py`, `reference_encoder.py`) before completing Phase 0 audits.
* **Justification / Rationale:** The development machine is a Windows host CPU without Intel Arc or NVIDIA CUDA hardware. Prototyping inspectable CPU references and memory estimators provides an early software sandbox while audits are conducted.
* **Waiver Scope:** The current CPU implementations must be treated strictly as **unverified provisional prototypes**, NOT as audited contract-conformant implementations. No milestone may claim conformance against upstream EXL3 until Phase 0 audits are completed and verified against upstream reference fixtures.

---

## 2. Inventory of Invented / Placeholder Defaults

The following implementation details in `exl3_quant/` are currently **speculative placeholders** and must not be treated as ground truth until audited against `turboderp-org/exllamav3`:

| Component | Current Prototype Value | Audit Requirement | Dependency |
| :--- | :--- | :--- | :--- |
| **Trellis State Transition** | `((s_prev << bits) | sym) & (S - 1)` (generic shift-register) | Verify exact bitshift order, polynomial, and state indexing in upstream CUDA kernels | P0-11 |
| **`mul1` Arithmetic Levels** | `scale * (c - zero_point)` with half-integer zero point (`1.5` at 2-bit, `7.5` at 4-bit) | Audit exact integer arithmetic sequence and scaling in native C++/CUDA | P0-11 |
| **Trellis Tail Closure** | Unconstrained `argmin_s C_T[s]` | Verify whether upstream forces tail state to 0 or uses terminal path constraints | P0-11 |
| **Quantization Objective** | Unweighted Mean Squared Error $\sum (w - \hat{w})^2$ | Upstream objective is Hessian-weighted error: $\text{Tr}((\hat{W} - W) H (\hat{W} - W)^T)$ | P0-12 |
| **Tile Permutation** | Missing / Not implemented | Extract exact 16×16 `tensor_core_perm` and inverse permutation matrices | P0-13 |
| **Bitpack Packing Layout** | Generic LSB-first contiguous bitpacking | Verify upstream safetensors bit/byte ordering and endianness | P0-14 |

---

## 3. Rectification Plan

1. **Un-check premature checklist items in `tasks.md`:** Ensure tasks requiring audited contracts or GPU execution (P1-02, P1-04, P2-13) are marked accurately.
2. **Execute Phase 0 Audits as Critical Path:** Complete `docs/compatibility_manifest.md` (P0-08), `docs/native_dependency_audit.md`, and `docs/algorithm_contract.md` (P0-16).
3. **Update Reference Code:** Align `reference_encoder.py`, `bitpack.py`, and test fixtures with the audited contracts.

---

## 4. Audit Update — 2026-10-20 (P0-11, P0-13, P0-14 partially closed)

> **Status of §2 changed from "speculative placeholder" to "confirmed incorrect".**
> The upstream source was fetched and read on the audit host (no GPU required):
> `turboderp-org/exllamav3` @ `4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f`.
> The audited contract is [`algorithm_contract.md`](./algorithm_contract.md);
> the audited primitives are [`exl3_quant/trellis_format.py`](../exl3_quant/trellis_format.py).

| Component | Audited upstream value | Prototype verdict | Source |
| :--- | :--- | :--- | :--- |
| **Trellis State Transition** | 16-bit states; `edges = 65536 >> K`; `Kr = 16 - K`; branch state `sigma = (k << Kr) | e`; predecessor `pred = sigma >> K` | **WRONG** (wrong state width, wrong branch count) | `quantize_tiles_kernel.cuh:37-39, 104-107` |
| **`mul1` Arithmetic Levels** | Pseudorandom LUT over the 16-bit state: `x = state*0x83DCD12D`; `sum = dp4a(x, 0x01010101, 0x6400)`; `__hfma(fp16(sum), 0x1EEE, 0xC931)` → **913 distinct levels in [-3.453125, 3.34765625]** | **WRONG** (uniform grid with half-integer zero point) | `codebook.cuh:76-89` |
| **Trellis Tail Closure** | Two-pass **tail-biting ring**: `forward(L/2,-1)` → `argmin_cost()` → `backward(L/2,false,…)` → `forward(0,end_state)` → `backward(0,true,…)`, with `pre_state` masking to `H_INF` | **WRONG** (unconstrained terminal `argmin`) | `.cuh:59-72, 99, 130, 231-313` |
| **Quantization Objective** | Per-element **fp16** squared error with **fused** accumulation (`__hfma2`); layer-level error weighting enters through the LDLQ feedback (`L` slices), not a Hessian-weighted objective inside the tile encoder | **WRONG precision** (fp64 metric, fp64 accumulation) | `.cuh:97-98, 180`; `quantize.py:566-604` |
| **Tile Permutation** | Fixed 256-entry table; applied **before** encoding and **not** undone on the index stream | **RESOLVED** — implemented as `tensor_core_perm()` / `tensor_core_perm_inv()` | `quantize.py:22-49, 579, 584-585` |
| **Bitpack Packing Layout** | MSB-first, 16 spans × 16 elements, **low `K` bits of each 16-bit state**, `SWAP16` half-word swap; the decoder recovers the remaining bits from a 16-bit window ending at bit `(t+1)*K` | **WRONG** (independent symbols, LSB-first) — this is a **different format**, not a bit-order difference | `pack.cu:9-57`; `exl3xpu/ref.py:81-99` |

### Consequences

1. `exl3_quant/reference_encoder.py` and `exl3_quant/bitpack.py` must be **replaced**, not reconciled.
   `exl3_quant/trellis_format.py` now carries the audited codebook LUT, the tile permutation,
   and the upstream packing/decoding layout, with 17 format-conformance tests in
   `tests/test_trellis_format.py`.
2. `exl3_quant/scratch_estimator.py` is **confirmed correct** against
   `get_temp_buffers` (`quantize.py:52-63`) and needs no rework.
3. The "unweighted MSE objective" row in §2 is **partly retired**: the tile encoder's
   objective is plain squared error; the Hessian weighting lives in the LDLQ feedback
   stage, which is documented separately in `algorithm_contract.md` §9.
4. Still blocked and still provisional: P0-08 (manifest), P0-12 (LDLQ *implementation* —
   the equations are now extracted, the code is not written), P0-15/P0-16 (native dependency
   inventory), P0-17–P0-23 (hardware), P0-24/P0-25 (CUDA ground truth).

---

## 5. Implementation Update — 2026-10-20 (P0-05, P0-08, P1-01, P1-02, P1-05 closed)

* **Platform Profiling:** Primary target device empirically queried and recorded: Intel Core Ultra 7 258V (Lunar Lake), Arc 140V integrated GPU (`0x64A0`, 8 Xe-cores / 64 XVEs, 128 KB SLM), 32 GB shared memory pool, PyTorch XPU `2.14.0+xpu` and oneAPI 2026.0.
* **Manifest (P0-08 / V140-01):** Published in `docs/compatibility_manifest.md`.
* **Contract-Conformant Reference Encoder (P1-01 / P1-02):** Replaced provisional prototype in `exl3_quant/reference_encoder.py` with `ReferenceTrellisEncoder` transcribed from `quantize_tiles_kernel.cuh:23-313`:
  - 16-bit states, $S = 65536 \gg K$ tracked edges, $Kr = 16 - K$, candidate state $\sigma = (k \ll Kr) \mid e$, predecessor $pred = \sigma \gg K$.
  - Two-pass tail-biting circular ring schedule ($roll = L/2$ unconstrained to find `end_state`, followed by $roll = 0$ constrained by `pre_state = end_state`).
  - fp16 metric with fused accumulation and strict `<` tie-breaking.
  - End-to-end format round-trip verified: encoded states bitpack into container, decode through `decode_states` and `decode_tile_values`, and reconstruct the original tile with 100% bit-exact parity.
* **Waiver DEV-2026-10-20-01 Status:** The reference encoder and container format components are resolved with respect to upstream CUDA source transcription, mathematical state inverse identities, and live round-trip decoding through the independent oracle (`exl3xpu.ref`). However, full waiver closure and final numerical acceptance remain contingent on verifying numerical parity against upstream CUDA execution fixtures (P0-24) and resolving Open Item O-1 (terminal cost buffer parity).
