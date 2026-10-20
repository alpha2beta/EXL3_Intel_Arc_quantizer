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
