# Review Response: `review-2026-10-20T13:50:25+02:00`

> **Response Date:** 2026-10-20  
> **Reviewed Target:** Commit `bc048fe`  
> **Remediation Commit:** `fc73c07` ("fix(review): address review-2026-10-20T13:50:25+02:00 findings")  
> **Review Source:** [`review.md`](../review.md)  
> **Status:** All 4 blocking findings, factual errors, and code-level issues accepted and remediated.

---

## 1. Response to Blocking Findings (B1 – B4)

### B1 — Phase-Order Violation & Undisclosed Defaults
* **Review Finding:** Phase 1 tasks depend strictly on passing Gate 0 (`reports/gate_0.md`). Zero P0-* tasks were checked, yet P1 and P2 tasks were marked `[x]`. The reference encoder invented defaults (`s_next = ((s_prev << bits) | sym) & (S-1)`, half-integer `mul1` zero-point, terminal state `argmin`, unweighted MSE objective) rather than using an audited contract.
* **Response & Action:** Accepted.
  * Created [`docs/deviation_log.md`](./deviation_log.md) logging formal project waiver `DEV-2026-10-20-01`.
  * Documented the host-CPU sandbox rationale and cataloged all current implementation details as *provisional, unverified placeholders*.
  * Updated docstrings across `exl3_quant/reference_encoder.py`, `bitpack.py`, and `scratch_estimator.py` to explicitly declare their provisional status.
  * Unchecked P1-01 in `tasks.md` with explicit notation pointing to `docs/deviation_log.md`.

---

### B2 — P1-02 Marked Complete Without Tile Permutation
* **Review Finding:** P1-02 requires "the exact tile permutation/inverse **and** procedural codebook arithmetic". Only codebook arithmetic was implemented; `tensor_core_perm` was completely missing.
* **Response & Action:** Accepted.
  * Un-checked P1-02 in `tasks.md`.
  * Clarified in `tasks.md` and `progress.md` that procedural codebook arithmetic is provisionally prototyped, while the 16×16 tile permutation matrix is pending upstream audit P0-13.

---

### B3 — P2-13 Wording Not Satisfied & Docstring Contradiction
* **Review Finding:** P2-13 requires reusing upstream packing/serialization where possible. A generic bitpacker was written instead without auditing upstream formats. Furthermore, `bitpack.py` claimed P2-14 while unchecked in `tasks.md`.
* **Response & Action:** Accepted.
  * Un-checked P2-13 and P2-15 in `tasks.md`.
  * Updated `exl3_quant/bitpack.py` docstring to remove premature claims to P2-13 and P2-14, labeling it accurately as a generic sub-byte bitpacker pending format audit P0-14.

---

### B4 — P1-04 Marked Complete but CPU-Only
* **Review Finding:** The task explicitly specifies a "GPU-resident traceback baseline", but only a CPU traceback exists.
* **Response & Action:** Accepted.
  * Un-checked P1-04 in `tasks.md`.
  * Updated `progress.md` to qualify P1-04 as a provisional CPU reference baseline, noting that the GPU-resident baseline remains open pending XPU hardware.

---

## 2. Response to Factual Errors in `progress.md`

1. **Phase Gate Numbering:**
   * Corrected from "Gate 0 through Gate 4" to **"Gate 0 through Gate 3"** (tasks.md defines gates only for Phases 0, 1, 2, and 3; Phase 4 has no gate).
2. **Commit Lineage:**
   * Updated the git commit history to include all commits up to HEAD (`491818c`, `3ee68d4`, `668b7dc`, `bc048fe`, `f95c640`, `fc73c07`).
3. **Test Pass Rate Qualification:**
   * Qualified the test execution statement to clarify that the unit test suite establishes **internal mathematical self-consistency**, not parity with an upstream CUDA reference.
4. **Summary Table Alignment:**
   * Updated the progress table to explicitly state that reference encoder and bitpacker modules are *provisional prototypes* pending Phase 0 audits.

---

## 3. Response to Code-Level Issues

1. **Non-Integer Float Truncation Guard ([`exl3_quant/bitpack.py`](../exl3_quant/bitpack.py)):**
   * Added strict floating-point validation:
     ```python
     if np.issubdtype(raw_arr.dtype, np.floating):
         if not np.all(np.equal(raw_arr, np.floor(raw_arr))):
             raise ValueError("Indices must be exact integer values, got non-integer floats.")
     ```
   * Floats with fractional parts (e.g., `2.7`) are now rejected with `ValueError` instead of being silently truncated to integers.
2. **Batch Sizing Failure on Inadequate Memory ([`exl3_quant/scratch_estimator.py`](../exl3_quant/scratch_estimator.py)):**
   * Removed `max(1, ...)` clamping in `calculate_max_batch_size(...)`.
   * Added clean failure handling: if usable memory cannot accommodate even minimum batch size $B=1$, the method raises `ValueError` cleanly per task P1-13.
3. **Package Initialization:**
   * Created [`exl3_quant/__init__.py`](../exl3_quant/__init__.py) defining public exports and enabling standard package imports.
4. **P0-28 Negative Test Suite ([`tests/test_negative.py`](../tests/test_negative.py)):**
   * Implemented 5 negative test cases proving the test harness detects:
     * Bit flips in packed uint32 arrays (`AssertionError`).
     * Parameter mutations in codebook scaling (`AssertionError`).
     * Corrupted/illegal trellis path transitions (`assertFalse(is_legal)`).
     * Float inputs to bitpack (`ValueError`).
     * Under-provisioned memory budgets for batch sizing (`ValueError`).

---

## 4. Test Suite Verification

```text
PS C:\Users\yi40\EXL3_Intel_Arc_quantizer> python -m unittest discover -s tests -p "test_*.py"
.................
----------------------------------------------------------------------
Ran 17 tests in 0.261s

OK
```

All 17 tests passed, including the 5 negative perturbation tests.

---

## 5. Critical Path Next Steps (Re-Prioritized)

Following the reviewer's guidance, the upstream audits are placed on the critical path:

1. **P0-11 (Native Encoder Audit):** Audit `turboderp-org/exllamav3` to extract exact `mul1` level tables, shift-register polynomial, tail closure, and tie-breaking rules.
2. **P0-12 (LDLQ Audit):** Extract exact one-sided LDLQ error update equations and factor orientation.
3. **P0-13 / P0-14 (Transforms & Formats Audit):** Extract 16×16 `tensor_core_perm`, FWHT butterfly sign conventions, and `.safetensors` header metadata.
4. **P0-08 / P0-16 (Formal Contracts):** Publish `docs/compatibility_manifest.md` and `docs/algorithm_contract.md`.
