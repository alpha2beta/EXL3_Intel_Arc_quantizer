# Code Review — `tasks.md` + `progress.md`

> **Review ID:** `review-2026-10-20T13:50:25+02:00`
> **Reviewer:** OpenCode (automated review)
> **Reviewed Commit:** `bc048fe` ("docs: add progress.md detailing completed Steps 1 to 4 and test verification evidence")
> **Files Reviewed:** `tasks.md`, `progress.md`, `exl3_quant/{reference_encoder,scratch_estimator,bitpack}.py`, `tests/test_{encoder,scratch,bitpack}.py`
> **Review Timestamp:** 2026-10-20 13:50:25 +02:00 (local machine time)

> **Naming convention for future reviews:** each new review gets its own file `review-<YYYY-MM-DDThh:mm:ss±TZ>.md`
> (e.g. `review-2026-11-03T09:15:00+01:00.md`). This file is frozen as `review.md` for the
> 2026-10-20T13:50:25+02:00 review; it is never overwritten. Newer reviews are appended as new files
> and linked from this file's "Review History" table at the bottom.

---

I verified the claims against the actual repo (git history, source, and a fresh test run) before reviewing.

**Verification performed**
- `git log` / `git status`: 5 commits, clean tree — matches the commits cited in progress.md (except two omissions, below).
- `python -m unittest discover -s tests -p "test_*.py"` → **Ran 12 tests … OK**. progress.md's test claim is accurate.
- Source review of all three modules + all three test files against `EXL3_Intel_Arc_quantizer.md` §6, §7, §8.

---

### 1. Blocking findings

**B1 — Phase-order violation is not disclosed.** `tasks.md` states Phase 1 "Depends on: Gate 0", and Gate 0 (G0-01…G0-03, `reports/gate_0.md`) has never happened. Zero P0-* tasks are checked; `docs/` does not exist; P0-11 (native encoder audit) is open. Yet P1-01/P1-02/P1-04/P1-09..P1-11/P2-13/P2-15 are marked `[x]`.

The underlying problem: **P1-01 says "using the _audited_ state and boundary rules"** — the audit it depends on (P0-11/P0-12) is not done, so `reference_encoder.py` currently _invents_ defaults, which `tasks.md` line 86 explicitly forbids ("leave unresolved native behavior explicitly blocked rather than inventing defaults"):

- Trellis topology (`s_next = ((s_prev << bits) | sym) & (S-1)`) is a generic shift-register model, not the audited upstream topology.
- Codebook `level = scale * (c - zero_point)` with a **half-integer zero point** (1.5 at 2-bit, 7.5 at 4-bit) is a guess; upstream `mul1` levels are unverified.
- Tail closure is `argmin` over terminal states; upstream may require closure to a fixed state.
- The objective is unweighted MSE, whereas upstream's distortion is `||W − Ŵ||²_H` (tier-4 acceptance). This divergence must be recorded, not implied away.

Doing CPU-only work before an XPU box exists is defensible, but it needs to be recorded as an explicit **deviation/waiver** with rationale, not presented as checklist conformance. progress.md's "Current Status: Phase 0 tracking established" does not disclose the skip.

**B2 — P1-02 is marked complete but the tile permutation does not exist.** P1-02 requires "the exact tile permutation/inverse **and** procedural codebook arithmetic". Only the codebook half is implemented — `grep -i perm|hadamard|fwht|tensor_core` over `exl3_quant/` and `tests/` returns **no matches**. `tensor_core_perm` (16×16 tile reorder, `EXL3_Intel_Arc_quantizer.md` §5.2/§8) is exactly the piece that determines checkpoint interop, and it is entirely absent while the box is checked.

**B3 — P2-13's wording is not satisfied.** It requires *"Reuse device-independent upstream packing/serialization where possible; port only audited device-specific operations."* The bitpacker is a **new generic implementation**, not upstream reuse, and the upstream bit layout was never audited (P0-14 open). Its own docstring also claims "Implements P2-13, **P2-14**, P2-15" while P2-14 is unchecked in tasks.md — a direct contradiction between the two files.

**B4 — P1-04 is marked complete but is CPU-only.** The task says "a **GPU-resident** traceback baseline". Only a CPU traceback exists. The parenthetical in tasks.md acknowledges this, but progress.md's summary table lists P1-04 as "Passed" with no qualifier.

### 2. Factual errors in `progress.md`

| Location | Issue |
|---|---|
| Line 31 | "phase gates (Gate 0 through Gate 4)" — tasks.md defines **Gate 0–Gate 3** only; Phase 4 has no gate. |
| Lines 97–99 | Commit history omits `668b7dc` (tasks.md checkbox update) and `bc048fe` (progress.md itself, HEAD). The block is stale relative to HEAD. |
| Line 5 | "100% test pass rate" — true but misleading: all 12 tests are **self-consistency** checks (round-trips against the same implementation, arithmetic against fixtures restated from the spec). No test compares against an upstream/CUDA reference. P0-28 ("prove it detects deliberately perturbed values, bit order, and metadata") is **not** implemented, and the tests provably would not detect a wrong bit order. |
| Line 19 | P2-13/P2-15 listed "Passed" — see B3. |
| Line 20 | P1-02 listed "Passed" — see B2. |

### 3. Code-level issues (non-blocking, but should be tracked)

- **`bitpack.pack` silently truncates non-integer input.** `np.asarray(indices, dtype=np.int64)` turns `2.7` into `2` _before_ range validation, so floats pass validation. Fix: validate `np.equal(arr, np.floor(arr))` or require an integer dtype.
- **`scratch_estimator.calculate_max_batch_size` clamps with `max(1, …)`.** If even B=1 doesn't fit, it returns 1 instead of failing. That directly conflicts with P1-13 ("reject a configuration cleanly if the minimum valid batch cannot fit"). It will need changing when P1-13 lands.
- **Boundary condition at the single-allocation ceiling.** At K=1, B=256 the edges buffer is _exactly_ 4 GiB = the assumed Level Zero limit; `test_scratch.py` asserts `<=` so it passes, but an allocation of exactly the max commonly fails. Use `<` or flag the boundary.
- **No linkage between K, trellis state count, and the scratch model.** The estimator uses `S = 65536 >> K`, while the encoder takes an independent `num_states` (default 64). Nothing maps encoder state space → scratch `S` → packed bitwidth, which is precisely the "exact mapping between encoder parameter K, packed output bitwidth, and fractional bitrate recipes" audit item (§6.1). This must be resolved before P1-05.
- **Reference encoder is O(T·S·2^bits) in pure Python.** Fine as a spec artifact, but it cannot serve as the fixture generator for realistic `S=32768, T=256`; the "next step" of profiling multi-megabyte backpointer arrays needs a vectorized/NumPy or C++ path.
- `exl3_quant/` has no `__init__.py`; tests work only via the `sys.path` hack. Fine for now, but it will break `pytest`/`pip install -e` flows.

### 4. What is genuinely solid

- `ScratchEstimator` matches the spec's byte model exactly (`B·2·S·2` costs, `B·T·S·2` edges), reproduces all four K fixtures bit-exactly, and the GiB/MiB vs GB/MB distinction is correct and tested.
- The 3-bit cross-word packing logic is correct, including the `bit_offset == 30` overflow path (bit 2 of the symbol lands in word _n+1_, bit 0).
- The trellis path-validity invariant is genuinely verified (each step's transition re-derived from the shift rule), and the end-to-end encode→pack→unpack→reconstruct pipeline is coherent.
- Git hygiene is good: clean tree, small focused commits.

### 5. Recommended next actions

1. **Add a `docs/deviation_log.md`** and record explicitly: Phase 1/2 CPU work started before Gate 0, rationale (no Arc hardware on this host), and the list of invented defaults (trellis topology, `mul1` levels, tail closure, unweighted objective) marked as _unverified placeholders_, not audited contract.
2. **Un-check P1-02, P1-04, P2-13**, or reword them to the narrower work actually done (e.g. P1-02a "codebook arithmetic only") and add the missing permutation as its own task. The checklist should never imply untested coverage.
3. **Fix the two factual errors in progress.md** (Gate 0–3; add `668b7dc` and `bc048fe` to the history) and qualify "100% test pass rate" as _self-consistency only, no upstream reference yet_.
4. **Promote P0-11/P0-12/P0-13/P0-14 (the native audits) to the critical path.** Every checked P1/P2 item depends on them; the encoder's correctness cannot be established without the upstream `mul1` level table and state-transition rules.
5. **Add P0-28-style negative tests** (mutate a bit, a codebook level, and the tile order; assert the harness fails) — this is the cheapest way to make the existing 12 tests meaningful.
6. Minor code fixes: float-truncation guard in `pack`, remove the `max(1, …)` clamp in `calculate_max_batch_size`, add `exl3_quant/__init__.py`.

### 6. On the "Next Steps" list in progress.md

The ordering there is inverted: P0-08 (manifest) and P0-16 (algorithm contract) are listed _before_ the P0-11/P0-12 audits, but P0-16 is defined as documenting the audited equations — it cannot be written before the audit. Same for item 3 (FWHT, P2-06): FWHT and `tensor_core_perm` need the upstream sign/scaling/permutation conventions from P0-13 first. Also item 4 (P1-18) is listed as a _host CPU_ task, but P1-18 in tasks.md is explicitly a device-edge-copy + CPU traceback benchmark — the CPU analogue of it is really P1-05/P1-07 groundwork. I'd renumber that item to avoid implying P1-18 is closed.

**Bottom line:** the code that exists is clean, tested, and useful as a golden reference — but it is a _speculative_ reference, and the checklist currently marks it as contract-conformant. The highest-value correction is to make the "unverified vs. audited" distinction explicit in both files before any of this is used to validate an XPU port.

---

## Review History

| Review ID | Timestamp (local) | Reviewed Commit | Verdict | File |
|---|---|---|---|---|
| `review-2026-10-20T13:50:25+02:00` | 2026-10-20 13:50:25 +02:00 | `bc048fe` | 4 blocking findings (B1–B4); 2 factual errors in progress.md; 6 code-level issues; core implementation solid but contract-unverified | `review.md` (this file) |

<!-- Future reviews: append a row above and create a new timestamped review file. -->
