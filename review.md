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
>
> **Addendum A1–A8 appended 2026-10-20 (post-audit).** After this review was filed, the pinned upstream
> source was fetched and audited on the same host (no GPU). Findings B1, B2 and B3 are **upgraded from
> "unverified assumption" to "confirmed incorrect"**, and three new findings (A4–A6) are added.
> See [§ Addendum](#addendum-a1--a8-post-audit-evidence) below and `docs/algorithm_contract.md`.

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

## Addendum A1–A8 (post-audit evidence)

**Audit performed:** 2026-10-20, on the same host, **without CUDA or Intel Arc hardware**.
The pinned upstream revision was fetched directly and read. This upgrades the earlier
findings from "unverified assumption" to "confirmed incorrect", and resolves the question
of how much of `tasks.md` is executable offline.

**Sources fetched** (upstream org is `turboderp-org`, **not** `turboderp` — the latter returns
"Repository not found"; the source spec's URLs at `EXL3_Intel_Arc_quantizer.md:359-360` are correct):

| Source | SHA-256 (first 16) |
|---|---|
| `exllamav3/modules/quant/exl3_lib/quantize.py` @ `4d7b69c7…` | `1B11658F5DE96BFE` |
| `exllamav3/exllamav3_ext/quant/quantize_tiles_kernel.cuh` | `F8399AA3D281625F` |
| `exllamav3/exllamav3_ext/quant/codebook.cuh` | `0E3C63B323F8D3CC` |
| `exllamav3/exllamav3_ext/quant/pack.cu` | `27606EED6650ACC3` |
| `exllamav3/exllamav3_ext/quant/reconstruct.cu` | `F47F121CDB452CF8` |
| `exllamav3/util/hadamard.py` | `6884841B61378788` |
| `exllamav3/conversion/convert_model.py` | `EE576CA6D2D56E55` |
| `sybil-solutions/exl3xpu` → `exl3xpu/ref.py` (CPU decode oracle) | cloned @ `15ded2f3` |

Full contract with line-level citations: **`docs/algorithm_contract.md`**.

### A2 — B1 upgraded: the reference encoder is **confirmed incorrect**, not merely unverified

The original B1 said the trellis topology, the `mul1` codebook, the tail closure and the metric
precision were *invented defaults*. The audit confirms each one is wrong:

| Repo assumption (`exl3_quant/reference_encoder.py`) | Audited upstream | Source |
|---|---|---|
| `Mul1Codebook`: `level = scale * (c - zero_point)`, uniform grid, half-integer zero point | **Pseudorandom LUT over the 16-bit state**: `x = state*0x83DCD12D`; `sum = dp4a(x,0x01010101,0x6400)`; `__hfma(fp16(sum), 0x1EEE, 0xC931)` | `codebook.cuh:76-89` |
| State space: `num_states` (default 64), `bits` symbols per transition | **16-bit states**, `edges = 65536 >> K`, `Kr = 16-K`, branch state `sigma = (k << Kr) | e`, predecessor `pred = sigma >> K` | `quantize_tiles_kernel.cuh:37-39, 104-107` |
| Metric in float64 | **fp16 throughout**: `__hsub2`/`__hmul2`/`__hfma2`, `H_INF = 0x7c00`, input rounded `__float2half_rn` | `.cuh:19-21, 56, 97-98, 180` |
| Termination: `argmin` over terminal states | **Two-pass tail-biting ring**: `forward(L/2,-1)` → `argmin_cost()` → `backward(L/2,false,…)` → `forward(0,end_state)` → `backward(0,true,…)`, with `pre_state` masking to `H_INF` | `.cuh:59-72, 99, 130, 231-313` |
| Tie-break: unspecified | **Strict `<`**, smallest `k` wins; terminal argmin preserves the "historical 1024-thread" even-then-odd edge order | `.cuh:131-140, 231-260` |

Consequence: **the 12 passing unit tests cannot detect any of this.** They are self-consistency
checks against the same implementation. The one test that would have caught it — comparing
`codebook_lut` against `Mul1Codebook` — does not exist, because the oracle (`exl3xpu/ref.py:57-63`)
was never consulted.

**Recommendation:** do not patch `reference_encoder.py`. Replace it with a transcription of
`quantize_tiles_kernel.cuh` (fp16 metric, tail-biting two-pass schedule, per-state LUT),
and validate it against `exl3xpu/ref.py` as a CPU oracle.

### A3 — B2 confirmed and now specified: the permutation table is obtained, still not implemented

`tensor_core_perm` is a fixed 256-entry table (`quantize.py:22-44`), identical in
`exl3xpu/ref.py:30-45`. It is applied **before** encoding and **not** undone on the index
stream (`quantize.py:579, 584-585`). The specification is now exact and implementable in
~15 lines; the check box must stay unchecked until it is.

### A4 — B3 confirmed and **widened**: the exported format is not a bit-permutation of the repo's format

The upstream packing is not "sub-byte codes packed into words" in the sense the repo
implemented:

- Only the **low `K` bits of each 16-bit state** are stored (`pack.cu:40`).
- Elements are packed **MSB-first** in **16 spans of 16 elements**, each uint16 emitted from a
  32-bit staging buffer (`pack.cu:28-52`).
- uint16 pairs are written with **`SWAP16`** (`pack.cu:55-56`).
- The decoder reconstructs each element's 16-bit state from a **16-bit window ending at bit
  `(t+1)*K`**, circular in the tile (`exl3xpu/ref.py:81-99`) — i.e. the packed stream is a
  **state-history encoding**, not an independent-symbol encoding.

`exl3_quant/bitpack.py` packs independent K-bit symbols LSB-first into uint32 words. It is
therefore **not a reordering of the upstream format; it is a different format**, and no amount of
bit-order adjustment will make it interoperable. The repo's packer must be redesigned as a
*state* packer.

### A5 — New finding: the encoder's output unit is wrong (independent of the codebook)

Upstream's encoder emits **16-bit states** (`output_indices[ri] = (uint16_t) encoded`,
`.cuh:298`), which are then packed. The repo's encoder emits **K-bit symbol indices**
(`optimal_indices`, `reference_encoder.py:174`). The exported tensor units differ, so the
repo's end-to-end test (`Viterbi → pack → unpack → reconstruct`) exercises a pipeline that
could never produce a valid checkpoint.

### A6 — New finding: the scratch estimator is the one component that survives audit unchanged

`get_temp_buffers` (`quantize.py:52-63`) confirms `temp_costs (B,2,65536>>K) float16` and
`temp_edges (B,tile_len,65536>>K) int16`, with `B = 256` or `max(256, 2*multi_processor_count)`
for `K >= 4`. The repo's `ScratchEstimator` matches this exactly, including the `S = 65536 >> K`
state dimension and the single-allocation bound. **No change required** beyond the
`K >= 4` batch-policy note.

### A7 — New finding: the LDLQ contract is now extractable, and it is not what one would guess

- `block_ldl` (`quantize.py:416-490`) is a **Cholesky** with a **retry-on-failure damping
  schedule** (`H.diagonal().add_(2*0.025*mean)`, up to 10 retries, `.py:447`), not a
  pre-damped LDL. Block-diagonal inverses are folded into `L` (`.py:476-481`).
- The error feedback is a **bottom-up span loop** with the trailing-weight update
  `prod_cache.addmm_(b_L.T, b_err)` (`.py:603-604`) and the corrected target
  `rows = b_weight[bi:bj] + compensation_term` (`.py:574`) — an **addition**, not a subtraction.
- Scale/sign vectors `su`/`sv` are part of the export format; `su` divides by
  `-codebook_scale = -1.24371088` (`.py:1215`).

None of this exists in the repo yet (P2-04/P2-08 are correctly unchecked).

### A8 — Scope answer: what is executable without CUDA/Arc

The audit changes the answer materially. **Network access to the upstream sources works from this
host**, so the following `tasks.md` items are executable offline and are the critical path:

- **P0-09 → P0-16** (source audit, dependency inventory, algorithm contract, output-tensor audit) — source reading; the contract is now written.
- **P0-01, P0-02, P0-03, P0-04, P0-06, P0-07, P0-08** — pinning, manifest, tolerances.
- **P0-28** — comparison harness; §7.4 of the contract already demonstrates the CPU-executable form.
- **P1-01, P1-02** — rewrite the reference encoder against the audited contract; add the permutation.
- **P2-04, P2-05, P2-06, P2-08, P2-09, P2-12** — LDLQ, Hadamard, block traversal, distortion (CPU).
- **P2-13, P2-14, P2-15** — rewrite the packer as a state packer (§7).
- **P2-16 (decode half)** — reconstruct via `exl3xpu/ref.py` on CPU.
- **P1-05, P1-06, P1-07, P1-12, P1-13, P1-15** — reconstruction, tie policy, batching, scratch arena with a fault-injecting allocator.
- **P3-01, P3-04 → P3-08** — conversion entry point, checkpoint integrity, resume semantics, fault injection (pure file I/O).
- **P4-10 tier 1** — CPU CI.

**Still blocked without CUDA:** P0-24, P0-25 (encoder ground truth), and P0-26/P0-27 provenance.
**Still blocked without Arc/XPU:** P0-18 → P0-23, P1-03/04/08/14/16 → P1-29, P2-01 → P2-03/07/11/17/18, P3-09 → P3-18.

The **one thing that cannot be substituted** is tier-4 acceptance (quantizer numerics vs. the CUDA
encoder). Offline work can prove *contract conformance* and validate the *decode* side bit-exactly;
it cannot prove that our encoder selects the same path as upstream's. That remains the first
hardware task.

### A9 — Artefacts landed with this addendum

| Artefact | Purpose |
|---|---|
| `docs/algorithm_contract.md` | The audited contract with line-level citations to the pinned upstream revision; 13 sections, 6 explicitly blocked open items |
| `exl3_quant/trellis_format.py` | Audited device-independent primitives: `codebook_lut` (mul1/mcg/3inst, fp16-exact), `tensor_core_perm` (+inverse), `pack_trellis`, `swap16_storage`, `decode_states`, `decode_tile_values` |
| `tests/test_trellis_format.py` | 17 tests: codebook exactness, permutation structure, packing layout identity for K∈{1,2,3,4,5,6,8}, the state-history property, and **negative tests** (flipped bit, reversed bit order, perturbed codebook) — the P0-28 harness requirement |

Test suite at this commit: **34 tests, all passing** — 12 original + 5 negative tests
added by the owner in `tests/test_negative.py` (commit `fc73c07`) + 17 trellis-format
tests. The new tests are *format-conformance* tests against the audited upstream
semantics, not encoder-parity tests — the latter still need P0-24.

Measured codebook facts now on record: mul1 → **913 distinct levels**, range
**[-3.453125, 3.34765625]**; 3INST → 10598 levels, [-3.957, 3.973]; mcg → 10746 levels,
[-3.949, 3.949]. The `mul1` codebook is **affine in the byte sum**, not a uniform grid.

### Reproduction

```text
# upstream source (org is turboderp-org)
https://raw.githubusercontent.com/turboderp-org/exllamav3/4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f/exllamav3/exllamav3_ext/quant/quantize_tiles_kernel.cuh
https://raw.githubusercontent.com/turboderp-org/exllamav3/4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f/exllamav3/exllamav3_ext/quant/codebook.cuh
https://raw.githubusercontent.com/turboderp-org/exllamav3/4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f/exllamav3/exllamav3_ext/quant/pack.cu
git clone --depth 1 https://github.com/sybil-solutions/exl3xpu.git      # CPU decode oracle: exl3xpu/ref.py
```

Environment note: the audit host has **numpy 2.0.2 / scipy 1.13.0, no torch**. The contract's
§7.4 experiment runs on numpy alone. `exl3xpu/ref.py` requires torch (CPU wheel) or a numpy
transcription.

---

## Review History

| Review ID | Timestamp (local) | Reviewed Commit | Verdict | File |
|---|---|---|---|---|
| `review-2026-10-20T13:50:25+02:00` | 2026-10-20 13:50:25 +02:00 | `bc048fe` | 4 blocking findings (B1–B4); 2 factual errors in progress.md; 6 code-level issues; core implementation solid but contract-unverified | `review.md` (this file) |
| ↳ addendum A1–A8 (same review ID) | 2026-10-20 (post-audit) | `bc048fe` + upstream `4d7b69c7…` | B1/B2/B3 upgraded to **confirmed incorrect**; new findings A4 (format is not a bit-permutation), A5 (wrong encoder output unit), A7 (LDLQ contract extractable); A8 lists the offline-executable task set | `review.md` + `docs/algorithm_contract.md` |

<!-- Future reviews: append a row above and create a new timestamped review file. -->
