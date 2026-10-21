# CUDA Ground-Truth Deferral

> **Decision ID:** `SKIP-CUDA-2026-10-21-01`
> **Recorded:** 2026-10-21 09:52:42 +02:00
> **Status:** **ACTIVE** — deferral in force; P0-24 / P0-25 remain open and blocked
> **Scope:** NVIDIA CUDA ground truth for the EXL3 tile encoder (upstream CUDA reference)
> **Waiver:** `DEV-2026-10-21-01` in [`docs/deviation_log.md`](./docs/deviation_log.md)
> **Affects:** `tasks.md` P0-24, P0-25, P0-27, G0-01, G1-01, G2-01, Gate 3 tier 4

---

## 1. Decision

**Defer the NVIDIA eGPU setup and continue Phase 2 / Gate 2 development on the CPU + Arc 140V
(XPU) path only.**

A capable GPU is already on hand — an **RTX 3060 12 GB** with a **Thunderbolt-to-PCIe dock**
and a **PSU** — and could be attached later. It is deliberately **not** being installed now.
The CUDA ground-truth requirement (P0-24 / P0-25) is therefore **not waived**: it stays open,
explicitly blocked, and must not be reported as satisfied.

This is a *deferral of execution*, not a relaxation of the acceptance criteria. Every artifact
that would carry CUDA-derived evidence must state which evidence class it rests on (§4).

**Rationale in one line:** the encoder's format, container, codebook and search-quality
properties are already verifiable against upstream-derived oracles on this host, so the
remaining CUDA-only gap (exact *path* parity) does not block the next development phase.

---

## 2. What the deferred setup would provide

Recorded now so the decision can be executed later without re-deriving the analysis.

### 2.1 Host capability (verified on this machine, 2026-10-21)

| Property | Value | Source |
|---|---|---|
| Host | Dell Latitude 5440, Intel Core i7-1365U (10c/12t), 32 GB RAM | `Win32_ComputerSystem` |
| OS | Windows 11 Enterprise, build 26100 | `Win32_OperatingSystem` |
| USB4/TB4 host controller | **Present** — "USB4(TM) Host Router (Microsoft)" | `Get-PnpDevice` |
| PCIe root ports | 2 | `Get-PnpDevice` |
| Free disk | 71.6 GB | `Get-PSDrive C` |
| NVIDIA GPU / driver / `nvcc` | **None** (Intel Iris Xe iGPU only) | `Win32_VideoController`, `nvidia-smi`, `nvcc` |

Windows eGPU support requires 1903 or newer ✓. The USB4 host router is the requirement for a
full-rate Thunderbolt link (a TB3-only link would halve the usable bandwidth).

### 2.2 GPU adequacy

| Property | Assessment |
|---|---|
| Architecture | Ampere GA106 (`sm_86`) — supports `lop3.b32` (`sm_80+`) and `__dp4a`, both used by upstream's `codebook.cuh` ✓. Upstream even carries an `sm_86`-specific code path in `mul_const_u32`, so the pinned kernel is expected to compile for it |
| VRAM | 12 GB — upstream's encoder needs `temp_edges = B·T·S·2` = **4 GiB @ K=1, 2 GiB @ K=2, 1 GiB @ K=3, 0.5 GiB @ K=4** (B=256, T=256, S = 65536>>K) plus `temp_costs ≤ 32 MiB`. 12 GB covers K=1…4 comfortably, and leaves room for a layer's weights and Hessians (e.g. d=4096 → 64 MiB fp32) |
| Bandwidth | Thunderbolt 4 ≈ 25–32 GB/s vs ≈ 64 GB/s for PCIe 4.0 x16. The `quantize_tiles` kernel is compute/VRAM-bound rather than link-bound, so fixture capture is practical; a full-model conversion would be roughly 2× slower than a desktop x16 slot |
| Power | Desktop RTX 3060 ≈ 170 W (laptop ≈ 130 W); the PSU is already available ✓ |
| Physical | Enclosure must accept the card's slot width and length; FE 3060 is 2-slot, some partner cards are 3-slot and may need a larger chassis or a riser |

**Verdict:** the deferred setup is adequate for P0-24 / P0-25 fixture capture at every K in
scope. Estimated effort when resumed: 1–3 hours setup + capture.

### 2.3 Known risks when resuming

1. **Windows eGPU enumeration order** — the NVIDIA driver should be installed *after* the
   eGPU is first attached; attaching an already-drivered card is often not enumerated.
2. **TB3 vs TB4 link rate** — confirm the negotiated link speed, not just the port label.
3. **Upstream build on Windows** — the pinned commit's CUDA extension needs MSVC build tools,
   `nvcc` and cuBLAS; upstream supports Windows builds, but the pinned revision has not been
   built on this host.
4. **Thermal/power throttling over a long capture run** — record clocks and power state with
   the fixtures.

---

## 3. Why deferring is acceptable at this point

The following evidence is already available **without any NVIDIA hardware**:

| Evidence | What it establishes | Where |
|---|---|---|
| Upstream's own CUDA-derived MSE ceilings (`max_mse_per_K`: 0.1 @ K=2–5, 0.07 @ K=6, 0.05 @ K=7, 0.04 @ K=8) reproduced on CPU | The reference encoder's **search quality** matches the bar upstream measured on its CUDA encoder | `tests/test_upstream_encode_oracle.py` |
| Upstream's tail-biting ring invariant `(states[0] & 0xFFFF) >> K == (states[255] & 0xFFFF) & ((1<<(16-K))-1)` | The **trellis wrap structure** is correct | same |
| Upstream's "ideal encoding" construction (`tests/test_quant_fn.py:98-112`) | The **state layout** (element symbol in the low K bits, history shifted up) | same |
| Bit-exact parity with the independent downstream decoder `exl3xpu.ref` across all 65 536 states × 3 codebooks × K ∈ {1..8} | The **codebook LUT, tile permutation and container layout** are byte-identical to an independent implementation | `tests/test_oracle_parity.py` |
| Independent re-transcription of the forward ACS pass, bit-for-bit agreement on the predecessor table and terminal costs | The **branch metric, fp16 accumulation, tie-breaking and masking rules** are implemented as specified | `tests/test_forward_semantics.py` |
| Encode → pack → decode → reconstruct round trip | The **exported container** is self-consistent and upstream-shaped | `tests/test_trellis_format.py` |

**What this evidence does *not* establish** (and must not be claimed):

- **Path parity** — that our encoder selects the *same* 16-bit states as upstream's CUDA
  kernel on the same input.
- **Open Item O-1** — the terminal `argmin_cost()` buffer parity in the upstream kernel remains
  unread from source alone.
- Upstream's own test-suite results (e.g. `tests/test_quant_fn.py::test_quant_dequant`), which
  are only executable against a CUDA build.

---

## 4. Evidence classes (use this wording in reports and gate documents)

| Class | Meaning | Permitted claim |
|---|---|---|
| **E1 — source-transcribed** | Behaviour derived from the pinned upstream source and re-implemented | "transcribed from upstream" |
| **E2 — independent-oracle** | Bit-exact agreement with an independent implementation (`exl3xpu.ref`) | "bit-exact vs the downstream oracle" |
| **E3 — upstream-threshold** | Meets upstream's own CUDA-derived ceilings/invariants | "meets upstream's CUDA-derived quality ceiling" |
| **E4 — CUDA-path-parity** | Identical selected states to upstream's CUDA kernel | "path-parity with upstream CUDA" — **NOT AVAILABLE while this deferral is active** |

Gate documents must label the highest class actually demonstrated. While this deferral is
active, no gate may claim E4, and any statement of "tier 4 passed" must be qualified as
**E3 (quality-bar parity), not E4 (path parity)**.

---

## 5. Consequences carried in the record

- `tasks.md` **P0-24**, **P0-25**, **P0-27** stay unchecked; **G0-01** and **G1-01** remain
  partial; **G2-01** cannot claim tier-4 exactness.
- `reports/gate_0.md`, `reports/gate_1.md` and any future gate report must not imply E4.
- **Open Item O-1** stays open in `docs/algorithm_contract.md` §12.
- The XPU encoder's own comparison obligations (P1-05, P2-09) remain satisfied by their
  internal-transcription evidence, as recorded.

---

## 6. Decision triggers — revisit this deferral when **any** of these fires

| ID | Trigger | Action |
|---|---|---|
| **T1** | The XPU encoder's reconstruction MSE exceeds upstream's `max_mse_per_K` for any K in scope | Suspect a real divergence; obtain CUDA ground truth to arbitrate |
| **T2** | A candidate checkpoint's reconstructed weights differ from the CPU reference beyond fp16 tolerance | Localise with CUDA states (encoder path vs container vs transform) |
| **T3** | A gate or reviewer requires explicit E4 (path-parity) evidence rather than E3 | Resume the eGPU setup and capture P0-24 fixtures |
| **T4** | A downstream runtime (`exl3xpu`, `exllamav3`) rejects or mis-decodes a produced checkpoint | Upstream's own CUDA test suite, run locally, is the fastest arbiter |
| **T5** | Upstream re-pins its commit (new encoder revision) | The thresholds and invariants in §3 must be re-derived; only then is a CUDA baseline comparable |
| **T6** | Upstream changes the encoder kernel, the codebook arithmetic or the container layout | E1–E3 evidence must be re-established before any further comparison |

T5 and T6 are the common triggers: the cost of a CUDA baseline is only justified while the
pinned revision stays the comparison target.

---

## 7. Resume checklist (execute when a trigger fires)

1. **Attach** the RTX 3060 to the Thunderbolt dock with the PSU connected; confirm the host
   negotiates a Thunderbolt 4 link.
2. **Install the NVIDIA driver** *after* first attach; verify:
   `nvidia-smi -L`, `nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv`,
   and `python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
3. **Install the CUDA Toolkit** (`nvcc`) matching the driver; verify `nvcc --version`.
4. **Build upstream at the pinned SHA** `4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f`
   (MSVC build tools + cuBLAS); verify the extension imports and `ext.quantize_tiles` runs.
5. **Cross-check the harness against upstream's own test** (`tests/test_quant_fn.py`) before
   trusting any capture.
6. **Capture fixtures**: seeded tiles (`torch.manual_seed(0)`, `randn((B,256))*scale`),
   recording per tile: the input tile, the reconstructed tile, the 16-bit states, and
   (where accessible) `temp_costs` / `temp_edges` snapshots — bounded, not multi-gigabyte
   dumps. Write `.npz` + SHA-256 + provenance (device, driver, clocks, upstream SHA).
7. **Register** the fixtures under `fixtures/` with checksums, and add a local test that
   compares the CPU/XPU encoder against them (states, path costs, reconstruction).
8. **Close** P0-24 / P0-25 / P0-27, resolve Open Item O-1, and update the affected gate
   reports to the E4 class.

---

## 8. Record

| Field | Value |
|---|---|
| Decision ID | `SKIP-CUDA-2026-10-21-01` |
| Recorded | 2026-10-21 09:52:42 +02:00 |
| Hardware deferred | RTX 3060 12 GB + Thunderbolt-to-PCIe dock + PSU (available, not installed) |
| Deferral owner | Project owner (host: `C:/Users/yi40/EXL3_Intel_Arc_quantizer`) |
| Reviewer | _pending_ |
| Review date | _pending_ |
| Supersedes | — |
| Superseded by | — |
| Reopen condition | Any trigger in §6 |

**Reversal note:** attaching the eGPU is a *reversible* operation — no repository state,
algorithm or acceptance criterion depends on it. Only the *evidence class* changes, from
E3 to E4, and the affected gate reports must be re-issued.
