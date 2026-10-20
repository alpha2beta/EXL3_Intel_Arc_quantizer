# EXL3 Intel Arc Quantizer: Implementation Tasks

**Basis:** `EXL3_Intel_Arc_quantizer.md`, version 3.2, reviewed October 20, 2026, amended by `docs/140v.md`.  
**Purpose:** Step-by-step implementation checklist for the quantization encoder and conversion pipeline, not an inference-engine rewrite.  
**Primary prototype platform:** Intel Core Ultra 7 258V with Intel Arc 140V integrated graphics (device ID `0x64A0`) and 32 GB shared system memory (30 GiB usable Linux pool). Discrete Arc devices (A770, B580, Pro B60) are retained as a separate, optional qualification track.  
**Evidence:** This checklist translates the attached assessment and 140V adaptation into concrete engineering work. It does not establish that a port has been implemented, benchmarked, or qualified without recorded evidence.

## How to use this checklist

- Work in task-ID order unless a dependency explicitly permits parallel work.
- Mark a task complete only when its implementation, applicable tests, and evidence are recorded.
- Record the owner, implementation commit, test command, result artifact, and unresolved limitations in the project tracker or task notes.
- A phase gate requires an explicit go/no-go decision. Do not treat a timebox expiring as a passed gate.
- Mark an inapplicable task with a written rationale; do not silently omit it or imply untested coverage.
- Keep exact checks for discrete representations separate from tolerance-based floating-point checks.
- All artifact paths below are proposed project-relative destinations, not files already created by this checklist.

## Scope and execution order

**Initial scope:** one dense transformer architecture, `mul1`, one integer encoder configuration, one primary Linux/XPU device (Intel Arc 140V / Core Ultra 7 258V / 32 GB shared memory), single-device execution, and unchanged LDLQ/format semantics.

**Capacity gate:** Tile encoding and complete single-layer conversion are the initial milestones. Attempt the selected full model only after measured layer, calibration, runtime, and system-memory requirements leave a documented operating reserve (preliminary 6–8 GiB reserve). Full-model fit and conversion speed remain unvalidated.

**Deferred:** alternate codebooks, fractional average-bitrate recipes, two-sided LDLQ, MoE, embedding/vision/MTP quantization, multi-GPU execution, Windows qualification, and discrete Arc qualification (moved to optional deployment gate). Preserve supported upstream behavior for components left unquantized; reject unsupported requested modes explicitly.

```text
Phase 0: pin scope, audit native code, establish fixtures and toolchain
  -> Gate 0
Phase 1: implement tile encoder, traceback, scratch management, profiling
  -> Gate 1
Phase 2: integrate complete single-layer conversion and export
  -> Gate 2
Phase 3: convert and qualify one complete supported model
  -> Gate 3
Phase 4: separately approved coverage and productionization
```

Calibration and factorization probes in Phase 0 may run in parallel with the encoder audit. Phase 2 integration depends on Gate 1. Do not parallelize dependency-sensitive calibration or LDLQ updates without proving independence.

## Phase 0: Source, scope, and toolchain audit

**Planning allowance from the assessment:** 1–2 person-weeks, not a delivery guarantee.  
**Source mapping:** assessment Sections 2–7, 8, and 11.

### 0.1 Freeze the compatibility contract

- [ ] **P0-01** Select one coherent upstream repository commit; reconcile the separately referenced quantizer and conversion-document revisions against that commit.
- [ ] **P0-02** Select the initial dense model architecture and exact source checkpoint/tokenizer revisions; record source weight dtypes.
- [ ] **P0-03** Select `mul1`, one supported integer encoder parameter K, and an explicit initial bitrate recipe; verify the mapping between K and exported representation in native code.
- [ ] **P0-04** Check the chosen model, codebook, bitrate, and dimensions against the pinned downstream decoder/runtime coverage before choosing the initial validation path.
- [x] **P0-05** Select the primary prototype device (Intel Core Ultra 7 258V with Arc 140V integrated graphics, 32 GB shared memory); record actual device identifiers, driver version, memory reporting semantics, and maintain discrete Arc (A770/B580/Pro B60) as a separate qualification track. *(Probed & recorded in `docs/compatibility_manifest.md`)*
- [ ] **P0-06** Pin calibration corpus revision, tokenizer settings, token/sample counts, sequence handling, masks, and RNG seeds.
- [ ] **P0-07** Define numerical acceptance criteria before tuning: exact discrete checks, floating-point absolute/relative tolerances, distortion checks, and full-model quality thresholds. Document how tolerances are justified.
- [x] **P0-08** Write `docs/compatibility_manifest.md` covering all pinned inputs, software versions, supported modes, integrated memory boundaries, and exclusions; do not claim universal model support. *(Published in `docs/compatibility_manifest.md`)*

### 0.2 Audit Python and native dependencies

- [ ] **P0-09** Trace the conversion entry point through calibration, statistics, transforms, factorization, tile encoding, traceback, feedback, packing, export, and resume.
- [ ] **P0-10** Inventory every custom-extension call and CUDA-specific stream, synchronization, allocation, event, and device-property dependency.
- [x] **P0-11** Audit the native encoder's state transitions, initialization, tail closure, codebook arithmetic, tie-breaking, metric precision, conversions, and any normalization/clamping. Do not infer arithmetic precision from FP16 scratch storage. *(Audited in `docs/algorithm_contract.md` §3–§5)*
- [x] **P0-12** Extract exact one-sided LDLQ equations, block sizes, factor orientation, diagonal normalization, damping, and residual-update order. *(Audited in `docs/algorithm_contract.md` §9)*
- [x] **P0-13** Extract transform dimensions, signs, scaling, tile permutation/inverse, and the order in which they are applied. *(Audited in `docs/algorithm_contract.md` §2, §8, implemented in `exl3_quant/trellis_format.py`)*
- [x] **P0-14** Audit output tensor names, dtypes, shapes, scales, metadata, bit ordering, and all native versus device-independent packing operations. *(Audited in `docs/algorithm_contract.md` §7, §10, implemented in `exl3_quant/trellis_format.py`)*
- [ ] **P0-15** Classify each dependency as reusable, XPU-native, custom-port-required, explicit CPU fallback, or unsupported; include owner and validation method.
- [x] **P0-16** Save `docs/native_dependency_audit.md` and `docs/algorithm_contract.md`; leave unresolved native behavior explicitly blocked rather than inventing defaults. *(Contract published in `docs/algorithm_contract.md`)*

### 0.3 Establish a reproducible environment

- [ ] **P0-17** Pin Linux, Intel driver/Compute Runtime, Level Zero, compiler/SYCL (oneAPI 2026.0), PyTorch XPU (2.14.0+xpu), Python, and build dependencies on the actual 140V platform; record versions and installation provenance.
- [ ] **P0-18** Build and execute a minimal custom XPU extension through the intended framework integration; verify device selection, output correctness, queue ordering, and tensor lifetime.
- [ ] **P0-19** Query supported subgroup sizes, local memory, work-group limits, and maximum allocation size; save device capabilities in `artifacts/environment/`.
- [ ] **P0-20** Test representative scratch allocations through the actual framework allocator; start small and bound tests by available memory and a configurable reserve (e.g. 6–8 GiB).
- [ ] **P0-21** Probe required calibration and factorization operations/dtypes on Arc 140V; record unsupported operations and intentional CPU fallbacks.
- [ ] **P0-22** Test any proposed Cholesky-derived LDLQ quantities against the extracted equations; benchmark CPU fallback transfer, memory, and wall-clock costs on the shared memory bus without silently changing the algorithm.
- [ ] **P0-23** Add an environment smoke test and reproducible build/test instructions; verify a clean rebuild.

### 0.4 Create reference fixtures and a test harness

- [ ] **P0-24** Generate or obtain trusted fixtures from the pinned CUDA reference: activations, Hessians, factors, transformed/permuted tiles, reconstructed tiles, feedback outputs, and packed tensors (local host need not contain an NVIDIA GPU).
- [ ] **P0-25** Capture representative native encoder costs/edges where practical; avoid dumping full multi-gigabyte scratch buffers when small fixtures or selected snapshots suffice.
- [ ] **P0-26** Include zeros, repeated values, outliers, seeded random tiles, ties/near-ties, boundary states, and representative real-weight tiles. Define expected behavior for invalid inputs from the audited contract.
- [ ] **P0-27** Attach checksums, provenance, settings, shapes, dtypes, and expected results to fixtures; do not treat fixtures from another commit as equivalent without verification.
- [x] **P0-28** Implement a reference comparison harness with separate exact and tolerance-based assertions; prove it detects deliberately perturbed values, bit order, and metadata. *(Implemented in `tests/test_negative.py` and `tests/test_trellis_format.py`)*
- [ ] **P0-29** Record unavailable reference coverage as a blocker or explicit limitation; do not mark unexecuted CUDA comparisons as passed.

### 0.5 Arc 140V Prototype Profile Checkboxes

- [x] **V140-01** Save actual hardware, software, allocation-limit, and memory-reporting observations in `docs/compatibility_manifest.md`. *(Recorded in `docs/compatibility_manifest.md`)*
- [ ] **V140-02** Establish an idle system baseline and configurable memory-pressure abort policy before large scratch tests.
- [ ] **V140-03** Validate scratch sizing at small B; increase batch capacity under measured system and runtime constraints.
- [ ] **V140-04** Demonstrate that CPU/GPU accounting does not double-count shared backing or omit distinct copies.
- [ ] **V140-05** Compare traceback candidates with identical fixtures and complete latency/pressure measurements.
- [ ] **V140-06** Pass the complete single-layer numerical and memory gate on 140V.
- [ ] **V140-07** Decide full-model admission from measured calibration/layer/runtime peaks, not the 32 GB label.
- [ ] **V140-08** Run full-model quality and runtime interop tests only for the supported configuration; publish remaining gaps.
- [ ] **V140-09** Publish local qualification separately from any later A770/B580/Pro B60 results.

### Gate 0: Audit readiness

- [ ] **G0-01** Confirm the compatibility manifest, algorithm contract, dependency inventory, working extension skeleton, and fixture harness are complete.
- [ ] **G0-02** Confirm factorization/operator coverage or a measured, explicit fallback exists for the initial scope.
- [ ] **G0-03** Record go/no-go, blockers, owners, and the refined Phase 1 plan in `reports/gate_0.md`.

## Phase 1: Tile encoder and scratch-memory prototype

**Depends on:** Gate 0.  
**Planning allowance from the assessment:** 2–4 additional person-weeks, subject to audit findings.  
**Source mapping:** assessment Sections 6–8, 10, and 11.

### 1.1 Implement a correctness-first encoder

- [x] **P1-01** Implement an inspectable reference encoder for the selected codebook/K using the audited state and boundary rules. *(Contract-conformant `ReferenceTrellisEncoder` implemented in `exl3_quant/reference_encoder.py` from `quantize_tiles_kernel.cuh:23-313`; verified in `tests/test_encoder.py`)*
- [x] **P1-02** Implement and test the exact tile permutation/inverse and procedural codebook arithmetic against fixtures. *(Transcribed in `exl3_quant/trellis_format.py` and verified in `tests/test_trellis_format.py` and `tests/test_encoder.py`)*
- [ ] **P1-03** Implement XPU forward ACS search with explicit precision/conversion policy; initially prefer understandable storage and synchronization over speculative fusion.
- [ ] **P1-04** Implement predecessor recording and a GPU-resident traceback baseline; preserve state and edge conventions. *(CPU reference baseline in `exl3_quant/reference_encoder.py`; GPU-resident baseline pending XPU kernel)*
- [x] **P1-05** Reconstruct encoded tiles and compare path validity, objective/distortion, and output weights against the reference. *(Verified in `tests/test_encoder.py` across K=2, 3, 4, CB_MUL1, CB_MCG, CB_3INST)*
- [ ] **P1-06** Test ties separately: allow alternate valid paths only under the defined objective/quality criteria; do not require identical state indices merely because floating-point paths differ.
- [ ] **P1-07** Test single tiles, multiple batches, partial final batches, repeated calls, and the supported input validation rules; reject unsupported shapes/configurations explicitly.
- [ ] **P1-08** Stress asynchronous invocation and tensor lifetime; test non-default framework streams/queues if supported, or document and enforce the initial restriction.

### 1.2 Implement bounded scratch management

- [x] **P1-09** Implement a checked scratch-size estimator using the pinned allocator contract; validate shapes/dtypes against native code before relying on it. *(Implemented in `exl3_quant/scratch_estimator.py`)*
- [x] **P1-10** Unit-test the estimator against the allocation examples below; distinguish GiB/MiB from decimal GB/MB. *(Verified 4/4 fixtures in `tests/test_scratch.py`)*
- [x] **P1-11** Implement adaptive scratch batch capacity B using measured memory availability, a configurable reserve, and validated single-allocation limits; do not substitute Xe-core count for CUDA multiprocessor count. *(Implemented in `ScratchEstimator.calculate_max_batch_size`)*
- [ ] **P1-12** Implement a bounded scratch cache or reusable arena with safe event-aware reuse/eviction; prevent unbounded residency across K/device changes. Account for shared-backing vs distinct memory copies on 140V.
- [ ] **P1-13** Test allocation failure handling and smaller-batch retries without changing tile semantics or silently dropping work; reject a configuration cleanly if the minimum valid batch cannot fit within the available memory envelope (leaving system reserve intact).
- [ ] **P1-14** Sweep B values such as 1, 16, 64, 128, and 256 where feasible; compare numerical results, actual allocations, throughput, and reserved memory under OS paging/pressure monitoring.
- [ ] **P1-15** Repeatedly allocate, reuse, release, and alternate supported configurations; verify no growing memory high-water mark, swap growth, or unsafe reuse.
- [ ] **P1-16** Log full live-memory context (framework device allocations, OS available RAM, swap activity) rather than claiming device fit from scratch size alone.

**Estimator regression fixtures from the assessment; not measured peak VRAM:**

```text
S = 65536 >> K; T = 256; B = 256
costs = B * 2 * S * 2 bytes
edges = B * T * S * 2 bytes

K=1: S=32768, edges=4 GiB,   costs=32 MiB
K=2: S=16384, edges=2 GiB,   costs=16 MiB
K=3: S=8192,  edges=1 GiB,   costs=8 MiB
K=4: S=4096,  edges=0.5 GiB, costs=4 MiB
```

Testing these arithmetic fixtures does not require allocating every configuration or expanding the initial encoder scope.

### 1.3 Compare traceback placements

- [ ] **P1-17** Benchmark GPU-resident search/traceback as the baseline on 140V, measuring search and traceback timings separately.
- [ ] **P1-18** Prototype device-edge copy plus CPU traceback; measure data-movement, synchronization, host RAM, CPU time, and shared-memory contention rather than assuming discrete PCIe behavior.
- [ ] **P1-19** Prototype direct host-accessible edge writes only where USM/host-unified memory is supported; measure link traffic, kernel stalls, and end-to-end latency.
- [ ] **P1-20** Compare identical fixtures and objectives across all supported candidates; document an unsupported candidate rather than forcing an unsafe workaround.
- [ ] **P1-21** Select a default from measured total latency, memory, and complexity; save `reports/traceback_decision.md`.

### 1.4 Profile and tune without changing semantics

- [ ] **P1-22** Record cold compilation, warmup, search, traceback, and transfer time separately under recorded AC power and thermal state; synchronize timing boundaries appropriately.
- [ ] **P1-23** Report weights/s and tiles/s with K, codebook, batch capacity, and shape; report state-transition rate only after defining and auditing the executed-work count.
- [ ] **P1-24** Inspect generated code/resource reports for register use, spills, local memory (SLM 128KB limit), barriers, and memory access patterns on Arc 140V.
- [ ] **P1-25** Compare supported subgroup/work-group geometries (SIMD16 vs SIMD32 on Arc 140V); do not mandate SIMD16 without empirical evidence.
- [ ] **P1-26** Compare global-memory, SLM, and register-resident cost designs where feasible; account for all local allocations and measured occupancy.
- [ ] **P1-27** Compare supported GRF modes and memory coalescing/vectorization options; retain only measured improvements that pass regression tests.
- [ ] **P1-28** Measure dispatch durations and chunk tile batches if needed; treat any under-500-ms budget as a project choice, not a universal watchdog guarantee.
- [ ] **P1-29** Re-run numerical, boundary, and memory tests after each optimization; preserve a known-correct baseline.

### Gate 1: Encoder readiness

- [ ] **G1-01** Confirm selected tile encoding passes reference path, reconstruction, and distortion tests under the agreed policy.
- [ ] **G1-02** Confirm bounded scratch behavior, allocation-failure handling, and measured throughput/resource reports.
- [ ] **G1-03** Record the traceback decision, remaining risks, and go/no-go in `reports/gate_1.md`; estimate Phase 2 from measured evidence.

## Phase 2: Complete single-layer conversion

**Depends on:** Gate 1; calibration/factorization probes from Phase 0.  
**Effort:** Re-estimate at Gate 1.  
**Source mapping:** assessment Sections 3–5, 7–9, and 11.

### 2.1 Integrate calibration and numerical preprocessing

- [ ] **P2-01** Execute one complete selected transformer decoder layer on XPU; verify operator coverage and log intentional CPU fallbacks/transfers within the shared 32 GB physical memory budget.
- [ ] **P2-02** Capture activations and accumulate Hessian statistics with upstream normalization, masks, dtype policy, and batching.
- [ ] **P2-03** Compare activations/Hessians with fixtures; measure live activation and statistics memory.
- [ ] **P2-04** Implement upstream regularization, scaling, factorization-derived quantities, and diagonal normalization without altering the quantization objective.
- [ ] **P2-05** Test inactive/ill-conditioned feature cases and factorization failure handling; measure CPU fallback rather than assuming it is zero-cost.
- [ ] **P2-06** Implement exact Hadamard/sign/scaling and tile-permutation conventions in a clear reference path; compare intermediate tensors.
- [ ] **P2-07** Optimize transforms only after parity; validate any fusion against the unfused path.

### 2.2 Integrate blockwise LDLQ and the encoder

- [ ] **P2-08** Implement the audited block traversal: corrected target tile, encoding, reconstruction, and trailing-weight feedback.
- [ ] **P2-09** Compare each block's inputs, factors, reconstructed outputs, and post-feedback trailing weights with reference fixtures.
- [ ] **P2-10** Preserve dependencies between blocks and projections; prove independence before any concurrency.
- [ ] **P2-11** Integrate scratch reuse with layer/module lifetimes and library workspaces; verify synchronization before reuse or eviction.
- [ ] **P2-12** Measure factorization residuals, reconstruction error, tile distortion, and layerwise proxy loss using the pinned objective definition.

### 2.3 Export and validate one layer

- [ ] **P2-13** Reuse device-independent upstream packing/serialization where possible; port only audited device-specific operations. *(Audited layout implemented in `exl3_quant/trellis_format.py`)*
- [ ] **P2-14** Validate output keys, shapes, dtypes, scales, signs, codebook metadata, bit ordering, and tensor permutations.
- [ ] **P2-15** Add exact synthetic pack/unpack round trips and malformed/truncated-output rejection tests. *(Audited state-history round trips tested in `tests/test_trellis_format.py`)*
- [ ] **P2-16** Export and reload the layer artifact; compare reconstructed weights using the pinned reference decoder and a supported XPU decoder (`exl3xpu`).
- [ ] **P2-17** Profile full-layer peak allocated/reserved/device-wide VRAM, host RSS, and workspace; account for simultaneous live tensors and fragmentation without double counting shared physical memory.
- [ ] **P2-18** Run repeated layer conversions to detect memory growth; verify the chosen memory reserve and device fit experimentally under sustained execution.

### Gate 2: Single-layer readiness

- [ ] **G2-01** Pass applicable acceptance tiers 1–4: structure, packing, reconstruction, and quantizer numerics.
- [ ] **G2-02** Confirm a complete decoder layer converts within measured memory limits without sustained paging, with all fallback and unsupported-mode restrictions documented.
- [ ] **G2-03** Save `reports/gate_2.md` with artifact checksums, numerical/resource results, go/no-go, and a Phase 3 estimate.

## Phase 3: Bounded full-model conversion and qualification

**Depends on:** Gate 2.  
**Effort:** Re-estimate at Gate 2.  
**Source mapping:** assessment Sections 8–11.

### 3.1 Orchestrate conversion and recovery

- [ ] **P3-01** Add a bounded conversion entry point/configuration for the selected model, codebook, and recipe; reject unsupported modes before expensive work.
- [ ] **P3-02** Preserve upstream calibration propagation and conversion order across layers; handle unchanged model components according to the pinned format contract.
- [ ] **P3-03** Integrate layer/shard streaming and memory-mapped loading where supported; measure real host memory pressure during streaming rather than assuming mapping guarantees capacity.
- [ ] **P3-04** Preserve working-directory, checkpoint interval, output sharding, and explicit resume semantics; record the compatibility manifest with progress state.
- [ ] **P3-05** Implement integrity checks and recoverable checkpoint publication; ensure interrupted writes cannot be mistaken for complete artifacts.
- [ ] **P3-06** Inject interruptions at representative calibration, encoding, packing, and checkpoint-write boundaries; verify safe restart without dropped/duplicated work.
- [ ] **P3-07** Reject incompatible resume arguments or require an explicitly documented safe override; verify source/fixture/configuration identity.
- [ ] **P3-08** Test insufficient disk space, missing shards, corrupted checkpoints, and minimum-batch allocation failure under measured host memory pressure.

### 3.2 Run the first complete model

- [ ] **P3-09** Convert one supported dense model at the fixed recipe; save logs, checksums, and full environment metadata. Attempt full model only after measured layer, calibration, runtime, and system-memory requirements leave a documented operating reserve.
- [ ] **P3-10** Measure total/per-stage wall-clock, compilation, transfers, throughput, peak device/host memory, and disk workspace under disclosed power/thermal and background-load conditions.
- [ ] **P3-11** Verify all exported tensors and reconstruction coverage; do not infer compatibility from plausible generated text.
- [ ] **P3-12** Evaluate teacher-forced logits/KLD, fixed-corpus NLL/perplexity, and selected task metrics against the matched CUDA-quantized reference and floating-point baseline.
- [ ] **P3-13** Load and execute in the pinned supported downstream runtime(s) (`exl3xpu`); document unavailable runtime coverage explicitly.
- [ ] **P3-14** Repeat conversion or representative deterministic stages to characterize reproducibility and explain acceptable tied-path differences.

### 3.3 Optional Discrete Arc Deployment Gate

- [ ] **P3-15** Rebuild/probe the pinned stack on discrete Arc targets (A770, B580, Pro B60) and capture device-specific capability/allocation limits.
- [ ] **P3-16** Re-run tile fixtures, scratch stress tests, full-layer tests, and declared full-model qualification on the discrete device.
- [ ] **P3-17** Tune device-specific batch/geometry settings without changing algorithm semantics; rerun numerical and quality acceptance tests.
- [ ] **P3-18** Publish a tested model/codebook/K/device/runtime matrix with explicit exclusions and measured results.

### Gate 3: Initial supported release

- [ ] **G3-01** Pass all six acceptance tiers for the declared supported configuration, including quality and runtime interoperability.
- [ ] **G3-02** Confirm fault recovery, bounded memory, repeatability; permit an explicitly 140V-only supported release if all local criteria pass (discrete Arc qualification remains an independent deployment gate).
- [ ] **G3-03** Publish reproducible build/conversion/test instructions, known limitations, benchmark methodology, and release artifact checksums.
- [ ] **G3-04** Record `reports/gate_3.md`; claim support only for configurations actually qualified, not arbitrary Hugging Face models.

## Phase 4: Separately scoped extensions and productionization

**Depends on:** Gate 3 and explicit approval of each extension. These tasks are not prerequisites for the initial bounded prototype.

- [ ] **P4-01** Add alternate codebooks one at a time, each with native arithmetic audit, fixtures, packing tests, and quality qualification.
- [ ] **P4-02** Add fractional average-bitrate recipes; validate actual integer encoder dispatch, allocation-cache transitions, budget allocation, and exported metadata.
- [ ] **P4-03** Add two-sided LDLQ/output-Hessian modes with their own numerical contract and reference tests.
- [ ] **P4-04** Add embedding, vision, and MTP conversion only for explicitly supported architectures; test their distinct calibration/packing behavior.
- [ ] **P4-05** Add MoE conversion with routing/calibration coverage and expert-specific resource accounting.
- [ ] **P4-06** Design a multi-device dependency graph for independent tensors/tile batches; preserve sequential calibration and within-block feedback dependencies.
- [ ] **P4-07** Measure multi-device transfer bytes, utilization, load balance, per-device memory, and end-to-end speedup; do not assume zero interconnect cost or independent odd/even layers.
- [ ] **P4-08** Qualify larger models and additional GPU families separately; measure host RAM, disk, and device fit for each workload.
- [ ] **P4-09** Qualify Windows separately, including build/runtime support and measured dispatch/watchdog behavior.
- [ ] **P4-10** Add CI tiers: CPU/reference tests, XPU tile tests, full-layer tests, and scheduled full-model/quality runs with owned fixtures.
- [ ] **P4-11** Define maintenance ownership, dependency-update policy, regression triage, and supported-version lifecycle.

## Acceptance evidence and task completion record

Use this record for completed work; do not fabricate measurements or mark tasks passed based only on compilation.

```text
Task ID:
Owner:
Implementation commit:
Pinned environment / device:
Test command or reproducible procedure:
Input fixture / model / configuration:
Expected criterion:
Observed result:
Artifact path and checksum:
Limitations / follow-up IDs:
Reviewer / review date:
```

**Six acceptance tiers:** (1) structure; (2) exact bit-packing round trip; (3) tile/weight reconstruction; (4) quantizer numerical fidelity; (5) model quality; (6) downstream runtime interoperability. Phases 1–2 use applicable lower tiers; full-model qualification must not skip tiers 5–6.

## Source traceability and guardrails

This checklist is derived from `EXL3_Intel_Arc_quantizer.md` version 3.2. The source document's references S1–S3 cover the quantizer, converter, and downstream runtime; S4–S7 cover architecture/capability/GRF/allocation references. This task list does not independently verify or expand those external claims.

- Preserve the source document alongside this file and record its checksum in the project manifest.
- Pin one coherent upstream commit rather than treating separate file-history revisions as a tested build.
- Treat scratch arithmetic as a sizing fixture, not a full-pipeline VRAM guarantee.
- Treat native arithmetic precision, executed search work, library support, and resource occupancy as audit/test questions.
- Do not replace LDLQ with a different algorithm under the label of backend portability.
- Do not infer production timelines or cross-vendor parity from prototype timeboxes.
- Keep inference-runtime use limited to reconstruction/quality/interoperability validation; the implementation deliverable is the quantizer.
