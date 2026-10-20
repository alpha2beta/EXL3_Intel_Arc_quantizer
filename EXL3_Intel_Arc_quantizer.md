# Feasibility Analysis: Porting the EXL3 Quantizer to Intel Arc GPUs

**Version:** 3.2, source-informed feasibility assessment and prototype plan  
**Review date:** October 20, 2026  
**Primary targets:** Intel Arc A770 (16 GB, Xe-HPG) and Arc B580 / Pro B60 (12 GB / 24 GB, Xe2-HPG); other Intel GPU families require separate qualification.  
**Scope:** Offline conversion of floating-point model weights into compatible EXL3 checkpoints, covering calibration, numerical preprocessing, trellis encoding, error feedback, bit-packing, and resumable orchestration.  
**Verdict:** **Technically plausible; production feasibility and effort are gated on a native source audit and a single-layer prototype.**

> **Evidence boundary:** This is a source-informed engineering assessment, not a benchmarked port. No XPU quantization kernels were compiled or executed for this assessment. The reviewed upstream Python source and conversion documentation establish concrete algorithmic and memory requirements, which must be paired with an audit of all native C++/CUDA kernels at a pinned commit. Statements marked *Proposal* define the target engineering design.

## Contents

1. [Executive summary](#1-executive-summary)
2. [Scope and compatibility contract](#2-scope-and-compatibility-contract)
3. [Quantization architecture and dependencies](#3-quantization-architecture-and-dependencies)
4. [Calibration and Hessian capture](#4-calibration-and-hessian-capture)
5. [Numerical preprocessing and error feedback](#5-numerical-preprocessing-and-error-feedback)
6. [Trellis encoding and traceback](#6-trellis-encoding-and-traceback)
7. [Scratch memory and total capacity](#7-scratch-memory-and-total-capacity)
8. [Packing and downstream validation](#8-packing-and-downstream-validation)
9. [Orchestration and multi-device scaling](#9-orchestration-and-multi-device-scaling)
10. [Evaluation framework and risks](#10-evaluation-framework-and-risks)
11. [Phased roadmap and resources](#11-phased-roadmap-and-resources)
12. [Conclusion](#12-conclusion)
13. [Changes and sources](#13-changes-and-sources)

---

## 1. Executive summary

The EXL3 quantizer is an offline conversion pipeline, not an inference backend. Porting it to Intel Arc GPUs requires preserving calibration accuracy, exact blockwise LDLQ error-feedback semantics, high-throughput Viterbi trellis encoding, bounded scratch-memory allocation, and bit-exact checkpoint formatting. Upstream conversion exposes controls for calibration datasets, bitrate recipes, codebook selection, Hessian modes, checkpoint/resume, and multi-device execution [S1][S2].

The existing `exl3xpu` project provides an Intel XPU inference implementation and reference weight-reconstruction tests. While it serves as a downstream compatibility oracle within its supported configurations, it does not implement the upstream quantization search or preprocessing pipeline. Because `exl3xpu` was archived on October 12, 2026, its commit and environment must be pinned if used as a validation target [S3].

**Recommended decision:** Fund a bounded Phase 0 source audit and Phase 1 tile-encoder and scratch-memory prototype, followed by complete single-layer conversion in Phase 2. Commit to full-model production timelines only after measuring numerical fidelity, scratch allocation behavior, SYCL/XPU encoder throughput, and full-layer peak VRAM on target hardware.

---

## 2. Scope and compatibility contract

### 2.1 Initial target scope

- **Architecture:** One explicitly supported dense transformer architecture.
- **Codebook & Encoder:** One codebook and integer encoder configuration initially (`mul1` is the upstream default) [S2].
- **Hardware:** Single-GPU conversion qualified on one primary Arc device (Xe-HPG or Xe2-HPG), followed by a cross-generation portability check.
- **Format Contract:** Drop-in EXL3 checkpoint compatibility without altering the upstream quantization objective.
- **Milestone Structure:** Complete single-layer conversion and numerical verification prior to full-model orchestration.

**Deferred to Phase 4 (Out of initial scope):** Alternate codebooks, fractional average-bitrate recipes, two-sided LDLQ modes, Mixture-of-Experts (MoE) routing, embedding/vision/MTP quantization, multi-GPU scaling, and Windows WDDM qualification [S2].

### 2.2 Compatibility manifest

**Proposal:** Lock a reproducible compatibility manifest prior to writing kernel code:
- Single unified upstream `exllamav3` repository commit (reconciling `quantize.py` rev `4d7b69c7` and `convert.md` rev `1d90befe`) [S1][S2].
- Source FP16/BF16 model revision, calibration corpus, tokenizer, and fixed RNG seeds.
- Target codebook identifier, encoder parameter $K$, bitrate recipe, and output safetensors schema.
- Pinned downstream validation runtime (`exl3xpu` or reference CUDA decoder fixtures) [S3].

---

## 3. Quantization architecture and dependencies

The following conceptual pipeline illustrates the dataflow; exact transform placement and factorization order must strictly follow the pinned upstream commit [S1][S2]:

```text
Floating-point checkpoint + calibration data + conversion settings
    -> calibration forwards and Hessian statistics capture
    -> scaling, regularization, Hadamard transforms, and LDLQ factorization
    -> blockwise quantization loop:
         prepare error-corrected target tile
         encode trellis (forward ACS) and recover selected path (traceback)
         reconstruct quantized tile
         apply blockwise error feedback to trailing weights
    -> pack quantized tensors and safetensors metadata
    -> layer checkpoint / disk export
    -> structural, numerical, and downstream runtime validation
```

### 3.1 Native dependency audit

The upstream quantizer imports a custom C++/CUDA extension, invokes explicit CUDA synchronization primitives, queries `multiprocessor_count` for scratch buffer sizing, and applies a hardware-specific $16 \times 16$ tile permutation (`tensor_core_perm`) expected by the CUDA trellis encoder [S1]. Porting therefore requires a systematic native call-site inventory rather than a simple PyTorch device-string substitution.

**Proposal:** Audit and classify every upstream call site across eight workstreams:

| Component | Audit question | Initial implementation direction |
| --- | --- | --- |
| **Calibration execution** | Which model operators or attention backends rely on CUDA-only paths? | Verify PyTorch XPU operator coverage; implement explicit fallbacks |
| **Statistics & dense algebra** | Are required accumulation dtypes and Cholesky/LDLQ routines supported on oneMKL? | Benchmark native XPU linear algebra vs. pinned-host CPU fallback |
| **Transforms & scaling** | Where are Hadamard transforms and `tensor_core_perm` applied? | Implement verified reference first; fuse into SYCL kernels later |
| **Trellis encoding** | What are the exact state transition rules, codebook arithmetic, and precision semantics? | Build inspectable reference encoder, then native SYCL/XPU kernel |
| **Traceback & reconstruction** | How are predecessor edges indexed and decoded into reconstructed tiles? | Preserve exact edge conventions before optimizing memory placement |
| **Error feedback** | What block sizes, triangular factors, and update orders govern residual propagation? | Preserve equations, block ordering, and numerical policy; compare floating-point intermediates under justified tolerances |
| **Packing & export** | Which bit-packing permutations occur in C++ vs. device-independent Python? | Reuse upstream Python serialization and packing wherever device-agnostic |
| **Runtime utilities** | Where does orchestration assume `torch.cuda` streams, allocators, or events? | Introduce an explicit `torch.xpu` / device abstraction layer |

---

## 4. Calibration and Hessian capture

For standard one-sided input statistics, the second-moment accumulation over calibration batches $b$ is:

$$H = \sum_{b} X_b^T X_b$$

Normalization, accumulation precision (typically FP32 or FP64 accumulators), sequence masking, and layer-to-layer activation propagation must match the pinned upstream implementation. Upstream also supports precomputed input/output Hessian artifacts and two-sided modes [S1][S2].

**Validation protocol:**
1. Execute forward passes for the target model architecture on PyTorch XPU.
2. Verify operator coverage and detect unintended host-device transfers; explicitly document and time any intentional CPU fallback.
3. Compare captured activation statistics and accumulated $H$ matrices against CUDA reference fixtures under defined FP32/FP64 relative error tolerances.
4. Measure peak activation VRAM alongside accumulator workspace.

### 4.1 Hessian memory scaling

A single dense FP32 square Hessian of input dimension $d$ requires:

$$M_H = 4d^2 \text{ bytes}$$

- $d = 4096 \implies 64\text{ MiB}$
- $d = 8192 \implies 256\text{ MiB}$
- $d = 16384 \implies 1024\text{ MiB}$ ($1\text{ GiB}$)

Total memory during calibration and factorization must account for the raw Hessian, regularized copies, triangular factorization outputs, workspace buffers, and live layer activations.

---

## 5. Numerical preprocessing and error feedback

### 5.1 Preserving LDLQ semantics

Upstream conversion distinguishes standard one-sided LDLQ from modes incorporating output-side factors and two-sided LDLQ, which directly alter weight scaling and rounding trajectories [S2].

**Proposal:**
- Extract the exact damping/regularization schedule, block factorization equations, diagonal normalization, and trailing-weight residual update rules from upstream source.
- If using standard oneMKL `potrf` (Cholesky) to derive the $L$ and $D$ factors, prove algebraic equivalence to the upstream LDLQ block formulation before integration.
- Treat any modification to damping constants or residual update order as an algorithmic deviation requiring full perplexity regression testing.
- Benchmark `torch.linalg` / oneMKL on XPU against a pinned-memory CPU factorization fallback, recording PCIe transfer latency and wall-clock impact.

### 5.2 Hadamard transforms and tile permutations

Upstream `quantize.py` includes Hadamard helpers and applies `tensor_core_perm` to reorder $16 \times 16$ tiles prior to encoding [S1]. 

**Proposal:** Preserve exact transform dimensions, sign conventions, scaling factors, and permutation indices so that exported checkpoints remain compatible with existing CUDA and XPU inference kernels [S1][S3]. Implement a verified PyTorch reference path first, then optimize using SYCL subgroup shuffles (targeting SIMD16/SIMD32 on Xe/Xe2) and Shared Local Memory (SLM) once numerical parity is established.

---

## 6. Trellis encoding and traceback

### 6.1 Encoder architecture and state space

Upstream `quantize_tiles` operates on batches of $16 \times 16$ weight tiles flattened into $T = 256$ elements following the encoder's permutation order [S1]. For encoder parameter $K$, the allocated scratch state dimension is:

$$S = 65536 \gg K = 2^{16 - K}$$

At $K = 2, 3, 4$, the allocated scratch state dimensions are $S = 16384, 8192, 4096$, respectively, with $T = 256$ tile elements. Path-cost scratch storage (`temp_costs`) uses `float16`, and predecessor scratch storage (`temp_edges`) uses `int16`. These shapes and dtypes do not establish every intermediate arithmetic precision or the exact number of executed state updates; verify those in the native encoder [S1].

**Required source audit items:**
- State transition topology and procedural codebook value generation (`mul1` arithmetic).
- Initial state boundary conditions and tail closure rules across the 256-step tile.
- Branch-metric arithmetic, intermediate accumulation precision, storage conversions, any normalization or clamping, and tie-breaking rules; do not infer implementation from scratch storage dtype [S1].
- Exact mapping between encoder parameter $K$, packed output bitwidth, and fractional bitrate recipes.

### 6.2 XPU kernel design considerations

When porting the forward Add-Compare-Select (ACS) loop and traceback to SYCL / Intel XPU:
- **SLM footprint and resource scope:** Calculated FP16 ping-pong cost storage for one tile is $4S$ bytes: $64\text{ KiB}$ at $K=2$, $32\text{ KiB}$ at $K=3$, and $16\text{ KiB}$ at $K=4$. Intel documents up to $128\text{ KB}$ SLM per Xe-HPG Xe-core, which is not a universal per-work-group limit or a Xe2 guarantee [S4]. **Proposal:** Query device local-memory and work-group limits, account for other kernel-local allocations, and inspect compiled resource use before evaluating SLM-resident costs [S5]. Measure occupancy, barriers, and runtime rather than inferring efficiency from capacity alone.
- **Global scratch bandwidth:** The allocated predecessor table contains $256S \times 2$ bytes, or $8\text{ MiB}$ per tile at $K=2$, across the complete 256-element tile, not at each step [S1]. **Proposal:** Measure actual reads and writes, including repeated search passes. Benchmark coalescing, vector widths, and alignment on each target; do not prescribe a universal 64-byte transaction or cache-line requirement.
- **GRF mode:** **Proposal:** Compare default, automatic, and supported larger-GRF configurations where available. More registers may reduce spills but can reduce resident-thread capacity; large-GRF mode is not inherently high occupancy. Confirm compiler/backend support and generated resource usage, then measure performance [S6].

### 6.3 Traceback placement experiments

Because traceback follows a serial pointer-chasing chain of length $T = 256$ per tile, while remaining embarrassingly parallel across the $B$ independent tiles in a batch, three placement designs should be benchmarked in Phase 1:

| Design | Mechanism | Key metric to profile |
| --- | --- | --- |
| **1. GPU-resident traceback** | Dedicated SYCL kernel (1 work-item or subgroup per tile) walks `temp_edges` in VRAM | Low-occupancy latency vs. zero PCIe transfer overhead |
| **2. Device-to-Host copy + CPU traceback** | Copy `temp_edges` to pinned host RAM; multi-threaded C++/OpenMP traceback on CPU | PCIe transfer time ($0.5\text{–}2.0\text{ GiB}$ per batch) and synchronization stalls |
| **3. Zero-copy host-mapped edge writes** | Encoder writes `temp_edges` directly to USM host-allocated pinned memory over PCIe | Kernel stall sensitivity to PCIe write bandwidth limits |

---

## 7. Scratch memory and total capacity

### 7.1 Source-backed scratch buffer model

From upstream `get_temp_buffers`, the trellis scratch allocator reserves two contiguous tensors per active $K$ [S1]:
- `temp_costs`: shape $(B, 2, S)$, dtype `float16`
- `temp_edges`: shape $(B, T, S)$, dtype `int16`

Where $S = 65536 \gg K$, $T = 256$, and $B$ is the scratch batch capacity. The exact byte allocations are:

$$M_{\text{costs}} = 4 B S \text{ bytes}, \qquad M_{\text{edges}} = 2 B T S \text{ bytes}$$

Upstream defaults to $B = 256$ (or $\max(256, 2 \times \text{multiprocessor\_count})$ for $K \ge 4$) and caches buffers by argument signature [S1].

### 7.2 Scratch footprint at $B = 256, T = 256$

| Encoder $K$ | State dimension $S$ | `temp_edges` $(B \times 256 \times S \times 2\text{ B})$ | `temp_costs` $(B \times 2 \times S \times 2\text{ B})$ |
| :---: | ---: | ---: | ---: |
| **1** | 32,768 | $4.00\text{ GiB}$ | $32\text{ MiB}$ |
| **2** | 16,384 | $2.00\text{ GiB}$ | $16\text{ MiB}$ |
| **3** | 8,192 | $1.00\text{ GiB}$ | $8\text{ MiB}$ |
| **4** | 4,096 | $0.50\text{ GiB}$ | $4\text{ MiB}$ |

### 7.3 Total peak VRAM accounting and XPU allocator constraints

Total device memory consumption at any point during layer conversion is governed by the live tensor high-water mark:

$$M_{\text{peak}} = \max_{t} \Big( M_{\text{weights}}(t) + M_{\text{calib}}(t) + M_{\text{Hessian/LDLQ}}(t) + M_{\text{transforms}}(t) + M_{\text{scratch}}(t) + M_{\text{packing}}(t) + M_{\text{runtime}}(t) \Big)$$

**Critical Intel XPU memory policies to implement:**
1. **Cache eviction across mixed-$K$ recipes:** Because upstream caches `get_temp_buffers` by argument, quantizing a layer with mixed bitrates (e.g., $K=2$ and $K=3$) can leave multiple multi-gigabyte `temp_edges` buffers simultaneously resident. The XPU port must explicitly release or reuse a single high-water scratch arena across $K$ transitions.
2. **Adaptive batch capacity ($B$):** Halving $B$ from 256 to 128 or 64 linearly reduces $M_{\text{edges}}$ and $M_{\text{costs}}$. On 12 GB cards (Arc B580) vs. 16 GB (A770) or 24 GB (Arc Pro B60), $B$ should be dynamically sized from queried free VRAM (`torch.xpu.mem_get_info()`) minus a safety reserve, decouple-tuned from raw Xe-core counts.
3. **Single-allocation limits:** Query the reported maximum allocation size and test representative allocations through the actual PyTorch/SYCL allocator on the pinned stack [S5]. **Proposal:** Prefer reducing scratch batch capacity when allocations fail or exceed validated limits. Level Zero defines a relaxed-allocation extension, but availability and successful allocation through the selected framework are not guaranteed [S7]. Use such mechanisms only when explicitly supported and validated; do not prescribe driver debug keys or device-partitioning settings as portable allocation workarounds.

---

## 8. Packing and downstream validation

Upstream defines an explicit encoder tile permutation (`tensor_core_perm`), and `exl3xpu` provides reference reconstruction tests comparing XPU-dequantized weights against CUDA outputs [S1][S3].

**Proposal:** Reuse upstream Python bit-packing and safetensors serialization unaltered wherever possible. Audit all tensor names, dtypes, shapes, bit-packing layouts, scale formats, Hadamard flags, and codebook metadata.

### 8.1 Six-tier acceptance gate

1. **Structural:** Verify output safetensors keys, tensor shapes, dtypes, and metadata dictionary completeness.
2. **Bit-packing round-trip:** Confirm lossless pack/unpack identity on synthetic integer state sequences.
3. **Tile reconstruction:** Dequantize packed tiles via the reference decoder and verify reconstructed FP16 weights.
4. **Quantizer numerics:** Compare tile distortion ($||W - \hat{W}||_H^2$), LDLQ factorization residuals, selected Viterbi path costs, and post-feedback trailing weights against CUDA fixtures. *(Note: legitimate floating-point tie-breaks may select alternate paths with identical distortion; validate path cost parity rather than demanding bit-identical state indices on tied branches).*
5. **Model quality:** Measure teacher-forced logit KL-divergence, fixed-corpus WikiText-2 / C4 Negative Log-Likelihood (NLL), and perplexity against a CUDA-quantized checkpoint of the same model.
6. **Downstream runtime interop:** Load and execute the generated checkpoint in a pinned inference runtime (`exl3xpu` for supported `mul1` 4/6 bpw configurations, and upstream `exllamav3` on CUDA) [S3].

---

## 9. Orchestration and multi-device scaling

### 9.1 Streaming, disk workspace, and resumability

Upstream conversion uses a working directory for intermediate state, configurable checkpoint intervals, explicit resume flags, and sharded safetensors output [S2].

**Proposal:** Preserve upstream working-directory and resume semantics. Validate fault recovery by simulating interrupted writes, verifying checksum/shape integrity on restart, and ensuring mismatched conversion arguments trigger an immediate abort rather than corrupting partial checkpoints. Use `safetensors` memory-mapped (`mmap`) layer loading so host RAM scales with the active layer footprint rather than the full 70B+ model size.

### 9.2 Multi-device strategy (Deferred to Phase 4)

Upstream supports distributing linear modules within a layer stage across multiple GPUs and falling back to multi-device tile encoding, subject to PCIe transfer overhead [S2].

**Proposal:** Keep Phases 0–3 strictly single-device. Because sequential calibration propagation and within-layer LDLQ error feedback create strict data dependencies across columns/layers, multi-GPU scaling must only parallelize mutually independent projection matrices (e.g., parallel $Q, K, V$ or independent tile batches) after profiling host-link transfer costs.

---

## 10. Evaluation framework and risks

### 10.1 Standardized reporting metrics

All prototype benchmarks must report the following metrics under disclosed hardware, driver, power limit, and thermal conditions:

| Category | Required metrics |
| --- | --- |
| **Wall-clock latency** | Total conversion time and per-stage breakdown (calibration, LDLQ, transform, trellis search, traceback, packing); report AOT/JIT kernel compilation separately |
| **Encoder throughput** | Quantized weights/sec and tiles/sec, parameterized by $(K, \text{codebook}, B, \text{matrix shape})$ |
| **Trellis search rate** | Billions of state-transition updates per second ($\frac{\text{tiles} \times T \times S}{\text{time}}$) |
| **Memory high-water marks** | Peak `torch.xpu` allocated, peak reserved, Level Zero device-wide VRAM, peak host RSS, and disk workspace |
| **Numerical fidelity** | Layerwise proxy loss $\text{Tr}((W - \hat{W}) H (W - \hat{W})^T)$, max/mean reconstruction error, and LDLQ residual norms |
| **Downstream quality** | Fixed-corpus perplexity ($\Delta\text{PPL}$ vs. FP16 and vs. CUDA EXL3) and logit KLD |

### 10.2 Risk register

| Risk | Priority | Mitigation |
| --- | --- | --- |
| Silent algorithmic divergence during LDLQ/feedback porting | **Critical** | Unit-test factorization and residual updates against dumped CUDA intermediate tensors |
| Encoder state-transition, normalization, or tie-break mismatch | **Critical** | Validate against deterministic single-tile CUDA fixtures before full-layer runs |
| Missing PyTorch XPU operator coverage during calibration | **High** | Run Phase 0 forward-pass audit on target architecture; fallback unsupported ops to CPU |
| Peak VRAM exceeds 12 GB / 16 GB device limits | **High** | Evict cached multi-$K$ buffers immediately; dynamically tune batch capacity $B$ |
| Scratch edge writes ($M_{\text{edges}}$) bottleneck VRAM bandwidth | **High** | Profile coalesced vector writes and test GPU-resident vs. host-mapped traceback |
| Register spilling or SLM bank conflicts in SYCL Viterbi kernel | **High** | Inspect IGC (Intel Graphics Compiler) assembly dumps; tune work-group size and GRF mode |
| Long kernel dispatch timeouts (Linux hangcheck / Windows TDR) | **Medium** | Profile dispatch duration and chunk batches as needed; any initial target such as under 500 ms is a proposed engineering budget, not a universal watchdog threshold |
| Upstream `exllamav3` or `exl3xpu` drift | **Medium** | Pin exact Git commits and maintain standalone fixture verification tests |

---

## 11. Phased roadmap and resources

**Planning convention:** Effort is expressed in person-weeks of dedicated engineering experiment budget. Calendar duration depends on concurrency and hardware availability. Downstream phases are re-estimated at each exit gate.

### Phase 0: Source and toolchain audit (1–2 person-weeks)
- Pin a single coherent `exllamav3` commit and document the compatibility manifest.
- Complete the C++/CUDA call-site inventory (`quantize.py` and native C++/CUDA extensions).
- Export deterministic reference fixtures (calibration activations, $H$, LDLQ factors, transformed tiles, `temp_costs`, `temp_edges`, and packed outputs) from a reference CUDA run.
- Verify `torch.xpu` / oneMKL Cholesky/LDLQ accuracy and benchmark pinned-CPU fallback latency.
- **Exit gate:** Complete dependency specification, working SYCL/PyTorch C++ extension build skeleton, and verified reference test harness.

### Phase 1: Encoder and scratch prototype (2–4 person-weeks)
- Implement the `mul1` trellis encoder and traceback kernel in SYCL/XPU for one target $K$ (e.g., $K=4$ and $K=2$).
- Verify exact tile ordering (`tensor_core_perm`), state transitions, reconstructed weights, and path costs against Phase 0 fixtures.
- Profile SLM usage, register pressure, scratch allocation sizes, and throughput across batch capacities $B \in \{64, 128, 256\}$.
- Benchmark the three traceback placement options (Section 6.3).
- **Exit gate:** Verified tile encoding matching reference distortion, profiled memory/throughput report, and refined effort estimate for Phases 2–3.

### Phase 2: Complete single-layer conversion (TBD at Gate 1)
- Integrate XPU calibration capture, Hadamard transforms, LDLQ preprocessing, and blockwise error feedback for one complete transformer decoder layer.
- Connect upstream packing and safetensors export; implement scratch-cache eviction across layers/modules.
- Measure end-to-end single-layer peak VRAM and numerical drift against the CUDA reference layer.
- **Exit gate:** Exported single-layer checkpoint passing all applicable Tier 1–4 acceptance tests within target VRAM limits.

### Phase 3: Bounded full-model conversion (TBD at Gate 2)
- Execute end-to-end conversion of one supported dense model (e.g., 7B–8B parameter class) at a fixed bitrate recipe.
- Validate checkpoint/resume fault recovery, peak host RAM, and total wall-clock conversion time.
- Verify the resulting model on `exl3xpu` / reference runtimes (Tier 5–6 acceptance tests) and repeat qualification on the second Arc architecture generation.
- **Exit gate:** Reproducible full-model EXL3 checkpoint meeting perplexity and runtime interoperability criteria.

### Phase 4: Extended coverage and productionization (Scoped independently)
- Additional codebooks, fractional bitrate recipes, two-sided LDLQ, MoE architectures, and multi-GPU tile distribution.

### Resource requirements
- **Engineering:** GPU kernel engineering (SYCL / Intel Xe architecture / IGC optimization) and numerical/PyTorch C++ extension integration.
- **Hardware:** Primary development workstation with Intel Arc A770 (16 GB) and/or Arc B580 (12 GB) / Arc Pro B60 (24 GB) on Linux with pinned Intel Compute Runtime, Level Zero, oneAPI DPC++/SYCL, and PyTorch XPU stacks. Access to an NVIDIA GPU (or pre-generated Phase 0 tensor fixtures) for differential numerical verification.

---

## 12. Conclusion

**Proceed with Phase 0 (Source Audit) and Phase 1 (Tile-Encoder and Scratch-Memory Prototype). Complete single-layer conversion is the Phase 2 milestone.**

Porting the EXL3 quantizer to Intel Arc GPUs is an offline numerical and combinatorial optimization challenge centered on the Viterbi trellis search (`quantize_tiles`), blockwise LDLQ error feedback, and multi-gigabyte scratch-buffer management [S1][S2]. No algorithmic showstopper has been identified in the reviewed material, but the native audit remains incomplete. Competitive encoding throughput, register/SLM efficiency across large state spaces (including $S=32768$ for the $K=1$ allocation example), and peak VRAM fit on 12 GB–16 GB consumer cards can only be established empirically. The phased gate structure above bounds engineering risk while directly testing the highest-uncertainty components first.

---

## 13. Changes and sources

### Changes in Version 3.2 (from Version 3.1)
- Reclassified the document as a feasibility assessment and prototype plan.
- Distinguished FP16 scratch storage from unverified native arithmetic precision and executed state-update counts.
- Separated physical SLM capacity, exposed device limits, and compiled kernel resource requirements.
- Made GRF tuning a measured spill/occupancy tradeoff, not a high-occupancy guarantee.
- Clarified that the 8 MiB predecessor example is a full-tile allocation, not per-step traffic.
- Replaced allocation debug-key prescriptions with capability queries and allocator tests.
- Reconciled floating-point tolerances with exact discrete-format requirements.
- Aligned Phase 1 with tile encoding and Phase 2 with complete single-layer conversion.
- Labelled the dispatch-duration target as a proposal and added architecture/runtime references.
- Qualified the absence of identified showstoppers and included the $K=1$ state-space example.

### Changes in Version 3.1 (from Version 3.0)
- **Removed revision residue from main body:** Stripped inline arguments refuting Version 2.0 claims from Sections 1, 4, 5, 6, 7, 9, 10, and 11 so the document reads as a clean, standalone engineering specification.
- **Added explicit Intel Xe/Xe2 architectural constraints:** Specified Shared Local Memory (SLM) capacity vs. ping-pong cost buffer sizes ($16\text{–}64\text{ KiB}$), large-GRF register considerations, SIMD16/SIMD32 subgroup shuffle alignment, Level Zero `maxMemAllocSize` single-allocation limits, and Linux hangcheck / Windows TDR dispatch chunking.
- **Expanded hardware qualification tier:** Added the 24 GB workstation tier (Arc Pro B60) alongside the 16 GB A770 and 12 GB B580 targets.
- **Formalized mathematical notation:** Upgraded plain-text equation fences in Sections 4, 6, and 7 to standard mathematical typesetting.

### Changes in Version 3.0 (from Version 2.0)
- Replaced the generic 64–256-state, full-channel-length model with the source-backed $T = 256$ tile encoder (`quantize_tiles`) and $S = 65536 \gg K$ state space [S1].
- Derived exact scratch buffer sizes (`temp_costs` and `temp_edges`) from `get_temp_buffers` and separated scratch allocation from full-pipeline peak VRAM (`M_peak`), withdrawing unverified 11.7 GB / 7.4 GB total memory claims [S1].
- Preserved upstream one-sided and two-sided LDLQ error-feedback semantics instead of assuming a naive Cholesky replacement is sufficient [S2].
- Replaced the "only two custom kernels" assumption with an eight-part native dependency audit.
- Reframed GPU-resident, CPU-copied, and host-mapped traceback as benchmark candidates rather than assuming 4096-step row-serial stalls.
- Corrected multi-device scaling claims to account for sequential calibration and LDLQ dependencies [S2].
- Withdrawn unsupported performance forecasts (1.5× RTX 4090 speed, 15M weights/s, 0.05 PPL delta) and inconsistent 2.5–4.5 engineer-month / 14-vs-16-week schedules in favor of gated experiment budgets.

### Sources
Reviewed on October 20, 2026. References [S1]–[S3] identify quantizer and downstream-runtime sources; [S4]–[S7] identify architecture, capability-query, GRF, and allocation-extension references. Python allocation facts do not establish native arithmetic semantics. Proposals and calculated examples are labelled separately; none of these references establishes measured performance for this port.

- **[S1] Upstream quantizer implementation (`exllamav3`):** `quantize.py` (rev `4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f`), including `tensor_core_perm`, `get_temp_buffers`, and `quantize_tiles`. https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/modules/quant/exl3_lib/quantize.py
- **[S2] Upstream conversion documentation (`exllamav3`):** `convert.md` (rev `1d90befe096bbf69e49b75d1bd0eb2e89416beae`), covering CLI options, Hessian/LDLQ modes, calibration, checkpoint/resume, and multi-device controls. https://github.com/turboderp-org/exllamav3/blob/master/doc/convert.md
- **[S3] `exl3xpu` repository and README:** Intel XPU inference implementation, reconstruction/logit validation tests, supported configuration matrix (`mul1` at 4/6 bpw by default), and October 12, 2026 archive status. https://github.com/sybil-solutions/exl3xpu
- **[S4] Intel, Introduction to the Xe-HPG Architecture:** Xe-core L1/SLM organization and physical SLM capacity; not a Xe2 or per-kernel resource guarantee. https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-the-xe-hpg-architecture.html
- **[S5] Khronos SYCL device reference:** capability queries, including local memory, work-group limits, and maximum allocation size. https://github.khronos.org/SYCL_Reference/iface/device.html
- **[S6] Intel LLVM experimental GRF-size extension:** configurable GRF properties, backend restrictions, and advanced tuning status. Resource and occupancy consequences require target-specific measurement. https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_intel_grf_size.asciidoc
- **[S7] Level Zero Relaxed Allocation Limits Extension:** extension API; not a guarantee of framework exposure or allocation success. https://oneapi-src.github.io/level-zero-spec/level-zero/1.13.0/core/EXT_Exp_RelaxedAllocLimits.html