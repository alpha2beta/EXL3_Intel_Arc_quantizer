# EXL3 Intel Arc Quantizer

An implementation and tooling suite for porting the **EXL3 quantization and model conversion pipeline** from NVIDIA CUDA to Intel Arc GPUs (Xe2-LPG / Xe-HPG / Xe2-HPG) using SYCL, oneAPI, and PyTorch XPU.

Compatible with upstream [ExLlamaV3](https://github.com/turboderp-org/exllamav3) format and downstream Intel Arc inference runtimes such as [exl3xpu](https://github.com/0xSero/exl3xpu).

---

## Current Status & Verification Baseline

| Metric | Status / Value | Verification Source |
|---|---|---|
| **Phase / Maturity** | **Gate 0, Gate 1, Gate 2 & Gate 3 Closed** (Phase 3 bounded orchestration P3-01..P3-08, full 32-layer conversion P3-09, P3-11 verification, P3-12 validation PPL, P3-13 vLLM runtime execution, P3-14 bit-identical repeat conversion, and initial Hugging Face release complete; discrete-Arc gate P3-15..P3-17 open per [`B60_B580.md`](./B60_B580.md); Phase 4 architecture extensions: P4-12 `[x]` / P4-13 `[x]` / P4-14 `[x]`) | [`reports/gate_0.md`](./reports/gate_0.md), [`reports/gate_1.md`](./reports/gate_1.md), [`reports/gate_2.md`](./reports/gate_2.md), [`reports/gate_3.md`](./reports/gate_3.md), `exl3_quant/converter.py` |
| **Unit Test Suite** | **277 / 277 passed (100%)** (1 platform-conditional skip, 236.9 s) | `python -m unittest discover -s tests -p "test_*.py"` |
| **CUDA Ground Truth (E4)** | **Deferred, not waived** — `skip_cuda.md` / `SKIP-CUDA-2026-10-21-01`; evidence capped at **E3** | [`skip_cuda.md`](./skip_cuda.md), [`docs/deviation_log.md`](./docs/deviation_log.md) §6 |
| **Primary Prototype Target** | Intel Core Ultra 7 258V / Arc 140V (32 GB UMA) | [`docs/compatibility_manifest.md`](./docs/compatibility_manifest.md) |
| **Downstream Oracle Parity** | **100% Bit-Exact** (CPU Ref & XPU SYCL vs `exl3xpu.ref`) | [`tests/test_oracle_parity.py`](./tests/test_oracle_parity.py), [`tests/test_xpu_quantizer.py`](./tests/test_xpu_quantizer.py) |
| **Upstream Pinned Commit** | `turboderp-org/exllamav3 @ 4d7b69c7` | [`docs/algorithm_contract.md`](./docs/algorithm_contract.md) |
| **Active Target Model** | [`empero-ai/Qwen3.8-9B-Distill`](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill) — **CONVERTED, QUALIFIED, AND RELEASED** as [`alpha2beta/Qwen3.8-9B-Distill-EXL3-K2`](https://huggingface.co/alpha2beta/Qwen3.8-9B-Distill-EXL3-K2) (32 layers, $K=2$, `mul1`; see [`reports/gate_3.md`](./reports/gate_3.md) & §13 below) | `tasks.md` §3 `P3-09` / [`reports/gate_3.md`](./reports/gate_3.md) |
| **XPU Encoder Throughput** | Up to **1,301.5 tiles/s** (43.67 G state transitions/s) | [`artifacts/benchmarks/phase1_encoder_bench.json`](./artifacts/benchmarks/phase1_encoder_bench.json) |
| **Sustained Layer Conversion** | Peak Alloc **649.7 MiB**, **0 B Swap Growth** across repeated iterations | [`artifacts/benchmarks/phase2_layer_bench.json`](./artifacts/benchmarks/phase2_layer_bench.json) |
| **Mmap Shard Streaming (P3-03)** | Streaming residual RSS **2.8 MiB** over a **424 MiB** sharded checkpoint | [`artifacts/benchmarks/phase3_stream_bench.json`](./artifacts/benchmarks/phase3_stream_bench.json) |

---

## Primary Prototype Platform: Intel Arc 140V (Lunar Lake)

Development is anchored on an integrated **Intel Core Ultra 7 258V** platform with **Intel Arc 140V integrated graphics** and 32 GB shared on-package memory (30.0 GiB available to Linux kernel).

- **GPU Architecture:** Xe2-LPG, device ID `0x64A0`, 8 Xe-cores (subslices), 64 XVEs, 128 KiB shared local memory (SLM) per workgroup, SIMD16/32 execution.
- **Software Toolchain:** Arch Linux x86_64, Intel oneAPI 2026.0 (`icpx`), Compute Runtime `26.35.39758.10`, Level Zero `1.17.39758`, and PyTorch XPU `2.14.0+xpu`.
- **Memory Model & Headroom:** Unified physical memory pool. Live headroom under the mandatory 6.0 GiB OS/driver reserve is **$\approx 4.0 \text{ GiB}$**. Dynamic scratch allocation is bounded to safe batch limits ($B \le 128$ for $K=2$, $B \le 64$ for $K=1$) to prevent paging and swap thrashing.
- **Discrete Arc Track:** Discrete targets (Arc A770, Arc B580, Pro B60) are qualified separately on an independent deployment track. The execution plan for the **Arc Pro B60 + Arc B580** machine (inventory, encoder probe, memory-policy regime, full-model qualification, evidence bundle, comparison against the 140V baseline) is [`B60_B580.md`](./B60_B580.md).

For architectural details, refer to [`docs/140v.md`](./docs/140v.md) and [`docs/compatibility_manifest.md`](./docs/compatibility_manifest.md).

---

## Core Implemented Architecture (`exl3_quant/` & `csrc/`)

### 1. SYCL XPU Tile Encoder ([`csrc/quantize_xpu/`](./csrc/quantize_xpu/) & [`exl3_quant/xpu_quantizer.py`](./exl3_quant/xpu_quantizer.py))
Native Intel Arc SYCL implementation of the EXL3 forward Trellis Viterbi search and circular traceback:
- **Exact fp16 Dynamic Programming:** Add-Compare-Select (ACS) dynamic programming using exact fp16 `__hsub2` and single-rounded `sycl::fma` accumulation (`O-6` on-device verified), bit-for-bit identical to the CPU reference.
- **Two-Pass Tail-Biting Circular Closure:** Pass 1 ($roll = 128$) unconstrained forward/backward to find `end_state`; Pass 2 ($roll = 0$) wrap-constrained by `pre_state = end_state` emitting the optimal path.
- **Traceback Placement Architectures:**
  - `gpu_resident`: GPU-resident circular traceback baseline with sub-millisecond traceback latency (<0.25 ms per wave). Selected as default (`reports/traceback_decision.md`).
  - `d2h_cpu`: Device-to-host edge transfer followed by CPU traceback.
  - `usm_host`: Direct host-accessible edge writes via SYCL host-allocated USM.
- **Geometry & Cache Sweeps:** Evaluated SIMD16 vs SIMD32 workgroups and SLM vs global cost storage. SLM achieves +49% higher throughput at $K \in \{3, 4\}$; global cost storage excels at $K=2$ by avoiding SLM occupancy bottlenecks.
- **Shape Support:** Fully supports $K \in \{1, 2, 3, 4\}$, all 3 codebooks (`mul1`, `mcg`, `3INST`), standard $L=256$ tiles, and $L=160$ n-gram rows.

### 2. Bounded Scratch Arena & Memory Policy ([`exl3_quant/xpu_quantizer.py`](./exl3_quant/xpu_quantizer.py), [`exl3_quant/memory_policy.py`](./exl3_quant/memory_policy.py))
Adaptive, pressure-aware memory manager replacing upstream's unbounded `get_temp_buffers`:
- **Single-Slot Stream Arena (`ScratchArena`):** Event-aware reuse and eviction synchronized with the current PyTorch XPU stream. Prevents cross-$K$ memory leaks.
- **Dual-View Accounting:** Jointly samples OS `/proc/meminfo` and framework allocator (`torch.xpu.mem_get_info` / `memory_allocated`). Enforces a strict 6.0 GiB OS/driver reserve.
- **Configurable reserve (P3-09):** the admission reserve can be overridden per run with `EXL3_MEMORY_RESERVE_GIB` (float GiB) or `EXL3_MEMORY_RESERVE_BYTES` (integer bytes); unset keeps the 6.0 GiB default. The reserve is *device-pool* headroom: on an integrated (UMA) device the scratch arena draws from the same DRAM as the host, so a host with a small device pool but plenty of RAM headroom can end up with no admissible budget at all while the system is in no danger. The override makes that trade-off explicit and auditable (the P3-09 full-model run uses 4 GiB with the measured headroom recorded in `progress.md` Step 26).
- **Pre-Admission Abort & Adaptive Fallback:** Pre-checks requests against usable headroom; raises `MemoryPressureError` before exhausting memory. Automatically retries with halved batch capacities ($B \to B/2$) without changing tile semantics.

### 3. Block LDLQ Decomposition & Preprocessing ([`exl3_quant/ldlq.py`](./exl3_quant/ldlq.py))
Direct contract transcription of the blockwise Cholesky/LDLQ factorization (`quantize.py:416-490`):
- **Contract-Conformant Inversion:** Inverts block diagonal $L_{bb}^{-1}$ and folds into $M_{:, block} = L_{:, block} L_{bb}^{-1}$.
- **Retry-Only Damping Schedule:** Preserves upstream's adaptive diagonal regularisation ($2 \cdot \sigma_{\text{reg}} \cdot \text{mean}(\text{diag}(H))$ with up to 10 retries) without artificial pre-emptive damping.
- **Operator Attribution Probe:** Probed 17 operator/dtype pairs on Lunar Lake XPU; catalogued explicit CPU fallback boundaries (e.g., `torch.linalg.cholesky` on XPU falls back to CPU).
- **One-Sided LDLQ Error Feedback (`P2-08`):** `ldlq()` reproduces upstream's bottom-up span/block traversal (`quantize.py:493-609`): additive target correction from the strictly-lower `L` slice, tensor-core tile permutation, un-permuted reconstruction, and trailing-span residual feedback. Tile encoding is a pluggable callback with bit-exact reference and Arc XPU adapters that reuse a single `ScratchArena`.

### 4. Contract-Conformant CPU Reference Encoder ([`exl3_quant/reference_encoder.py`](./exl3_quant/reference_encoder.py))
Direct host-CPU transcription of the upstream CUDA kernel `quantize_tiles_kernel.cuh:23-313`:
- **16-bit State Space:** $S = 65536 \gg K$ tracked states, $Kr = 16 - K$, candidate branch states $\sigma = (k \ll Kr) \mid e$, predecessor $pred = \sigma \gg K$.
- **Exact fp16 Fused Metric:** Single-rounding `__hfma2` emulation using float64 intermediate accumulation with strict `<` first-branch tie-breaking.
- **Vectorized NumPy Acceleration:** Precomputes candidate branch tables, executing 256-step dynamic programming in ~15 ms per tile on CPU.

### 5. Format Primitives & Bitpacking ([`exl3_quant/trellis_format.py`](./exl3_quant/trellis_format.py))
Device-independent format contract transcribed from `pack.cu`, `codebook.cuh`, and `quantize.py`:
- **Procedural Codebooks:** Bit-exact LUTs for `mul1` (913 distinct levels in `[-3.453125, 3.34765625]`), `mcg` (10,746 levels), and `3INST` (10,598 levels).
- **Tensor-Core Permutation:** Exact 256-element permutation table `tensor_core_perm()` and inverse `tensor_core_perm_inv()`.
- **State-History Container:** Upstream MSB-first packing (16 spans $\times$ 16 elements, low $K$ bits per state) with `SWAP16` half-word swap (`pack_trellis`, `swap16_storage`, `decode_states`, `decode_tile_values`).

### 6. Scratch Memory Estimator ([`exl3_quant/scratch_estimator.py`](./exl3_quant/scratch_estimator.py))
Predicts exact VRAM requirements based on the pinned allocator contract:
- Costs buffer: $B \times 2 \times S \times 2$ bytes (`float16`).
- Edges buffer: $B \times T \times S \times 2$ bytes (`int16`).
- Batch capacity solver `calculate_max_batch_size(...)` with configurable UMA/system reserves and Level Zero single-allocation bounds.

### 7. Sylvester Hadamard Transform ([`exl3_quant/hadamard.py`](./exl3_quant/hadamard.py))
Contract §8 reference path, transcribed from upstream `util/hadamard.py`:
- **Natural Sylvester `H_128`**, orthonormalized by `1/sqrt(128)`; symmetric and self-inverse.
- **Blockwise application** (`H @ x` per 128 rows, `x @ H` per 128 columns), matching upstream `blockwise_preapply_had_{l,r}_` — not a full Kronecker transform.
- **Decode ordering** `H_l → ×su → H_r → ×sv` (`weight_from_quantized`), verified against `exl3xpu.ref.weight_orig`.

### 8. Weight Regularization & Global-Scale Search ([`exl3_quant/regularize.py`](./exl3_quant/regularize.py), [`exl3_quant/gss.py`](./exl3_quant/gss.py))
Contract §9.3, transcribed from `quantize.py:1079-1236`:
- `block_rms` with upstream's 32-slice accumulation order.
- Output-scale skew heuristic (top 2% of `sqrt(H_diag)` vs 15% cutoff), out-channel RMS normalization, `sv`/`su` sign+scale folding, and the zero-channel floor (`0.1`) / `sv = 0` rule.
- Global-scale (`g_scale_gss`) search ([`exl3_quant/gss.py`](./exl3_quant/gss.py), P2-04 complete): wrapped-diagonal tile sampling with highest/lowest-RMS outlier tiles, coarse grid (0.1–1.9, strided subsample) → fine grid (0.075 around coarse minimum, full sample) → parabolic refinement with a `0.01` floor.

### 9. Transformer Layer Calibration & Hessian Harvesting ([`exl3_quant/calibration.py`](./exl3_quant/calibration.py))
Complete layer forward execution and activation covariance capture (P2-01..P2-03, P2-05):
- Complete `TransformerDecoderLayer` (RMSNorms, Rotary Embedding, Attention $Q/K/V/O$, SwiGLU $Gate/Up/Down$) executing natively on Intel XPU with zero silent CPU fallback.
- `HessianCollector` groups inputs into 4 architectural locations (attention in, attention out, mlp in, mlp down), reducing Hessian memory by 43% (976 MiB total in fp32 for Llama-3.1-8B).
- Second-moment accumulation $H = \sum X^T X$ with token normalization, damping $\lambda \cdot \text{mean}(\text{diag}(H))$ ($\lambda = 0.01$), and floor clamping for inactive/ill-conditioned channels.

### 10. Single-Layer Quantization & Distortion Metrics ([`exl3_quant/layer_quant.py`](./exl3_quant/layer_quant.py))
End-to-end layer quantization with bounded scratch reuse and proxy loss (P2-11, P2-12):
- Single-slot `ScratchArena` and `XPUTrellisEncoder` reused across all 7 linear projections (`q, k, v, o, gate, up, down`) with explicit `torch.xpu.synchronize()` between projections.
- Evaluates relative Frobenius error, reconstruction MSE, tile distortion, and Hessian-weighted proxy loss $\text{tr}\left((W - \hat{W}) H (W - \hat{W})^T\right)$ via $(H @ E \cdot E).\text{sum}()$.

### 11. Trellis Packing & Safetensors Serialization ([`exl3_quant/serialization.py`](./exl3_quant/serialization.py))
Export, reload, and verification of EXL3-compliant checkpoint artifacts (P2-13..P2-16):
- `pack_module_trellis` applies audited `pack_trellis` and `swap16_storage` into `int16 [k/16, n/16, 16*K]`.
- Exports compliant `.safetensors` checkpoints with `quantization_config.json` and index weight map.
- `load_and_verify_layer_safetensors` decodes weights via `exl3xpu.ref.weight_orig` and validates state-history round trips and malformed-file rejection.

### 12. Bounded Full-Model Conversion Orchestrator & Recovery ([`exl3_quant/converter.py`](./exl3_quant/converter.py))
Phase 3 driver preserving the audited upstream conversion semantics (P3-01..P3-08):
- **`ConversionConfig` + `validate()`** reject unsupported modes (MoE / vision / MTP / pre-quantized input, non-dense architectures, fractional or out-of-range K, non-`mul1` codebooks, non-128-divisible geometry, partial/scaled rotary, sliding window) in pure Python before any expensive work (P3-01).
- **`convert_model`** executes the pinned upstream per-module loop: per decoder layer in ascending order — grouped-Hessian capture on the running state produced by all previously *quantized* layers, quantization, reload of quantized numerics (`w_hat` into `nn.Linear` layout), and advance of the calibration state through the quantized layer. Unchanged components (`embed_tokens`, `lm_head`, RMSNorm) are bit-exact fp16 pass-through with per-layer `qtensors/` artifacts and a merged `quantization_config.json` manifest (P3-02).
- **`ShardedSafetensorsIndex`** streams HF-style sharded/single-file checkpoints through memory-mapped `safetensors.safe_open` handles, materializing only the layer under conversion; host RSS is measured (`read_rss_kib`) rather than assumed (see [`artifacts/benchmarks/phase3_stream_bench.json`](./artifacts/benchmarks/phase3_stream_bench.json)) (P3-03).
- **Checkpointing, Resumability & Recovery (P3-04..P3-08):**
  - Atomic publication (`atomic_save_safetensors`) writing `.tmp` before POSIX rename, preventing incomplete artifacts on crash (P3-05).
  - Checkpoint integrity verification (`verify_layer_checkpoint`) checking all 30 expected tensors per layer (P3-05).
  - Fault injection testing (`fault_injector`) verifying safe restart across calibration, quantization, and publishing boundaries without dropped or duplicated work (P3-06).
  - Strict resume argument validation (`_validate_resume`) against `quantization_config.json` with documented `force_resume` override (P3-07).
  - Pre-checked disk space limits (`check_disk_space`), missing shard handling, and corrupted checkpoint detection (`CorruptedCheckpointError`) (P3-08).
- **Real-corpus calibration (P3-09):** `ConversionConfig.calibration_token_ids` accepts a pre-tokenized `exl3_quant.calibration_token_ids.v1` JSON corpus file (rows of `batch_size * seq_len` token ids + provenance metadata), prepared reproducibly by [`scripts/prepare_calibration_tokens.py`](./scripts/prepare_calibration_tokens.py) from a local corpus copy and the target tokenizer. The full contract (format stamp, row count/shapes, int token bounds, `seq_len` match) is validated in `validate()` before any weight byte; the manifest records `calibration.corpus` + `calibration.token_ids {path, sha256, metadata}` and `calibration_state.pt` records `corpus_sha256`, so resume can never mix corpora (cross-corpus resume raises `IncompatibleResumeError`). The synthetic deterministic corpus remains the unchanged default when the field is `None`.
- **Streaming memory hygiene (P3-09):** [`exl3_quant/memory_hygiene.py`](./exl3_quant/memory_hygiene.py) releases what a finished streaming step no longer needs — the page cache of the consumed source shards (`POSIX_FADV_DONTNEED`, advisory: clean pages only, dirty pages are written back first, file content is never affected) and freed host arenas (`malloc_trim(0)`). The converter calls it after every pass-through group, every layer read, every published layer artifact and the calibration state, so a 32-layer run over a 19.3 GB checkpoint does not pin the checkpoint in DRAM (on this shared-memory device the P0-20 admission policy binds on host pressure, so an unpinned page cache aborts the run with `MemoryPressureError`).
- **Output verification (P3-11):** [`scripts/verify_conversion_output.py`](./scripts/verify_conversion_output.py) checks a finished bundle without any inference: manifest contract (status, layer list, recipe, corpus sha256), the exact per-layer tensor key set (quantized module tensors + layer norms + per-layer pass-through tensors), dtypes, trellis tile geometry against the *source* checkpoint shapes, bit-exact comparison of every pass-through tensor against the source checkpoint, and reconstruction of every quantized module with its relative Frobenius error (must lie strictly inside (0, 1)).
- **Runtime export (P3-13):** [`scripts/export_for_runtime.py`](./scripts/export_for_runtime.py) publishes a converted bundle as an ordinary checkpoint directory for the pinned downstream runtime: the per-layer artifacts and pass-through groups become shards (hardlinked by default — they are immutable once published), `model.safetensors.index.json` is built from the *actual* tensor names (duplicate keys across shards are refused), the model's config/tokenizer files are copied, and the conversion manifest is published as `quantization_config.json` (it already carries the `quant_method`/`bits`/`codebook`/`tensor_storage` schema the `exl3xpu` vLLM plugin reads). It also embeds that manifest into the exported `config.json`, because the plugin reports no config filenames of its own. **Verified end to end (P3-13):** the exported 6.72 GiB bundle (35 shards, 1375 tensors) loads in vLLM `0.31.0+xpu` with the `exl3xpu` plugin (`exl3: 200 EXL3 modules (0 in MTP head)`, 52 s load, text-only mode) and decodes coherent greedy text. The runtime harness needs the same three exports as `vllm.xpu/scripts/launch_vllm.sh`: `PATH=/opt/intel/oneapi/compiler/latest/bin:$PATH`, `LD_LIBRARY_PATH=<venv>/lib:$LD_LIBRARY_PATH` (oneCCL's legacy plugin `libccl.so.1`, which the RUNPATH-less `libccl.so.2` loads by bare name) and `CPATH=<vllm.xpu>/tools/sysroot/usr/include:$CPATH` (`level_zero/ze_api.h` for the Triton JIT).
- **Quality evaluation (P3-12):** [`scripts/evaluate_quality.py`](./scripts/evaluate_quality.py) measures teacher-forced NLL/perplexity, KL divergence and argmax agreement between the floating-point source and the decoded artifacts inside the repo's own reference forward (the published per-layer trellis is decoded and forwarded exactly like the calibration batches, so only the weights differ); [`scripts/evaluate_perplexity.py`](./scripts/evaluate_perplexity.py) measures the same teacher-forced NLL inside the pinned runtime through `prompt_logprobs`. Measured on 4 held-out `wikitext-2` validation sequences (2044 scored positions, disjoint from the test-split calibration corpus): floating-point PPL **6.368**, quantized PPL **7.554** (+18.6 %; NLL +9.2 %), mean KL **0.254** over 128 positions, argmax agreement **0.750**, and the runtime reproduces the repo-side quantized NLL to **1.4e-5** relative. The exact quantization loss is attributed against the floating-point baseline; the matched CUDA-quantized reference stays deferred (`skip_cuda.md`).

### 13. Hybrid Qwen3.5 Text-Tower Extension ([`exl3_quant/hybrid.py`](./exl3_quant/hybrid.py))
Opt-in conversion of the declared Active Target's hybrid architecture (`architecture_mode="qwen3_5_text_tower"`, P4-12/P4-13; requires `transformers >= 5.17` for the pinned `qwen3_5` reference model):
- **Quantization contract (P4-12):** every bias-free linear projection of both layer kinds (`linear_attention`: `linear_attn.in_proj_qkv`/`in_proj_z`/`out_proj` + `mlp.*`; `full_attention`: `self_attn.q/k/v/o_proj` + `mlp.*`) is quantized unchanged because all k/n widths are multiples of the 128-wide Hadamard block. `linear_attn.in_proj_a`/`in_proj_b` (n = num_value_heads) violate that encoder contract and — together with `conv1d.weight`/`A_log`/`dt_bias`/`linear_attn.norm` and the full-attention `q_norm`/`k_norm` — are emitted **bit-exact pass-through** in source dtype.
- **Reference forward:** calibration runs the pinned `transformers.models.qwen3_5` decoder layers (eager attention, causal mask on full-attention blocks, partial rotary 0.25 + interleaved mrope) with recorded reference-PyTorch fallbacks for `causal_conv1d`/`flash-linear-attention`; shared-input Hessian groups cover all four `in_proj_*` members.
- **Text-tower-only semantics (P4-13):** vision (`model.visual.*`) and MTP (`mtp.*`) tensors are preserved, never quantized, and written to separate `qtensors/untouched*.safetensors` group files so no single serialization holds all pass-through bytes; the manifest records the groups plus a `hybrid` provenance block (multimodal token ids, rope parameters, norm semantics, reference-forward environment).
- **Text-only container (P4-14):** the same layer contract is also accepted in its **text-only** packaging — `Qwen3_5ForCausalLM` / `model_type: qwen3_5_text` with a *flat* config (no `text_config` block) and the decoder stack at the top-level `model.layers.` prefix, as shipped by [`autotrust/JEV-9B`](https://huggingface.co/autotrust/JEV-9B). The container layout derives the checkpoint prefixes (`nested → model.language_model.layers`, `flat → model.layers`), is recorded in the manifest as `hybrid.text_tower_container`, and is accepted only on the unambiguous text-only markers — a multimodal config that is merely missing its text block stays a hard rejection. Pass-through groups remain key-derived, so a config that declares an MTP head the checkpoint does not carry (JEV-9B does) produces no MTP group file. Converting a JEV-9B-class checkpoint is a separate, owner-approved task (its System-1 LoRA + decision head live outside the backbone safetensors — see `tasks.md` P4-14).
- **Gating:** the default `"dense"` mode keeps every historical acceptance and rejection byte-identical; hybrid support availability and geometry are validated in pure Python before any weight byte is read, and resume refuses cross-mode manifests.

---

## Test Suite & Verification

The test harness enforces exact discrete checks, perturbation detection, and live oracle cross-validation:

```bash
# Set PYTHONPATH to project root and run all unit tests
PYTHONPATH=. /home/yanchun/Projects/vllm.xpu/.venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

### Complete Test Breakdown (277 Tests Total)

- **[`tests/test_calibration.py`](./tests/test_calibration.py) (7 tests):** RMSNorm, Rotary embeddings complex unit norm, Attention forward, SwiGLU forward, complete layer forward on XPU, exact second-moment Hessian accumulation matching explicit formula, dead channel floor and damping safeguard.
- **[`tests/test_layer_quant.py`](./tests/test_layer_quant.py) (4 tests):** Proxy loss trace formula equivalence, factorization residual check, single module XPU quantization, and complete decoder layer quantization on XPU with ScratchArena reuse.
- **[`tests/test_serialization.py`](./tests/test_serialization.py) (6 tests):** Trellis packing round trip with state-history low K-bit preservation, safetensors schema validation, export/reload verification with `exl3xpu.ref.weight_orig`, malformed/truncated safetensors rejection, and tile encoder bit-width mismatch guards (`TestEncoderBitWidthGuard`).
- **[`tests/test_upstream_encode_oracle.py`](./tests/test_upstream_encode_oracle.py) (4 tests):** CPU-runnable tier-4 oracle derived from upstream's CUDA test recipe (E3 evidence): reproduces upstream's own `max_mse_per_K` ceilings for K = 2..8, the tail-biting ring invariant, upstream's "ideal encoding" state layout, and encode→dequant round trips — proving *search-quality parity* without any NVIDIA hardware.
- **[`tests/test_forward_semantics.py`](./tests/test_forward_semantics.py) (5 tests):** independent audit of the reference encoder's forward ACS pass (branch metric, fp16 accumulation, predecessor table and terminal costs, tie-breaking and masking rules) — supports the E1 source-transcription claim.
- **[`tests/test_converter.py`](./tests/test_converter.py) (36 tests):** Phase 3 bounded conversion (P3-01/02/03) against deterministic synthetic mini-Llama checkpoints (`tests/fixtures/`): config rejection of MoE/vision/MTP/pre-quantized/fractional-K/bounded-K/bad-geometry/partial-rotary/rope-scaling/sliding-window/bad-output cases, plus hardening against the declared targets' architecture shape (hybrid `layer_types`, `mtp_num_hidden_layers`, top-level multimodal token ids incl. `vision_end_token_id`, and the full `Qwen3.8-9B-Distill` + `MiMo-V2.6-Distill` config replicas asserted deterministically rejected in the default mode — their opt-in acceptance under `qwen3_5_text_tower` is covered by `tests/test_hybrid_converter.py`); mmap shard index scoping, bit-exact streaming reads, single-file indexing, poisoned-shard isolation, RSS measurement; full two-layer XPU conversion with a recorder proving ascending layer order, upstream module order, layer-0 capture equality vs. independent calibration, layer-1 propagation through layer 0's **quantized** numerics (differential vs. unquantized control), reload replay equivalence, transposing `w_hat` reload, bit-exact unchanged pass-through, per-layer `qtensors` artifact layout (30 tensors), merged manifest consistency, and per-layer RSS recording; plus P3-09 real-corpus wiring (ordered row loading + metadata passthrough, synthetic default needing no corpus file, 11 malformed-corpus rejections in `validate()`, manifest `calibration.token_ids` provenance with sha256, `calibration_state.pt` corpus identity, cross-corpus resume rejection, same-corpus resume success) and the P3-09 streaming memory-hygiene integration check (source shards, published layer artifacts and the calibration state are released during a conversion).
- **[`tests/test_hybrid_converter.py`](./tests/test_hybrid_converter.py) (36 tests):** P4-12/P4-13 hybrid extension — validator acceptance of the synthetic hybrid fixture and both declared-target config replicas in the opt-in `qwen3_5_text_tower` mode (with the default-mode rejection preserved), rejection of unknown modes/MoE/unknown layer types/non-128 widths, `full_attention_interval` derivation and legacy layer-type remapping; P4-14 text-only container (flat `qwen3_5_text`, e.g. `autotrust/JEV-9B`) — config-replica acceptance with derived `model.layers` prefixes, dense-mode rejection preserved, the missing-`text_config`-block guard, key coverage of the flat fixture's pass-through groups (no vision/MTP group despite the stale MTP declaration), and a strict reference-layer build from the flat layout; quantized/pass-through module tables exhaustive over reference `named_parameters()` with encoder-divisibility proof; shared-input Hessian grouping with forward pre-hook input-identity; XPU end-to-end conversion of a 3-layer bfloat16 hybrid fixture asserting 32-tensor layer artifacts (6/7 quantized modules × 4 tensors), bit-exact per-layer and per-group pass-through with preserved source dtype, manifest provenance (`architecture_mode`, `layer_types`, pass-through groups, `hybrid` block, `environment`), decoded-projection quantization band, deterministic rebuilt-layer forwards, fault-injected publish failure resumed to bit-identical artifacts, and tampered-manifest mode mismatch refusal; dense-profile default guards.
- **[`tests/test_xpu_quantizer.py`](./tests/test_xpu_quantizer.py) (13 tests):** End-to-end XPU quantizer verification: all 65,536 codebook states, bit-exact parity vs `ReferenceTrellisEncoder` across $K \in \{1, 2, 3, 4\}$ and all 3 codebooks, $L=160$ n-gram rows, live oracle decode via `exl3xpu.ref`, edge cases & ties (zeros, constant, outliers), partial batches & waves, input validation, stream-queue ordering, all 4 traceback placements & kernel variants (`SG16`/`SG32`, SLM/global), `ScratchArena` bounded eviction across $K$, automatic batch halving retry, no double scratch residency during retry (B1), and zero resident scratch leakage on failed acquisition (B2).
- **[`tests/test_xpu_extension.py`](./tests/test_xpu_extension.py) (7 tests):** Custom SYCL XPU extension sanity, stream execution, memory lifetime, device cap queries, and on-device measurement of Open Item `O-6` (exact fp16 single-rounding FMA verified against 65,536 random and 11 adversarial inputs).
- **[`tests/test_hadamard.py`](./tests/test_hadamard.py) (11 tests):** Sylvester values/symmetry/orthogonality, non-power-of-two rejection, blockwise-vs-block-diagonal equivalence, involution, in-place validation, bit-exact `exl3xpu.ref.hadamard128`, and decode-order parity with `weight_orig`.
- **[`tests/test_gss.py`](./tests/test_gss.py) (19 tests):** wrapped-diagonal coverage/value verification against manual extraction, extreme-RMS tile inclusion, stage-by-stage recomputation of the coarse→fine→parabolic pipeline, batching/chunking invariance, MSE-below-`g_scale=1` with the real encoder, determinism, huge-weight scale walk-down, `select_refined_scale` floor/basin tests, and `regularize` wiring.
- **[`tests/test_ldlq_loop.py`](./tests/test_ldlq_loop.py) (13 tests):** identity-`L` equivalence with independent tile quantization, exact additive target correction, independent dense transcription for `buf_size_k ∈ {16,32,64}`, feedback objective improvement on a correlated Hessian, span invariance, upstream diagonal-zeroing equivalence, encoded layout/dtype, input immutability, validation, and XPU↔reference parity.
- **[`tests/test_regularize.py`](./tests/test_regularize.py) (13 tests):** `block_rms` accuracy and slice-boundary accumulation, full regularize→decode round-trip, skew heuristic both directions, zero-channel flooring, all-zero `zeros` flag, non-mutation, and input validation.
- **[`tests/test_converter_recovery.py`](./tests/test_converter_recovery.py) (18 tests):** Phase 3 checkpointing/resumability/fault recovery (P3-04..P3-08) against 3-layer synthetic checkpoints: working-directory layout, configurable checkpoint interval with manifest progress tracking (P3-04); atomic `.tmp` + POSIX-replace publication and 30-tensor-per-layer integrity verification with `CorruptedCheckpointError` on truncation (P3-05); interruption-safe recovery at calibration/quantization/reload/publication boundaries with clean resume — no dropped or duplicated layers (P3-06); rigorous `_validate_resume` invariance validation (K, codebook, calibration, geometry) with documented `force_resume` override (P3-07); and resilient failure handling — `check_disk_space` margins, missing shards, corrupted checkpoints (P3-08).
- **[`tests/test_memory_policy.py`](./tests/test_memory_policy.py) (28 tests):** OS `/proc/meminfo` parsing, reserve budget arithmetic, admission boundary enforcement, outstanding allocation tracking, `MemoryPressureError` context, post-commit reserve-breach detection, allocator-growth verification (M1), platform-native memory counters across OS variants (1 Linux/macOS conditional skip for Windows API), live GPU allocation under reserve constraints, and the P3-09 reserve override (default 6 GiB, GiB/byte overrides, precedence, invalid-value fallback, and a strictly smaller reserve admitting more).
- **[`tests/test_memory_hygiene.py`](./tests/test_memory_hygiene.py) (5 tests):** P3-09 advisory memory-release helpers — page-cache eviction preserves file content byte-for-byte, missing files and directories are not errors, host-heap trimming is best-effort and never raises, and the combined release entry point accepts a missing path.
- **[`tests/test_ldlq.py`](./tests/test_ldlq.py) (10 tests):** Algebraic verification of `block_ldl` ($L L^T = H$, block-diagonal inversion fold, identity block diagonals), adaptive retry-damping schedule on ill-conditioned Hessians, input validation, and XPU vs CPU parity.
- **[`tests/test_oracle_parity.py`](./tests/test_oracle_parity.py) (4 tests):** Live verification against `exl3xpu.ref` proving bit-exact identity for `tensor_core_perm`, all 3 codebooks across 65,536 states, `decode_states` across $K \in \{1, 2, 3, 4, 5, 6, 8\}$, and end-to-end encoded tile reconstruction.
- **[`tests/test_encoder.py`](./tests/test_encoder.py) (8 tests):** Validates shift-register inverse identity $((pred \ll K) \mid e) \equiv state$, continuous circular ring topology, bit-exact codebook reconstruction, end-to-end container round-trips for $K \in \{2, 3, 4\}$, path cost tracking, and bounded distortion.
- **[`tests/test_trellis_format.py`](./tests/test_trellis_format.py) (17 tests):** Format-conformance tests covering MSB-first bitstreams, `SWAP16`, state-history properties, and codebook range invariants.
- **[`tests/test_negative.py`](./tests/test_negative.py) (5 tests):** Perturbation harness detecting 1-bit container flips, scale distortions, illegal trellis transitions, non-integer inputs, and insufficient memory budgets.
- **[`tests/test_scratch.py`](./tests/test_scratch.py) (4 tests):** Arithmetic regression fixtures for $K \in \{1, 2, 3, 4\}$ and batch sizing edge cases.
- **[`tests/test_bitpack.py`](./tests/test_bitpack.py) (4 tests):** Legacy sub-byte container round-trip shims.

---

## Phase Reports & Governance Documents

The project maintains an evidence-backed governance model:

- [`reports/gate_0.md`](./reports/gate_0.md) — Gate 0 audit closure report (operator coverage, `block_ldl` benchmark, conditional GO to Phase 1).
- [`reports/gate_1.md`](./reports/gate_1.md) — Gate 1 encoder readiness report (reconstruction parity, bounded scratch, throughput sweeps, GO to Phase 2).
- [`reports/gate_2.md`](./reports/gate_2.md) — Gate 2 single-layer readiness report (Acceptance Tiers 1–4, sustained memory stability with 0 B swap growth, GO to Phase 3).
- [`reports/traceback_decision.md`](./reports/traceback_decision.md) — Comprehensive comparative analysis of `gpu_resident`, `d2h_cpu`, and `usm_host` traceback placements on Lunar Lake UMA.
- [`artifacts/benchmarks/phase2_layer_bench.json`](./artifacts/benchmarks/phase2_layer_bench.json) — Full single-layer sustained conversion benchmark across 3 iterations (peak alloc 649.7 MiB, peak reserved 672.0 MiB, 0 B swap growth).
- [`artifacts/benchmarks/phase1_encoder_bench.json`](./artifacts/benchmarks/phase1_encoder_bench.json) — Complete machine-readable benchmark results across $K \in \{1, 2, 3, 4\}$, codebooks, subgroup sizes, and batch capacities.
- [`EXL3_Intel_Arc_quantizer.md`](./EXL3_Intel_Arc_quantizer.md) — Initial technical assessment, kernel feasibility, and roadmap.
- [`docs/140v.md`](./docs/140v.md) — Arc 140V / Core Ultra 7 258V adaptation guide.
- [`docs/compatibility_manifest.md`](./docs/compatibility_manifest.md) — Pinned hardware/software environment, model target, calibration corpus, 6-tier acceptance criteria, and memory risk analysis.
- [`docs/algorithm_contract.md`](./docs/algorithm_contract.md) — Authoritative 13-section mathematical contract audited from upstream ExLlamaV3.
- [`docs/native_dependency_audit.md`](./docs/native_dependency_audit.md) — Complete classification of all upstream `ext.*` conversion sites and XPU porting strategy.
- [`docs/extension_build.md`](./docs/extension_build.md) — Build and linkage requirements for oneAPI 2026.0 / PyTorch XPU C++ extensions.
- [`docs/deviation_log.md`](./docs/deviation_log.md) — Formal deviation log and review waivers (`DEV-2026-10-20-01`).
- [`tasks.md`](./tasks.md) — Implementation task list and phase gates (Gate 0 through Gate 3; Phase 4 has no gate).
- [`progress.md`](./progress.md) — Detailed implementation progress log with per-entry commit timestamps.
- [`review.md`](./review.md) — Independent code review conclusions and formal remediation responses.

---

## Roadmap & Next Milestones (Phase 3)

With Gate 0, Gate 1, and Gate 2 successfully closed, Phase 3 focuses on bounded full-model conversion and qualification:

1. ~~**P3-01 / P3-02 / P3-03 — Bounded Model Orchestrator & Shard Streaming**~~ **DONE** — `exl3_quant/converter.py` (Step 21, 2026-10-21).
2. **P3-04 / P3-05 / P3-06 / P3-07 / P3-08 — Checkpointing, Resumability & Fault Recovery:** Implement atomic checkpoint publication, manifest recording, interrupted-write recovery, and resume argument validation.
3. **P3-09 .. P3-16 & Gate 3 — First Complete Model Conversion & Evaluation:** Execute full offline conversion of the declared Active Target `empero-ai/Qwen3.8-9B-Distill` — **UNBLOCKED** (Step 24) via the opt-in `architecture_mode="qwen3_5_text_tower"` (P4-12 hybrid quantization contract `[x]`, P4-13 text-tower-only pass-through semantics `[x]`; qwen3_5 hybrid 3:1 `linear_attention`/`full_attention` layers, 27-layer ViT tower preserved as pass-through, partial interleaved mrope, MTP head; sibling of the first-declared `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`). Default-mode rejection stays guarded by `test_declared_target_mimo_is_rejected`, opt-in acceptance by `tests/test_hybrid_converter.py`. Start with a measured single-layer run and documented operating reserve, then the full 32-layer attempt with checkpoint/resume; evaluate perplexity against downstream runtime (`exl3xpu`) and publish `reports/gate_3.md`.

> **CUDA ground truth (`skip_cuda.md` / `SKIP-CUDA-2026-10-21-01`):** deliberately deferred — no eGPU installed; gate evidence is capped at class **E3** (E4 path parity is not claimed). Reopen triggers (MSE-ceiling divergence, reconstruction divergence, an E4-requiring gate, runtime rejection, upstream re-pin/kernel change) and the resume checklist live in [`skip_cuda.md`](./skip_cuda.md) §6–§7.
