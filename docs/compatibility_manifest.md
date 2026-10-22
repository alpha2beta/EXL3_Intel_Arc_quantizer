# Compatibility Manifest: Intel Core Ultra 7 258V / Arc 140V (Lunar Lake)

> **Document ID:** `docs/compatibility_manifest.md`  
> **Profile:** `lunar_lake_140v_258v_32gb`  
> **Date:** 2026-10-20  
> **Tasks:** P0-05, P0-08, V140-01  
> **Status:** Platform environment pinned and empirically probed; encoder/layer qualification in progress.

---

## 1. Hardware Specification & Queried Device Identity

| Parameter | Specification / Probed Value | Provenance / Query Source |
|---|---|---|
| **CPU Model** | Intel(R) Core(TM) Ultra 7 258V | `lscpu` (8 cores, family 6, model 189, stepping 1) |
| **System Memory** | 32 GB on-package LPDDR5x | `free -h` (30.0 GiB available to Linux kernel, 30.0 GiB swap) |
| **Memory Architecture** | Unified Physical Pool (UMA / shared system memory) | Integrated graphics with shared memory controller |
| **Integrated GPU** | Intel Arc 140V Graphics | PCI `00:02.0` (device ID `0x64A0`, rev 04) |
| **Neural Processing Unit** | Intel Core Ultra 200V NPU | PCI `00:0b.0` (device ID `0x643e`, rev 04) |
| **GPU Compute Units (EUs)** | 64 Vector Engines (XVEs) / 8 Xe-cores (subslices) | `torch.xpu.get_device_properties(0)` (`max_compute_units=64`, `gpu_subslice_count=8`) |
| **Local Memory (SLM)** | 128 KiB per work-group | `torch.xpu.get_device_properties(0).local_mem_size` |
| **Last-Level Cache (L3)** | 8192 KiB (8 MiB) | `torch.xpu.get_device_properties(0).last_level_cache_size` |
| **Supported Subgroups** | SIMD16, SIMD32 (`[16, 32]`) | `torch.xpu.get_device_properties(0).sub_group_sizes` |
| **Max Work-Group Size** | 1024 work-items | `torch.xpu.get_device_properties(0).max_work_group_size` |
| **Native Dtype Flags** | `has_fp16=1`, `has_fp64=1`, `has_atomic64=1` | `torch.xpu.get_device_properties(0)` |
| **Device Classification** | `is_integrated_gpu=1` | `torch.xpu.get_device_properties(0)` |

---

## 2. Software Stack & Toolchain Provenance

| Component | Pinned Version / Path | Build & Package Provenance |
|---|---|---|
| **Operating System** | Linux `7.2.11-1-cachyos-deckify` x86_64 | Arch Linux / CachyOS kernel |
| **Python Runtime** | `3.12.14` | `/home/yanchun/Projects/vllm.xpu/.venv/bin/python` (`uv` CPython build) |
| **Build System** | Ninja `1.13.2` + `torch.utils.cpp_extension` | Bundled `pybind11` headers via PyTorch `2.14.0+xpu` |
| **Intel Compute Runtime** | `26.35.39758.10-1.1` | `local/intel-compute-runtime` (Level Zero & OpenCL NEO driver) |
| **Level Zero Loader** | `1.32.0-1` | `local/level-zero-loader` |
| **Unified Runtime Driver** | `1.17.39758` | Intel Level-Zero V2 driver reported by PyTorch XPU |
| **oneAPI DPC++/C++ Compiler** | `2026.0.0_947-1.1` | `/opt/intel/oneapi/compiler/2026.0/bin/icpx` |
| **oneAPI MKL & SYCL BLAS** | `2026.0.0_908-1.1` | `/opt/intel/oneapi/mkl/2026.0` |
| **PyTorch XPU Runtime** | `2.14.0+xpu` | Verified working on `xpu:0` in `/home/yanchun/Projects/vllm.xpu/.venv` |
| **Triton XPU** | `3.8.0+xpu` | Installed in active Python environment |
| **Validation Decoder Oracle** | `exl3xpu 0.1.0` | Co-located at `/home/yanchun/Projects/exl3xpu` (`exl3xpu.ref`) |

---

## 3. Pinned Upstream Repository Revisions

- **Upstream Repository:** `https://github.com/turboderp-org/exllamav3`
- **Pinned Commit:** `4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f`
- **Audited Modules:**
  - `exllamav3/modules/quant/exl3_lib/quantize.py` (`1B11658F5DE96BFE`)
  - `exllamav3/exllamav3_ext/quant/quantize_tiles_kernel.cuh` (`F8399AA3D281625F`)
  - `exllamav3/exllamav3_ext/quant/codebook.cuh` (`0E3C63B323F8D3CC`)
  - `exllamav3/exllamav3_ext/quant/pack.cu` (`27606EED6650ACC3`)
  - `exllamav3/exllamav3_ext/quant/reconstruct.cu` (`F47F121CDB452CF8`)
  - `exllamav3/util/hadamard.py` (`6884841B61378788`)

---

## 4. Memory Accounting & Admission Model

### 4.1 Physical vs Accounting Boundaries

1. **One Physical Pool:** 32 GB system memory is dynamically partitioned between CPU host processes and the integrated GPU driver. There is no discrete VRAM bank.
2. **Runtime Memory Query:** `torch.xpu.mem_get_info()` reports `(free, total)`. Queried total is `30,692,810,752` bytes (~28.6 GiB); measured idle free is **~8.0–8.4 GiB** (it varies with background workloads) while `/proc/meminfo` reports **~13.0–13.9 GiB** `MemAvailable`. The gap is the driver's own pool, which is not visible as free system memory.
3. **Counters Lag Commitments (measured 2026-10-20):** a touched 1 GiB XPU tensor moved `mem_get_info` free by only **~9 MiB** and `MemAvailable` by **~200 MiB**, while `torch.xpu.memory_allocated` moved by **exactly 1024 MiB**. Admission checks must therefore subtract the framework's live allocation total (`memory_allocated`) explicitly; free counters alone over-admit. See `exl3_quant/memory_policy.py` and `artifacts/environment/scratch_allocations.json`.
4. **No Double Counting:** CPU RSS, GPU allocations, shared mappings, and driver accounting overlap. Do not sum RSS and GPU-allocated bytes as independent physical quantities. Measured device-free and `MemAvailable` deltas track each other (within noise) during allocations, confirming one shared backing.
5. **Guardrail Reserve:** A minimum configurable reserve of **6.0 GiB** system memory must remain untouched by live allocations to prevent OS thrashing, page faults, and swap swapping. Enforced by `MemoryPolicy` (`vram_reserve` configurable per run).
6. **Level Zero Single Allocation Boundary:** Tested allocations must obey Level Zero max single allocation queries (measured `max_mem_alloc_size` = `30,692,810,752` B, i.e. the full reported device total).

---

## 5. Pinned Model Target & Calibration Specification (P0-02, P0-06)

| Parameter | Pinned Specification |
|---|---|
| **Primary Dense Architecture** | Meta Llama 3.1 8B (`meta-llama/Meta-Llama-3.1-8B`) |
| **Source Checkpoint Revision** | `48d6d03f933bc570042267aa51a83355a67738d4` (fp16 / bfloat16 base weights) |
| **Secondary Target Architecture**| Qwen 2.5 7B (`Qwen/Qwen2.5-7B`, commit `d4385dd2a52377042ab34a883a451e505ea6bfda`) |
| **Calibration Corpus** | `wikitext-2-raw-v1` (test split) |
| **Calibration Tokens / Samples** | 128 samples $\times$ 2048 sequence length (262,144 tokens total) |
| **Sequence Handling** | Deterministic non-overlapping chunks, left-truncated, seed `42` |
| **Hessian Covariance Dtype** | float32 accumulate ($H = \sum X^T X$), damped with $\lambda = 0.01 \cdot \text{mean}(\text{diag}(H))$ |

> **Amendment (2026-10-21, owner decisions — first "Keep MiMo, flag as blocked", then target swap):** the **declared Active Target** for the first complete conversion (`P3-09`) is now [`empero-ai/Qwen3.8-9B-Distill`](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill), succeeding the owner's first declaration [`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) (2026-10-21). Both are dense BF16 9B (≈ 18 GB; `config.json` of each verified 2026-10-21) and **sibling distills of the same Qwen3.5-VL base** — architecturally **outside this contract and upstream EXL3**: `Qwen3_5ForConditionalGeneration` / `qwen3_5` with 24 of 32 `text_config` layers of type `linear_attention` (3:1 hybrid SSM pattern, `full_attention_interval` 4), a 27-layer ViT `vision_config` plus top-level `image_token_id`/`video_token_id`/`vision_start_token_id`/`vision_end_token_id`, `partial_rotary_factor` 0.25 with interleaved mrope (`mrope_section` [11, 11, 10]), and `mtp_num_hidden_layers` 1. Status: **BLOCKED** on the `P4-12`/`P4-13` architecture-extension track (`tasks.md` §4.1); the converter deterministically rejects both exact config shapes (regression-tested in `tests/test_converter.py::test_declared_target_mimo_is_rejected`). The pinned primary/secondary architectures above are **retained as the contract-compatible fallbacks** for an owner-approved substitute and remain the reference for the pinned calibration recipe.

> **Amendment 2 (2026-10-21, P4-12/P4-13 landed — status change to UNBLOCKED):** the architecture-extension track now exists, so the declared Active Target above is **convertible** through the opt-in `architecture_mode="qwen3_5_text_tower"` (default `"dense"` behavior and every historical rejection are unchanged and still regression-guarded). The extended contract, evidenced in `tests/test_hybrid_converter.py` (31 tests; full suite 249/249) and recorded in the conversion manifest:
>
> - **Quantized (unchanged EXL3 chain):** every bias-free linear projection of both hybrid layer kinds — `linear_attn.in_proj_qkv`/`in_proj_z`/`out_proj` + `mlp.gate/up/down` (`linear_attention` layers), `self_attn.q/k/v/o_proj` + `mlp.gate/up/down` (`full_attention` layers). All k/n widths are multiples of the 128-wide Hadamard block, satisfying the pinned encoder's `ldlq`/`regularize`/`blockwise_preapply_had_r_` constraints.
> - **Bit-exact pass-through (source dtype preserved, never quantized):** `linear_attn.in_proj_a`/`in_proj_b` (n = num_value_heads — fails the 128-block contract; 0.09 % of the text tower's linear bytes), `linear_attn.conv1d.weight`, `A_log`, `dt_bias`, `linear_attn.norm`, `self_attn.q_norm`/`k_norm`, both layer norms per layer, the text embed/final-norm/lm_head, the whole ViT tower (`model.visual.*`), and the MTP head (`mtp.*` — quantization remains parked at P4-04). Pass-through tensors are partitioned into per-group files (`qtensors/untouched.safetensors`, `qtensors/untouched.vision.safetensors`, `qtensors/untouched.mtp.safetensors`) so no single serialization holds all pass-through bytes; each group's file and keys are recorded in the manifest (`pass_through`, `quant_format: source-dtype`).
> - **Pinned reference forward:** calibration uses `transformers.models.qwen3_5` decoder layers (transformers ≥ 5.17, eager attention, causal mask on full-attention blocks, partial rotary 0.25 + interleaved mrope, zero-centered RMSNorm) with reference-PyTorch fallbacks for `causal_conv1d`/`flash-linear-attention` recorded under `hybrid.reference_forward`; the environment block records python/platform/torch.
> - **Out of scope (unchanged):** quantizing ViT/MTP weights (P4-04), MoE (P4-05), and `exl3xpu` runtime interoperability for hybrid checkpoints (tier 6 — verified under P3-13 with real conversion output, see **Amendment 4**). The pinned primary/secondary architectures above remain the contract reference for the dense recipe.

> **Amendment 3 (2026-10-22, P4-14 landed — the *text-only* container of the same layer contract is now accepted):** the extension track above validated only the **multimodal** packaging (`Qwen3_5ForConditionalGeneration` / `qwen3_5`, mandatory `text_config` block, decoder stack nested under `model.language_model`). The identical layer contract also ships in a **text-only** packaging — `Qwen3_5ForCausalLM` / `model_type: qwen3_5_text`, a *flat* `config.json` that **is** the text-tower config (no `text_config` block, no vision tower, no MTP weights), and the decoder stack at the top-level `model.layers.` prefix. Reference checkpoint: [`autotrust/JEV-9B`](https://huggingface.co/autotrust/JEV-9B) — the Qwen3.5-9B text backbone with the vision tower and MTP head dropped (427-key weight index verified 2026-10-22). Contract details:
>
> - **Accepted only unambiguously:** the flat layout is recognized by the explicit text-only markers (`architectures` containing `Qwen3_5ForCausalLM` or `model_type: qwen3_5_text`). A multimodal `qwen3_5` config that is merely missing its `text_config` block remains a hard rejection — it is never reinterpreted as a text-tower config.
> - **Prefixes are derived, not hardcoded:** `nested → model.language_model.layers` / `model.language_model`, `flat → model.layers` / `model`; the layout actually converted is recorded in the manifest (`hybrid.text_tower_container`). All geometry, MoE, 128-wide Hadamard-width, layer-type and rope validation applies verbatim to both containers, and the P4-12 quantized/pass-through module tables are unchanged (one layer contract, two packagings).
> - **Provenance vs. emission:** `has_vision`/`has_mtp` record what the *config declares*; emitted pass-through groups remain *key-derived*. JEV-9B declares `mtp_num_hidden_layers: 1` yet ships no `mtp.*` tensors, so no MTP group file is produced and the manifest cannot claim one.
> - **Downstream runtime:** no change required — `exl3xpu`'s key normalization reduces any prefix to `layers.N.*`, and with no `mtp.*` tensors the runtime's MTP path is skipped. Runtime interoperability is now verified for a converted **multimodal** checkpoint (Amendment 4, text-only mode); a converted checkpoint in *this* text-only-container packaging has not had an end-to-end run yet.
> - **Not qualified by this amendment:** converting a JEV-9B-class checkpoint end to end (owner-approved target decision + a free XPU), and its **System-1 block** — the LoRA r=16 (`adapter/`, 154 MB) and the 24-slot fp32 decision head (`head.safetensors`) live *outside* the backbone safetensors. Options and their measured trade-offs are recorded in `tasks.md` P4-14 (quantize the pristine backbone only / merge the LoRA as a separate recipe with its own calibration and quality evaluation / serve the LoRA on the quantized base, which needs `exl3xpu` work). Estimated output for such a run, to be measured: ≈ 5.2–5.3 bpw / ≈ 5.8 GiB (8.95 B parameters, no vision bytes).
> - **Evidence:** `tests/test_hybrid_converter.py` (36 tests; new `TestTextOnlyContainer` covering replica acceptance, the dense-mode rejection, the missing-text-block guard, key coverage of the flat fixture's pass-through partition, and a strict reference-layer build from the flat layout); CPU-only subsets 20/20 hybrid + 18/18 converter config/index; the real JEV-9B `config.json` validated end to end (commit `046b079`). Default `dense` mode and the multimodal path are unchanged.

> **Amendment 4 (2026-10-22, P3-09/P3-11/P3-12/P3-13 evidence — the declared target converts, verifies and runs in the pinned runtime):** the 32-layer conversion of `empero-ai/Qwen3.8-9B-Distill` completed (`status: completed`, 8.8 GB bundle, `rc=0`); one late device-lost (`UR_RESULT_ERROR_DEVICE_LOST`, after layer 30 was published) was recovered by the supervisor at layer 31 with zero re-quantization. Independent verification (P3-11, [`scripts/verify_conversion_output.py`](./../scripts/verify_conversion_output.py)) passed **1047/1047 checks**: 32 layers, 1024 artifact tensors, 351 pass-through tensors bit-exact against the source checkpoint, 200 quantized modules reconstructed, decode relative error mean 0.318 / max 0.406. The runtime export (P3-13: 35 shards / 1375 tensors / 6.72 GiB) loads in vLLM `0.31.0+xpu` with the `exl3xpu` plugin (`exl3: 200 EXL3 modules (0 in MTP head)`, 52 s load) and produced coherent greedy text for a single request and for a 2-request batch in text-only mode (`language_model_only=True`, `dtype=float16`, `enforce_eager=True`).
>
> - **Tier-6 coverage actually exercised:** text-only decode of the *converted* checkpoint. `language_model_only=True` means the vision pass-through group is never loaded by the runtime, and the MTP head is neither quantized nor loaded — no claim is made about the vision or MTP paths in the runtime.
> - **Quality (P3-12)** on 2044 held-out `wikitext-2` validation positions (4 × 512, disjoint from the test-split calibration corpus): floating-point baseline NLL **1.8513** / PPL **6.368**; quantized (repo-side decode + reference forward) NLL **2.02208** / PPL **7.5540** (+9.2 % NLL, +18.6 % PPL); mean KL **0.254** over 128 sampled positions; argmax agreement **0.750**. The *runtime* teacher-forced NLL on the identical token ids is **2.022106** (PPL 7.5542) — agreement with the repo-side reconstruction to **1.4e-5** relative, i.e. the runtime executes the published trellis faithfully.
> - **Runtime coverage that is *not* available (explicit):** (a) the floating-point baseline *inside the runtime* is blocked on this device: the fp16 text tower plus top-level tensors is 17.13 GiB (14.78 GiB `model.language_model.*` + 2.35 GiB) against 16.07/28.58 GiB reported free at engine startup, so vLLM refuses the load before weights are read; the floating-point reference is the repo-side forward on identical tokens, and the 1.4e-5 runtime/repo agreement above shows that is a valid reference for the runtime path; (b) the matched **CUDA-quantized reference** stays deferred under [`skip_cuda.md`](./../skip_cuda.md) / `SKIP-CUDA-2026-10-21-01`; (c) Tier 5's CUDA-referenced ΔPPL criterion remains deferred with it — the ΔPPL above is measured against the floating-point baseline.
> - **Runtime environment prerequisites (measured):** `PATH=/opt/intel/oneapi/compiler/latest/bin:$PATH`, `LD_LIBRARY_PATH=<venv>/lib:$LD_LIBRARY_PATH` — the oneCCL V2 core `libccl.so.2` has no `RUNPATH` and `dlopen`s its legacy plugin `libccl.so.1` by bare name, so without it engine startup aborts with `oneCCL could not be initialized! Could not load any plugin` — and `CPATH=<vllm.xpu>/tools/sysroot/usr/include:$CPATH` — the distro level-zero headers ship no `level_zero/ze_api.h`, so the Triton JIT fails with `fatal error: 'level_zero/ze_api.h' file not found`. [`scripts/evaluate_perplexity.py`](./../scripts/evaluate_perplexity.py) now fails fast with the exact export lines.

---

## 6. Prototype Quantizer Scope & Target Recipes (P0-03)

- **Initial Scope:** Single dense decoder layer conversion (attention projections + MLP).
- **Codebook:** `mul1` (`cb = 2`), affine pseudorandom byte-sum LUT over 16-bit state (913 distinct levels in `[-3.453125, 3.34765625]`).
- **Bitrate Parameter K:** Integer $K \in \{2, 3, 4\}$.
- **Tile Format:** $16 \times 16$ elements ($L = 256$), pre-permuted into `tensor_core_perm` order.
- **Trellis States:** $S = 65536 \gg K$ (16,384 for $K=2$; 8,192 for $K=3$; 4,096 for $K=4$).
- **Packing Format:** State-history container: 16 spans $\times$ 16 elements, low $K$ bits MSB-first into uint16 words, uint32 little-endian pairs with `SWAP16` half-word swap.
- **Downstream Oracle Compatibility:** Bit-exact decode verified live against `exl3xpu.ref` (`tests/test_oracle_parity.py`).

---

## 7. Numerical Acceptance Criteria & Tolerances (P0-07)

| Acceptance Tier | Scope & Metric | Required Threshold | Justification |
|---|---|---|---|
| **Tier 1: Container Structure** | Tensor names, dtypes, metadata | 100% exact match | Format compatibility with ExLlamaV3 `.safetensors` |
| **Tier 2: Discrete Bitpacking** | State-history round-trip (`pack→decode`) | 0 bit difference | Lossless container round trip for low $K$ bits |
| **Tier 3: Weight Reconstruction** | Reconstructed fp16 weights vs LUT | 0 bit difference | Exact codebook index lookup |
| **Tier 4: Quantizer Numerics** | Viterbi path validity & distortion | 100% valid transitions; $\text{MSE} < 0.15$ on $N(0,1)$ at $K=2$ | Monotonic distortion improvement with increasing $K$ |
| **Tier 5: Model Perplexity** | Full model Wikitext-2 perplexity | $\Delta \text{PPL} \le 0.15$ vs CUDA reference | Quality gate preventing catastrophic degradation |
| **Tier 6: Oracle Runtime Interop** | Live decode through `exl3xpu.ref` | 100% bit-exact across all shapes & $K$ | Prevents runtime decode divergence |

> **Status of the runtime-facing tiers (2026-10-22):** Tier 4 and Tier 6 numerics are verified bit-exact against the oracle (`tests/test_oracle_parity.py`, `tests/test_xpu_quantizer.py`). Tier 5 is measured against the **floating-point baseline** on the declared target (P3-12: PPL 6.368 → 7.554 at `K=2`/`mul1`, qwen3_5 hybrid); its CUDA-referenced ΔPPL criterion stays deferred with `SKIP-CUDA-2026-10-21-01`, and the runtime-load coverage (text-only mode; vision/MTP paths unexercised) is documented in Amendment 4.

---

## 8. Explicit Exclusions & Out-of-Scope Modes

The following modes are explicitly excluded from the Arc 140V initial prototype:
- **MoE (Mixture of Experts):** Non-dense routing topologies deferred to Phase 4.
- **Vision / Audio / Multimodal:** Projections with specialized tokenizers excluded.
- **Multi-Token Prediction (MTP):** Speculative decoding auxiliary heads deferred.
- **Two-Sided LDLQ:** Output Hessian / column damping modes deferred.
- **Fractional Average Bitrates:** Sub-integer mixed-bitrate scheduling deferred.
- **Discrete Arc / Windows:** Relocated to separate optional deployment tracks.

---

## 9. Arc 140V Live Headroom & Allocation Risk (V140-02, V140-07)

- **Measured Idle Baseline [2026-10-20 20:01:23 +0200]:** `MemAvailable` 13.0–13.9 GiB, `mem_get_info` free 8.0–8.4 GiB, swap free 29.5 GiB (≈1.4 GiB already used by other processes). Sampling: 10 snapshots at 1 s intervals (`artifacts/environment/idle_baseline.json`).
- **Configured System Reserve:** 6.0 GiB (mandatory to avoid kernel swapping and driver thrashing).
- **Usable Live Headroom:** **≈ 2.0–2.4 GiB** = `min(device free, MemAvailable) − 6.0 GiB`, before subtracting live framework allocations. This is lower than the earlier 4.0 GiB estimate because the measured device free is ~8.0–8.4 GiB, not ~10 GiB.
- **Measured Allocations Through the Framework Allocator** (`artifacts/environment/scratch_allocations*.json`, `K=1`, `T=256`):
  - `B=1` (16.1 MiB), `B=16` (258 MiB), `B=64` (1.01 GiB): allocated and touched successfully at the 6 GiB reserve (3–51 ms allocation).
  - `B=128` (2.02 GiB): succeeds only when device free ≥ ~8.0 GiB; with a 4 GiB reserve it completes in 79 ms.
  - `B=192` (3.02 GiB): succeeds at a 4 GiB reserve (1.55 s allocation), denied at 6 GiB.
  - `B=256` (4.03 GiB): denied at both reserves (measured bound `B ≤ 222` at 4 GiB reserve; `B ≈ 94` at 6 GiB on the sampled idle state).
  - Oversized 20 GiB request: refused by policy before any allocation.
  - No swap growth in any case; `torch.xpu.memory_allocated` matched the requested bytes exactly.
- **Policy Directive:** On Arc 140V, clamp scratch batch capacity to what the *live* budget admits (`MemoryPolicy.check`), with `B ≤ 64` for `K=1` (1.0 GiB) during full-layer execution to leave headroom for activations, Hessian accumulation, and PyTorch workspace. Do not hard-code batch sizes from the 32 GB label; the device-visible free pool is the binding constraint.
