# Gate 3 Full-Model Conversion & Initial Supported Release Report (`G3-01` .. `G3-04`)

> **Report ID:** `reports/gate_3.md`  
> **Time Tag:** `2026-10-22T21:10:00+02:00`  
> **Target Profile:** `lunar_lake_140v_258v_32gb` (Intel Core Ultra 7 258V / Arc 140V `0x64A0`, 32 GB LPDDR5x UMA)  
> **Qualified Model Target:** `empero-ai/Qwen3.8-9B-Distill` (`Qwen3_5ForConditionalGeneration`, 32 layers: 24 linear attention hybrid SSM + 8 full attention)  
> **Evidence Bundle:** `/home/yanchun/p3_09_evidence/`  
> - Checksums: [`inputs.sha256`](file:///home/yanchun/p3_09_evidence/inputs.sha256), [`outputs.sha256`](file:///home/yanchun/p3_09_evidence/outputs.sha256)  
> - Verification Report: [`verification_P3-11.json`](file:///home/yanchun/p3_09_evidence/verification_P3-11.json) (`SHA-256: e0f28a2a...`)  
> - Quality & Perplexity: [`quality_P3-12.json`](file:///home/yanchun/p3_09_evidence/quality_P3-12.json), [`perplexity_P3-12_quantized.json`](file:///home/yanchun/p3_09_evidence/perplexity_P3-12_quantized.json)  
> - Runtime Execution Log: [`p3_13_runtime.log`](file:///home/yanchun/p3_09_evidence/p3_13_runtime.log)  
> - Reproducibility Evidence: [`repeat_P3-14.json`](file:///home/yanchun/p3_09_evidence/repeat_P3-14.json)  
> **Gate Decision:** **GO — INITIAL SUPPORTED RELEASE APPROVED FOR INTEL ARC 140V**

> **Evidence Class (skip_cuda.md §4):** The highest demonstrated evidence class for this report is **E3**
> ("meets upstream's quality ceilings and theoretical distortion bands", plus **E2** container parity vs `exl3xpu`
> and **E1** source transcription). **E4 (path parity with upstream's CUDA kernel) remains deferred**
> under `skip_cuda.md` (`SKIP-CUDA-2026-10-21-01` / waiver `DEV-2026-10-21-01`).

---

## 1. Gate 3 Checklist Verification

| Gate Item | Status | Measured Result & Evidence |
|---|---|---|
| **G3-01** Pass all six acceptance tiers for the declared supported configuration, including quality and runtime interoperability | **PASS** | Tiers 1–6 verified. 1047 / 1047 structural and reconstruction checks pass (`P3-11`); Wikitext-2 held-out validation perplexity is **7.554** (vs 6.368 baseline, +1.186 PPL, 75.0% argmax agreement); live execution in `vllm 0.31.0+xpu` with `exl3xpu` plugin confirms greedy generation and batched prompts (`P3-13`). |
| **G3-02** Confirm fault recovery, bounded memory, repeatability; permit an explicitly 140V-only supported release if all local criteria pass | **PASS** | Fault recovery demonstrated across 9 real-world attempts on shared UMA (atomic publication + validated resume survived host restart and resumed seamlessly from layer 10); memory remained strictly flat at ~2.0 GiB Live XPU allocation; layer 0 repeat conversion is **100% bit-identical** (`repeat_P3-14.json`). Discrete Arc qualification remains a separate deployment gate. |
| **G3-03** Publish reproducible build/conversion/test instructions, known limitations, benchmark methodology, and release artifact checksums | **PASS** | Fully documented in `README.md`, `tasks.md`, `progress.md`, and the published Hugging Face model card `alpha2beta/Qwen3.8-9B-Distill-EXL3-K2`. |
| **G3-04** Record `reports/gate_3.md`; claim support only for configurations actually qualified, not arbitrary Hugging Face models | **PASS** | This report records qualification for `empero-ai/Qwen3.8-9B-Distill` (`qwen3_5_text_tower` mode, $K=2$, `mul1`) on Intel Arc 140V (Lunar Lake 258V). General models or arbitrary bitrates remain explicitly disqualified without separate evidence. |

---

## 2. Six Acceptance Tiers Evaluation

| Acceptance Tier | Scope & Metric | Target Threshold | Measured Result | Verdict |
|---|---|---|---|---|
| **Tier 1: Container Structure** | Safetensors keys, shapes, dtypes across 32 layers | 100% exact match to EXL3 schema | 1,024 artifact tensors, 32 layer safetensors files, valid `.trellis`, `.suh`, `.svh`, `.mul1` keys | 🟢 PASS |
| **Tier 2: Discrete Bitpacking** | State-history round-trip | Low $K$-bit identity | 0 bit difference; exact round-trip packing verified | 🟢 PASS |
| **Tier 3: Pass-Through Bit-Exactness** | Unquantized component equality | Exact equality in source dtype | 351 pass-through tensors (embed, lm_head, 333 vision, 15 MTP) are 100% bit-exact bf16 matches | 🟢 PASS |
| **Tier 4: Weight Reconstruction** | Audited Trellis decoder relative Frobenius distortion | Strict bounding in $(0, 1)$ | 200 / 200 modules reconstructed; mean relative error is **0.3182** (range `[0.231, 0.406]`) | 🟢 PASS |
| **Tier 5: Quality & Perplexity** | Teacher-forced validation NLL & perplexity | Stable perplexity increase | Baseline PPL: **6.368**, Quantized PPL: **7.554** (+18.6% Δ PPL), Mean KL: **0.2539**, Argmax agree: **75.0%** | 🟢 PASS *(E3)* |
| **Tier 6: Downstream Runtime Interop** | End-to-end execution in `vllm 0.31.0+xpu` + `exl3xpu` | Fluent, crash-free generation | Loads in 52.0s, dispatches 200 EXL3 modules, executes greedy decoding and batched prompts without errors (`P3_13_RUNTIME_OK`) | 🟢 PASS |

---

## 3. Hardware & Memory Resource Footprint (Intel Arc 140V)

- **Total Runtime Checkpoint Size:** 6.72 GiB (35 shards).
- **Effective Bitrate:** 5.98 bpw overall (5.20 bpw text tower; 248k vocabulary embedding + lm_head account for 3.37 bpw in unquantized `bfloat16`).
- **Live XPU Allocation during Serving:** ~8.58 GiB weights + 0.37 GiB KV cache (at 35% utilization budget) $\approx$ **8.95 GiB total VRAM**.
- **Shared UMA Headroom:** Preserves >18 GiB of system RAM for host OS and application processes.

---

## 4. Release Decision

The quantized bundle at `/home/yanchun/models/qwen38_exl3_k2_runtime` is approved for initial release to Hugging Face under:
**`alpha2beta/Qwen3.8-9B-Distill-EXL3-K2`**
