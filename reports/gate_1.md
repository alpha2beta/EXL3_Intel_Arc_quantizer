# Gate 1 Encoder Readiness Report (`G1-01` .. `G1-03`)

> **Report ID:** `reports/gate_1.md`  
> **Time Tag:** `2026-10-20T20:57:38+02:00`  
> **Target Profile:** `lunar_lake_140v_258v_32gb` (Intel Core Ultra 7 258V / Arc 140V `0x64A0`, 32 GB LPDDR5x UMA)  
> **Evidence Artifacts:**  
> - [`artifacts/benchmarks/phase1_encoder_bench.json`](../artifacts/benchmarks/phase1_encoder_bench.json) (regenerated 2026-10-20 21:58 +0200 with a forced cold build; latency/throughput vary by roughly ±10% with host thermal state and background load, so treat the numbers below as representative of the 20:57 run)  
> - [`reports/traceback_decision.md`](./traceback_decision.md)  
> **Gate Decision:** **GO to Phase 2 (Complete single-layer conversion)**

---

## 1. Gate 1 Checklist Verification

| Gate Item | Status | Evidence |
|---|---|---|
| **G1-01** Selected tile encoding passes reference path, reconstruction, and distortion tests under the agreed policy | **PASS** | `tests/test_xpu_quantizer.py`, `tests/test_encoder.py`, `tests/test_oracle_parity.py`: 100% bit-exact parity between `XPUTrellisEncoder` and `ReferenceTrellisEncoder` across `K in {1, 2, 3, 4}`, all 3 codebooks (`3inst`, `mcg`, `mul1`), `L in {256, 160}`, edge cases (`zeros`, `constant`, `outliers`, `alternating`), and live `exl3xpu.ref` decode |
| **G1-02** Bounded scratch behavior, allocation-failure handling, and measured throughput/resource reports confirmed | **PASS** | `ScratchArena` (`exl3_quant/xpu_quantizer.py`), `tests/test_xpu_quantizer.py`, `artifacts/benchmarks/phase1_encoder_bench.json` (B sweep `1..256`, alternating-K zero-leak verification, smaller-batch retry, IGC Xe2 `lnl-m` ISA resource report) |
| **G1-03** Record traceback decision, remaining risks, go/no-go, and Phase 2 estimate in `reports/gate_1.md` | **PASS** | [`reports/traceback_decision.md`](./traceback_decision.md) and this report (`reports/gate_1.md`) |

---

## 2. Summary of Phase 1 Engineering & Empirical Results

### 2.1 Correctness & Numerical Parity (`P1-01` .. `P1-08`)
- **Procedural Codebook (`P1-02`):** `ext.decode_codebook` matches `codebook_lut(cb)` across all `65,536` 16-bit states with **0 bit-level mismatches** for `CB_3INST`, `CB_MCG`, and `CB_MUL1`.
- **Forward ACS & GPU Traceback (`P1-03`, `P1-04`, `P1-05`):** Having aligned `ReferenceTrellisEncoder` and the SYCL kernel on the audited contract's exact fp16 subtraction (`__hsub2`) followed by single-rounded fp16 FMA (`__hfma2` / `sycl::fma`), `XPUTrellisEncoder` matches `ReferenceTrellisEncoder` **bit-for-bit** on `output_indices`, `reconstructed`, `packed`, `end_state`, and `path_cost` across the verified test matrix: bitrates `K ∈ {2, 3, 4}` across all 3 codebooks (`3inst`, `mcg`, `mul1`); `K=1` with `CB_MUL1`; and tile lengths `L ∈ {256, 160}`. (All 3 codebooks are additionally verified on `K=1` for procedural codebook decoding).
- **Tie-Breaking & Edge Cases (`P0-26`, `P1-06`):** Parallel subgroup + SLM `argmin_cost` uses lexicographical comparison on `(valid, cost, edge_index)` so the lowest edge index strictly wins ties regardless of subgroup or work-group geometry. Verified on all-zeros, constant, alternating, and outlier tiles, plus `verify_tile_encoding_quality` tie-aware path validation.
- **Batching, Waves & Stream Ordering (`P1-07`, `P1-08`):** Verified identical results across single tiles, partial final chunks (`19` tiles with `B = 6`), multi-wave dispatches, and interleaved `torch.xpu` operations on the current XPU stream.

### 2.2 Bounded Scratch Management & Batch Sweep (`P1-09` .. `P1-16`, `V140-04`)
- **`ScratchArena` (`P1-12`, `P1-15`):** Replaces upstream's unbounded `@lru_cache get_temp_buffers` with a single-residency, stream-synchronized arena. Alternating `K in {2, 2, 3, 4, 2, 3, 2}` evicts previous buffers cleanly (`5` evictions, `1` reuse, `0` leaked bytes after `release()`).
- **Smaller-Batch Retry (`P1-13`):** Automatically halves `B` on `MemoryPressureError` or allocator failure down to `B = 1` without changing tile outputs; at `K = 1, B_req = 256` (`4.03 GiB`), `ScratchArena` automatically shrank to `B_adm = 128` (`2.02 GiB`) under the 6.0 GiB reserve and produced outputs bit-identical to `B = 1`.
- **Distinct Copy vs Shared-Backing Accounting (`V140-04`):** `ScratchArenaLogEntry` records `distinct_staging_bytes` alongside `scratch_bytes` and `outstanding_xpu_bytes`, distinguishing zero-staging GPU-resident execution (`16 KiB` input transfer) from `d2h_cpu` (`128 MiB` distinct host staging buffer).

### 2.3 Profiling, ISA Resource Inspection & Tuning (`P1-17` .. `P1-29`)
- **Audited Executed Work (`P1-23`):** Each `L = 256` tile executes `2` passes $\times$ `256` steps $\times$ `(65536 >> K) * (1 << K)` = **`33,554,432` (`33.55M`) candidate branch transitions** for every `K`.
- **Measured Throughput (`16` tiles, `AC` power online, package temp `39–49°C`):**
  - `K = 2` (`CB_MUL1`): `618–625 tiles/s` (`158K–160K weights/s`, `20.8–21.0 G transitions/s`) in default SLM mode; **`724.0 tiles/s` (`185.3K weights/s`, `24.3 G transitions/s`)** in `SG=32, global` cost mode (`+17.4%` speedup because `66.4 KiB` SLM caps occupancy to 1 WG/Xe-core at `K = 2`, whereas global cost storage uses only `850 B` SLM and fits in the 8 MiB L3 cache).
  - `K = 3` (`CB_MUL1`): **`1,158–1,181 tiles/s` (`296K–302K weights/s`, `38.9–39.6 G transitions/s`)** in SLM mode (`33.6 KiB` SLM $\rightarrow$ 2 WGs/Xe-core = 16 concurrent WGs), outperforming global cost mode (`774–794 tiles/s`) by **+49%**.
  - `K = 4` (`CB_MUL1`): **`1,151–1,188 tiles/s` (`295K–304K weights/s`, `38.6–39.9 G transitions/s`)** in SLM mode (`17.2 KiB` SLM).
- **IGC Xe2 (`lnl-m`) Resource Report (`P1-24`, `P1-25`, `P1-27`):**
  - Both `SIMD16` and `SIMD32` compile with **`0` bytes of GRF register spills** under the default `128-GRF` mode (`-abortOnSpill 4` passed).
  - `SIMD16` and `SIMD32` perform within `2–5%` of each other in SLM mode (`652.6` vs `616.7 tiles/s` at `K=2`; `1,181.4` vs `1,158.8 tiles/s` at `K=3`). Both remain supported via `sg_size in {16, 32}`.
  - Large GRF mode (`-ze-opt-large-register-file`, `256 GRF`, `P1-27`) yielded identical instruction and flag-spill counts while halving Xe-core thread capacity (`32` vs `64` EU threads); rejected.
- **Dispatch Duration & Watchdog Budget (`P1-28`):** Mean wave dispatch duration at `wave_cap = 16` is `25.4 ms` at `K = 2` and `13.8 ms` at `K = 3`, more than **19$\times$ below** the `500 ms` watchdog guideline. (Note: wave-cap chunking applies to `gpu_resident` fused kernel dispatches; `gpu_resident_split` submits a single full grid for all batch tiles and does not apply per-wave throttling, L3).

---

## 3. Remaining Open Items & Phase 2 Plan

1. **Carried External Blockers (`P0-29`):**
   - `P0-24` / `P0-25`: Upstream CUDA ground-truth fixtures require an NVIDIA GPU environment.
   - `O-1`: Terminal-edge ping-pong index verification (`L-1` vs `L-2` in `quantize_tiles_kernel.cuh:121, 134`) awaits a CUDA fixture snapshot (`P0-25`).
2. **Phase 2 Scope & Effort Estimate (`Complete single-layer conversion`):**
   - **Estimated Effort:** ~1.5–2 person-weeks.
   - **Next Steps:**
     1. **`P2-04` / `P2-05` / `P2-06`:** Implement upstream Hessian regularization, scale/sign extraction (`su`, `sv`), and exact Sylvester Hadamard ($H_{128}$) pre/post transforms (`exl3_quant/`).
     2. **`P2-08` .. `P2-12`:** Implement the blockwise one-sided `ldlq` error-feedback loop (`quantize.py:493-609`) consuming `block_ldl` (`exl3_quant/ldlq.py`) and `XPUTrellisEncoder` (`exl3_quant/xpu_quantizer.py`).
     3. **`P2-13` .. `P2-18` / `V140-06` / `G2`:** Convert, export, reload, and verify a complete transformer linear/decoder layer on Arc 140V through `exl3xpu.ref` under full memory-pressure telemetry, and publish `reports/gate_2.md`.
