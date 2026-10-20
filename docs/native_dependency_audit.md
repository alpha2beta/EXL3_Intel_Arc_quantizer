# Native Dependency Audit (P0-15, P0-16)

> **Document ID:** `docs/native_dependency_audit.md`
> **Date:** 2026-10-20
> **Tasks:** P0-15 (classify every dependency), P0-16 (publish the audit)
> **Related:** [`algorithm_contract.md`](./algorithm_contract.md), [`compatibility_manifest.md`](./compatibility_manifest.md), [`140v.md`](./140v.md)
> **Status:** Source-derived inventory complete. Runtime validation of every `reusable`/`verify` item is still owned by P0-17–P0-23 and V140-02/03 and is **not** claimed here.

---

## 0. Provenance and method

Upstream revision re-fetched and hash-verified for this audit:

```text
https://github.com/turboderp-org/exllamav3 @ 4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f
```

| File | SHA-256 (first 16) | Verified |
|---|---|---|
| `exllamav3/modules/quant/exl3_lib/quantize.py` | `1B11658F5DE96BFE` | yes |
| `exllamav3/exllamav3_ext/quant/quantize.cu` | `D8A84DD31CBF674A` | yes |
| `exllamav3/exllamav3_ext/quant/quantize_tiles_kernel.cuh` | `F8399AA3D281625F` | yes |
| `exllamav3/exllamav3_ext/quant/codebook.cuh` | `0E3C63B323F8D3CC` | yes |
| `exllamav3/exllamav3_ext/quant/pack.cu` | `27606EED6650ACC3` | yes |
| `exllamav3/conversion/convert_model.py` | `EE576CA6D2D56E55` | yes |
| `exllamav3/util/hadamard.py` | `6884841B61378788` | yes |
| `exllamav3/util/memory.py`, `util/misc.py` | fetched, not hashed in the contract | — |

**Method:** read the pinned source and the Intel-side runtime surface
(`torch.xpu`, Level Zero). No CUDA execution and no SYCL kernel was built for
this document. Every classification below is therefore a *source-derived
plan*, not a measured result. Where behaviour must be observed, the row names
the task that will observe it.

**Owner convention:** named ownership is assigned at Gate 0 (`G0-03`). Until
then each row is owned by the indicated workstream: `KERNEL` (SYCL port),
`PIPELINE` (Python conversion path), `ENV` (platform/memory).

### Classification vocabulary

| Label | Meaning |
|---|---|
| **Reusable** | Works on XPU unchanged, or with a backend rename (`cuda` → `xpu`) and no algorithm change. Still needs an XPU smoke test. |
| **XPU-native** | Has a direct `torch.xpu` / oneAPI equivalent; no custom kernel. |
| **Custom port required** | Needs a new SYCL kernel or a substantive device-side rewrite. |
| **Explicit CPU fallback** | Allowed, but only if P0-22 measures transfer, host RAM and wall-clock cost. Never silent. |
| **Deferred / not on initial path** | Out of the bounded initial scope with a written rationale. |

---

## 1. Executive result

The custom-extension surface is small: **six `ext.*` entry points**, of which
**one** (`quantize_tiles`) is on the critical path and requires a real port.

| Finding | Consequence |
|---|---|
| `ext.quantize_tiles` is the only mandatory custom kernel | Port effort is concentrated, not spread across the pipeline |
| `ext.pack_trellis` has a device-independent implementation already in `exl3_quant/trellis_format.py` | CPU reuse is viable for the first layer; a SYCL port is optional |
| `ext.unpack_trellis` is debug-only (call site commented out) | Not required |
| `ext.pack_signs` call sites are commented out in the current encode path | Dormant; not required for the first layer |
| `ext.had_paley` / `ext.had_paley2` are **not reached** for `H_128` | The Paley branches are dead for the initial transform (128 is even → Sylvester) |
| `torch.xpu` exposes `Stream`, `Event`, `synchronize`, `current_stream`, `mem_get_info`, `set_per_process_memory_fraction` | Stream/event/memory plumbing is a rename, not a rewrite |
| `torch.xpu` device properties have **no** `multi_processor_count` | `get_temp_buffers()` fails as written for `K >= 4`; the SM analogue must be chosen explicitly (see R11) |

---

## 2. Custom-extension inventory

| ID | Symbol | Call sites | First layer? | Classification | Owner | Validation |
|---|---|---|---|---|---|---|
| **D1** | `ext.quantize_tiles` | `quantize.py:88` (single device), `quantize.py:273` (multi-GPU) | **yes** | **Custom port required** | KERNEL | P1-03, P1-04, P1-14, P1-22…P1-27 |
| **D2** | `ext.pack_trellis` | `quantize.py:939` via `:1408`, `:1665` | **yes** | **Reusable (CPU)** now; optional SYCL port | PIPELINE | P2-15, oracle parity, P2-17 |
| **D3** | `ext.unpack_trellis` | `quantize.py:941` (commented self-check) | no | **Not needed** (debug only) | — | — |
| **D4** | `ext.pack_signs` | `quantize.py:950`; callers commented at `:1412`, `:1414` | no | **Dormant**; reuse CPU if re-enabled | PIPELINE | P2-14 only if re-enabled |
| **D5** | `ext.had_paley` | `hadamard.py:67` | no | **Not on initial path** | — | — |
| **D6** | `ext.had_paley2` | `hadamard.py:102` | no | **Not on initial path** | — | — |

### D1 — `ext.quantize_tiles` (custom port required)

C++ entry point (`quantize.cu:35-100`):

```text
quantize_tiles(input_tiles, output_tiles, output_indices,
               temp_costs, temp_edges, K, mcg, mul1)
```

- `input_tiles`: `(num_tiles, L)` float32, `L ∈ {256, 160}`; `L = 160` requires `mul1`.
- `output_tiles`: float32 reconstructed tile; `output_indices`: int16 16-bit states.
- `temp_costs`: `(B, 2, 65536 >> K)` float16; `temp_edges`: `(B, L, 65536 >> K)` int16.
- Device guard + **current CUDA stream**; launches `num_tiles / wave` sequential waves.
- Wave cap: `min(temp_costs.size(0), 2 * DevCtx::get_num_sms(device))`.
- Per-(`K`, `cb`, `L`) instantiations: 24 for `L = 256` (`8 K × 3 codebooks`)
  plus 8 for `L = 160` (`mul1` only), selected in `quantize.cu:81-85`.

Kernel facts (`quantize_tiles_kernel.cuh:18-313`):

- `__launch_bounds__(512, 2)`; one block per tile; 512 threads.
- Dynamic shared memory: `(K >= 2 ? 2*edges*2 : 0) + L*2 + 64 + 128` bytes.
  On the Arc 140V (128 KiB SLM):

  | K | SLM per work-group | Fits 2 WGs/SLM? |
  |---|---|---|
  | 1 | 704 B (costs in **global**) | n/a |
  | 2 | 66,240 B (64.7 KiB) | **no** (2 × 64.7 KiB > 128 KiB) |
  | 3 | 33,472 B (32.7 KiB) | yes |
  | 4 | 17,088 B (16.7 KiB) | yes |

  The upstream `launch_bounds(..., 2)` occupancy target is therefore not
  reachable at `K = 2` on this device. That is a tuning constraint, not a
  correctness change; confirm under P1-24/P1-26.
- Reduction uses `__shfl_down_sync(0xffffffff, …)`, i.e. a 32-wide warp. Map
  to a 32-wide subgroup (`sycl::reqd_sub_group_size(32)`); the device reports
  `sub_group_sizes = [16, 32]`.

Device intrinsics that must be reproduced:

| Upstream | Semantics | Port target | Risk |
|---|---|---|---|
| `__hsub2`, `__hmul2` | fp16 pairwise sub/mul, one rounding | SYCL `half2` ops | rounding mode must be checked (O-6) |
| `__hfma2(dh, dh, cost)` | **fused** fp16 multiply-add, one rounding | `sycl::fma` on half2, if it emits a true FMA | **highest numerical risk** |
| `__hadd2` | fp16 pair add | SYCL half2 add | mcG/3INST path only |
| `__hlt` | strict fp16 compare | `sycl::half` `<` | tie-break depends on strict `<` |
| `__dp4a(x, 0x01010101, 0x6400)` | unsigned byte-sum + accumulator | `sycl::ext::oneapi` dp4a if available, else plain `(x&0xff)+…` | compiler must not reorder/alter wrapping |
| `lop3.b32` inline PTX | `(x & 0x8FFF8FFF) ^ 0x3B603B60` | plain C++ bitwise expression | none (already audited in §3.2) |
| `__funnelshift_r` | funnel shift | `(a >> s) | (b << (32-s))` | unpack path only |
| `__float2half_rn`, `__half2float` | fp16 round-trip | `sycl::half` conversions | none |
| `__ushort_as_half(0x7c00)` | +inf | bit-cast | none |

What must be preserved exactly (from `algorithm_contract.md` §4–§5):

1. 16-bit state space, `Kr = 16 - K`, `sigma = (k << Kr) | e`, `pred = sigma >> K`.
2. fp16 metric **and fp16 accumulation**; do not silently accumulate in fp32.
3. Strict `<` tie-break, candidates ascending in `k`, lowest edge on terminal argmin.
4. Two-pass tail-biting schedule and `H_INF` masking at `pre_state`.
5. State-history output: `output_indices` are full 16-bit states, not sub-byte symbols.
6. `L = 160` n-gram rows are `mul1`-only (out of initial model scope, but the API accepts them).

Recommended port shape: template on `K`, `cb`, `L`; 32-wide subgroup;
local-accessor SLM sized by the formula above; global `temp_costs` fallback at
`K = 1`; wave cap `min(B, 2 * subslices)`; torch XPU current queue; explicit
device guard. Do not fuse passes or change the metric in the first port.

### D2 — `ext.pack_trellis` (reusable on CPU)

`pack.cu:66-94`; 128 threads, pure `uint16` bit manipulation, 8 template
instantiations; no fp16 arithmetic. `exl3_quant/trellis_format.py` already
implements the same layout and is tested against `decode_states` and the live
`exl3xpu.ref` oracle. Reuse the CPU path for the first layer; port only if
profiling shows the device→host→device round trip matters. The packing is
per-layer, not per-tile, so the expected cost is small.

### D3 — `ext.unpack_trellis` (not needed)

`pack.cu:147-175`; the only call site is commented out (`quantize.py:941`).
Keep as a test-only oracle if a device implementation is ever needed.

### D4 — `ext.pack_signs` (dormant)

`pack.cu:203-227`; packs fp16 sign bits, no arithmetic. The call sites in
`quantize.py:1412/1414` are commented out. Not required for the first layer.
Re-enable only with a defined sign-export contract and a round-trip test.

### D5/D6 — `ext.had_paley`, `ext.had_paley2` (not on initial path)

`get_hadamard(128)` takes the Sylvester branch (`n % 2 == 0` →
`get_hadamard(64)` → … → cached constant), so the Paley branches are not
reached for the pinned `H_128`. `hadamard.py` also ships cached Hadamard
constants. The Paley kernels stay deferred unless a non-power-of-two transform
enters scope.

---

## 3. Framework and runtime dependency inventory

| ID | Upstream dependency | Call sites | Classification | Notes / action | Owner | Validation |
|---|---|---|---|---|---|---|
| **R1** | current CUDA stream | `quantize.cu:48`, `pack.cu:74/155/210` | **XPU-native** | `torch.xpu.current_stream()`, SYCL queue | KERNEL | P0-18 |
| **R2** | `OptionalCUDAGuard` | same | **XPU-native** | XPU device guard / queue device | KERNEL | P0-18 |
| **R3** | `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` + `extern __shared__` | `quantize.cu:83` | **Custom port required** | SYCL local accessor; request less than 128 KiB | KERNEL | P0-19, P1-24 |
| **R4** | 512 threads, `__launch_bounds__(512,2)`, `__syncthreads` | kernel | **Custom port required** | `reqd_work_group_size(512)`, `group_barrier`; occupancy target not reachable at K=2 | KERNEL | P1-24, P1-26 |
| **R5** | `__shfl_down_sync` reduction | kernel `:254-277` | **Custom port required** | 32-wide subgroup shuffle; do not assume warp==subgroup 16 | KERNEL | P1-25 |
| **R6** | CUDA half/half2 math intrinsics | kernel, `codebook.cuh` | **Custom port required** | SYCL half ops; **true FMA** equivalence unproven (O-6) | KERNEL | P1-27, O-6 |
| **R7** | `__dp4a` | `codebook.cuh:35-36, 84` | **Custom port required** | oneAPI dp4a or explicit byte-sum | KERNEL | P1-27 |
| **R8** | `lop3.b32` inline PTX | `codebook.cuh:47-48, 63, 72, 101-102` | **Reusable** | rewrite as `(x & 0x8FFF8FFF) ^ 0x3B603B60` | KERNEL | unit test vs LUT |
| **R9** | `__funnelshift_r` | `pack.cu:130` | **Reusable** | shift/or rewrite | PIPELINE | P2-15 |
| **R10** | `cuda_check`, `cudaPeekAtLastError` | all `.cu` | **Reusable** | SYCL exception handling; fail loudly | KERNEL | P0-18 |
| **R11** | `multi_processor_count` / `DevCtx::get_num_sms` | `quantize.py:59` (`K >= 4` only), `quantize.cu:74` | **Custom decision (fails on XPU at `K >= 4`)** | `multi_processor_count` is **absent** on `torch.xpu`; `gpu_subslice_count=8`, `max_compute_units=64`, `gpu_eu_count=64`. Recommended SM analogue for wave capping: **subslices (8)**, not EUs. Record the choice and measure under P1-14 | ENV | P0-19, V140-03 |
| **R12** | pinned memory, events, per-device streams, `AutoSplit` | `quantize.py:155-345` | **Deferred** | Multi-GPU path; out of initial single-device scope. `torch.xpu` has Stream/Event if revisited | ENV | P4-06/P4-07 |
| **R13** | `cuda_sync_active` | `quantize.py:9`, `util/misc.py:35-43` | **XPU-native** | `torch.xpu.synchronize()` over active devices | PIPELINE | P0-18 |
| **R14** | `mem_get_info`, `memory_reserved`, `set_per_process_memory_fraction`, `empty_cache` | `util/memory.py` | **XPU-native, semantics unverified** | All symbols exist on `torch.xpu`; shared-memory accounting must be validated, not assumed | ENV | V140-02, V140-04 |
| **R15** | `malloc_trim` (glibc arenas) | `util/memory.py:78` | **Reusable** | Host-side; unchanged on Linux | ENV | V140-02 |
| **R16** | `@lru_cache get_temp_buffers` | `quantize.py:53` | **Resolved (`ScratchArena`)** | Unbounded residency replaced by `ScratchArena` (`exl3_quant/xpu_quantizer.py`) with single-slot reuse and stream synchronization | KERNEL | P1-12, P1-15 |
| **R17** | `torch.linalg.cholesky` / `inv`, `einsum`, matmul, `norm` | `quantize.py:350-481, 732-739`; `convert_model.py:387` | **XPU-native (probe)** | Coverage/dtype policy unverified on Arc; CPU fallback permitted only with measured cost | PIPELINE | P0-21, P0-22 |

---

## 4. Calibration / factorization operator coverage

These are ordinary torch ops, not `ext.*` calls, but they sit on the first
single-layer path. Coverage is a P0-21 probe, not an assumption:

| Operation | Used by | Fallback if unsupported |
|---|---|---|
| `@` / matmul (fp16, fp32) | Hadamard pre/post-apply, LDLQ | CPU fallback, measured |
| `torch.linalg.cholesky` | `block_ldl` (`quantize.py:428, 462`) | CPU fallback, measured |
| `torch.linalg.inv` | `block_ldl` diagonal normalization (`:476`) | CPU fallback, measured |
| `torch.einsum` | proxy error (`:739`) | reformulate as matmul |
| `torch.linalg.norm` | conversion diagnostics (`convert_model.py:387`) | CPU |
| fp16 elementwise / rounding | entire path | must be deterministic and match CPU reference |
| safetensors load/save (`safetensors_alt`) | checkpoint I/O | host-side, reusable |

---

## 5. Initial-scope closure

**Required for the first complete decoder layer**

- D1 (SYCL `quantize_tiles`), D2 (CPU `pack_trellis`).
- R1, R2, R3, R4, R5, R6, R7, R8, R10, R13, R14, R16, R17.
- A decision on R11 (subslice vs EU wave cap).

**Not required, with rationale**

- D3: debug-only, no live call site.
- D4: dormant, call sites commented out.
- D5/D6: Paley branches unreachable for `H_128`.
- R12: multi-GPU is Phase 4.
- Pinned-memory copy pipeline: single device has no host↔device split on UMA.

**Deferred with explicit reason, not dropped**

- `L = 160` n-gram rows: accepted by the kernel, outside the initial dense-model scope.
- Multi-GPU and discrete Arc: separate deployment gate.

---

## 6. Numerical contract to carry into the port

These are the non-negotiable behaviours from `algorithm_contract.md` §4–§5.
A port that changes any of them is a different quantizer.

1. **fp16 metric, fp16 accumulation.** `__hfma2` is fused; emulation with an
   fp32 intermediate is not bit-exact in general (see `reference_encoder.py`,
   now using an exact intermediate).
2. **Strict `<` tie-break.** First `k` wins; terminal argmin keeps the lowest
   edge index.
3. **`H_INF = 0x7C00`.** Disallowed predecessors are masked to +inf, not skipped.
4. **Two-pass tail-biting.** `forward(L/2,-1)` → `argmin` → `backward(false)` →
   `forward(0,end_state)` → `backward(true)`.
5. **State history.** `output_indices` carry full 16-bit states; the container
   stores only the low `K` bits per element.
6. **Permutation before encode.** The index stream stays in tensor-core order.

Open numerical risks that this audit does **not** close:

- **O-1:** whether terminal `argmin` reads step `L-1` or `L-2` costs (ping-pong parity).
- **O-6:** whether the XPU backend emits a true fp16 FMA for the SYCL expression used.

---

## 7. Validation plan

| Dependency | Test / measurement | Task |
|---|---|---|
| Extension build, queue ordering, tensor lifetime | minimal SYCL extension executing a trivial op on `xpu:0` | P0-18 |
| Capability limits, SLM, max allocation | save Level Zero/torch XPU properties | P0-19 |
| Scratch allocation through the framework allocator | small-B sweep with 6 GiB reserve | P0-20, V140-03 |
| Operator coverage | run the layer ops on XPU, record fallbacks | P0-21 |
| Cholesky/LDLQ quantities | compare against audited equations; measure CPU fallback | P0-22 |
| Kernel numerics | path/objective parity vs `ReferenceTrellisEncoder`, then CUDA fixtures | P1-03…P1-05, P1-29 |
| Batch/occupancy | B ∈ {1,16,64,128,256}, latency and pressure | P1-14, P1-17…P1-27 |
| Packing/serialization | exact round trips, malformed input | P2-15, P2-16 |
| Memory accounting | device-wide vs RSS vs shared backing | V140-02, V140-04 |

---

## 8. Decisions and blockers

**Decisions required before the kernel work**

1. **R11 — SM analogue.** Use `gpu_subslice_count` (8) for the wave cap and
   record it in the manifest; do not use `max_compute_units` (64) without
   measurement. The scratch estimator already refuses to infer `B` from SM count.
2. **D2 — pack path.** CPU reuse for the first layer (recommended) versus a
   SYCL port. If CPU reuse, add an explicit transfer measurement to P2-17.
3. **K set.** Initial recipe is `K ∈ {2,3,4}`; `K = 1` keeps the global cost
   buffer path and should not be exposed until tested.
4. **Precision policy.** Define the XPU FMA expression and prove it against the
   CPU reference before claiming parity (O-6).

**Blockers carried forward**

- P0-24/P0-25: no CUDA ground-truth fixtures; encoder parity remains unproven.
- O-1: terminal-edge parity unresolved.
- No XPU extension exists yet (P0-18).
- Shared-memory accounting and the 6 GiB reserve are policies, not measurements (V140-02/V140-04).

---

## 9. Task disposition

- **P0-15:** complete — every `ext.*` symbol and the supporting
  stream/event/allocation/device-property dependencies are classified above,
  with owner and validation method.
- **P0-16:** complete — this document plus
  [`algorithm_contract.md`](./algorithm_contract.md) are published; unresolved
  native behaviour is listed explicitly rather than defaulted.
- Runtime validation is **not** claimed and remains with P0-17–P0-23,
  V140-02/03/04, and P1-03 onward.
