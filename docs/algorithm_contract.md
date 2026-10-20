# EXL3 Algorithm Contract (audited from pinned upstream source)

> **Status:** Source-audited contract. Every normative statement below is cited to a
> file and line range in the pinned upstream revision. Items that could **not** be
> settled from source alone are listed explicitly in [§12 Open items](#12-open-items--explicitly-blocked)
> and are **not** to be implemented by guessing.
>
> **Task mapping:** P0-11, P0-12, P0-13, P0-14, P0-16 (partial), and the input to
> P1-01, P1-02, P2-04, P2-06, P2-08, P2-13, P2-14.

---

## 0. Provenance

**Upstream repository:** `https://github.com/turboderp-org/exllamav3`
(the `turboderp/*` org paths do **not** resolve; use `turboderp-org`)

**Pinned revision:** `4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f`
(referenced by the source specification as revision `4d7b69c7…` for `quantize.py`)

**Files audited** (re-fetchable via
`https://raw.githubusercontent.com/turboderp-org/exllamav3/<sha>/<path>`):

| File | Role | SHA-256 (first 16 hex) |
|---|---|---|
| `exllamav3/modules/quant/exl3_lib/quantize.py` | tile permutation, scratch buffers, LDLQ, Hadamard, packing entry points | `1B11658F5DE96BFE` |
| `exllamav3/exllamav3_ext/quant/quantize_tiles_kernel.cuh` | the trellis (Viterbi) encoder kernel | `F8399AA3D281625F` |
| `exllamav3/exllamav3_ext/quant/codebook.cuh` | procedural codebook arithmetic | `0E3C63B323F8D3CC` |
| `exllamav3/exllamav3_ext/quant/pack.cu` | trellis + sign packing | `27606EED6650ACC3` |
| `exllamav3/exllamav3_ext/quant/reconstruct.cu` | reconstruction path | `F47F121CDB452CF8` |
| `exllamav3/exllamav3_ext/quant/hadamard_inner.cuh` | Hadamard kernel internals | `63CA981A4A706924` |
| `exllamav3/exllamav3_ext/quant/quantize.cu` | quantize entry points | `D8A84DD31CBF674A` |
| `exllamav3/util/hadamard.py` | Hadamard matrix construction (device-independent) | `6884841B61378788` |
| `exllamav3/conversion/convert_model.py` | conversion entry point / orchestration | `EE576CA6D2D56E55` |

**Secondary (independent) reference — downstream XPU runtime:**
`https://github.com/sybil-solutions/exl3xpu` @ `15ded2f3add148c4db3c90b7036fe31a7281f`(?), file
`exl3xpu/ref.py` — a **bit-exact pure-PyTorch decoder** for the EXL3 linear format.
It is used here only as a *decode-side* cross-check; it is not the encoder contract.

**Not audited in this revision:** `exl3xpu` XPU kernels, upstream `eval/`, upstream
`tests/`. The upstream `tests/test_reconstruct_had.py` and `tests/test_quant_fn.py` exist
and are candidates for P0-24 fixture provenance.

---

## 1. Pipeline overview (P0-09)

```
convert_model.py
  └─ calibration → H = Σ XᵀX  (per layer, fp32/fp64 accumulate)
      └─ regularize()          quantize.py:1130-1236   → damping, scales, signs (su, sv)
          └─ blockwise_preapply_had_{l,r}_(H)          quantize.py:902-904
              └─ block_ldl(H, b)                      quantize.py:416-490   → L (block LDL)
                  └─ ldlq(weight, L)                   quantize.py:493-609   → weight_q, encoded
                      └─ quantize_tiles(tiles, args)   quantize.py:66-98    → quant_w, quant_idx
                          └─ ext.quantize_tiles  →  quantize_tiles_kernel.cuh:23-313
                  └─ preapply_had_{l,r}(weight_q)       quantize.py:1395-1397
                      └─ pack_trellis / pack_signs     quantize.py:932-951 → pack.cu
```

**All of the above is device-independent Python except the `ext.*` calls**, which are the
custom-extension call sites to be ported (P0-10): `ext.quantize_tiles`,
`ext.pack_trellis`, `ext.unpack_trellis`, `ext.pack_signs`, `ext.had_paley`, `ext.had_paley2`.

---

## 2. Tile layout and the `tensor_core_perm` permutation

- Weights are quantized in **16×16 tiles**, flattened to **T = 256** elements per tile
  (`quantize.py:66-76`, `quantize_tiles_kernel.cuh:6-8`).
- Tile construction from a weight span (`quantize.py:576`):
  `tiles = rows.reshape(16, tiles_n, 16).permute(1, 0, 2).reshape(tiles_n, 256)`.
- **Pre-permutation to tensor-core layout** (`quantize.py:579`):
  `tiles = tiles[:, tensor_core_perm(device)]`
- **Post-permutation of the reconstructed tile only** (`quantize.py:585`):
  `quant_w = quant_w[:, tensor_core_perm_i(device)]`
  — the *encoded indices stay in tensor-core order* (`quantize.py:584`).
- `tensor_core_perm` (`quantize.py:22-44`) is the following fixed 256-entry table
  (identical in `exl3xpu/ref.py:30-45`):

  ```text
  for t in 0..31:
      r0 = (t % 4) * 2 ; r1 = r0+1 ; r2 = r0+8 ; r3 = r0+9
      c0 = t // 4      ; c1 = c0+8
      perm[t*8 + 0..3] = [r0*16+c0, r1*16+c0, r2*16+c0, r3*16+c0]
      perm[t*8 + 4..7] = [r0*16+c1, r1*16+c1, r2*16+c1, r3*16+c1]
  ```

- The inverse is `torch.argsort(tensor_core_perm(device))` (`quantize.py:47-49`).
- Element `t` of the *encoded* tile corresponds to original tile position `perm[t]`
  (confirmed by `exl3xpu/ref.py:12` and its `tile[:, perm] = vals` scatter at
  `exl3xpu/ref.py:113`).

> **Normative:** the permutation is applied **before** encoding and **not** undone on the
> index stream. Any implementation that permutes after encoding will produce
> checkpoints that decode to permuted weights.

---

## 3. Procedural codebook arithmetic

Three codebooks exist; `mul1` (`cb == 2`) is the one in initial scope.
Constants (`quantize.py:16-19`):

```text
codebook_scale   = 1.24371088
codebook_mcg_mult  = 0xCBAC1FED
codebook_mul1_mult = 0x83DCD12D
```

The codebook is a **function of a 16-bit state**, not of a small symbol index, and it is
**pseudorandom**, not a uniform grid.

### 3.1 `mul1` (cb = 2) — `codebook.cuh:76-89`

```text
x   = (state * 0x83DCD12D) mod 2^32
sum = dp4a(x, 0x01010101, 0x6400)      # byte-wise sum of x's 4 bytes, plus 0x6400
h   = fp16(sum)                        # 0x6400 -> 1024.0 ... 0x67FF -> 2047.0
val = __hfma(h, fp16(0x1EEE), fp16(0xC931))
     # 0x1EEE = 0.00676727294921875  = 1/147.7
     # 0xC931 = -10.3828125          = (-1024.0 - 510.0) * 0.006767
```

`__dp4a` is an **unsigned byte-wise dot product with a u32 accumulator**: `sum` equals
`0x6400 + b0 + b1 + b2 + b3` where `b0..b3` are the four bytes of `x`. The accumulator
value `1024 + (b0+b1+b2+b3)` is exactly representable in fp16 (the fp16 bit pattern is
`0x6400 + byte_sum`), so `fp16(sum)` is lossless. `__hfma` is **fused** (single
rounding). The `exl3xpu/ref.py:57-63` transcription uses float64 as the exact intermediate
and rounds once to fp16, which reproduces it bit-exactly.

**Measured on this host** (numpy transcription in `exl3_quant/trellis_format.py`):

| Quantity | Value |
|---|---|
| distinct levels over all 65536 states | **913** |
| level range | **[-3.453125, 3.34765625]** |
| `fp16(0x1EEE)` | 0.00676727294921875 |
| `fp16(0xC931)` | -10.3828125 |

(The codebook is *affine in the byte sum*, so it has at most 1021 levels; only 913 byte
sums are reachable through the multiplier.)

For comparison, the other two codebooks are also pseudorandom and much finer-grained:
`3INST` → 10598 distinct levels in [-3.957, 3.973], `mcg` → 10746 distinct levels in
[-3.949, 3.949].

### 3.2 `mcg` (cb = 1) — `codebook.cuh:67-75`

```text
x   = (state * 0xCBAC1FED) mod 2^32
x   = (x & 0x8FFF8FFF) ^ 0x3B603B60      # lop3 immLut 0x6a
val = __hadd( low_half(x), high_half(x) )   # single fp16 rounding
```

### 3.3 `3INST` (cb = 0) — `codebook.cuh:59-66`

```text
x   = (state * 89226354 + 64248484) mod 2^32
x   = (x & 0x8FFF8FFF) ^ 0x3B603B60
val = __hadd( low_half(x), high_half(x) )
```

### 3.4 Structural property used by the kernel

For `mcg` and `mul1`, the codebook index multiplication is **linear mod 2^32** across the
branch states of a step, so the kernel advances the product by a constant stride
instead of re-multiplying (`quantize_tiles_kernel.cuh:110-123`, `196-207`):

```text
product_step = MULT << Kr          where Kr = 16 - K
```

For `3INST` the kernel instead decodes the two neighbouring states directly
(`quantize_tiles_kernel.cuh:125-127`).

---

## 4. Trellis state space and the forward ACS pass

Source: `quantize_tiles_kernel.cuh:23-228`.

### 4.1 Parameters

```text
K      = encoder parameter (bits per element), 1..8 supported by the packer
max_q  = 1 << K                        # branches per edge
Kr     = 16 - K
edges  = 65536 >> K                    # = 2^(16-K) states tracked per step
L      = 256 (tile length; 160 for n-gram rows, mul1 only)   # .cuh:6-8
threads = 512 per block, launch_bounds(512, 2)               # .cuh:18,24
```

`edges` is exactly the `S` of the scratch-memory model in the project specification
(`65536 >> K`). The scratch estimator in `exl3_quant/scratch_estimator.py` is **correct**
on this point.

### 4.2 State and transition

For a step, the kernel iterates over an *edge index* `e ∈ [0, edges)`. For each edge it
enumerates `k ∈ [0, max_q)` candidate branch states (`.cuh:104-141`):

```text
sigma   = (k << Kr) | e                 # 16-bit candidate state
pred    = sigma >> K                    # predecessor edge index
value   = codebook(sigma)               # cb-dependent, §3
metric  = (value - w)²                  # fp16
```

The predecessor relation `pred = sigma >> K` is the standard shift-register inverse:
given the new 16-bit state, the predecessor is determined modulo `2^(16-K)`, and the
`2^K` branches enumerate the `K` low bits of the predecessor's history.

### 4.3 Branch metric and accumulation precision

```text
dh2   = __hsub2(decoded2, w2)                     # fp16 subtract      (.cuh:97, 128)
err2  = __hmul2(dh2, dh2)                         # fp16 multiply     (.cuh:98)
acc2  = __hfma2(dh2, dh2, cost_in[pred])          # fp16 FMA          (.cuh:180, 210)
H_INF = __ushort_as_half(0x7c00)                  # +inf in fp16      (.cuh:19-21)
```

- `w2` is the **fp16-rounded** input tile element (`.cuh:56`: `__float2half_rn`).
- The forward cost buffer is **fp16** (`.cuh:52-54`, `quantize.py:61`).
- **All arithmetic is fp16**, including the accumulation. `__hfma2` is fused (one
  rounding). A port that accumulates in fp32 will produce different path selections and
  must not be presented as bit-exact.
- `H_INF` is used to mask disallowed predecessors (`.cuh:99`, `.cuh:130`).

### 4.4 Tie-breaking

```text
if (__hlt(low(err2),  low(min_err2)))  { ... }    # .cuh:131
if (__hlt(high(err2), high(min_err2))) { ... }    # .cuh:136
```

- Comparisons are **strict** (`<`), so on an exact tie the **first** candidate wins.
- Candidates are enumerated `k = 0, 1, …, max_q-1` in increasing order, therefore
  **the smallest `k` wins ties**.
- The terminal `argmin` over edges (`.cuh:231-284`) preserves "the historical
  1024-thread tie-breaking order": two strided sweeps (even edges first, then odd edges)
  followed by a warp-shuffle reduction, each comparison also strict `<`. Net effect:
  **the lowest edge index wins ties**.

### 4.5 DP recurrence

With `cost_i(e)` the cost of the best path arriving at edge `e` at step `i`:

```text
cost_0(e) = min over k of metric_0(e, k)                     # first step, no predecessor term
cost_i(e) = min over k of ( metric_i(e, k) + cost_{i-1}(pred(e,k)) )
```

The kernel implements this with two ping-pong fp16 buffers of `edges` entries each
(`.cuh:52-54`, `.cuh:69-72`, `.cuh:150-154`). The first loop of `forward()` computes
`i = 0` (local metric only, `.cuh:74-146`); the second loop computes `i = 1 … L-1`
with accumulation (`.cuh:149-228`).

> **Normative:** a port must reproduce fp16 metric + fp16 fused accumulation. Path-cost
> parity (not state-index parity) is the acceptance criterion for tied branches, per
> the project specification §8.1 tier 4.

---

## 5. Tail-biting closure and traceback

Source: `quantize_tiles_kernel.cuh:59-72`, `231-313`.

### 5.1 Ring

The tile is a **circular (tail-biting) ring**: `ri = ring(i, roll) = (i + roll) mod L`,
with `roll ∈ {0, L/2}` (`.cuh:59-65`).

### 5.2 The two-pass schedule

```text
forward(L / 2, -1);                        # .cuh:309   pass 1: rotation by L/2, unconstrained
int end_state = backward(L / 2, false, argmin_cost());   # .cuh:310
forward(0, end_state);                     # .cuh:311   pass 2: natural order, wrap constrained
backward(0, true, end_state);              # .cuh:312   emit indices + reconstructed tile
```

- Pass 1 traverses the ring in the rotated order to obtain the edge that the ring must
  close onto (`end_state`).
- Pass 2 re-runs the DP in natural order with `pre_state = end_state`, which **masks
  every predecessor other than `end_state` to `H_INF`** (`.cuh:99`, `.cuh:130`), thereby
  enforcing the tail-biting wrap constraint at step 0.
- `argmin_cost()` (`.cuh:231-284`) picks the terminal edge from the cost buffer; it
  returns edge `0` when every cost is `inf`/NaN, to stay in bounds on degenerate input
  (`.cuh:280-283`).

### 5.3 Backward traceback

```text
for (int i = L - 1; i >= 0; --i) {
    const int ri      = ring(i, roll);
    const int prev    = (int) temp_edges[edges * ri + edge];
    const int encoded = (prev << K) | edge;
    edge = prev;
    if (write) {
        output_indices[ri] = (uint16_t) encoded;      # .cuh:298
        output_tile[ri]   = __half2float(decode_3inst<cb>(encoded));   # .cuh:299
    } else if (ri == 0) break;
}
```

- `temp_edges[edges * ri + edge]` is the **predecessor edge index** recorded during the
  forward pass (`.cuh:144-145`, `224-225`).
- `encoded` is the **16-bit state** for that element — this is what is written to the
  `encoded` buffer and later packed.
- The first `backward(..., write=false)` call only returns the edge at ring position 0.
- The reconstruction value is `codebook(encoded)`, i.e. the codebook is indexed by the
  **full 16-bit state**, consistent with §3.

> **Open item O-1** — see §12: the exact buffer parity used by `argmin_cost()` (it reads
> `temp_costs_inc`, which after the final loop holds the second-to-last step's costs)
> must be confirmed against the CUDA reference before a port relies on it.

---

## 6. Scratch memory

Source: `quantize.py:52-63`, kernel `.cuh:46`, `.cuh:52-54`.

```text
temp_costs : (B, 2, 65536 >> K)  dtype float16
temp_edges : (B, tile_len, 65536 >> K)  dtype int16
B          = 256, or max(256, 2 * multi_processor_count) when K >= 4   (quantize.py:57-60)
```

- `B` is cached by argument signature (`@lru_cache` on `get_temp_buffers`), so **mixed-K
  recipes can leave several multi-gigabyte `temp_edges` buffers resident simultaneously**.
  This is the cache-eviction hazard called out in the project specification §7.3.
- The kernel caps the wave at `min(temp_costs.size(0), 2 * SMs)` (`.cuh:54-56`).
- `temp_costs` lives in **shared memory** when `K >= 2`, otherwise in global memory
  (`.cuh:52-54`); SLM footprint per tile is `4 * edges` bytes.
- **The project's `ScratchEstimator` matches this model exactly** (costs `B*2*S*2`,
  edges `B*T*S*2`, `S = 65536 >> K`). Its `calculate_max_batch_size` correctly refuses
  to substitute Xe-core count for `multi_processor_count` (upstream only scales `B` with
  SM count when `K >= 4`).

---

## 7. Packing and serialization

Source: `pack.cu:9-57` (`pack_trellis`), `pack.cu:176-201` (`pack_signs`),
`quantize.py:932-951`.

### 7.1 `pack_trellis`

```text
input  : (…, 256)      int16   # the 16-bit states from the encoder
output : (…, 256*K/16) int16   # the packed bitstream
```

Algorithm (`.cu:28-52`):

1. The 256 elements are divided into **16 spans of 16 elements**.
2. Within a span, the **low `K` bits** of each element's state are appended
   **MSB-first** into a 32-bit staging buffer (`.cu:39-42`).
3. Whenever the buffer has ≥ 16 bits, one uint16 word is emitted
   (`s_packed[j] = buf >> 16`), and the buffer is shifted down by 16 (`.cu:43-49`).
   Each span therefore emits `16*K/16 = K` uint16 words.
4. The uint16 array is written out as uint32 words with `SWAP16`
   (`.cu:55-56`), i.e. each pair of uint16 is byte/half-word swapped. The resulting
   little-endian uint32 word is `p[2t+1] << 16 | p[2t]`.

**Consequence (important):** only `K` bits per element are stored. The remaining
`16-K` bits of each 16-bit state are **not** stored; they are recovered by the decoder
from the neighbouring elements' bits. This is why the packed stream is *not* an
independent-symbol encoding.

### 7.2 `pack_signs`

`.cu:176-201`: 16 fp16 sign bits per input element are packed **LSB-first** into one
uint16 word.

### 7.3 Decoder-side semantics (from `exl3xpu/ref.py`, independent XPU runtime)

```text
trellis : int16 [k/16, n/16, 16*K]        # one 16x16 tile per (i, j)
suh     : fp16  [k]                       # input sign/scale vector
svh     : fp16  [n]                       # output sign/scale vector
mcg/mul1: int32 scalar                   # presence selects the codebook
```

- `exl3xpu/ref.py:90-99` (`decode_states`): the uint16 stream is read as uint32 pairs
  (`t16[0::2] | (t16[1::2] << 16)`), then **MSB-first** into `256*K` bits.
- `exl3xpu/ref.py:81-87` (`_state_index`): element `t` is decoded from the **16-bit
  window that ends at bit `(t+1)*K`**, circular within the tile:

  ```text
  b0  = (t*K + K - 16 + 256*K) mod (256*K)
  pos = (b0 + 0..15) mod (256*K)
  ```

- `exl3xpu/ref.py:102-115` (`reconstruct_inner`): the 256 decoded values are scattered
  back with `tile[:, perm] = vals`, then re-assembled into the 16×16 tile grid.
- `exl3xpu/ref.py:133-142` (`weight_orig`): full reconstruction

  ```text
  W_orig = diag(suh) · H128_blocks(rows) · W_inner · H128_blocks(cols) · diag(svh)
  ```

  with `H128` the **normalized** Sylvester Hadamard matrix (`h / sqrt(128)`).

### 7.4 Verified on this host (CPU, no GPU)

Implemented in this repository as `exl3_quant/trellis_format.py`, exercised by
`tests/test_trellis_format.py` (17 tests) and live oracle cross-check `tests/test_oracle_parity.py` (4 tests).

**Check 1 — layout identity (passes).** For every `K ∈ {1,2,3,4,5,6,8}`, take a random
*bit stream*, decode it to 256 states, re-pack it, and require the stream to be
reproduced bit for bit. This holds because the decoder's 16-bit window always ends at
bit `(t+1)*K`, so the window's low `K` bits are exactly element `t`'s `K` bits. The check
exercises the `SWAP16` pairing (`pack.cu:55-56`) and the MSB-first span layout
(`pack.cu:28-52`) together and is sensitive to a flipped bit order or a wrong half-word
swap (both are covered by explicit negative tests).

**Check 2 — the format is a state-history encoding (passes).** Arbitrary 16-bit states
do **not** survive a round trip; only the low `K` bits of each state do. This is the
direct demonstration that an independent-symbol sub-byte packer — which is what
`exl3_quant/bitpack.py` implements — **cannot** be made compatible with upstream by
reordering bits. It is a different format and needs a different data model.

> **Open item O-2** — RESOLVED: The host-CPU reference encoder is implemented as
> `ReferenceTrellisEncoder` in `exl3_quant/reference_encoder.py`. Its generated
> trellis states round-trip through `pack_trellis` -> `swap16_storage` -> `decode_states`
> bit-for-bit across K in {2, 3, 4}, verified in `tests/test_encoder.py` and against
> the independent oracle `exl3xpu.ref` in `tests/test_oracle_parity.py`.

---

## 8. Hadamard transforms

Source: `util/hadamard.py`, `quantize.py:345-392`, `quantize.py:15`.

```text
had_k, had_n = 128, 128                                  # quantize.py:15
had = get_hadamard_dt(dim, device, dtype, 1/sqrt(dim))   # util/hadamard.py:133-137
```

- `get_hadamard(n)` (`util/hadamard.py:107-131`) builds Sylvester matrices recursively
  (`util/hadamard.py:34-42`) for `n % 2 == 0`, and falls back to Paley constructions for
  other sizes. For `n = 128` the Sylvester branch applies: `H_128` is the Sylvester
  Hadamard matrix, then scaled by `1/sqrt(128)`.
- The scale `1/sqrt(dim)` is applied by `get_hadamard_dt` (`.py:136`) — i.e. the
  transform is **orthonormal**, not unnormalized. `exl3xpu/ref.py:118-123` reproduces
  exactly this (`h / math.sqrt(128)`).
- Applications:
  - `preapply_had_l(x, had_dim)` (`.py:345-352`) — `had @ x` on the **input** dimension.
  - `preapply_had_r(x, had_dim)` (`.py:355-362`) — `x @ had` on the **output** dimension.
  - `blockwise_preapply_had_{l,r}_` (`.py:365-391`) — same, applied blockwise to `H`
    (`.py:902-904`) and to the weight (`.py:1210`, `1217`).
  - After quantization, `preapply_had_l/r` are applied to `weight_q`
    (`.py:1395-1397`) — the inverse direction, applied to the *reconstructed* weight.

> **Normative:** the Hadamard is applied **before** quantization and its inverse is
> applied to the reconstruction. Skipping the post-quantization transform yields weights
> that differ by the transform.

---

## 9. One-sided LDLQ and error feedback

### 9.1 `block_ldl` (P0-12) — `quantize.py:416-490`

```text
n = H.shape[0];  b = block size;  m = n // b

1. L = torch.linalg.cholesky(H)                                  # .py:428
   on failure: increase diagonal damping and retry, up to 10 times (.py:437-448)
       H.diagonal().add_(2.0 * sigma_reg * H.diagonal().mean())  # .py:447
       sigma_reg default = 0.025
   on OOM: fall back to CPU Cholesky                            # .py:450-467
2. DL = diagonal blocks of L.reshape(m, b, m, b)                 # .py:470
3. DL = torch.linalg.inv(DL)                                     # .py:476
4. L.view(n, m, b)[:, i, :] = L.view(n, m, b)[:, i, :] @ DL[i]   # .py:479-481
5. insert identity matrices on the block diagonal                # .py:486-488
```

- The damping schedule is a **retry loop on Cholesky failure**, not a pre-emptive
  damping constant. Any port that adds damping up-front changes the algorithm.
- The "LDL" is a Cholesky factor `L` with **block-diagonal inverses folded in** — it is
  *not* an `L D Lᵀ` factorization in the classical sense.
- **Measured Arc 140V operator behavior (P0-21 / P0-22, `artifacts/environment/operator_coverage.json`):**
  - `torch.linalg.cholesky` (fp32) silently falls back to CPU (`aten::linalg_cholesky_ex.L`), while `torch.linalg.inv`, `torch.linalg.solve`, `torch.linalg.solve_triangular`, `matmul`, and `einsum` execute natively on XPU. Low-precision (`fp16`/`bf16`) decompositions are unsupported on both XPU and CPU.
  - At `n = 4096` (64 MiB fp32 matrix), CPU Cholesky takes `114.1 ms`, implicit XPU fallback takes `131.2 ms` (H2D transfer `8.8 ms` at `7.6 GB/s`), and full `block_ldl` takes `142.2 ms` on CPU vs `231.2 ms` on XPU (`max |XPU - CPU| = 1.91e-6`, zero swap growth). Reference implementation: `exl3_quant/ldlq.py` (`tests/test_ldlq.py`).

### 9.2 `ldlq` (error feedback) — `quantize.py:493-609`

```text
for j in range(size_k, 0, -buf_size_k):            # spans processed from the bottom up
    i = j - buf_size_k
    b_weight   = weight[i:j]                        # fp32 copy on device
    b_prod_cache = prod_cache[i:j]                  # running feedback term
    for bj in range(buf_size_k, 0, -16):            # 16-row blocks, bottom-up
        bi = bj - 16
        bb_err = b_weight[bj:] - b_weight_q[bj:]     # error of the already-quantized rows
        bb_L   = b_L[bj:, i+bi : i+bj]              # matching slice of L
        compensation_term = b_prod_cache[bi:bj]
        compensation_term.addmm_(bb_L.T, bb_err, alpha=1.0, beta=1.0)
        rows = b_weight[bi:bj] + compensation_term # corrected target weights
        tiles = rows.reshape(16, tiles_n, 16).permute(1,0,2).reshape(tiles_n, 256)
        tiles = tiles[:, tensor_core_perm(device)]
        quant_w, quant_i = quantize_tiles_multigpu(tiles, quant_args)
        quant_w = quant_w[:, tensor_core_perm_i(device)]
        b_weight_q[bi:bj]  = quant_w.reshape(tiles_n,16,16).permute(1,0,2).reshape(16,size_n)
        b_encoded[bi//16 : bj//16] = quant_i.unsqueeze(0)
    b_err = b_weight - b_weight_q
    b_prod_cache.addmm_(b_L.T, b_err, alpha=1.0, beta=1.0)   # trailing-weight feedback
```

- The **trailing-weight feedback** is the final `addmm_` at `.py:603-604`: the residual
  of the just-quantized span is propagated into `prod_cache`, which the next span
  consumes through the same `L` slices. This is the "one-sided LDLQ error feedback"
  the project specification requires to be preserved exactly.
- The correction applied to the rows being quantized is
  `rows = b_weight[bi:bj] + compensation_term` (`.py:574`) — note the **addition**, and
  that the feedback term is *not* subtracted.
- `buf_size_k` defaults to 128 (`quant_args.get("buf_size_k", 128)`, `.py:538`) and must
  be a multiple of 16; `size_n % buf_size_k == 0` is asserted.
- The traversal order is **bottom-up in `k`**, which is what makes the feedback
  sequential and non-parallelizable across blocks of a layer.

### 9.3 Regularization, scales and signs — `quantize.py:1130-1236`

- `out_channel_scales = block_rms(weight, dim=0)` (`.py:1184`), normalized by its mean
  (`.py:1185-1187`), floored at `0.1` where the channel scale is degenerate
  (`.py:1193-1196`).
- `sv = (sv * out_channel_scales + 1e-10)` (`.py:1197`) → the **output** sign/scale
  vector.
- `su = (su * in_channel_scales / (-codebook_scale) + 1e-10)` (`.py:1215`) → the
  **input** sign/scale vector. The division by `-codebook_scale = -1.24371088` maps the
  weight range onto the codebook range.
- Both are stored as fp16 in the checkpoint: `suh`, `svh` (`quantize.py:1406-1415`).
- `apply_out_scales` is auto-selected by a skew heuristic when not forced
  (`.py:1172-1181`).

> **Normative:** `su`/`sv` are part of the exported format and must be reproduced exactly
> for the reconstruction to match. `exl3xpu/ref.py:133-142` consumes them directly.

---

## 10. Reconstruction / decode path

- `exl3xpu/ref.py` is a bit-exact pure-PyTorch reference for the **linear** format and
  is the best available CPU-side oracle for tier-3 acceptance (tile reconstruction).
- `reconstruct.cu` is the upstream CUDA reconstruction kernel (13.5 KB, audited but not
  summarized here; it is the production path for the CUDA runtime).
- The upstream `quantize.py` `pack_trellis` wrapper (`.py:932-943`) calls
  `ext.pack_trellis`; the commented-out `unpack_trellis` self-check (`.py:940-942`) is
  the natural oracle for the packing round trip.

---

## 11. Implications for the current implementation in this repository

| Component | Status vs. this contract | Action |
|---|---|---|
| `exl3_quant/trellis_format.py` | **Audited** — `codebook_lut` (§3), `tensor_core_perm` (§2), `pack_trellis` + `swap16_storage` + `decode_states` (§7) | Verified against live `exl3xpu.ref` oracle |
| `exl3_quant/scratch_estimator.py` | **Correct** — matches §6 exactly | Keep; reserve configured for 140V UMA profile |
| `exl3_quant/reference_encoder.py` | **Implemented** — `ReferenceTrellisEncoder` transcribed from `quantize_tiles_kernel.cuh:23-313`: 16-bit states, 2-pass tail-biting ring, fp16 fused metric | Verified against `decode_states` and `exl3xpu.ref`; CUDA numerical parity pending P0-24 |
| `exl3_quant/bitpack.py` | **Superseded** by `exl3_quant/trellis_format.py` | Retained for legacy negative tests only |
| `tensor_core_perm` | **Implemented** — §2 | Transcribed in `trellis_format.py` |
| Hadamard reference | **Missing** — §8 | Implement (normalized Sylvester `H128`) |
| LDLQ reference | **Missing** — §9 | Implement in a device-independent path |
| Objective / distortion | §4 uses plain squared error; upstream's is the same metric per element but accumulated in fp16, and the *layer-level* distortion is Hessian-weighted via `L` (§9) | Document the distinction explicitly |

---

## 12. Open items (explicitly blocked)

Per the project specification (§"Source traceability and guardrails"): unresolved
native behaviour is left blocked rather than replaced with an invented default.

| ID | Item | Why it is blocked | Closes when |
|---|---|---|---|
| **O-1** | Exact `argmin_cost()` buffer parity (it reads `temp_costs_inc`, which after the final loop of `forward()` holds the *second-to-last* step's costs) — is the terminal edge selected from step `L-2` or `L-1`? | Cannot be determined by reading alone; the ping-pong parity and the `roll` interaction must be observed | A CUDA reference run produces the terminal edge for a known tile (P0-24), or the upstream author's intent is confirmed |
| **O-2** | Bit-exactness of the packed stream for **trellis-produced** states | **RESOLVED** — `ReferenceTrellisEncoder` generates states that round-trip bit-for-bit through `pack_trellis` and `exl3xpu.ref` | Closed by `tests/test_encoder.py` and `tests/test_oracle_parity.py` |
| **O-3** | Whether `codebook_scale = 1.24371088` is applied anywhere else besides `su` (`.py:1215`) | `quantize.py` was read end-to-end for the encode path, but `convert_model.py` may add its own scaling | `convert_model.py` scale/sign path is audited (P0-09) |
| **O-4** | `H_INF` + NaN behaviour on degenerate tiles | Kernel guards only the argmin index (`.cuh:280-283`); metric arithmetic with `inf` inputs is undefined | Degenerate-input fixtures exist (P0-26) |
| **O-5** | Numerical ground truth for the encoder itself (tier 4) | Requires the CUDA reference (P0-24) | CUDA machine available |
| **O-6** | Exact fp16 emulation of `__hfma2` on the target (does the XPU backend fuse?) | **Resolved (measured on Arc 140V, 2026-10-20)**: `sycl::fma(half)` — and the plain `a*a+c` expression, which the compiler fuses anyway — match *exact fp16 single rounding* on all inputs tested, including 11 adversarial pairs where fp32 double-rounding differs from exact. Device NEVER showed the fp32 double-rounding behaviour, so the CPU-side port must use exact fp64-FMA→fp16 rounding (not fp32 double-rounding emulation) to achieve parity. Evidence: `artifacts/environment/extension_probe.json` (`hfma_o6_probe`: 65,536 random pairs, 0 mismatches; `hfma_o6_adversarial`: 8→11 discriminating pairs, `fma_matches_exact` = n, `fma_matches_fp32_emul` = 0). Source: `csrc/probe_xpu/probe_kernels.sycl` (`ProbeHfma`), `exl3_quant/xpu_probe.run_hfma_probe` / `run_hfma_adversarial_probe` | 

---

## 13. Method notes

- All statements above were obtained by reading the pinned upstream source; no upstream
  code was executed (no CUDA device is available on the audit host).
- Line numbers refer to the pinned revision; re-fetch with the URLs in §0 and verify the
  SHA-256 prefixes before relying on them.
- The CPU experiment in §7.4 is reproducible with numpy alone (no torch required).
