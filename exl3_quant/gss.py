"""Global-scale search (P2-04 completion, upstream ``quantize.py:895-1076``).

Closes the ``g_scale`` hook that :func:`exl3_quant.regularize.regularize`
exposes as ``g_scale_fn``:

* ``sample_scale_tiles`` (``.py:895-916``): wrapped-diagonal tile sampling so
  every tile row *and* column is covered, plus the highest/lowest-RMS tiles as
  explicit outlier insurance.
* ``g_scale_search_batch`` (``.py:918-1047``): rather than a golden-section
  search (the sampled error curve is flat and not perfectly unimodal), a
  **coarse grid** over 0.1..1.9 on a strided subsample, then a **fine grid**
  around the coarse minimum on the full sample, refined by **parabolic
  interpolation** (offset clamped to ±0.5 steps, result floored at 0.01).

The only deliberate deviation from upstream is the test-quantization backend:
upstream calls ``quantize_tiles_multigpu(batch, quant_args)`` directly, while
this module takes a pluggable ``encode_tiles`` callback — the same
``(tiles_n, 256) tensor-core-order -> (reconstruction, indices)`` protocol the
P2-08 LDLQ adapters implement. Batching of grid evaluations into encoder calls
is preserved (fewer host round trips on XPU); the per-tile MSE arithmetic is
independent of batching.

Upstream returns ``(best_scale, mse_tensor)``; this module returns plain
``float`` values for both.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from exl3_quant.trellis_format import tensor_core_perm

#: Range of the coarse grid: 0.1 + 0.2 * i, i in 0..9 (same bracket as the old
#: golden-section search upstream replaced).
COARSE_START = 0.1
COARSE_STEP = 0.2
COARSE_POINTS = 10

#: Fine grid: 5 points at 0.075 spacing around the coarse minimum, then a
#: parabolic offset clamped to ±0.5 of a step. The curve's flatness makes the
#: grid spacing precise enough.
FINE_POINTS = 5
FINE_STEP = 0.075
MIN_SCALE = 0.01

#: Upstream batches grid evaluations into kernel calls of at most this many
#: tiles (two host syncs per group regardless of group size on CUDA; the same
#: bound bounds our encoder-call tile count).
MAX_TILES_PER_CALL = 65536

TileEncoder = Callable[[torch.Tensor], "tuple[torch.Tensor, torch.Tensor]"]


def sample_scale_tiles(weight_r: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Sample tiles for the global-scale search (``.py:895-916``).

    A wrapped diagonal of the tile grid guarantees every tile row and column is
    sampled at least once (outliers are typically whole input or output
    channels), plus the tiles with the highest and lowest tile-RMS as explicit
    outlier insurance for anything a diagonal of the given width misses.

    Returns a ``(num_tiles, 256)`` float32 tensor already in tensor-core order,
    exactly what the P2-08 encoder adapters consume.
    """
    device = weight_r.device
    tiles_k = weight_r.shape[0] // 16
    tiles_n = weight_r.shape[1] // 16
    w4 = weight_r.reshape(tiles_k, 16, tiles_n, 16)

    diag_len = max(tiles_k, tiles_n)
    ii = torch.arange(diag_len, device=device).repeat_interleave(width)
    ww = torch.arange(width, device=device).repeat(diag_len)
    kk = ii % tiles_k
    nn = (ii + ww) % tiles_n

    # Tiles with extreme RMS, by flat tile index
    num_x = max(8, (diag_len * width) // 16)
    tile_ms = w4.square().mean(dim=(1, 3)).flatten()
    num_x = min(num_x, (tile_ms.shape[0] + 1) // 2)
    hi = torch.topk(tile_ms, num_x).indices
    lo = torch.topk(tile_ms, num_x, largest=False).indices
    xk = torch.cat((hi, lo)) // tiles_n
    xn = torch.cat((hi, lo)) % tiles_n

    tiles = w4[torch.cat((kk, xk)), :, torch.cat((nn, xn)), :].reshape(-1, 256)
    perm = torch.as_tensor(tensor_core_perm(), dtype=torch.long, device=device)
    return tiles[:, perm].contiguous()


def select_refined_scale(fine_scales: "list[float]", fine_mses: "list[float]") -> float:
    """Fine-stage selection + parabolic refinement (``.py:1038-1046``).

    Picks the minimum of the fine grid and, when it is interior, adds a
    parabolic offset of at most half a step. The result is floored at
    :data:`MIN_SCALE` (upstream ``max(..., 0.01)``), so a fine grid that runs
    past zero still yields a usable scale.
    """
    best = min(range(len(fine_mses)), key=lambda i: fine_mses[i])
    if 0 < best < len(fine_scales) - 1:
        y0, y1, y2 = fine_mses[best - 1], fine_mses[best], fine_mses[best + 1]
        denom = y0 - 2.0 * y1 + y2
        offset = 0.5 * (y0 - y2) / denom if denom > 0 else 0.0
        offset = max(-0.5, min(0.5, offset))
    else:
        offset = 0.0
    return max(fine_scales[best] + offset * FINE_STEP, MIN_SCALE)


def g_scale_search_batch(
    samples: Sequence[torch.Tensor],
    encode_tiles: TileEncoder,
    max_tiles: int = MAX_TILES_PER_CALL,
) -> "list[tuple[float, float]]":
    """Global scale search over a group of tensors' tile samples (``.py:918-1047``).

    Evaluations are batched into as few ``encode_tiles`` calls as possible,
    chunked at ``max_tiles`` tiles per call — mirroring upstream's kernel-call
    chunking (two host syncs per group regardless of group size).

    Args:
        samples: one TC-ordered ``(T, 256)`` float32 tile sample per tensor.
        encode_tiles: P2-08 encoder protocol; the reconstruction is in
            tensor-core order, matching the samples.
        max_tiles: tiles per encoder call (upstream uses 65536).

    Returns:
        One ``(best_scale, best_mse)`` pair per sample tensor.
    """
    n_t = len(samples)
    if n_t == 0:
        return []
    for t, s in enumerate(samples):
        if s.dim() != 2 or s.shape[1] != 256 or s.shape[0] < 1:
            raise ValueError(
                f"samples[{t}] must be (T, 256) tensor-core-order tiles, "
                f"got {tuple(s.shape)}"
            )

    device = samples[0].device

    def eval_pairs(pairs, tile_sets):
        # pairs: list of (tensor_idx, scale); one mse per pair, encoder calls
        # chunked to the batch limit
        out = torch.empty(len(pairs), dtype=torch.float32, device=device)
        i = 0
        while i < len(pairs):
            j, tot = i, 0
            while j < len(pairs) and tot + tile_sets[pairs[j][0]].shape[0] <= max_tiles:
                tot += tile_sets[pairs[j][0]].shape[0]
                j += 1
            j = max(j, i + 1)
            batch = torch.cat([tile_sets[t] * s for t, s in pairs[i:j]])
            quant_w, _ = encode_tiles(batch)
            offset = 0
            for p, (t, s) in enumerate(pairs[i:j]):
                cnt = tile_sets[t].shape[0]
                out[i + p] = (quant_w[offset:offset + cnt] / s - tile_sets[t]) \
                    .square().mean()
                offset += cnt
            i = j
        return out

    # Stage 1: coarse grid over the full search range, on a strided subsample
    # of each tensor's tiles
    coarse = [COARSE_START + COARSE_STEP * i for i in range(COARSE_POINTS)]
    subs = [s[::3] for s in samples]
    pairs1 = [(t, s) for t in range(n_t) for s in coarse]
    mse1 = eval_pairs(pairs1, subs).reshape(n_t, len(coarse))
    centers = [coarse[c] for c in mse1.argmin(dim=1).tolist()]

    # Stage 2: fine grid around each tensor's coarse minimum, on its full sample
    fine = [[c + FINE_STEP * (i - 2) for i in range(FINE_POINTS)] for c in centers]
    pairs2 = [(t, s) for t in range(n_t) for s in fine[t]]
    mse2 = eval_pairs(pairs2, samples).reshape(n_t, FINE_POINTS)
    mse2_h = mse2.tolist()

    results = []
    for t in range(n_t):
        best_scale = select_refined_scale(fine[t], mse2_h[t])
        best = min(range(FINE_POINTS), key=lambda i: mse2_h[t][i])
        results.append((best_scale, mse2_h[t][best]))
    return results


def g_scale_gss(
    weight_r: torch.Tensor,
    encode_tiles: TileEncoder,
    width: int = 3,
) -> "tuple[float, float]":
    """Upstream ``g_scale_gss`` for a single tensor (``.py:1050-1076``).

    Returns ``(best_scale, best_mse)``.
    """
    tiles = sample_scale_tiles(weight_r, width)
    return g_scale_search_batch([tiles], encode_tiles)[0]


def g_scale_fn_from_encoder(
    encode_tiles: TileEncoder,
    width: int = 3,
) -> Callable[[torch.Tensor], float]:
    """Adapt :func:`g_scale_gss` to the ``g_scale_fn`` hook of ``regularize``."""
    def g_scale_fn(weight_r: torch.Tensor) -> float:
        return g_scale_gss(weight_r, encode_tiles, width)[0]
    return g_scale_fn
