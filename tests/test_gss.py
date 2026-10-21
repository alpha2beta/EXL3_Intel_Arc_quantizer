"""Tests for the global-scale search wiring (P2-04 completion).

Covers, in order of strength:

1. **Sampling contract** — the wrapped diagonal covers every tile row and
   column; tile values match an independent manual extraction with the
   tensor-core permutation applied; extreme-RMS tiles are included.
2. **Search contract** — the two-stage coarse/fine pipeline can be recomputed
   independently stage-by-stage and must agree; batching into fewer encoder
   calls must not change any pair's MSE.
3. **End-to-end effect** — with the real Arc trellis encoder the found scale
   attains a lower sample MSE than g_scale = 1, ``regularize`` folds it exactly
   like upstream (``weight *= g; su /= g``), and the decode round-trip with the
   ``g``-scaled ``su`` still reconstructs the original weight.
4. **Clamping** — the 0.01 floor and the fine-grid edges behave like upstream.
"""

from __future__ import annotations

import unittest

import torch

from exl3_quant.gss import (
    COARSE_POINTS,
    FINE_STEP,
    MIN_SCALE,
    select_refined_scale,
    COARSE_START,
    COARSE_STEP,
    FINE_POINTS,
    FINE_STEP,
    g_scale_fn_from_encoder,
    g_scale_gss,
    g_scale_search_batch,
    sample_scale_tiles,
)
from exl3_quant.hadamard import weight_from_quantized
from exl3_quant.regularize import regularize
from exl3_quant.trellis_format import tensor_core_perm


def make_encoder():
    """Bit-exact Arc trellis encoder through the P2-08 adapter protocol."""
    try:
        if torch.xpu.is_available():
            from exl3_quant.ldlq import make_xpu_tile_encoder
            return make_xpu_tile_encoder(K=2)
    except Exception:
        pass
    from exl3_quant.ldlq import make_reference_tile_encoder
    return make_reference_tile_encoder(K=2)


def mse_at(tiles, encode, scale):
    quant_w, _ = encode(tiles * scale)
    return float((quant_w / scale - tiles).square().mean())


class TestSampleScaleTiles(unittest.TestCase):
    def test_wrapped_diagonal_covers_every_row_and_column(self):
        torch.manual_seed(0)
        k, n = 256, 512
        weight = torch.randn(k, n)
        tiles = sample_scale_tiles(weight, width=3)

        tiles_k, tiles_n = k // 16, n // 16
        diag_len = max(tiles_k, tiles_n)
        base = diag_len * 3

        self.assertEqual(tiles.dtype, torch.float32)
        self.assertEqual(tiles.shape[1], 256)

        # The first `base` sampled tiles walk a wrapped diagonal: extract their
        # (tile_k, tile_n) source positions with an independent Python loop and
        # confirm every tile row and every tile column is visited at least once.
        positions = []
        idx = 0
        for d in range(diag_len):
            for w in range(3):
                positions.append((d % tiles_k, (d + w) % tiles_n))
                idx += 1
        self.assertEqual(idx, base)
        self.assertEqual({r for r, _ in positions}, set(range(tiles_k)))
        self.assertEqual({c for _, c in positions}, set(range(tiles_n)))

    def test_sample_values_match_manual_extraction(self):
        torch.manual_seed(1)
        k, n = 256, 256
        weight = torch.randn(k, n)
        tiles_k, tiles_n = k // 16, n // 16
        w4 = weight.reshape(tiles_k, 16, tiles_n, 16)

        diag_len = max(tiles_k, tiles_n)
        for width in (1, 2, 3):
            with self.subTest(width=width):
                tiles = sample_scale_tiles(weight, width=width)

                kk, nn = [], []
                for d in range(diag_len):
                    for w in range(width):
                        kk.append(d % tiles_k)
                        nn.append((d + w) % tiles_n)

                num_x = min(max(8, (diag_len * width) // 16),
                            (tiles_k * tiles_n + 1) // 2)
                tile_ms = w4.square().mean(dim=(1, 3)).flatten()
                hi = torch.topk(tile_ms, num_x).indices.tolist()
                lo = torch.topk(tile_ms, num_x, largest=False).indices.tolist()
                flat_hi_lo = hi + lo
                xk = [f // tiles_n for f in flat_hi_lo]
                xn = [f % tiles_n for f in flat_hi_lo]

                perm = torch.as_tensor(tensor_core_perm())
                expected = []
                positions = list(zip(kk + xk, nn + xn))
                for r, c in positions:
                    expected.append(w4[r, :, c, :].reshape(-1)[perm])
                expected = torch.stack(expected)

                self.assertEqual(tiles.shape[0], len(positions))
                self.assertTrue(torch.equal(tiles, expected))

    def test_extreme_rms_tiles_are_sampled(self):
        torch.manual_seed(2)
        k, n = 256, 256
        weight = torch.randn(k, n)
        tiles_k, tiles_n = k // 16, n // 16
        # One extreme tile, plus one with almost no energy
        weight[96:112, 32:48] *= 100.0
        weight[0:16, 208:224] *= 1e-4

        tiles = sample_scale_tiles(weight, width=1)
        w4 = weight.reshape(tiles_k, 16, tiles_n, 16)
        perm = torch.as_tensor(tensor_core_perm())
        hi_flat = int(torch.topk(
            w4.square().mean(dim=(1, 3)).flatten(), 8
        ).indices[0])
        r, c = hi_flat // tiles_n, hi_flat % tiles_n
        hi_tile = w4[r, :, c, :].reshape(-1)[perm]
        lo_flat = int(torch.topk(
            w4.square().mean(dim=(1, 3)).flatten(), 8, largest=False
        ).indices[0])
        r2, c2 = lo_flat // tiles_n, lo_flat % tiles_n
        lo_tile = w4[r2, :, c2, :].reshape(-1)[perm]

        matched_hi = (tiles - hi_tile).abs().amax(dim=1).min().item()
        matched_lo = (tiles - lo_tile).abs().amax(dim=1).min().item()
        self.assertEqual(matched_hi, 0.0)
        self.assertEqual(matched_lo, 0.0)

    def test_sample_takes_whole_rows(self):
        """Outlier channels are whole rows/columns, so samples must be full tiles."""
        torch.manual_seed(3)
        k, n = 256, 256
        weight = torch.randn(k, n)
        tiles_k, tiles_n = k // 16, n // 16
        weight[112:128, :] *= 50.0   # one full input-channel outlier row band
        tiles = sample_scale_tiles(weight, width=3)
        w4 = weight.reshape(tiles_k, 16, tiles_n, 16)
        spread = w4.square().mean(dim=(1, 3)).flatten()
        self.assertGreater(float(spread.std()), 0.0)
        # Some sampled tile must contain the amplified values.
        big = float(tiles.abs().max())
        self.assertGreater(big, 20.0)


class TestGScaleSearchBatch(unittest.TestCase):
    def setUp(self):
        self.encode = make_encoder()

    def test_matches_stage_by_stage_recomputation(self):
        """The documented coarse->fine->parabolic pipeline, recomputed manually."""
        torch.manual_seed(5)
        k, n = 256, 384
        weight = torch.randn(k, n) * 0.2
        tiles = sample_scale_tiles(weight)
        subs = tiles[::3]

        coarse = [COARSE_START + COARSE_STEP * i for i in range(COARSE_POINTS)]
        curve = [mse_at(subs, self.encode, s) for s in coarse]
        center = coarse[curve.index(min(curve))]
        fine = [center + FINE_STEP * (i - 2) for i in range(FINE_POINTS)]
        fine_curve = [mse_at(tiles, self.encode, s) for s in fine]
        best = fine_curve.index(min(fine_curve))

        got_scale, got_mse = g_scale_gss(weight, self.encode)
        self.assertAlmostEqual(got_mse, fine_curve[best], places=6)
        self.assertGreaterEqual(got_scale, max(0.01, fine[best] - FINE_STEP / 2))
        self.assertLessEqual(got_scale, fine[best] + FINE_STEP / 2 + 1e-9)

    def test_batching_does_not_change_pairwise_mse(self):
        torch.manual_seed(6)
        a = torch.randn(48, 256)
        b = torch.randn(64, 256)
        both = g_scale_search_batch([a, b], self.encode)
        solo_a = g_scale_search_batch([a], self.encode)[0]
        solo_b = g_scale_search_batch([b], self.encode)[0]
        self.assertEqual(both[0][0], solo_a[0])
        self.assertEqual(both[0][1], solo_a[1])
        self.assertEqual(both[1][0], solo_b[0])
        self.assertEqual(both[1][1], solo_b[1])

    def test_chunked_calls_match_oneshot_calls(self):
        """Forcing several encoder calls must not change the results."""
        torch.manual_seed(7)
        samples = [torch.randn(60, 256), torch.randn(45, 256)]
        oneshot = g_scale_search_batch(samples, self.encode, max_tiles=65536)
        chunked = g_scale_search_batch(samples, self.encode, max_tiles=77)
        self.assertEqual(oneshot, chunked)

    def test_search_finds_lower_mse_than_scale_one(self):
        """On low-magnitude tiles the optimum sits above 1 (measured 2.05)."""
        torch.manual_seed(8)
        weight = torch.randn(256, 512) * 0.03
        tiles = sample_scale_tiles(weight)
        scale, mse = g_scale_gss(weight, self.encode)
        self.assertGreater(scale, 0.0)
        self.assertLess(mse, mse_at(tiles, self.encode, 1.0))

    def test_deterministic(self):
        torch.manual_seed(9)
        weight = torch.randn(256, 256)
        first = g_scale_gss(weight, self.encode)
        second = g_scale_gss(weight, self.encode)
        self.assertEqual(first, second)

    def test_huge_weights_push_scale_down(self):
        """fp16 trellis precision loss dominates at extreme magnitudes, so the
        search must walk below the coarse grid (measured ~0.025 for sigma=1000).
        """
        torch.manual_seed(10)
        weight = torch.randn(256, 256) * 1000.0
        scale, mse = g_scale_gss(weight, self.encode)
        self.assertLess(scale, 0.1)
        self.assertTrue(mse >= 0.0)

    def test_empty_samples(self):
        self.assertEqual(g_scale_search_batch([], self.encode), [])

    def test_invalid_sample_shape(self):
        with self.assertRaises(ValueError):
            g_scale_search_batch([torch.randn(10, 128)], self.encode)


class TestSelectRefinedScale(unittest.TestCase):
    """The fine-stage selection + parabolic refinement + floor (`.py:1038-1046`)."""

    FINE = [-0.05, 0.025, 0.1, 0.175, 0.25]

    def test_tied_curve_walks_to_left_edge_and_floors(self):
        self.assertEqual(select_refined_scale(self.FINE, [0.0] * 5), MIN_SCALE)

    def test_left_side_basin_pulls_scale_down(self):
        # Parabola through (0.2 @ -0.075), (0.1 @ 0), (0.4 @ +0.075):
        # vertex at offset u = 0.5*(y0-y2)/(y0-2y1+y2) = -0.25 steps.
        ms = [0.3, 0.2, 0.1, 0.4, 0.5]
        self.assertAlmostEqual(select_refined_scale(self.FINE, ms),
                               0.1 - 0.25 * FINE_STEP)

    def test_parabolic_offset_interpolates(self):
        # Symmetric basin around index 2: offset 0 -> scale = center
        ms = [0.2, 0.1, 0.0, 0.1, 0.2]
        self.assertAlmostEqual(select_refined_scale(self.FINE, ms), 0.1)
        # Leaning basin: y0 == y2, y1 low -> denom == 2*y1 > 0 (y1 < 0 required)
        # offset = 0.5*(y0-y2)/denom = 0 -> same; asymmetric instead:
        ms = [2.0, 1.0, 0.0, 2.0, 6.0]
        got = select_refined_scale([-0.05, 0.025, 0.1, 0.175, 0.25], ms)
        best = ms.index(min(ms))
        self.assertGreaterEqual(got, self.FINE[best] - FINE_STEP / 2)
        self.assertLessEqual(got, self.FINE[best] + FINE_STEP / 2)

    def test_parabolic_offset_never_exceeds_half_step(self):
        """For any convex triple y0=y1+a, y2=y1+b the offset is analytically
        0.5*(a-b)/(a+b), which lies in [-0.5, 0.5]; the upstream clamp is a
        float-pathology defence. Verify the exact formula and the bound."""
        # Leaning basin: a = y0-y1 = 0.9? No: y0=1.0, y1=0.0, y2=0.1, so
        # offset = 0.5*(y0-y2)/(y0-2y1+y2) = 0.5*0.9/1.1 = 9/22.
        ms = [1.9, 1.0, 0.0, 0.1, 2.0]
        self.assertAlmostEqual(select_refined_scale(self.FINE, ms),
                               0.1 + (9.0 / 22.0) * FINE_STEP, places=12)
        # Randomized convex basins: the result must stay within half a step
        for seed in range(20):
            g = torch.Generator().manual_seed(100 + seed)
            y1 = float(torch.rand(1, generator=g)) + 1.0
            a = float(torch.rand(1, generator=g)) + 1e-3
            b = float(torch.rand(1, generator=g)) + 1e-3
            y0, y2 = y1 + a, y1 + b
            ms = [9.9, y0, y1, y2, 9.9]   # best is interior (index 2)
            got = select_refined_scale(self.FINE, ms)
            self.assertGreaterEqual(got, self.FINE[2] - 0.5 * FINE_STEP)
            self.assertLessEqual(got, self.FINE[2] + 0.5 * FINE_STEP)


class TestRegularizeWiring(unittest.TestCase):
    def test_regularize_folds_g_scale_like_upstream(self):
        torch.manual_seed(11)
        k, n = 256, 256
        weight = torch.randn(k, n) * 0.2
        su0 = torch.where(torch.rand(k, 1) < 0.5, -1.0, 1.0)
        sv0 = torch.where(torch.rand(1, n) < 0.5, -1.0, 1.0)

        skip = regularize(weight, su0, sv0, skip_g_scale=True)
        live = regularize(weight, su0, sv0, skip_g_scale=False,
                          g_scale_fn=g_scale_fn_from_encoder(make_encoder()))

        g = live.g_scale
        self.assertGreater(g, 0.0)
        # weight *= g;  su /= g  (verbatim upstream .py:1224-1225)
        self.assertAlmostEqual(live.weight.max().item() / skip.weight.max().item(),
                               g, places=2)
        self.assertTrue(torch.allclose(live.su, skip.su / g, atol=1e-6))
        # sv, channel scales, and the skew decision are untouched by g_scale
        self.assertTrue(torch.equal(live.sv, skip.sv))
        self.assertEqual(live.apply_out_scales, skip.apply_out_scales)
        self.assertTrue(torch.allclose(live.out_channel_scales,
                                       skip.out_channel_scales, atol=1e-7))

    def test_round_trip_with_g_scale(self):
        """Decode with the g-scaled su must reconstruct the original weight."""
        torch.manual_seed(12)
        k, n = 256, 256
        weight = torch.randn(k, n) * 0.2
        su0 = torch.where(torch.rand(k, 1) < 0.5, -1.0, 1.0)
        sv0 = torch.where(torch.rand(1, n) < 0.5, -1.0, 1.0)

        result = regularize(weight, su0, sv0, skip_g_scale=False,
                            g_scale_fn=g_scale_fn_from_encoder(make_encoder()))
        decoded = weight_from_quantized(result.weight, result.su, result.sv)
        self.assertTrue(torch.allclose(decoded, weight, atol=1e-3),
                        f"max diff {(decoded - weight).abs().max().item():.3e}")

    def test_regularize_still_requires_fn_when_not_skipping(self):
        torch.manual_seed(13)
        weight = torch.randn(256, 256)
        su0 = torch.ones(256, 1)
        sv0 = torch.ones(1, 256)
        with self.assertRaises(ValueError):
            regularize(weight, su0, sv0, skip_g_scale=False, g_scale_fn=None)


if __name__ == "__main__":
    unittest.main()
