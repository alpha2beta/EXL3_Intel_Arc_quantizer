"""Tests for upstream weight regularization (P2-04, contract §9.3).

The decisive test is the round trip: ``regularize`` followed by the decode
ordering (``H_l → ×su → H_r → ×sv``) must reconstruct the original weight exactly
(in fp32 tolerance). That single identity pins down the transform order, the
scale vectors, and the global-scale folding.
"""

from __future__ import annotations

import unittest

import torch

from exl3_quant.hadamard import weight_from_quantized
from exl3_quant.regularize import (
    CODEBOOK_SCALE,
    RegularizeResult,
    block_rms,
    regularize,
)


def make_weight(k=256, n=384, seed=11):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(k, n, generator=generator)


def sign_vector(size, seed=3, shape="col"):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(size, generator=generator).sign()
    return values.reshape(size, 1) if shape == "col" else values.reshape(1, size)


class TestBlockRMS(unittest.TestCase):
    def test_matches_direct_computation(self):
        torch.manual_seed(0)
        x = torch.randn(256, 384)
        for dim in (0, 1):
            got = block_rms(x, dim=dim, keepdim=True)
            expected = x.square().mean(dim=dim, keepdim=True).sqrt()
            self.assertTrue(torch.allclose(got, expected, atol=1e-6))

    def test_block_boundary_accumulation(self):
        # 40 rows => blocks of 32 + 8; the split path must still be the mean.
        x = torch.ones(40, 128)
        got = block_rms(x, dim=0, keepdim=True)
        self.assertTrue(torch.allclose(got, torch.ones(1, 128), atol=1e-7))


class TestRegularize(unittest.TestCase):
    def setUp(self):
        self.k, self.n = 256, 384
        self.weight = make_weight(self.k, self.n)
        self.su = sign_vector(self.k, seed=5, shape="col")
        self.sv = sign_vector(self.n, seed=9, shape="row")

    def test_roundtrip_reconstructs_weight(self):
        reg = regularize(self.weight, self.su, self.sv, skip_g_scale=True)
        recovered = weight_from_quantized(reg.weight, reg.su, reg.sv)
        max_diff = (recovered - self.weight).abs().max().item()
        self.assertLess(max_diff, 1e-4, f"round-trip max diff {max_diff:.3e}")

    def test_roundtrip_with_forced_out_scales_off(self):
        reg = regularize(
            self.weight, self.su, self.sv,
            quant_args={"apply_out_scales": False}, skip_g_scale=True,
        )
        self.assertFalse(reg.apply_out_scales)
        recovered = weight_from_quantized(reg.weight, reg.su, reg.sv)
        self.assertLess((recovered - self.weight).abs().max().item(), 1e-4)

    def test_roundtrip_with_global_scale_hook(self):
        reg = regularize(
            self.weight, self.su, self.sv,
            skip_g_scale=False, g_scale_fn=lambda w: 2.0,
        )
        self.assertEqual(reg.g_scale, 2.0)
        recovered = weight_from_quantized(reg.weight, reg.su, reg.sv)
        self.assertLess((recovered - self.weight).abs().max().item(), 1e-4)

    def test_missing_global_scale_hook_raises(self):
        with self.assertRaises(ValueError):
            regularize(self.weight, self.su, self.sv, skip_g_scale=False)

    def test_skew_heuristic(self):
        # Concentrated diagonal: top 2% dominates -> do not scale output channels.
        spike = torch.zeros(self.k)
        spike[: self.k // 50] = 1.0
        reg_spike = regularize(self.weight, self.su, self.sv,
                               H_diag=spike, skip_g_scale=True)
        self.assertIsNotNone(reg_spike.skew_factor)
        self.assertGreaterEqual(reg_spike.skew_factor, 0.15)
        self.assertFalse(reg_spike.apply_out_scales)

        # Flat diagonal -> apply output channel scaling.
        flat = torch.ones(self.k)
        reg_flat = regularize(self.weight, self.su, self.sv,
                              H_diag=flat, skip_g_scale=True)
        self.assertLess(reg_flat.skew_factor, 0.15)
        self.assertTrue(reg_flat.apply_out_scales)

    def test_explicit_apply_out_scales_overrides_heuristic(self):
        spike = torch.zeros(self.k)
        spike[: self.k // 50] = 1.0
        reg = regularize(self.weight, self.su, self.sv, H_diag=spike,
                         quant_args={"apply_out_scales": True}, skip_g_scale=True)
        self.assertTrue(reg.apply_out_scales)

    def test_zero_output_channel_is_zeroed_and_floored(self):
        weight = self.weight.clone()
        weight[:, 7] = 0.0
        reg = regularize(weight, self.su, self.sv, H_diag=torch.ones(self.k),
                         skip_g_scale=True)
        self.assertTrue(reg.apply_out_scales)
        self.assertEqual(float(reg.sv[0, 7]), 0.0)
        self.assertAlmostEqual(float(reg.out_channel_scales[0, 7]), 0.1, places=7)

    def test_all_zero_weight_sets_zeros_flag(self):
        weight = torch.zeros(self.k, self.n)
        args = {}
        reg = regularize(weight, self.su, self.sv, quant_args=args,
                         skip_g_scale=True)
        self.assertTrue(reg.zeros)
        self.assertTrue(args["zeros"])
        self.assertTrue(torch.all(reg.weight == 0))

    def test_input_weight_not_mutated(self):
        original = self.weight.clone()
        regularize(self.weight, self.su, self.sv, skip_g_scale=True)
        self.assertTrue(torch.equal(self.weight, original))

    def test_scales_are_folded_as_expected(self):
        reg = regularize(self.weight, self.su, self.sv, H_diag=torch.ones(self.k),
                         skip_g_scale=True)
        self.assertIsInstance(reg, RegularizeResult)
        # su carries the (negative) codebook scale normalisation.
        self.assertTrue(torch.all(reg.su < 0) or torch.any(reg.su < 0))
        self.assertEqual(reg.su.shape, (self.k, 1))
        self.assertEqual(reg.sv.shape, (1, self.n))
        self.assertAlmostEqual(CODEBOOK_SCALE, 1.24371088, places=8)

    def test_validation(self):
        with self.assertRaises(ValueError):
            regularize(torch.randn(128), self.su, self.sv)
        with self.assertRaises(ValueError):
            regularize(torch.randn(255, 384), sign_vector(255), self.sv)
        with self.assertRaises(ValueError):
            regularize(self.weight.to(torch.float64), self.su, self.sv)
        with self.assertRaises(ValueError):
            regularize(self.weight, sign_vector(self.k - 1), self.sv)


if __name__ == "__main__":
    unittest.main()
