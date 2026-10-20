"""Tests for the Sylvester Hadamard reference path (P2-06, contract §8).

Covers:
  * Sylvester construction values, symmetry and orthonormality;
  * bit-exact agreement with the independent decode oracle
    (``exl3xpu.ref.hadamard128``);
  * blockwise application equals an explicit block-diagonal matmul (i.e. the
    transform is *not* a full Kronecker matrix);
  * involution (``H`` is its own inverse) which the encode/decode pair relies on;
  * the decode ordering ``H_l → ×su → H_r → ×sv`` against
    ``exl3xpu.ref.weight_orig`` on an identical trellis.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

from exl3_quant.hadamard import (
    HAD_DIM,
    blockwise_preapply_had_l_,
    blockwise_preapply_had_r_,
    hadamard,
    preapply_had_l,
    preapply_had_r,
    sylvester_hadamard,
    weight_from_quantized,
)

EXL3XPU_PATH = Path("/home/yanchun/Projects/exl3xpu")
if EXL3XPU_PATH.exists() and str(EXL3XPU_PATH) not in sys.path:
    sys.path.insert(0, str(EXL3XPU_PATH))
try:
    import exl3xpu.ref as oracle_ref
    ORACLE_AVAILABLE = True
except Exception:  # pragma: no cover
    ORACLE_AVAILABLE = False


class TestSylvesterConstruction(unittest.TestCase):
    def test_values_symmetry_and_orthogonality(self):
        for n in (2, 8, 128):
            raw = sylvester_hadamard(n)
            self.assertEqual(raw.shape, (n, n))
            self.assertTrue(torch.all(raw.abs() == 1.0))
            self.assertTrue(torch.equal(raw, raw.T), f"H_{n} must be symmetric")
            self.assertTrue(
                torch.equal(raw @ raw.T, torch.eye(n, dtype=torch.float64) * n),
                f"H_{n} H_{n}^T must equal {n} I",
            )

    def test_orthonormal_scaling(self):
        h = hadamard(128, torch.float64)
        self.assertTrue(torch.allclose(h @ h.T, torch.eye(128, dtype=torch.float64),
                                       atol=1e-12))
        self.assertTrue(torch.allclose(h.abs(), torch.full_like(h, 1 / math.sqrt(128)),
                                       atol=0.0))

    def test_rejects_non_power_of_two(self):
        for n in (0, 3, 96, 200):
            with self.assertRaises(ValueError):
                sylvester_hadamard(n)

    def test_fast_path_cache_is_consistent(self):
        a = hadamard(128, torch.float32)
        b = hadamard(128, torch.float32)
        self.assertIs(a, b)


class TestBlockwiseApplication(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.k, self.n = 256, 384
        self.x = torch.randn(self.k, self.n, dtype=torch.float32)

    def test_left_equals_block_diagonal_matmul(self):
        h = hadamard(128, torch.float32)
        block = torch.block_diag(*([h] * (self.k // 128)))
        expected = block @ self.x
        got = preapply_had_l(self.x, 128)
        self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    def test_right_equals_block_diagonal_matmul(self):
        h = hadamard(128, torch.float32)
        block = torch.block_diag(*([h] * (self.n // 128)))
        expected = self.x @ block
        got = preapply_had_r(self.x, 128)
        self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    def test_involution_both_directions(self):
        left = preapply_had_l(preapply_had_l(self.x, 128), 128)
        right = preapply_had_r(preapply_had_r(self.x, 128), 128)
        self.assertTrue(torch.allclose(left, self.x, atol=1e-5))
        self.assertTrue(torch.allclose(right, self.x, atol=1e-5))

    def test_inplace_variants_match_out_of_place_and_validate(self):
        a = self.x.clone()
        b = self.x.clone()
        blockwise_preapply_had_l_(a, 128)
        blockwise_preapply_had_r_(b, 128)
        self.assertTrue(torch.allclose(a, preapply_had_l(self.x, 128), atol=1e-6))
        self.assertTrue(torch.allclose(b, preapply_had_r(self.x, 128), atol=1e-6))
        with self.assertRaises(ValueError):
            blockwise_preapply_had_l_(self.x.to(torch.float16), 128)

    def test_rejects_non_divisible_shapes(self):
        with self.assertRaises(ValueError):
            preapply_had_l(torch.zeros(100, 128), 128)
        with self.assertRaises(ValueError):
            preapply_had_r(torch.zeros(128, 100), 128)


@unittest.skipUnless(ORACLE_AVAILABLE, "exl3xpu.ref oracle not available")
class TestOracleParity(unittest.TestCase):
    def test_hadamard128_bit_exact_vs_oracle(self):
        ours = hadamard(128, torch.float64)
        theirs = oracle_ref.hadamard128(torch.device("cpu"), torch.float64)
        self.assertTrue(torch.equal(ours, theirs))

    def test_decode_order_matches_weight_orig(self):
        torch.manual_seed(7)
        k, n = 256, 384
        K = 2
        cb = 1  # mcg in the oracle's numbering (CB_MCG)
        trellis = torch.randint(
            -32768, 32767, (k // 16, n // 16, 16 * K), dtype=torch.int16
        )
        su = (torch.randn(k).sign() * torch.rand(k) * 2 + 0.5).reshape(k, 1)
        sv = (torch.randn(n).sign() * torch.rand(n) * 2 + 0.5).reshape(1, n)

        w_hat = oracle_ref.reconstruct_inner(trellis, K, cb).float()
        # The oracle expects 1-D vectors (it applies unsqueeze itself).
        expected = oracle_ref.weight_orig(
            trellis, su.reshape(-1), sv.reshape(-1), K, cb
        )
        got = weight_from_quantized(w_hat, su, sv)
        self.assertTrue(torch.allclose(got, expected, atol=1e-4),
                        f"max diff {(got - expected).abs().max().item():.3e}")


if __name__ == "__main__":
    unittest.main()
