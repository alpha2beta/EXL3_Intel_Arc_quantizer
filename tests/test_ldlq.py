"""Tests for the block-LDL preparation (P0-22, contract §9.1).

The assertions check the *extracted equations* directly:
  * ``L Lᵀ = H`` (undamped case),
  * the fold formula ``M[:, block] = L[:, block] @ inv(L_bb)``,
  * identity insertion on the block diagonal,
  * the damping-retry schedule (no pre-emptive damping).
"""

import unittest

import torch

from exl3_quant.ldlq import block_ldl

try:
    XPU_AVAILABLE = torch.xpu.is_available()
except Exception:  # pragma: no cover
    XPU_AVAILABLE = False


def spd_matrix(n, seed=0, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=generator)
    H = A @ A.T / n + torch.eye(n)
    return (H * scale).to(torch.float32)


def near_indefinite(n, seed=0):
    """Symmetric matrix with one small negative eigenvalue and large diag."""
    generator = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=generator))
    eigenvalues = torch.full((n,), 100.0)
    eigenvalues[-1] = -0.5
    return (Q @ torch.diag(eigenvalues) @ Q.T).to(torch.float32)


class TestBlockLDL(unittest.TestCase):
    def setUp(self):
        self.n = 64
        self.b = 16
        self.H = spd_matrix(self.n, seed=7)

    def test_cholesky_matches_h(self):
        result = block_ldl(self.H, block_size=self.b)
        factor = result.cholesky
        # lower triangular with positive diagonal
        self.assertTrue(torch.allclose(factor, torch.tril(factor), atol=0.0))
        self.assertTrue(torch.all(factor.diagonal() > 0))
        # L Lᵀ = H
        reconstructed = factor @ factor.T
        self.assertTrue(torch.allclose(reconstructed, self.H, atol=1e-4),
                        f"max err {(reconstructed - self.H).abs().max().item():.3e}")

    def test_block_diagonal_becomes_identity(self):
        result = block_ldl(self.H, block_size=self.b)
        identity = torch.eye(self.b)
        for block in range(self.n // self.b):
            start = block * self.b
            got = result.transformed[start:start + self.b, start:start + self.b]
            self.assertTrue(torch.allclose(got, identity, atol=1e-6),
                            f"block {block} not identity: {got}")

    def test_fold_formula(self):
        result = block_ldl(self.H, block_size=self.b)
        L = result.cholesky
        m = self.n // self.b
        for col in range(m):
            start = col * self.b
            stop = start + self.b
            expected = L[:, start:stop] @ result.block_inverses[col]
            # identity insertion replaces the diagonal block of this column slice
            expected[start:stop, :] = torch.eye(self.b)
            got = result.transformed[:, start:stop]
            self.assertTrue(torch.allclose(got, expected, atol=1e-6),
                            f"fold mismatch in column block {col}")

    def test_inverse_matches_diagonal_block(self):
        result = block_ldl(self.H, block_size=self.b)
        for block in range(self.n // self.b):
            start = block * self.b
            stop = start + self.b
            block_matrix = result.cholesky[start:stop, start:stop]
            product = block_matrix @ result.block_inverses[block]
            self.assertTrue(torch.allclose(product, torch.eye(self.b), atol=1e-5))

    def test_positive_definite_needs_no_retries(self):
        result = block_ldl(self.H, block_size=self.b)
        self.assertEqual(result.retries, 0)
        self.assertFalse(result.damping_applied)

    def test_retry_damping_succeeds(self):
        H = near_indefinite(32, seed=3)
        # precondition: raw matrix is not positive definite
        with self.assertRaises(torch.linalg.LinAlgError):
            torch.linalg.cholesky(H)

        result = block_ldl(H, block_size=16)
        self.assertGreaterEqual(result.retries, 1)
        self.assertLessEqual(result.retries, 10)
        self.assertTrue(result.damping_applied)

        # LLᵀ equals H plus a scalar multiple of the identity (the damping).
        difference = result.cholesky @ result.cholesky.T - H
        diagonal = difference.diagonal()
        self.assertGreater(diagonal.min().item(), 0.0)
        self.assertLess((diagonal - diagonal.mean()).abs().max().item(), 1e-2)
        off_diagonal = difference - torch.diag(diagonal)
        self.assertLess(off_diagonal.abs().max().item(), 1e-2)

    def test_input_not_mutated_by_damping(self):
        H = near_indefinite(32, seed=4)
        original = H.clone()
        block_ldl(H, block_size=16)
        self.assertTrue(torch.equal(H, original))

    def test_fails_after_retry_budget(self):
        H = -torch.eye(16)
        with self.assertRaises(torch.linalg.LinAlgError):
            block_ldl(H, block_size=16, max_retries=2)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            block_ldl(torch.randn(63, 63), block_size=16)   # not divisible
        with self.assertRaises(ValueError):
            block_ldl(torch.randn(64, 63), block_size=16)   # not square
        with self.assertRaises(ValueError):
            block_ldl(self.H.to(torch.float64), block_size=16)  # dtype policy
        with self.assertRaises(ValueError):
            block_ldl(self.H, block_size=0)
        with self.assertRaises(ValueError):
            block_ldl(self.H, block_size=128)               # larger than n
        with self.assertRaises(ValueError):
            block_ldl(self.H, block_size=16, max_retries=-1)

    @unittest.skipUnless(XPU_AVAILABLE, "torch.xpu device not available")
    def test_xpu_matches_cpu(self):
        H = spd_matrix(64, seed=11)
        cpu = block_ldl(H, block_size=16)
        xpu = block_ldl(H.to("xpu"), block_size=16)
        torch.xpu.synchronize()

        self.assertEqual(xpu.retries, cpu.retries)
        self.assertTrue(torch.allclose(xpu.transformed.cpu(), cpu.transformed,
                                       atol=1e-5),
                        "XPU fold differs from CPU reference")
        self.assertTrue(torch.allclose(xpu.cholesky.cpu(), cpu.cholesky,
                                       atol=1e-5))


if __name__ == "__main__":
    unittest.main()
