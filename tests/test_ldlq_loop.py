"""Tests for the one-sided LDLQ error-feedback loop (P2-08, contract §9.2).

The loop is the part of EXL3 where a silent index/sign error still produces
plausible-looking weights, so the tests pin it down from four directions:

1. **No-feedback equivalence** — with an identity ``L``, the loop must reduce to
   independent per-tile quantization.
2. **Exact target correction** — a recording encoder checks that the correction
   is *added* to the target rows and uses the strictly-lower ``L`` slice.
3. **Independent naive implementation** — a dense, index-explicit transcription
   (no views, explicit tile loops) must agree tensor-for-tensor.
4. **Objective improvement** — on a correlated Hessian the trailing-weight
   feedback must lower ``tr(E H Eᵀ)`` versus the same encoder with feedback
   disabled.

Plus layout/dtype checks, input immutability, and XPU↔reference parity.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from exl3_quant.ldlq import (
    LDLQResult,
    block_ldl,
    ldlq,
    make_reference_tile_encoder,
)
from exl3_quant.trellis_format import tensor_core_perm, tensor_core_perm_inv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class GridEncoder:
    """Deterministic fast encoder: uniform scalar quantization to ``step``."""

    def __init__(self, step: float = 0.25):
        self.step = step
        self.inputs: list[torch.Tensor] = []

    def __call__(self, tiles: torch.Tensor):
        self.inputs.append(tiles.detach().clone())
        quantized = torch.round(tiles / self.step) * self.step
        indices = (
            torch.arange(tiles.numel(), dtype=torch.int64)
            .reshape(tiles.shape) % 65536
        )
        return quantized, indices


class LinearEncoder:
    """Continuous deterministic encoder: ``q = gain * tiles + bias``.

    Unlike :class:`GridEncoder` this has no rounding boundaries, so a 1-ulp
    difference between two arithmetic formulations cannot flip a tile and hide
    (or fake) an indexing bug.
    """

    def __init__(self, gain: float = 1.1, bias: float = 0.05):
        self.gain = gain
        self.bias = bias

    def __call__(self, tiles: torch.Tensor):
        indices = (
            torch.arange(tiles.numel(), dtype=torch.int64)
            .reshape(tiles.shape) % 65536
        )
        return tiles * self.gain + self.bias, indices


class RecordingEncoder:
    """Reconstructs zeros and records every target tile it was given."""

    def __init__(self):
        self.inputs: list[torch.Tensor] = []

    def __call__(self, tiles: torch.Tensor):
        self.inputs.append(tiles.detach().clone())
        return (
            torch.zeros_like(tiles),
            torch.zeros(tiles.shape, dtype=torch.int16),
        )


def rows_to_tiles(rows: torch.Tensor) -> torch.Tensor:
    """(16, n) rows -> (tiles_n, 256), row-major within each 16x16 tile."""
    tiles_n = rows.shape[1] // 16
    out = torch.empty(tiles_n, 256, dtype=rows.dtype)
    for t in range(tiles_n):
        out[t] = rows[:, t * 16:(t + 1) * 16].reshape(-1)
    return out


def tiles_to_rows(tiles: torch.Tensor, n: int) -> torch.Tensor:
    tiles_n = n // 16
    out = torch.empty(16, n, dtype=tiles.dtype)
    for t in range(tiles_n):
        out[:, t * 16:(t + 1) * 16] = tiles[t].reshape(16, 16)
    return out


def naive_ldlq(weight, L, encode, buf_size_k):
    """Dense, index-explicit transcription used as an independent oracle.

    Deliberately avoids tensor views and reshape/permute arithmetic: tiles are
    packed with explicit Python loops and the feedback cache is a plain k x n
    matrix with global indices.
    """
    k, n = weight.shape
    tiles_k, tiles_n = k // 16, n // 16
    perm = torch.as_tensor(tensor_core_perm(), dtype=torch.long)
    inv = torch.as_tensor(tensor_core_perm_inv(), dtype=torch.long)

    prod = torch.zeros((k, n), dtype=torch.float32)
    weight_q = torch.zeros((k, n), dtype=torch.float32)
    encoded = torch.zeros((tiles_k, tiles_n, 256), dtype=torch.int16)

    for i in range(k - buf_size_k, -1, -buf_size_k):
        b_weight = weight[i:i + buf_size_k].clone()
        b_weight_q = torch.zeros_like(b_weight)

        for bi in range(buf_size_k - 16, -1, -16):
            bj = bi + 16
            bb_err = b_weight[bj:] - b_weight_q[bj:]
            compensation = prod[i + bi:i + bj].clone()
            if bb_err.shape[0] > 0:
                compensation = (
                    compensation
                    + L[i + bj:i + buf_size_k, i + bi:i + bj].T @ bb_err
                )
                prod[i + bi:i + bj] = compensation

            rows = b_weight[bi:bj] + compensation
            tiles = rows_to_tiles(rows)[:, perm]
            quant_w, quant_i = encode(tiles)
            quant_w = quant_w[:, inv]
            b_weight_q[bi:bj] = tiles_to_rows(quant_w, n)
            encoded[(i + bi) // 16] = quant_i

        prod = prod + L[i:i + buf_size_k].T @ (b_weight - b_weight_q)
        weight_q[i:i + buf_size_k] = b_weight_q

    return weight_q, encoded


def correlated_L(k: int, rank: int = 8, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(k, rank, generator=generator)
    H = a @ a.T + 0.05 * torch.eye(k)
    return H, block_ldl(H).transformed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIdentityL(unittest.TestCase):
    def test_identity_L_matches_independent_quantization(self):
        """No coupling => each 16-row block is quantized in isolation."""
        torch.manual_seed(1)
        k, n = 64, 128
        weight = torch.randn(k, n)
        L = block_ldl(torch.eye(k)).transformed
        self.assertTrue(torch.equal(L, torch.eye(k)))

        encode = make_reference_tile_encoder(K=2)
        result = ldlq(weight, L, encode, buf_size_k=64)

        perm = torch.as_tensor(tensor_core_perm())
        inv = torch.as_tensor(tensor_core_perm_inv())
        expected = torch.zeros_like(weight)
        expected_encoded = torch.zeros(k // 16, n // 16, 256, dtype=torch.int16)
        for block in range(k // 16):
            rows = weight[block * 16:(block + 1) * 16]
            tiles = rows_to_tiles(rows)[:, perm]
            quant_w, quant_i = encode(tiles)
            expected[block * 16:(block + 1) * 16] = tiles_to_rows(quant_w[:, inv], n)
            expected_encoded[block] = quant_i

        self.assertTrue(torch.equal(result.weight_q, expected))
        self.assertTrue(torch.equal(result.encoded, expected_encoded))

    def test_identity_quantizer_reproduces_weight(self):
        torch.manual_seed(2)
        k, n = 64, 128
        weight = torch.randn(k, n)
        L = torch.eye(k)
        result = ldlq(weight, L, lambda tiles: (tiles, torch.zeros(tiles.shape,
                                                                  dtype=torch.int16)),
                      buf_size_k=32)
        self.assertTrue(torch.equal(result.weight_q, weight))


class TestErrorFeedback(unittest.TestCase):
    def test_target_correction_is_added(self):
        """The upper block target is ``weight + L_lower.T @ error`` (added)."""
        torch.manual_seed(3)
        k, n = 32, 128
        weight = torch.randn(k, n)
        _, L = correlated_L(k, seed=4)
        self.assertGreater(L[16:32, 0:16].abs().max().item(), 1e-3)

        encoder = RecordingEncoder()
        result = ldlq(weight, L, encoder, buf_size_k=32)

        inv = torch.as_tensor(tensor_core_perm_inv())
        self.assertEqual(len(encoder.inputs), 2)
        # Step 1 (bottom block) sees the uncorrected rows.
        bottom = tiles_to_rows(encoder.inputs[0][:, inv], n)
        self.assertTrue(torch.allclose(bottom, weight[16:32], atol=0, rtol=0))
        # Step 2: error is the whole bottom block (it reconstructed to zero),
        # compensated through the strictly-lower L slice, then ADDED.
        expected = weight[0:16] + L[16:32, 0:16].T @ weight[16:32]
        top = tiles_to_rows(encoder.inputs[1][:, inv], n)
        self.assertTrue(torch.allclose(top, expected, atol=1e-6),
                        f"max diff {(top - expected).abs().max().item():.3e}")
        self.assertTrue(torch.equal(result.weight_q[16:32], torch.zeros(16, n)))

    def test_trailing_feedback_reaches_next_span(self):
        torch.manual_seed(5)
        k, n = 64, 128
        weight = torch.randn(k, n)
        _, L = correlated_L(k, seed=6)
        encoder = LinearEncoder()
        result = ldlq(weight, L, encoder, buf_size_k=32)

        expected_wq, expected_enc = naive_ldlq(weight, L, encoder, 32)
        self.assertTrue(torch.allclose(result.weight_q, expected_wq, atol=1e-5),
                        f"max diff {(result.weight_q - expected_wq).abs().max().item():.3e}")
        self.assertTrue(torch.equal(result.encoded, expected_enc))

    def test_matches_naive_implementation_for_multiple_spans(self):
        torch.manual_seed(7)
        k, n = 64, 128
        weight = torch.randn(k, n)
        _, L = correlated_L(k, seed=8)
        for buf in (16, 32, 64):
            with self.subTest(buf=buf):
                encoder = LinearEncoder()
                result = ldlq(weight, L, encoder, buf_size_k=buf)
                expected_wq, expected_enc = naive_ldlq(weight, L, encoder, buf)
                self.assertTrue(
                    torch.allclose(result.weight_q, expected_wq, atol=1e-5),
                    f"buf={buf} max diff "
                    f"{(result.weight_q - expected_wq).abs().max().item():.3e}",
                )
                self.assertTrue(torch.equal(result.encoded, expected_enc))

    def test_feedback_reduces_weighted_error(self):
        torch.manual_seed(9)
        k, n = 64, 128
        weight = torch.randn(k, n) * 0.5
        H, L = correlated_L(k, rank=6, seed=10)

        encoder_step = 0.25
        with_feedback = ldlq(weight, L, GridEncoder(encoder_step), buf_size_k=32)
        # Diagonal-only L: every strictly-lower slice is zero -> no feedback.
        without = ldlq(weight, torch.diag(torch.diag(L)),
                       GridEncoder(encoder_step), buf_size_k=32)

        def objective(result):
            err = weight - result.weight_q
            numerator = float(((H @ err) * err).sum())
            denominator = float(((H @ weight) * weight).sum())
            return numerator / denominator

        self.assertLess(objective(with_feedback), objective(without))

    def test_result_is_invariant_to_span_size(self):
        """Span buffering is a summation-order choice, not part of the math.

        With exact arithmetic the two-level update covers every previously
        quantized row below the target, whatever the span height. (The encoder
        is discontinuous, so compare with a small tolerance rather than bitwise.)
        """
        torch.manual_seed(11)
        k, n = 128, 128
        weight = torch.randn(k, n)
        _, L = correlated_L(k, seed=12)
        small = ldlq(weight, L, GridEncoder(0.3), buf_size_k=16)
        large = ldlq(weight, L, GridEncoder(0.3), buf_size_k=128)
        self.assertTrue(
            torch.allclose(small.weight_q, large.weight_q, atol=1e-5),
            f"max diff {(small.weight_q - large.weight_q).abs().max().item():.3e}",
        )
        self.assertTrue(torch.equal(small.encoded, large.encoded))


class TestLayoutAndContract(unittest.TestCase):
    def test_encoded_layout_and_dtype(self):
        torch.manual_seed(13)
        k, n = 32, 256

        class TileIdEncoder:
            def __call__(self, tiles):
                indices = torch.arange(tiles.shape[0], dtype=torch.int64)
                indices = indices.reshape(-1, 1).expand(-1, 256)
                return tiles, indices

        result = ldlq(torch.randn(k, n), torch.eye(k), TileIdEncoder(),
                      buf_size_k=32)
        self.assertIsInstance(result, LDLQResult)
        self.assertEqual(result.encoded.shape, (k // 16, n // 16, 256))
        self.assertEqual(result.encoded.dtype, torch.int16)
        self.assertEqual((result.tiles_k, result.tiles_n), (2, 16))
        for row in range(k // 16):
            for tile in range(n // 16):
                self.assertTrue(torch.all(result.encoded[row, tile] == tile))

    def test_zeroed_diagonal_is_equivalent(self):
        """Upstream zeroes ``L``'s diagonal before the loop; it must not matter."""
        torch.manual_seed(18)
        k, n = 64, 128
        weight = torch.randn(k, n)
        _, L = correlated_L(k, seed=19)
        zeroed = L.clone()
        zeroed.diagonal().zero_()
        a = ldlq(weight, L, LinearEncoder(), buf_size_k=32)
        b = ldlq(weight, zeroed, LinearEncoder(), buf_size_k=32)
        self.assertTrue(torch.allclose(a.weight_q, b.weight_q, atol=1e-5))
        self.assertTrue(torch.equal(a.encoded, b.encoded))

    def test_inputs_are_not_mutated(self):
        torch.manual_seed(14)
        k, n = 32, 128
        weight = torch.randn(k, n)
        _, L = correlated_L(k, seed=15)
        weight_copy = weight.clone()
        L_copy = L.clone()
        ldlq(weight, L, GridEncoder(0.5), buf_size_k=32)
        self.assertTrue(torch.equal(weight, weight_copy))
        self.assertTrue(torch.equal(L, L_copy))

    def test_validation_errors(self):
        k, n = 32, 128
        weight = torch.randn(k, n)
        L = torch.eye(k)
        encode = GridEncoder()
        with self.assertRaises(ValueError):
            ldlq(torch.randn(k), L, encode)
        with self.assertRaises(ValueError):
            ldlq(weight.to(torch.float64), L, encode)
        with self.assertRaises(ValueError):
            ldlq(torch.randn(30, n), torch.eye(30), encode)
        with self.assertRaises(ValueError):
            ldlq(weight, torch.eye(k + 16), encode)
        with self.assertRaises(ValueError):
            ldlq(weight, L.to(torch.float64), encode)
        with self.assertRaises(ValueError):
            ldlq(weight, L, encode, buf_size_k=24)
        with self.assertRaises(ValueError):
            ldlq(weight, L, encode, buf_size_k=48)      # 128 % 48 != 0
        with self.assertRaises(ValueError):
            ldlq(torch.randn(48, n), torch.eye(48), encode, buf_size_k=32)

    def test_encoder_output_validation(self):
        k, n = 32, 128
        with self.assertRaises(ValueError):
            ldlq(torch.randn(k, n), torch.eye(k),
                 lambda tiles: (tiles[:-1], torch.zeros(tiles.shape)), 32)


class TestXPUParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.xpu_available = torch.xpu.is_available()
        except Exception:
            cls.xpu_available = False

    @unittest.skipUnless(torch.xpu.is_available(), "torch.xpu not available")
    def test_xpu_matches_reference_encoder(self):
        from exl3_quant.ldlq import make_xpu_tile_encoder

        torch.manual_seed(16)
        k, n = 32, 128
        weight = torch.randn(k, n) * 0.5
        _, L = correlated_L(k, seed=17)

        reference = ldlq(weight, L, make_reference_tile_encoder(K=2), buf_size_k=32)
        xpu = ldlq(weight, L, make_xpu_tile_encoder(K=2), buf_size_k=32)

        self.assertTrue(torch.equal(xpu.weight_q, reference.weight_q))
        self.assertTrue(torch.equal(xpu.encoded, reference.encoded))


if __name__ == "__main__":
    unittest.main()
