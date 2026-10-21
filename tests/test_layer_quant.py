"""Unit tests for layer quantization, scratch arena reuse, and proxy loss metrics.

Tasks: P2-11, P2-12.
"""

from __future__ import annotations

import unittest
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exl3_quant.calibration import TransformerDecoderLayer
from exl3_quant.layer_quant import (
    compute_factorization_residual,
    compute_proxy_loss,
    quantize_decoder_layer,
    quantize_linear_module,
)
from exl3_quant.ldlq import make_reference_tile_encoder, make_xpu_tile_encoder
from exl3_quant.memory_policy import MemoryPolicy
from exl3_quant.xpu_quantizer import CB_MUL1


class TestLayerQuantizationMetrics(unittest.TestCase):
    """P2-12: Measure factorization residuals, reconstruction error, and proxy loss."""

    def test_proxy_loss_formula_matches_matrix_trace(self):
        k, n = 128, 128
        torch.manual_seed(42)
        W = torch.randn(k, n)
        W_hat = W + 0.05 * torch.randn(k, n)

        # Generate a positive-definite Hessian
        X = torch.randn(256, k)
        H = X.T @ X / 256.0
        H.diagonal().add_(0.1)

        proxy_loss, norm_loss = compute_proxy_loss(W, W_hat, H)

        # Compare with explicit trace: tr(E^T H E)
        E = W - W_hat
        explicit_trace = torch.trace(E.T @ H @ E).item()
        self.assertAlmostEqual(proxy_loss, explicit_trace, places=4)
        self.assertGreater(norm_loss, 0.0)

    def test_factorization_residual(self):
        k = 128
        torch.manual_seed(7)
        X = torch.randn(256, k)
        H = X.T @ X / 256.0
        H.diagonal().add_(0.1)

        chol = torch.linalg.cholesky(H)
        res = compute_factorization_residual(H, chol)
        self.assertLess(res, 1e-5)


class TestSingleModuleQuantization(unittest.TestCase):
    """Test quantization of one linear module through regularize -> block_ldl -> ldlq."""

    @unittest.skipUnless(torch.xpu.is_available(), "Intel XPU device required")
    def test_quantize_linear_module_xpu(self):
        device = torch.device("xpu:0")
        k, n = 128, 128
        torch.manual_seed(10)
        W = torch.randn(k, n, device=device)
        X = torch.randn(256, k, device=device)
        H = X.T @ X / 256.0
        H.diagonal().add_(0.05)

        encoder = make_xpu_tile_encoder(K=2, device_index=0)
        res = quantize_linear_module(
            name="test_proj",
            weight=W,
            H=H,
            K=2,
            cb=CB_MUL1,
            tile_encoder=encoder,
            skip_g_scale=True,
            buf_size_k=64,
            block_size=16,
        )

        self.assertEqual(res.encoded.shape, (k // 16, n // 16, 256))
        self.assertEqual(res.w_hat.shape, (k, n))
        self.assertGreater(res.metrics.relative_frobenius_error, 0.0)
        self.assertLess(res.metrics.relative_frobenius_error, 1.0)
        self.assertGreater(res.metrics.proxy_loss, 0.0)
        self.assertLess(res.metrics.factorization_residual, 1e-4)


class TestDecoderLayerQuantization(unittest.TestCase):
    """P2-11: Full layer quantization with single ScratchArena reuse."""

    @unittest.skipUnless(torch.xpu.is_available(), "Intel XPU device required")
    def test_quantize_decoder_layer_xpu(self):
        """P2-11: Full layer quantization on Intel XPU with ScratchArena reuse."""
        device = torch.device("xpu:0")
        # Ensure all projection dimensions are multiples of 128
        k = 256
        d_mlp = 256
        layer = TransformerDecoderLayer(
            hidden_size=k,
            intermediate_size=d_mlp,
            num_heads=2,
            num_kv_heads=1,  # head_dim = 128, num_kv_heads * head_dim = 128
        ).to(device)

        torch.manual_seed(123)
        H_k = torch.eye(k, device=device) * 2.0 + 0.05 * torch.randn(k, k, device=device)
        H_k = H_k.T @ H_k

        H_mlp = torch.eye(d_mlp, device=device) * 2.0 + 0.05 * torch.randn(d_mlp, d_mlp, device=device)
        H_mlp = H_mlp.T @ H_mlp

        hessians = {
            "self_attn.q_proj": H_k,
            "self_attn.k_proj": H_k,
            "self_attn.v_proj": H_k,
            "self_attn.o_proj": H_k,
            "mlp.gate_proj": H_k,
            "mlp.up_proj": H_k,
            "mlp.down_proj": H_mlp,
        }

        report = quantize_decoder_layer(
            layer=layer,
            hessians=hessians,
            K=2,
            cb=CB_MUL1,
            device=device,
            skip_g_scale=True,
            buf_size_k=64,
            block_size=16,
        )

        self.assertEqual(len(report.module_results), 7)
        self.assertGreater(report.total_proxy_loss, 0.0)
        self.assertGreater(report.peak_device_memory_bytes, 0)
        self.assertTrue(report.scratch_arena_reused)

        # Check each projection result
        for name, res in report.module_results.items():
            self.assertEqual(res.encoded.dtype, torch.int16)
            self.assertGreater(res.metrics.proxy_loss, 0.0)


if __name__ == "__main__":
    unittest.main()
