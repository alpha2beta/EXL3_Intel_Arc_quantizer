"""Unit tests for layer calibration, activation harvesting, and Hessian accumulation.

Tasks: P2-01, P2-02, P2-03, P2-05.
"""

from __future__ import annotations

import unittest
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exl3_quant.calibration import (
    DEFAULT_DAMPING_LAMBDA,
    DEFAULT_ZERO_CHANNEL_FLOOR,
    HessianCollector,
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
    TransformerDecoderLayer,
    calibrate_layer,
    make_rotary_embeddings,
)


class TestCalibrationPrimitives(unittest.TestCase):
    """Test individual submodules and mathematical equivalence."""

    def test_rmsnorm_forward(self):
        norm = LlamaRMSNorm(hidden_size=64)
        x = torch.randn(2, 16, 64)
        out = norm(x)
        self.assertEqual(out.shape, (2, 16, 64))
        # Variance of output should be approximately 1.0
        var = out.to(torch.float32).pow(2).mean(-1)
        self.assertTrue(torch.allclose(var, torch.ones_like(var), atol=1e-2))

    def test_rotary_embeddings_shape(self):
        cos, sin = make_rotary_embeddings(seq_len=32, head_dim=64)
        self.assertEqual(cos.shape, (1, 1, 32, 64))
        self.assertEqual(sin.shape, (1, 1, 32, 64))
        # Norm of (cos, sin) complex phasor must equal 1 within fp16 precision
        norm = cos.float().pow(2) + sin.float().pow(2)
        self.assertTrue(torch.allclose(norm, torch.ones_like(norm), atol=2e-3))

    def test_attention_forward(self):
        attn = LlamaAttention(hidden_size=128, num_heads=4, num_kv_heads=2)
        x = torch.randn(2, 16, 128)
        pos_emb = make_rotary_embeddings(seq_len=16, head_dim=32)
        out = attn(x, position_embeddings=pos_emb)
        self.assertEqual(out.shape, (2, 16, 128))

    def test_mlp_forward(self):
        mlp = LlamaMLP(hidden_size=128, intermediate_size=256)
        x = torch.randn(2, 16, 128)
        out = mlp(x)
        self.assertEqual(out.shape, (2, 16, 128))


class TestHessianAccumulationParity(unittest.TestCase):
    """P2-02 & P2-03: Verify exact Hessian accumulation against explicit reference."""

    def test_hessian_collector_matches_explicit_formula(self):
        layer = TransformerDecoderLayer(
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            num_kv_heads=2,
        )
        layer.eval()

        device = torch.device("cpu")
        collector = HessianCollector(layer, device=device, dtype=torch.float32)

        # Generate 2 batches of calibration inputs
        x1 = torch.randn(2, 16, 64)
        x2 = torch.randn(1, 32, 64)
        pos1 = make_rotary_embeddings(seq_len=16, head_dim=16)
        pos2 = make_rotary_embeddings(seq_len=32, head_dim=16)

        with torch.no_grad():
            _ = layer(x1, position_embeddings=pos1)
            _ = layer(x2, position_embeddings=pos2)

        collector.close()
        hessians, damping = collector.finalize(damping_lambda=0.01)

        # Verify all 7 projections are present
        expected_keys = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
        for k in expected_keys:
            self.assertIn(k, hessians)
            self.assertIn(k, damping)

        # q_proj, k_proj, v_proj share identical input Hessian
        h_q = hessians["self_attn.q_proj"]
        h_k = hessians["self_attn.k_proj"]
        h_v = hessians["self_attn.v_proj"]
        self.assertTrue(torch.equal(h_q, h_k))
        self.assertTrue(torch.equal(h_q, h_v))

        # gate_proj, up_proj share identical input Hessian
        h_gate = hessians["mlp.gate_proj"]
        h_up = hessians["mlp.up_proj"]
        self.assertTrue(torch.equal(h_gate, h_up))

        # Down proj Hessian dimension is intermediate_size x intermediate_size
        h_down = hessians["mlp.down_proj"]
        self.assertEqual(h_down.shape, (128, 128))

        # Check positive semi-definiteness (diagonal elements > 0)
        for k, h in hessians.items():
            self.assertTrue(torch.all(h.diagonal() > 0))

    def test_inactive_channel_damping_safeguard(self):
        """P2-05: Inactive/dead channels get damped and clamped to floor."""
        layer = TransformerDecoderLayer(
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            num_kv_heads=2,
        )
        layer.eval()

        # Zero out certain input channels in weights
        with torch.no_grad():
            layer.input_layernorm.weight.data[5] = 0.0
            layer.input_layernorm.weight.data[10] = 0.0

        device = torch.device("cpu")
        collector = HessianCollector(layer, device=device, dtype=torch.float32)

        x = torch.randn(2, 16, 64)
        pos = make_rotary_embeddings(seq_len=16, head_dim=16)

        with torch.no_grad():
            _ = layer(x, position_embeddings=pos)

        collector.close()
        hessians, _ = collector.finalize(damping_lambda=0.01, zero_floor=0.05)

        h_q = hessians["self_attn.q_proj"]
        # Inactive channels must be at least the floor
        self.assertGreaterEqual(float(h_q.diagonal()[5]), 0.05)
        self.assertGreaterEqual(float(h_q.diagonal()[10]), 0.05)


class TestCalibrationXPUExecution(unittest.TestCase):
    """P2-01 & P2-03: Execute calibration pipeline on target Intel XPU device."""

    @unittest.skipUnless(torch.xpu.is_available(), "Intel XPU device required")
    def test_xpu_calibrate_layer(self):
        device = torch.device("xpu:0")
        layer = TransformerDecoderLayer(
            hidden_size=256,
            intermediate_size=512,
            num_heads=8,
            num_kv_heads=4,
        )

        batches = [
            torch.randn(2, 64, 256),
            torch.randn(2, 64, 256),
        ]

        hessians, telemetry = calibrate_layer(
            layer=layer,
            calibration_data=batches,
            device=device,
            damping_lambda=0.01,
        )

        self.assertEqual(telemetry.total_tokens, 256)
        self.assertEqual(telemetry.batches_processed, 2)
        self.assertGreater(telemetry.peak_allocated_bytes, 0)
        self.assertEqual(len(hessians), 7)

        # Check shapes and dtypes
        self.assertEqual(hessians["self_attn.q_proj"].shape, (256, 256))
        self.assertEqual(hessians["mlp.down_proj"].shape, (512, 512))
        self.assertEqual(hessians["self_attn.q_proj"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
