"""Unit tests for EXL3 layer packing, safetensors serialization, and reload verification.

Tasks: P2-13, P2-14, P2-15, P2-16.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exl3_quant.layer_quant import (
    quantize_linear_module,
)
from exl3_quant.ldlq import make_reference_tile_encoder, make_xpu_tile_encoder
from exl3_quant.serialization import (
    build_layer_tensors,
    decode_weight_from_storage,
    export_layer_safetensors,
    load_and_verify_layer_safetensors,
    pack_module_trellis,
    unpack_module_trellis,
)
from exl3_quant.trellis_format import CB_MUL1


class TestTrellisPackingRoundTrip(unittest.TestCase):
    """P2-13 & P2-15: Pack/unpack state-history round trips."""

    def test_trellis_pack_unpack_shape_and_dtype(self):
        tiles_k, tiles_n = 4, 8
        K = 2
        # Valid states in 0..65535
        states = torch.randint(0, 65536, (tiles_k, tiles_n, 256), dtype=torch.int64)

        packed = pack_module_trellis(states, K=K)
        self.assertEqual(packed.shape, (tiles_k, tiles_n, 16 * K))
        self.assertEqual(packed.dtype, torch.int16)

        unpacked = unpack_module_trellis(packed, K=K)
        self.assertEqual(unpacked.shape, (tiles_k, tiles_n, 256))

        # State-history property: low K bits must match exactly
        mask = (1 << K) - 1
        self.assertTrue(
            torch.equal(
                (unpacked.long() & mask),
                (states & mask),
            )
        )


class TestSafetensorsExportAndReload(unittest.TestCase):
    """P2-14 & P2-16: Validate keys, shapes, dtypes, export, reload, and weight reconstruction."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_layer_export_and_reload_verification(self):
        k, n = 128, 128
        torch.manual_seed(42)
        device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")
        W = torch.randn(k, n, device=device)
        X = torch.randn(256, k, device=device)
        H = (X.T @ X / 256.0) + torch.eye(k, device=device) * 0.1

        if device.type == "xpu":
            encoder = make_xpu_tile_encoder(K=2, device_index=0)
        else:
            # Must match the K passed to quantize_linear_module below: the
            # container is packed at the requested bit width, so a mismatched
            # encoder would silently corrupt the checkpoint.
            encoder = make_reference_tile_encoder(K=2)

        q_res = quantize_linear_module(
            name="q_proj",
            weight=W,
            H=H,
            K=2,
            cb=CB_MUL1,
            tile_encoder=encoder,
            skip_g_scale=True,
            buf_size_k=64,
            block_size=16,
        )

        module_results = {"self_attn.q_proj": q_res}
        norm_weights = {"input_layernorm.weight": torch.ones(k, dtype=torch.float16)}

        # Build tensors and check schema
        tensors, qc_storage = build_layer_tensors(
            module_results=module_results,
            layer_idx=0,
            prefix="model.layers",
            norm_weights=norm_weights,
        )

        # Check expected keys
        self.assertIn("model.layers.0.self_attn.q_proj.trellis", tensors)
        self.assertIn("model.layers.0.self_attn.q_proj.suh", tensors)
        self.assertIn("model.layers.0.self_attn.q_proj.svh", tensors)
        self.assertIn("model.layers.0.self_attn.q_proj.mul1", tensors)
        self.assertIn("model.layers.0.input_layernorm.weight", tensors)

        # Check dtypes and shapes
        self.assertEqual(tensors["model.layers.0.self_attn.q_proj.trellis"].dtype, torch.int16)
        self.assertEqual(tensors["model.layers.0.self_attn.q_proj.trellis"].shape, (k // 16, n // 16, 32))
        self.assertEqual(tensors["model.layers.0.self_attn.q_proj.suh"].dtype, torch.float16)
        self.assertEqual(tensors["model.layers.0.self_attn.q_proj.svh"].dtype, torch.float16)

        # Export to safetensors
        sf_file, qc_file = export_layer_safetensors(
            module_results=module_results,
            output_dir=self.temp_dir,
            filename="test_layer.safetensors",
            layer_idx=0,
            prefix="model.layers",
            norm_weights=norm_weights,
        )

        self.assertTrue(sf_file.exists())
        self.assertTrue(qc_file.exists())

        # Reload and verify reconstructed weight matches w_hat
        reconstructed = load_and_verify_layer_safetensors(
            safetensors_path=sf_file,
            expected_results=module_results,
        )

        self.assertIn("model.layers.0.self_attn.q_proj", reconstructed)
        reconstructed_w = reconstructed["model.layers.0.self_attn.q_proj"]
        self.assertEqual(reconstructed_w.shape, (k, n))

        # Check difference against q_res.w_hat (difference <= 5e-3 due to float16 storage of suh/svh)
        diff = (reconstructed_w.float() - q_res.w_hat.cpu().float()).abs().max().item()
        self.assertLess(diff, 5e-3)


class TestEncoderBitWidthGuard(unittest.TestCase):
    """The container is packed at the requested K, so a mismatched encoder
    must fail loudly rather than emit a silently corrupt checkpoint."""

    def test_encoder_adapters_expose_their_bit_width(self):
        self.assertEqual(make_reference_tile_encoder(K=3).K, 3)

    def test_mismatched_encoder_k_is_rejected(self):
        k = n = 128
        torch.manual_seed(1)
        W = torch.randn(k, n)
        X = torch.randn(64, k)
        H = (X.T @ X / 64.0) + torch.eye(k) * 0.1

        with self.assertRaises(ValueError) as ctx:
            quantize_linear_module(
                name="q_proj",
                weight=W,
                H=H,
                K=2,
                cb=CB_MUL1,
                tile_encoder=make_reference_tile_encoder(K=4),
                skip_g_scale=True,
                buf_size_k=64,
                block_size=16,
            )
        self.assertIn("does not match the requested K", str(ctx.exception))


class TestSafetensorsMalformedRejection(unittest.TestCase):
    """P2-15: Malformed, truncated, or invalid file rejection."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_file_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            load_and_verify_layer_safetensors(Path(self.temp_dir) / "nonexistent.safetensors")

    def test_truncated_safetensors_raises_error(self):
        bad_file = Path(self.temp_dir) / "truncated.safetensors"
        with open(bad_file, "wb") as f:
            f.write(b"\x00\x00\x00\x00badheader")

        with self.assertRaises(Exception):
            load_and_verify_layer_safetensors(bad_file)


if __name__ == "__main__":
    unittest.main()
