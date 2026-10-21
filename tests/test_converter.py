"""Tests for the bounded Phase 3 conversion orchestrator.

Covers P3-01 (bounded entry point / early rejection of unsupported modes),
P3-02 (upstream calibration propagation + conversion order + unchanged
pass-through), and P3-03 (memory-mapped shard streaming with measured host
RSS), using deterministic synthetic mini-Llama checkpoints under
``tests/fixtures`` — no network, no real model, no CUDA.
"""

from __future__ import annotations

import json
import shutil
import sys
import unittest
import unittest.mock
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from fixtures import (  # noqa: E402
    FIXTURE_GEOMETRY,
    build_fixture_tensors,
    layer_shard_file,
    write_fixture_config,
    write_sharded_model,
)

import exl3_quant.converter as conv  # noqa: E402
from exl3_quant.calibration import (  # noqa: E402
    calibrate_layer,
    make_rotary_embeddings,
)

_TEST_ROOT = Path("/tmp/opencode/exl3_converter_tests")

_SHARDS = 2
_NUM_LAYERS = 2


def _fresh(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _relative_frobenius(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()
    diff = (a - b).norm().item()
    denom = max(b.norm().item(), 1e-12)
    return diff / denom


class TestStreamingIndex(unittest.TestCase):
    """P3-03: lazy mmap index over a HF-style sharded checkpoint."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_dir = _fresh(_TEST_ROOT / "model_sharded")
        cls.source = write_sharded_model(cls.model_dir, num_layers=_NUM_LAYERS,
                                         shards=_SHARDS)
        cls.single_dir = _fresh(_TEST_ROOT / "model_single")
        write_sharded_model(cls.single_dir, num_layers=_NUM_LAYERS, shards=1)

    def test_index_parses_weight_map(self):
        index = conv.ShardedSafetensorsIndex(self.model_dir)
        self.assertTrue(index.is_sharded)
        self.assertEqual(len(index.shards), _SHARDS)
        self.assertEqual(len(index.weight_map), len(self.source))
        # Round-robin layer placement matches the documented assignment.
        for idx in range(_NUM_LAYERS):
            shard = layer_shard_file(idx, _SHARDS)
            self.assertEqual(index.weight_map[f"model.layers.{idx}.self_attn.q_proj.weight"],
                             shard)
        # Untied head / embed / final norm land in the last shard.
        for key in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            self.assertIn(key, index.weight_map)
            self.assertEqual(index.weight_map[key], layer_shard_file(_SHARDS - 1, _SHARDS))

    def test_layer_keys_are_scoped(self):
        index = conv.ShardedSafetensorsIndex(self.model_dir)
        keys0 = index.layer_keys(0)
        self.assertTrue(all(k.startswith("model.layers.0.") for k in keys0))
        self.assertEqual(len(keys0), 9)  # 7 projections + 2 layernorms
        keys1 = index.layer_keys(1)
        self.assertEqual(len(keys1), 9)
        self.assertFalse(set(keys0) & set(keys1))

    def test_unchanged_keys_cover_contract_components(self):
        index = conv.ShardedSafetensorsIndex(self.model_dir)
        unchanged = index.unchanged_keys()
        self.assertIn("model.embed_tokens.weight", unchanged)
        self.assertIn("model.norm.weight", unchanged)
        self.assertIn("lm_head.weight", unchanged)
        self.assertIn("model.layers.0.input_layernorm.weight", unchanged)
        self.assertIn("model.layers.1.post_attention_layernorm.weight", unchanged)
        self.assertNotIn("model.layers.0.self_attn.q_proj.weight", unchanged)

    def test_read_layer_tensors_bit_exact(self):
        index = conv.ShardedSafetensorsIndex(self.model_dir)
        tensors = index.read_layer_tensors(1)
        self.assertEqual(len(tensors), 9)
        for key, value in tensors.items():
            self.assertTrue(torch.equal(value, self.source[key]),
                            f"{key} not bit-exact from mmap stream")
        self.assertNotIn("model.embed_tokens.weight", tensors)

    def test_single_file_model_indexed(self):
        index = conv.ShardedSafetensorsIndex(self.single_dir)
        self.assertFalse(index.is_sharded)
        self.assertEqual(len(index.shards), 1)
        tensors = index.read_layer_tensors(0)
        self.assertEqual(len(tensors), 9)

    def test_streaming_does_not_touch_other_shards(self):
        # Poison the *other* shard with garbage, then re-read layer 0:
        # a streaming index must never consult shard 1 for layer 0 tensors.
        index = conv.ShardedSafetensorsIndex(self.model_dir)
        other = layer_shard_file(1, _SHARDS)
        poisoned = load_file(str(self.model_dir / other))
        for key in poisoned:
            if key.startswith("model.layers.1."):
                poisoned[key] = torch.full_like(poisoned[key], 999.0)
        from safetensors.torch import save_file
        save_file(poisoned, str(self.model_dir / other))
        try:
            tensors = index.read_layer_tensors(0)
            for key, value in tensors.items():
                self.assertTrue(torch.equal(value, self.source[key]),
                                f"{key} was affected by a poisoned different shard")
        finally:
            # Restore the shared fixture so other tests see pristine values.
            pristine = {key: tensor for key, tensor in self.source.items()
                        if index.weight_map[key] == other}
            save_file(pristine, str(self.model_dir / other))

    def test_read_rss_kib_positive(self):
        rss = conv.read_rss_kib()
        self.assertGreater(rss, 0)


class TestConversionConfig(unittest.TestCase):
    """P3-01: early rejection of unsupported modes (no weights touched)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.valid_dir = _fresh(_TEST_ROOT / "valid_model")
        write_fixture_config(cls.valid_dir)
        cls.variant_dirs = {}
        variants: Dict[str, dict] = {
            "moe": {"num_local_experts": 8},
            "vision": {"vision_config": {"depth": 1}},
            "mtp": {"mtp_num_heads": 1},
            "prequantized": {"quantization_config": {"quant_method": "exl3"}},
            "bad_hidden": {"hidden_size": 100},
            "bad_intermediate": {"intermediate_size": 200},
            "bad_heads": {"num_attention_heads": 3, "num_key_value_heads": 2},
            "bad_gqa": {"num_attention_heads": 4, "num_key_value_heads": 3},
            "partial_rotary": {"partial_rotary_factor": 0.5},
            "rope_scaling": {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
            "sliding_window": {"sliding_window": 4096},
            "single_layer_zero": {"num_hidden_layers": 0},
            "odd_head_dim": {"head_dim": 48},
        }
        for name, overrides in variants.items():
            d = _fresh(_TEST_ROOT / f"variant_{name}")
            write_fixture_config(d, overrides=overrides)
            cls.variant_dirs[name] = d

    def _valid_config(self, **overrides) -> conv.ConversionConfig:
        cfg = conv.ConversionConfig(
            model_dir=str(self.valid_dir),
            output_dir=str(_TEST_ROOT / "out_unused"),
        )
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def test_valid_config_produces_geometry(self):
        geometry = self._valid_config().validate()
        self.assertEqual(geometry.architecture, "llama")
        self.assertEqual(geometry.hidden_size, FIXTURE_GEOMETRY["hidden_size"])
        self.assertEqual(geometry.num_layers, _NUM_LAYERS)
        self.assertEqual(geometry.head_dim, FIXTURE_GEOMETRY["head_dim"])
        self.assertEqual(geometry.vocab_size, FIXTURE_GEOMETRY["vocab_size"])
        # Qwen2-family checkpoints resolve through the same window.
        qwen_dir = _fresh(_TEST_ROOT / "variant_qwen")
        write_fixture_config(qwen_dir, overrides={
            "architectures": ["Qwen2ForCausalLM"]})
        geo_q = conv.ConversionConfig(
            model_dir=str(qwen_dir),
            output_dir=str(_TEST_ROOT / "out_unused")).validate()
        self.assertEqual(geo_q.architecture, "qwen2")

    def test_rejects_out_of_range_or_fractional_K(self):
        cases = [
            (5, "K above bounded max"),
            (1, "K below bounded min"),
            (2.5, "fractional K"),
            (True, "bool K"),
        ]
        for value, label in cases:
            with self.subTest(label=label):
                cfg = self._valid_config(K=value)
                with self.assertRaises(conv.UnsupportedModeError):
                    cfg.validate()

    def test_rejects_unsupported_codebook(self):
        with self.assertRaises(conv.UnsupportedModeError):
            self._valid_config(cb="mcg").validate()

    def test_rejects_bad_output_dirs(self):
        with self.assertRaises(conv.UnsupportedModeError):
            self._valid_config(output_dir=self.valid_dir).validate()
        occupied = _fresh(_TEST_ROOT / "out_occupied")
        (occupied / "stray.txt").write_text("not empty")
        with self.assertRaises(conv.UnsupportedModeError):
            self._valid_config(output_dir=str(occupied)).validate()

    def test_rejects_missing_model_dir(self):
        with self.assertRaises(conv.UnsupportedModeError):
            conv.ConversionConfig(
                model_dir=str(_TEST_ROOT / "does_not_exist"),
                output_dir=str(_TEST_ROOT / "out_unused"),
            ).validate()

    def test_rejects_variant_configs(self):
        cases = [
            ("moe", "MoE expert config"),
            ("vision", "vision submodule"),
            ("mtp", "MTP head"),
            ("prequantized", "already-quantized checkpoint"),
            ("bad_hidden", "hidden not divisible by 128"),
            ("bad_intermediate", "intermediate not divisible by 128"),
            ("bad_heads", "q dim not divisible by 128"),
            ("bad_gqa", "GQA divisibility violated"),
            ("partial_rotary", "partial rotary"),
            ("rope_scaling", "scaled rotary"),
            ("sliding_window", "sliding window attention"),
            ("single_layer_zero", "no layers"),
            ("odd_head_dim", "head_dim != hidden // heads"),
        ]
        for kind, label in cases:
            with self.subTest(label=label):
                cfg = conv.ConversionConfig(
                    model_dir=str(self.variant_dirs[kind]),
                    output_dir=str(_TEST_ROOT / "out_unused"),
                )
                with self.assertRaises(conv.UnsupportedModeError):
                    cfg.validate()

    def test_rejects_unsupported_calibration_bounds(self):
        for field, value, label in [
            ("seq_len", 100, "seq_len not a multiple of 16"),
            ("seq_len", 8, "seq_len below minimum"),
            ("damping_lambda", 0.0, "non-positive damping"),
            ("zero_floor", -1e-9, "non-positive zero floor"),
            ("num_calibration_rows", 0, "no calibration rows"),
            ("batch_size", 0, "no batch"),
        ]:
            with self.subTest(label=label):
                with self.assertRaises(conv.UnsupportedModeError):
                    self._valid_config(**{field: value}).validate()


@unittest.skipUnless(torch.xpu.is_available(), "Intel XPU device required")
class TestConverterPipeline(unittest.TestCase):
    """P3-02: full bounded conversion of the synthetic 2-layer model.

    One shared conversion run (with a recorder patched around
    ``quantize_decoder_layer``) feeds every assertion, so the expensive XPU
    path executes exactly once for the whole class.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("xpu:0")
        cls.model_dir = _fresh(_TEST_ROOT / "model_pipe")
        cls.source = write_sharded_model(cls.model_dir, num_layers=_NUM_LAYERS,
                                         shards=_SHARDS)
        cls.out_dir = _fresh(_TEST_ROOT / "out_pipe")
        cls.config = conv.ConversionConfig(
            model_dir=str(cls.model_dir),
            output_dir=str(cls.out_dir),
            K=2,
            num_calibration_rows=2,
            batch_size=1,
            seq_len=128,
            calibration_seed=0,
            damping_lambda=0.01,
            zero_floor=1e-7,
            skip_g_scale=True,   # GSS wiring itself is covered by P2 tests
        )
        cls.captured: List[Dict[str, torch.Tensor]] = []
        cls.module_reports = []
        original = conv.quantize_decoder_layer
        geometry = cls.config.validate()

        def recorder(layer, hessians, **kwargs):
            result = original(layer, hessians, **kwargs)
            cls.captured.append(
                {k: v.detach().cpu().clone() for k, v in hessians.items()})
            cls.module_reports.append(result)
            return result

        with unittest.mock.patch.object(conv, "quantize_decoder_layer", recorder):
            cls.report = conv.convert_model(cls.config)

        cls.index = conv.ShardedSafetensorsIndex(cls.model_dir)
        cls.embed = torch.nn.Embedding(geometry.vocab_size, geometry.hidden_size)
        cls.embed.weight.data.copy_(cls.source["model.embed_tokens.weight"])
        cls.embed.to(device=cls.device, dtype=torch.float16)
        generator = torch.Generator().manual_seed(cls.config.calibration_seed)
        cls.rows = [
            torch.randint(0, geometry.vocab_size, (cls.config.batch_size,
                                                  cls.config.seq_len),
                          generator=generator)
            for _ in range(cls.config.num_calibration_rows)
        ]
        cls.pos_emb = make_rotary_embeddings(seq_len=cls.config.seq_len,
                                             head_dim=geometry.head_dim,
                                             device=cls.device,
                                             dtype=torch.float16)

    @classmethod
    def tearDownClass(cls) -> None:
        if torch.xpu.is_available():
            torch.xpu.synchronize(0)
            torch.xpu.empty_cache()

    def _layer_states(self):
        """Pre-layer running state exactly as the driver builds it."""
        with torch.no_grad():
            return [self.embed(row.to(self.device)) for row in self.rows]

    def _fresh_layer(self, layer_idx: int):
        tensors = self.index.read_layer_tensors(layer_idx)
        return conv._build_layer(self.report.geometry, tensors, layer_idx,
                                 self.device)

    # -- conversion order ---------------------------------------------------
    def test_layers_converted_in_ascending_order(self):
        self.assertEqual([e.layer_index for e in self.report.layers],
                         list(range(_NUM_LAYERS)))

    def test_module_keys_match_upstream_projection_order(self):
        expected = list(conv._QUANTIZED_MODULES)
        for entry in self.report.layers:
            self.assertEqual(entry.module_keys, expected)

    # -- capture is performed on the running (propagated) state --------------
    def test_layer0_capture_matches_independent_calibration(self):
        """Driver capture for layer 0 == independent capture on embed states."""
        layer0 = self._fresh_layer(0)
        hidden = self._layer_states()
        h_ref, _ = calibrate_layer(
            layer=layer0, calibration_data=hidden, device=self.device,
            damping_lambda=self.config.damping_lambda,
            zero_floor=self.config.zero_floor)
        for key, ref in h_ref.items():
            got = self.captured[0][key]
            self.assertEqual(got.shape, ref.shape)
            self.assertLess(_relative_frobenius(got, ref), 5e-3,
                            f"layer-0 capture deviates for {key}")

    def test_propagation_uses_quantized_numerics(self):
        """Layer 1's Hessian comes from layer 0's *quantized* output.

        A control capture is run against layer-1 fed by the *unquantized*
        layer-0 outputs; the driver's recorded capture must differ from that
        control, i.e. quantization error propagated forward (upstream
        semantics), rather than the raw checkpoint numerics.
        """
        layer1 = self._fresh_layer(1)
        layer0 = self._fresh_layer(0)   # original, unquantized weights
        with torch.no_grad():
            hidden_orig = [layer0(x, position_embeddings=self.pos_emb)
                           for x in self._layer_states()]
        h_orig, _ = calibrate_layer(
            layer=layer1, calibration_data=hidden_orig, device=self.device,
            damping_lambda=self.config.damping_lambda,
            zero_floor=self.config.zero_floor)
        key = "self_attn.q_proj"
        rel = _relative_frobenius(self.captured[1][key], h_orig[key])
        self.assertGreater(rel, 5e-4,
                           "layer-1 Hessian identical to unquantized control; "
                           "quantized numerics were NOT propagated")

    def test_layer1_capture_derived_from_quantized_layer0_forward(self):
        """End-to-end equivalence: re-running layer 0 with quantized weights
        reloaded and then re-capturing layer 1 must match the driver run."""
        reports = self.module_reports
        layer0 = self._fresh_layer(0)
        conv._reload_quantized(layer0, reports[0].module_results)
        layer1 = self._fresh_layer(1)
        with torch.no_grad():
            hidden_quant = [layer0(x, position_embeddings=self.pos_emb)
                            for x in self._layer_states()]
        h_quant, _ = calibrate_layer(
            layer=layer1, calibration_data=hidden_quant, device=self.device,
            damping_lambda=self.config.damping_lambda,
            zero_floor=self.config.zero_floor)
        for key, ref in h_quant.items():
            got = self.captured[1][key]
            self.assertLess(_relative_frobenius(got, ref), 5e-3,
                            f"layer-1 capture not reproducible for {key}")

    # -- quantized numerics swap ---------------------------------------------
    def test_reload_swaps_transposed_w_hat_into_linear(self):
        """``w_hat`` is (in, out); nn.Linear expects (out, in) — transpose."""
        layer = conv.TransformerDecoderLayer(hidden_size=128,
                                             intermediate_size=256,
                                             num_heads=2, num_kv_heads=2)
        layer.to(torch.float16)
        out_features = layer.self_attn.q_proj.weight.shape[0]
        w_hat = torch.arange(128 * out_features, dtype=torch.float32) \
            .reshape(128, out_features) + 1.0
        stub = type("S", (), {"w_hat": w_hat})()
        conv._reload_quantized(layer, {"self_attn.q_proj": stub})
        expected = w_hat.T.to(torch.float16)
        self.assertTrue(torch.equal(layer.self_attn.q_proj.weight.data,
                                    expected))

    def test_quantized_weights_differ_from_original(self):
        """The reload step really replaced weights (sanity vs. no-op)."""
        original = self.source["model.layers.0.self_attn.q_proj.weight"]
        w_hat = self.module_reports[0].module_results["self_attn.q_proj"].w_hat
        rel = _relative_frobenius(w_hat.cpu(), original.T.float())
        self.assertGreater(rel, 1e-4)

    # -- manifest + artifacts -------------------------------------------------
    def test_report_metrics_finite(self):
        for entry in self.report.layers:
            self.assertTrue(all(
                v == v and abs(v) != float("inf")
                for v in (entry.proxy_loss, entry.normalized_proxy_loss,
                          entry.mean_mse, entry.max_factorization_residual)))
            self.assertEqual(len(entry.g_scales), 7)
            self.assertTrue(all(g > 0 for g in entry.g_scales.values()))

    def test_metrics_consistent_with_module_reports(self):
        entry = self.report.layers[0]
        rep = self.module_reports[0]
        losses = [r.metrics.proxy_loss for r in rep.module_results.values()]
        self.assertAlmostEqual(entry.proxy_loss,
                               sum(losses) / len(losses), places=4)

    def test_manifest_contents(self):
        manifest = json.loads((self.out_dir / "quantization_config.json").read_text())
        self.assertEqual(manifest["quant_method"], "exl3")
        self.assertEqual(manifest["bits"], 2)
        self.assertEqual(manifest["codebook"], "mul1")
        self.assertEqual(len(manifest["layers"]), _NUM_LAYERS)
        self.assertEqual(manifest["calibration"]["seed"], 0)
        self.assertTrue(manifest["calibration"]["skip_g_scale"])
        for layer_manifest in manifest["layers"]:
            self.assertEqual(len(layer_manifest["modules"]), 7)
            self.assertEqual(layer_manifest["quantized_tensors"], 30)
        # unchanged pass-through recorded per the format contract
        storage = manifest["tensor_storage"]
        self.assertEqual(storage["model.embed_tokens.weight"],
                         {"quant_format": "fp16", "unchanged": True})
        self.assertTrue(storage["lm_head.weight"]["unchanged"])
        self.assertEqual(storage["model.layers.0.self_attn.q_proj"]["quant_format"],
                         "exl3")

    def test_per_layer_artifacts_match_upstream_qtensor_layout(self):
        for idx in range(_NUM_LAYERS):
            path = self.out_dir / "qtensors" / f"model.layers.{idx}.safetensors"
            self.assertTrue(path.is_file())
            tensors = load_file(str(path))
            self.assertEqual(len(tensors), 30)  # 7 modules x 4 + 2 norms
            for module in conv._QUANTIZED_MODULES:
                base = f"model.layers.{idx}.{module}"
                self.assertIn(f"{base}.trellis", tensors)
                self.assertIn(f"{base}.suh", tensors)
                self.assertIn(f"{base}.svh", tensors)
                self.assertIn(f"{base}.mul1", tensors)
                k = tensors[f"{base}.suh"].shape[0]
                n = tensors[f"{base}.svh"].shape[0]
                self.assertEqual(tensors[f"{base}.trellis"].ndim, 3)
                self.assertEqual(tensors[f"{base}.suh"].dtype, torch.float16)
                self.assertEqual(tensors[f"{base}.mul1"].dtype, torch.int32)

    def test_untouched_components_bit_exact_pass_through(self):
        path = self.out_dir / "qtensors" / "untouched.safetensors"
        self.assertTrue(path.is_file())
        untouched = load_file(str(path))
        for key, value in untouched.items():
            self.assertTrue(torch.equal(value, self.source[key]),
                            f"unchanged tensor {key} was modified")
        self.assertEqual(set(untouched),
                         set(self.index.unchanged_keys()))

    # -- host RSS is measured around each layer (P3-03) -----------------------
    def test_rss_measurements_recorded_per_layer(self):
        for entry in self.report.layers:
            self.assertGreater(entry.rss_kib_before, 0)
            self.assertGreaterEqual(entry.rss_kib_after, 0)
        self.assertGreater(self.report.total_elapsed_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
