"""Tests for the P4-12/P4-13 hybrid Qwen3.5 text-tower extension.

Covers:

* **P4-12** — hybrid linear-attention support: exhaustive quantized-module
  tables per layer kind, the documented bit-exact pass-through boundary
  (``in_proj_a``/``in_proj_b`` fail the 128-wide Hadamard/tiling contract),
  shared-input Hessian grouping for GatedDeltaNet layers, and full
  conversion runs of both layer kinds through the reference
  (``transformers.models.qwen3_5``) forward.
* **P4-13** — multimodal text-tower-only semantics: opt-in validator mode,
  bit-exact pass-through of vision/MTP/text-tower tensors (source dtype
  preserved), manifest provenance (multimodal token ids, mrope/partial
  rotary, reference-forward environment), and resume behavior.

Everything runs on deterministic synthetic checkpoints (``tests/fixtures``)
— no network, no real model download.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from fixtures import (  # noqa: E402
    HYBRID_FIXTURE_GEOMETRY,
    build_hybrid_fixture_tensors,
    write_qwen35_hybrid_model,
)

import exl3_quant.converter as conv  # noqa: E402
from exl3_quant import hybrid  # noqa: E402
from exl3_quant.calibration import (  # noqa: E402
    calibrate_layer,
    enumerate_linear_modules,
    hessian_groups,
)
# Declared-target config replicas (default-mode rejection guards live in
# test_converter.py; the opt-in acceptance assertions live here).
from test_converter import (  # noqa: E402
    _MIMO_V2_6_CONFIG,
    _QWEN38_9B_DISTILL_CONFIG,
)

_TEST_ROOT = Path("/tmp/opencode/exl3_hybrid_tests")

_HYBRID_MODE = "qwen3_5_text_tower"

# The hybrid text-tower reference needs transformers with Qwen3_5 support
# (>= 5.17). exl3_quant.hybrid sets Qwen3_5DecoderLayer to None when that
# import fails, and require_support() then raises HybridSupportUnavailable.
HYBRID_AVAILABLE = getattr(hybrid, "Qwen3_5DecoderLayer", None) is not None
_skip_no_hybrid = unittest.skipUnless(
    HYBRID_AVAILABLE,
    "transformers with Qwen3_5 support (>= 5.17) required",
)


def _write_config(model_dir: Path, raw: dict) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "config.json", "w") as f:
        json.dump(raw, f, indent=2)
    return model_dir


class TestHybridValidator(unittest.TestCase):
    """P3-01-style early rejection/acceptance for the opt-in hybrid mode."""

    def setUp(self) -> None:
        self.root = _TEST_ROOT / "validator"
        shutil.rmtree(self.root, ignore_errors=True)
        self.model_dir = self.root / "hybrid"
        write_qwen35_hybrid_model(self.model_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_dense_mode_still_rejects_hybrid_fixture(self) -> None:
        with self.assertRaises(conv.UnsupportedModeError):
            conv.load_model_geometry(self.model_dir)
        with self.assertRaises(conv.UnsupportedModeError):
            conv.ConversionConfig(
                model_dir=str(self.model_dir),
                output_dir=str(self.root / "out"),
            ).validate()

    def test_optin_accepts_hybrid_fixture_geometry(self) -> None:
        geo = conv.load_model_geometry(self.model_dir, mode=_HYBRID_MODE)
        self.assertEqual(geo.architecture, "qwen3_5")
        self.assertEqual(geo.architecture_mode, _HYBRID_MODE)
        self.assertTrue(geo.is_hybrid)
        self.assertEqual(geo.key_prefix, "model.language_model.layers")
        self.assertEqual(geo.text_prefix, "model.language_model")
        self.assertEqual(geo.hidden_size, 256)
        self.assertEqual(geo.intermediate_size, 512)
        self.assertEqual(geo.num_layers, 3)
        self.assertEqual(
            geo.layer_types,
            ["linear_attention", "linear_attention", "full_attention"])
        self.assertTrue(geo.has_vision)
        self.assertTrue(geo.has_mtp)
        self.assertFalse(geo.tie_word_embeddings)
        self.assertEqual(geo.embed_key,
                         "model.language_model.embed_tokens.weight")
        self.assertEqual(geo.final_norm_key, "model.language_model.norm.weight")
        self.assertEqual(geo.lm_head_key, "lm_head.weight")
        self.assertEqual(
            geo.hybrid_meta["multimodal"],
            {"image_token_id": 999, "video_token_id": 1000,
             "vision_start_token_id": 997, "vision_end_token_id": 998})
        self.assertEqual(geo.hybrid_meta["rope"]["partial_rotary_factor"], 0.25)
        self.assertEqual(geo.hybrid_meta["rope"]["mrope_section"], [6, 5, 5])

    def test_optin_accepts_both_declared_target_replicas(self) -> None:
        for name, raw in (("qwen38_9b", _QWEN38_9B_DISTILL_CONFIG),
                          ("mimo_v2_6", _MIMO_V2_6_CONFIG)):
            with self.subTest(replica=name):
                d = self.root / f"cfg_{name}"
                _write_config(d, copy.deepcopy(raw))
                geo = conv.load_model_geometry(d, mode=_HYBRID_MODE)
                self.assertEqual(geo.num_layers, 32)
                self.assertEqual(geo.hidden_size, 4096)
                self.assertEqual(geo.vocab_size, 248320)
                self.assertEqual(geo.layer_types[3], "full_attention")
                self.assertEqual(
                    geo.quantized_modules(0)[0], "linear_attn.in_proj_qkv")
                self.assertEqual(
                    geo.quantized_modules(3)[0], "self_attn.q_proj")
                self.assertTrue(geo.has_vision)
                self.assertTrue(geo.has_mtp)
                # 3:1 hybrid pattern of the declared target
                self.assertEqual(
                    geo.layer_types.count("linear_attention"), 24)
                self.assertEqual(geo.layer_types.count("full_attention"), 8)

    def test_rejects_unknown_architecture_mode(self) -> None:
        cfg = conv.ConversionConfig(
            model_dir=str(self.model_dir),
            output_dir=str(self.root / "out"),
            architecture_mode="dense_moe_hybrid",
        )
        with self.assertRaises(conv.UnsupportedModeError):
            cfg.validate()

    def test_rejects_missing_text_config(self) -> None:
        raw = copy.deepcopy(HYBRID_FIXTURE_GEOMETRY)
        raw.pop("text_config")
        d = _write_config(self.root / "no_text", raw)
        with self.assertRaises(conv.UnsupportedModeError):
            conv.load_model_geometry(d, mode=_HYBRID_MODE)

    def test_rejects_moe_text_config(self) -> None:
        raw = copy.deepcopy(HYBRID_FIXTURE_GEOMETRY)
        raw["text_config"]["num_local_experts"] = 4
        d = _write_config(self.root / "moe", raw)
        with self.assertRaises(conv.UnsupportedModeError) as ctx:
            conv.load_model_geometry(d, mode=_HYBRID_MODE)
        self.assertIn("MoE", str(ctx.exception))

    def test_rejects_unknown_layer_type(self) -> None:
        raw = copy.deepcopy(HYBRID_FIXTURE_GEOMETRY)
        raw["text_config"]["layer_types"] = [
            "linear_attention", "sliding_attention", "full_attention"]
        d = _write_config(self.root / "bad_layer_type", raw)
        with self.assertRaises(conv.UnsupportedModeError) as ctx:
            conv.load_model_geometry(d, mode=_HYBRID_MODE)
        self.assertIn("sliding_attention", str(ctx.exception))

    def test_rejects_layer_types_length_mismatch(self) -> None:
        raw = copy.deepcopy(HYBRID_FIXTURE_GEOMETRY)
        raw["text_config"]["layer_types"] = ["full_attention"]
        d = _write_config(self.root / "short_types", raw)
        with self.assertRaises(conv.UnsupportedModeError):
            conv.load_model_geometry(d, mode=_HYBRID_MODE)

    def test_rejects_widths_outside_hadamard_block(self) -> None:
        raw = copy.deepcopy(HYBRID_FIXTURE_GEOMETRY)
        # 8 value heads x 24 head_dim = 192, not divisible by 128
        raw["text_config"]["linear_num_value_heads"] = 8
        raw["text_config"]["linear_value_head_dim"] = 24
        d = _write_config(self.root / "narrow", raw)
        with self.assertRaises(conv.UnsupportedModeError) as ctx:
            conv.load_model_geometry(d, mode=_HYBRID_MODE)
        self.assertIn("Hadamard", str(ctx.exception))

    def test_layer_types_derived_from_interval_when_absent(self) -> None:
        raw = copy.deepcopy(HYBRID_FIXTURE_GEOMETRY)
        raw["text_config"].pop("layer_types")
        raw["text_config"].pop("full_attention_interval", None)
        d = _write_config(self.root / "interval", raw)
        geo = conv.load_model_geometry(d, mode=_HYBRID_MODE)
        # Mirrors Qwen3_5TextConfig.__post_init__: layer i is full-attention
        # iff (i + 1) % 4 == 0, so a 3-layer model has none.
        self.assertEqual(
            geo.layer_types,
            ["linear_attention", "linear_attention", "linear_attention"])
        raw["text_config"]["full_attention_interval"] = 2
        d = _write_config(self.root / "interval2", raw)
        geo = conv.load_model_geometry(d, mode=_HYBRID_MODE)
        self.assertEqual(
            geo.layer_types,
            ["linear_attention", "full_attention", "linear_attention"])

    def test_legacy_layer_type_names_are_remapped(self) -> None:
        raw = copy.deepcopy(HYBRID_FIXTURE_GEOMETRY)
        raw["text_config"]["layer_types"] = [
            "mamba", "mamba", "attention"]
        d = _write_config(self.root / "legacy", raw)
        geo = conv.load_model_geometry(d, mode=_HYBRID_MODE)
        self.assertEqual(
            geo.layer_types,
            ["linear_attention", "linear_attention", "full_attention"])


@_skip_no_hybrid
class TestHybridModuleTables(unittest.TestCase):
    """P4-12 decision evidence: which projections are quantized vs passed
    through, and why (encoder divisibility constraints)."""

    def test_tables_are_disjoint_and_exhaustive(self) -> None:
        for layer_type, quantized in hybrid.QUANTIZED_MODULES.items():
            with self.subTest(layer_type=layer_type):
                passed = set(hybrid.PASS_THROUGH_MODULE_NAMES[layer_type])
                self.assertTrue(passed.isdisjoint(quantized))
                self.assertEqual(
                    set(hybrid.quantized_module_names(layer_type)),
                    set(quantized))

    def test_pass_through_covers_non_quantized_parameters(self) -> None:
        """Every reference-layer parameter is either quantized, a layer norm,
        or in the pass-through table — nothing can be silently dropped."""
        raw = HYBRID_FIXTURE_GEOMETRY["text_config"]
        text_config = hybrid.make_text_config(raw)
        for layer_type in hybrid.SUPPORTED_LAYER_TYPES:
            layer_idx = raw["layer_types"].index(layer_type)
            names = {name for name, _ in hybrid.Qwen3_5DecoderLayer(
                text_config, layer_idx).named_parameters()}
            quantized = {m + ".weight"
                         for m in hybrid.QUANTIZED_MODULES[layer_type]}
            norms = {"input_layernorm.weight",
                     "post_attention_layernorm.weight"}
            remainder = names - quantized - norms
            self.assertEqual(
                remainder, set(hybrid.PASS_THROUGH_MODULE_NAMES[layer_type]),
                f"{layer_type}: unaccounted parameters")

    def test_quantized_widths_satisfy_encoder_contract(self) -> None:
        """ldlq/regularize require k % 128 and n % 128 (Hadamard block =
        buf_size_k = 128) for every quantized projection, and n % 16 for
        tiling. The pass-through table exists precisely because
        in_proj_a/in_proj_b (n = num_value_heads) violate this."""
        raw = HYBRID_FIXTURE_GEOMETRY["text_config"]
        text_config = hybrid.make_text_config(raw)
        for layer_type in hybrid.SUPPORTED_LAYER_TYPES:
            layer_idx = raw["layer_types"].index(layer_type)
            layer = hybrid.Qwen3_5DecoderLayer(text_config, layer_idx)
            params = dict(layer.named_parameters())
            for name in hybrid.QUANTIZED_MODULES[layer_type]:
                out_k, in_k = params[name + ".weight"].shape  # (n, k)
                with self.subTest(layer_type=layer_type, module=name):
                    self.assertEqual(in_k % 128, 0,
                                     f"{name}: k={in_k} not % 128")
                    self.assertEqual(out_k % 128, 0,
                                     f"{name}: n={out_k} not % 128")
            for name in hybrid.PASS_THROUGH_MODULE_NAMES[layer_type]:
                if name not in params or params[name].dim() != 2:
                    continue
                out_k, _in_k = params[name].shape
                if name.startswith(("linear_attn.in_proj_a",
                                    "linear_attn.in_proj_b")):
                    self.assertNotEqual(out_k % 128, 0,
                                        "pass-through premise changed")

    def test_geometry_tables_follow_layer_type(self) -> None:
        model_dir = _TEST_ROOT / "tables_model"
        shutil.rmtree(model_dir, ignore_errors=True)
        try:
            write_qwen35_hybrid_model(model_dir)
            geo = conv.load_model_geometry(model_dir, mode=_HYBRID_MODE)
            self.assertEqual(geo.quantized_modules(0),
                             list(hybrid.QUANTIZED_MODULES["linear_attention"]))
            self.assertEqual(geo.quantized_modules(2),
                             list(hybrid.QUANTIZED_MODULES["full_attention"]))
            self.assertEqual(
                geo.layer_pass_through_keys(0),
                [f"{geo.key_prefix}.0.{t}"
                 for t in hybrid.PASS_THROUGH_MODULE_NAMES["linear_attention"]])
            self.assertEqual(
                geo.layer_pass_through_keys(2),
                [f"{geo.key_prefix}.2.{t}"
                 for t in hybrid.PASS_THROUGH_MODULE_NAMES["full_attention"]])
            # Dense profile keeps the historical table.
            self.assertEqual(
                conv.ModelGeometry(
                    architecture="llama", hidden_size=128,
                    intermediate_size=256, num_layers=2, num_heads=2,
                    num_kv_heads=2, head_dim=64, rms_norm_eps=1e-6,
                    vocab_size=96).layer_pass_through_keys(0), [])
        finally:
            shutil.rmtree(model_dir, ignore_errors=True)


@_skip_no_hybrid
class TestHybridCalibrationGroups(unittest.TestCase):
    """Shared-input Hessian grouping for both hybrid layer kinds (P2-02
    semantics generalized in P4-12)."""

    @classmethod
    def setUpClass(cls) -> None:
        raw = HYBRID_FIXTURE_GEOMETRY["text_config"]
        cls.text_config = hybrid.make_text_config(raw)
        cls.types = list(raw["layer_types"])

    def _layer(self, layer_type: str):
        idx = self.types.index(layer_type)
        layer = hybrid.Qwen3_5DecoderLayer(self.text_config, idx)
        layer.to(torch.float16)  # reference fp16 forward, as in conversion
        return layer

    def test_group_membership_linear_layer(self) -> None:
        layer = self._layer("linear_attention")
        groups = hessian_groups(layer)
        self.assertEqual(
            set(groups["linear_attn.input"]),
            {"linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
             "linear_attn.in_proj_b", "linear_attn.in_proj_a"})
        self.assertEqual(groups["linear_attn.out_proj"],
                         ["linear_attn.out_proj"])
        self.assertEqual(set(groups["mlp.input"]),
                         {"mlp.gate_proj", "mlp.up_proj"})
        self.assertEqual(groups["mlp.down_proj"], ["mlp.down_proj"])
        self.assertNotIn("self_attn.input", groups)

    def test_group_membership_full_layer(self) -> None:
        layer = self._layer("full_attention")
        groups = hessian_groups(layer)
        self.assertEqual(
            set(groups["self_attn.input"]),
            {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"})
        self.assertEqual(groups["self_attn.o_proj"], ["self_attn.o_proj"])
        self.assertEqual(set(groups["mlp.input"]),
                         {"mlp.gate_proj", "mlp.up_proj"})
        self.assertEqual(groups["mlp.down_proj"], ["mlp.down_proj"])

    def test_every_linear_module_is_assigned(self) -> None:
        for layer_type in hybrid.SUPPORTED_LAYER_TYPES:
            layer = self._layer(layer_type)
            modules = enumerate_linear_modules(layer)
            groups = hessian_groups(layer)
            assigned = {m for members in groups.values() for m in members}
            self.assertEqual(assigned, set(modules))

    def test_group_members_receive_identical_input_tensor(self) -> None:
        """Grouping is only exact if all members consume the same tensor;
        assert input identity per group on a real forward."""
        for layer_type in hybrid.SUPPORTED_LAYER_TYPES:
            layer = self._layer(layer_type)
            groups = hessian_groups(layer)
            modules = dict(layer.named_modules())
            seen: dict = {}

            def hook(module, inputs, _seen=seen):
                _seen[id(module)] = inputs[0] if inputs else None

            handles = [modules[m].register_forward_pre_hook(hook)
                       for members in groups.values() for m in members]
            try:
                runner = hybrid.make_calibration_runner(
                    self.text_config, torch.device("cpu"))
                x = torch.randn(1, 16, HYBRID_FIXTURE_GEOMETRY[
                    "text_config"]["hidden_size"], dtype=torch.float16)
                with torch.no_grad():
                    runner(layer, x)
            finally:
                for h in handles:
                    h.remove()
            for key, members in groups.items():
                tensors = [seen[id(modules[m])] for m in members]
                for other in tensors[1:]:
                    self.assertIs(
                        tensors[0], other,
                        f"{layer_type}/{key}: group members do not share "
                        "one input tensor")

    def test_calibrated_hessians_cover_all_quantized_modules(self) -> None:
        model_dir = _TEST_ROOT / "calibrate_model"
        shutil.rmtree(model_dir, ignore_errors=True)
        try:
            tensors = write_qwen35_hybrid_model(model_dir)
            geo = conv.load_model_geometry(model_dir, mode=_HYBRID_MODE)
            device = torch.device("cpu")
            runner = hybrid.make_calibration_runner(
                hybrid.make_text_config(geo.text_config), device)
            hidden = torch.randn(1, 32, geo.hidden_size, dtype=torch.float16)
            for layer_idx, layer_type in enumerate(geo.layer_types):
                layer_tensors = {
                    k: v for k, v in tensors.items()
                    if k.startswith(f"{geo.key_prefix}.{layer_idx}.")}
                layer = conv._build_layer(geo, layer_tensors, layer_idx, device)
                modules = enumerate_linear_modules(layer)
                hessians, telemetry = calibrate_layer(
                    layer=layer, calibration_data=[hidden], device=device,
                    layer_runner=runner)
                with self.subTest(layer_idx=layer_idx, layer_type=layer_type):
                    for name in geo.quantized_modules(layer_idx):
                        self.assertIn(name, hessians)
                        k_in = modules[name].in_features
                        self.assertEqual(
                            tuple(hessians[name].shape), (k_in, k_in))
                    self.assertGreater(telemetry.total_tokens, 0)
        finally:
            shutil.rmtree(model_dir, ignore_errors=True)


@unittest.skipUnless(torch.xpu.is_available(), "Intel XPU device required")
@_skip_no_hybrid
class TestHybridConversion(unittest.TestCase):
    """Full opt-in conversion of the hybrid fixture on XPU (P4-12 + P4-13)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _TEST_ROOT / "conversion"
        shutil.rmtree(cls.root, ignore_errors=True)
        cls.model_dir = cls.root / "model"
        cls.out_dir = cls.root / "out"
        cls.source = write_qwen35_hybrid_model(cls.model_dir)
        cls.config_kwargs = dict(
            K=2, num_calibration_rows=2, batch_size=1, seq_len=32,
            architecture_mode=_HYBRID_MODE, device="xpu:0",
            skip_g_scale=True, min_disk_free_bytes=0,
        )
        cls.report = conv.convert_model(conv.ConversionConfig(
            model_dir=str(cls.model_dir), output_dir=str(cls.out_dir),
            **cls.config_kwargs))
        with open(cls.out_dir / "quantization_config.json") as f:
            cls.manifest = json.load(f)
        cls.geometry = conv.load_model_geometry(
            cls.model_dir, mode=_HYBRID_MODE)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)

    def _artifact(self, layer_idx: int) -> dict:
        return load_file(str(
            self.out_dir / "qtensors" / f"model.layers.{layer_idx}.safetensors"))

    def test_manifest_records_hybrid_contract(self) -> None:
        m = self.manifest
        self.assertEqual(m["status"], "completed")
        self.assertEqual(m["architecture"], "qwen3_5")
        self.assertEqual(m["architecture_mode"], _HYBRID_MODE)
        self.assertEqual(m["key_prefix"], "model.language_model.layers")
        self.assertEqual(
            m["layer_types"],
            ["linear_attention", "linear_attention", "full_attention"])
        # P4-13 provenance
        self.assertEqual(
            set(m["pass_through"]), {"text", "vision", "mtp"})
        for info in m["pass_through"].values():
            self.assertTrue((self.out_dir / info["file"]).is_file())
            self.assertEqual(info["quant_format"], "source-dtype")
            self.assertTrue(info["unchanged"])
        hy = m["hybrid"]
        self.assertTrue(hy["text_tower_only"])
        self.assertEqual(
            hy["multimodal"],
            {"image_token_id": 999, "video_token_id": 1000,
             "vision_start_token_id": 997, "vision_end_token_id": 998})
        self.assertEqual(hy["rope"]["partial_rotary_factor"], 0.25)
        self.assertEqual(hy["rope"]["mrope_section"], [6, 5, 5])
        self.assertIn("zero_centered_rmsnorm", hy["norm_semantics"])
        self.assertEqual(hy["linear_attention"]["quantized"],
                         list(hybrid.QUANTIZED_MODULES["linear_attention"]))
        self.assertEqual(hy["linear_attention"]["pass_through"],
                         list(hybrid.PASS_THROUGH_MODULE_NAMES["linear_attention"]))
        self.assertEqual(hy["reference_forward"]["reference"],
                         "transformers.models.qwen3_5")
        self.assertIn("torch", m["environment"])
        self.assertIn("python", m["environment"])

    def test_layer_entries_record_kind_and_pass_through(self) -> None:
        for entry, layer_type in zip(self.manifest["layers"],
                                     self.manifest["layer_types"]):
            self.assertEqual(entry["layer_type"], layer_type)
            expected_extra = self.geometry.layer_pass_through_keys(
                entry["layer_index"])
            self.assertEqual(entry["pass_through_keys"],
                             sorted(expected_extra))
            self.assertEqual(entry["quantized_tensors"], 32)

    def test_layer_artifacts_expected_key_sets(self) -> None:
        for layer_idx, layer_type in enumerate(self.manifest["layer_types"]):
            art = self._artifact(layer_idx)
            modules = self.geometry.quantized_modules(layer_idx)
            trellis_keys = {k for k in art if k.endswith(".trellis")}
            expected = {f"{self.geometry.key_prefix}.{layer_idx}.{m}.trellis"
                        for m in modules}
            self.assertEqual(trellis_keys, expected)
            # 4 tensors per quantized module + 2 norms + pass-through
            self.assertEqual(len(art), 4 * len(modules) + 2 + len(
                self.geometry.layer_pass_through_keys(layer_idx)))
            if layer_type == "linear_attention":
                self.assertNotIn(
                    f"{self.geometry.key_prefix}.{layer_idx}."
                    "linear_attn.in_proj_a.trellis", trellis_keys)
                self.assertNotIn(
                    f"{self.geometry.key_prefix}.{layer_idx}."
                    "linear_attn.in_proj_b.trellis", trellis_keys)

    def test_per_layer_pass_through_is_bit_exact(self) -> None:
        for layer_idx in range(self.geometry.num_layers):
            art = self._artifact(layer_idx)
            for key in self.geometry.layer_pass_through_keys(layer_idx):
                with self.subTest(key=key):
                    self.assertEqual(art[key].dtype, torch.bfloat16)
                    self.assertTrue(torch.equal(art[key], self.source[key]))
            for tail in ("input_layernorm.weight",
                         "post_attention_layernorm.weight"):
                key = f"{self.geometry.key_prefix}.{layer_idx}.{tail}"
                self.assertEqual(art[key].dtype, torch.bfloat16)
                self.assertTrue(torch.equal(art[key], self.source[key]))

    def test_pass_through_group_files_are_bit_exact(self) -> None:
        expected_files = {
            "text": "untouched.safetensors",
            "vision": "untouched.vision.safetensors",
            "mtp": "untouched.mtp.safetensors",
        }
        for group, info in self.manifest["pass_through"].items():
            payload = load_file(str(self.out_dir / "qtensors" /
                                    expected_files[group]))
            self.assertEqual(sorted(payload), sorted(info["keys"]))
            for key in info["keys"]:
                with self.subTest(group=group, key=key):
                    self.assertEqual(payload[key].dtype,
                                     self.source[key].dtype)
                    self.assertTrue(torch.equal(payload[key],
                                                self.source[key]))

    def test_quantized_projection_weights_are_reconstructed(self) -> None:
        """Decode an artifact projection and check it is a real (finite)
        approximation of the source weight — proving the hybrid path produced
        actual EXL3 quantization, not pass-through.

        The round-trip decode itself is exact (verified separately against
        ``w_hat`` at ~1e-7); what varies with K=2 is the quantization error,
        measured at 0.3-0.6 relative Frobenius for the fixture's unstructured
        weights. The asserted band mirrors the established metric contract of
        test_layer_quant (0 < rel < 1.0) plus a lower bound proving the
        weights were genuinely quantized.
        """
        from exl3_quant.serialization import decode_weight_from_storage

        for layer_idx in (0, 2):
            art = self._artifact(layer_idx)
            for module in self.geometry.quantized_modules(layer_idx):
                base = f"{self.geometry.key_prefix}.{layer_idx}.{module}"
                w = decode_weight_from_storage(
                    art[f"{base}.trellis"], art[f"{base}.suh"],
                    art[f"{base}.svh"], K=2)
                src = self.source[f"{base}.weight"].float()
                src_kn = src.T  # (k, n)
                with self.subTest(layer_idx=layer_idx, module=module):
                    self.assertTrue(torch.isfinite(w).all())
                    self.assertEqual(tuple(w.shape), tuple(src_kn.shape))
                    rel = (w - src_kn).norm() / src_kn.norm()
                    self.assertGreater(rel.item(), 1e-2,
                                       "weight looks pass-through, not quantized")
                    self.assertLess(rel.item(), 1.0)

    def test_rebuilt_layer_forward_is_deterministic_and_finite(self) -> None:
        device = torch.device("xpu:0")
        runner = hybrid.make_calibration_runner(
            hybrid.make_text_config(self.geometry.text_config), device)
        for layer_idx, layer_type in enumerate(self.manifest["layer_types"]):
            art = self._artifact(layer_idx)
            layer = conv._build_quantized_layer(
                self.geometry, art, layer_idx, device, K=2)
            x = torch.randn(1, 16, self.geometry.hidden_size,
                            dtype=torch.float16, device=device)
            with torch.no_grad():
                first = runner(layer, x)
                second = runner(layer, x)
            with self.subTest(layer_idx=layer_idx, layer_type=layer_type):
                self.assertEqual(tuple(first.shape),
                                 (1, 16, self.geometry.hidden_size))
                self.assertTrue(torch.isfinite(first).all())
                self.assertTrue(torch.equal(first, second))

    def test_resume_completes_after_publish_fault(self) -> None:
        out = self.root / "fault_out"

        def boom(stage: str, layer_idx: int) -> None:
            if stage == "before_publish" and layer_idx == 1:
                raise RuntimeError("injected before_publish fault")

        with self.assertRaises(RuntimeError):
            conv.convert_model(conv.ConversionConfig(
                model_dir=str(self.model_dir), output_dir=str(out),
                **self.config_kwargs), fault_injector=boom)

        report = conv.convert_model(conv.ConversionConfig(
            model_dir=str(self.model_dir), output_dir=str(out),
            resume=True, **self.config_kwargs))
        self.assertEqual(len(report.layers), 3)
        with open(out / "quantization_config.json") as f:
            manifest = json.load(f)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["architecture_mode"], _HYBRID_MODE)
        self.assertEqual(
            [e["layer_index"] for e in manifest["layers"]], [0, 1, 2])
        # resumed artifacts are bit-identical to the uninterrupted run
        for layer_idx in range(3):
            resumed = load_file(str(
                out / "qtensors" / f"model.layers.{layer_idx}.safetensors"))
            expected = self._artifact(layer_idx)
            self.assertEqual(sorted(resumed), sorted(expected))
            for key, value in resumed.items():
                self.assertTrue(torch.equal(value, expected[key]), key)

    def test_resume_validates_architecture_mode(self) -> None:
        """A manifest recorded under another architecture_mode must refuse
        resume. The written manifest is tampered with instead of flipping the
        config, because the default ``dense`` mode rejects the hybrid fixture
        before ``_validate_resume`` can ever run (that guard is asserted in
        ``test_dense_mode_still_rejects_hybrid_fixture``)."""
        out = self.root / "mode_out"
        conv.convert_model(conv.ConversionConfig(
            model_dir=str(self.model_dir), output_dir=str(out),
            **self.config_kwargs))
        manifest_path = out / "quantization_config.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.assertEqual(manifest["architecture_mode"], _HYBRID_MODE)
        manifest["architecture_mode"] = "dense"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        with self.assertRaises(conv.IncompatibleResumeError) as ctx:
            conv.convert_model(conv.ConversionConfig(
                model_dir=str(self.model_dir), output_dir=str(out),
                resume=True, **self.config_kwargs))
        self.assertIn("architecture_mode", str(ctx.exception))


class TestDenseProfileUnchanged(unittest.TestCase):
    """The dense fixture path must remain byte-identical after P4-12/P4-13."""

    def test_dense_geometry_defaults(self) -> None:
        geo = conv.ModelGeometry(
            architecture="llama", hidden_size=128, intermediate_size=256,
            num_layers=2, num_heads=2, num_kv_heads=2, head_dim=64,
            rms_norm_eps=1e-6, vocab_size=96)
        self.assertFalse(geo.is_hybrid)
        self.assertEqual(geo.architecture_mode, "dense")
        self.assertEqual(geo.key_prefix, "model.layers")
        self.assertEqual(geo.text_prefix, "model")
        self.assertEqual(geo.layer_type(0), "full_attention")
        self.assertEqual(list(geo.quantized_modules(0)),
                         list(conv._QUANTIZED_MODULES))
        self.assertEqual(geo.embed_key, "model.embed_tokens.weight")
        self.assertEqual(geo.final_norm_key, "model.norm.weight")
        self.assertEqual(geo.layer_pass_through_keys(0), [])

    def test_hessian_groups_dense_profile_match_legacy(self) -> None:
        from exl3_quant.calibration import TransformerDecoderLayer

        layer = TransformerDecoderLayer(
            hidden_size=128, intermediate_size=256, num_heads=2,
            num_kv_heads=2, rms_norm_eps=1e-6)
        groups = hessian_groups(layer)
        self.assertEqual(
            groups,
            {
                "self_attn.input": ["self_attn.q_proj", "self_attn.k_proj",
                                    "self_attn.v_proj"],
                "self_attn.o_proj": ["self_attn.o_proj"],
                "mlp.input": ["mlp.gate_proj", "mlp.up_proj"],
                "mlp.down_proj": ["mlp.down_proj"],
            })


if __name__ == "__main__":
    unittest.main()
