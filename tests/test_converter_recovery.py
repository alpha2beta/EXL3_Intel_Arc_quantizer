"""Tests for checkpointing, resumability, and fault recovery (P3-04..P3-08).

Covers:
- P3-04: Working-directory layout (qtensors/), checkpoint interval, manifest progress tracking
- P3-05: Atomic checkpoint publication (atomic_save_safetensors), layer integrity verification (verify_layer_checkpoint)
- P3-06: Interruption-safe fault recovery across calibration, quantization, and publishing boundaries
- P3-07: Rigorous resume argument validation (_validate_resume) with force_resume override
- P3-08: Failure mode resilience: insufficient disk space, missing shards, corrupted checkpoints
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
import unittest.mock
from pathlib import Path
from typing import Dict, List

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from fixtures import (  # noqa: E402
    FIXTURE_GEOMETRY,
    write_fixture_config,
    write_sharded_model,
)

import exl3_quant.converter as conv  # noqa: E402

_TEST_ROOT = Path("/tmp/opencode/exl3_converter_recovery_tests")
_SHARDS = 2
_NUM_LAYERS = 3


def _fresh(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class TestAtomicPublicationAndIntegrity(unittest.TestCase):
    """P3-05: Atomic publication and checkpoint integrity verification."""

    def setUp(self) -> None:
        self.test_dir = _fresh(_TEST_ROOT / "atomic_pub")

    def test_atomic_save_safetensors_creates_valid_file(self):
        target = self.test_dir / "test.safetensors"
        tensors = {"a": torch.randn(4, 4), "b": torch.ones(2, 2)}
        conv.atomic_save_safetensors(tensors, target)

        self.assertTrue(target.is_file())
        self.assertFalse(target.with_suffix(".safetensors.tmp").exists())
        loaded = load_file(str(target))
        self.assertEqual(set(loaded.keys()), {"a", "b"})
        self.assertTrue(torch.equal(loaded["b"], tensors["b"]))

    def test_atomic_save_safetensors_cleans_tmp_on_failure(self):
        target = self.test_dir / "fail.safetensors"
        tensors = {"a": torch.randn(4, 4)}

        # Simulate failure during verification / corruption
        with unittest.mock.patch("exl3_quant.converter.safe_open", side_effect=RuntimeError("corrupted tmp")):
            with self.assertRaises(conv.CorruptedCheckpointError):
                conv.atomic_save_safetensors(tensors, target)

        self.assertFalse(target.exists())
        self.assertFalse(target.with_suffix(".safetensors.tmp").exists())

    def test_verify_layer_checkpoint_success(self):
        # Create a complete layer checkpoint with all 30 expected tensors
        layer_file = self.test_dir / "model.layers.0.safetensors"
        layer_tensors: Dict[str, torch.Tensor] = {}
        for mod in conv._QUANTIZED_MODULES:
            base = f"model.layers.0.{mod}"
            layer_tensors[f"{base}.trellis"] = torch.zeros((8, 8, 32), dtype=torch.int16)
            layer_tensors[f"{base}.suh"] = torch.ones(128, dtype=torch.float16)
            layer_tensors[f"{base}.svh"] = torch.ones(128, dtype=torch.float16)
            layer_tensors[f"{base}.mul1"] = torch.zeros(128, dtype=torch.int32)
        for tail in conv._NORM_KEYS:
            layer_tensors[f"model.layers.0.{tail}"] = torch.ones(128, dtype=torch.float16)

        save_file(layer_tensors, str(layer_file))
        self.assertTrue(conv.verify_layer_checkpoint(layer_file, 0))

    def test_verify_layer_checkpoint_missing_file_returns_false(self):
        missing = self.test_dir / "nonexistent.safetensors"
        self.assertFalse(conv.verify_layer_checkpoint(missing, 0))

    def test_verify_layer_checkpoint_rejects_empty_file(self):
        empty_file = self.test_dir / "model.layers.0.safetensors"
        empty_file.touch()
        with self.assertRaises(conv.CorruptedCheckpointError):
            conv.verify_layer_checkpoint(empty_file, 0)

    def test_verify_layer_checkpoint_rejects_missing_keys(self):
        layer_file = self.test_dir / "model.layers.0.safetensors"
        # Only save a subset of keys (e.g. missing norm tensors)
        incomplete_tensors = {"model.layers.0.self_attn.q_proj.trellis": torch.zeros(2, 2)}
        save_file(incomplete_tensors, str(layer_file))
        with self.assertRaises(conv.CorruptedCheckpointError):
            conv.verify_layer_checkpoint(layer_file, 0)

    def test_verify_layer_checkpoint_rejects_truncated_file(self):
        layer_file = self.test_dir / "model.layers.0.safetensors"
        layer_file.write_bytes(b"truncated safetensors header \x00\x00\x01\x02")
        with self.assertRaises(conv.CorruptedCheckpointError):
            conv.verify_layer_checkpoint(layer_file, 0)


class TestResumeArgumentValidation(unittest.TestCase):
    """P3-07: Resume argument validation against existing checkpoint manifest."""

    def setUp(self) -> None:
        self.model_dir = _fresh(_TEST_ROOT / "valid_model")
        write_fixture_config(self.model_dir)
        self.out_dir = _fresh(_TEST_ROOT / "resume_validation_out")
        self.geometry = conv.load_model_geometry(self.model_dir)

        # Baseline manifest
        self.manifest = {
            "status": "in_progress",
            "completed_layers": [0],
            "last_completed_layer": 0,
            "quant_method": "exl3",
            "bits": 2,
            "codebook": "mul1",
            "architecture": self.geometry.architecture,
            "num_layers": self.geometry.num_layers,
            "device": "cpu",
            "calibration": {
                "num_rows": 2,
                "batch_size": 1,
                "seq_len": 128,
                "seed": 0,
                "damping_lambda": 0.01,
                "zero_floor": 1e-7,
                "skip_g_scale": True,
            },
        }

    def _config(self, **overrides) -> conv.ConversionConfig:
        cfg = conv.ConversionConfig(
            model_dir=str(self.model_dir),
            output_dir=str(self.out_dir),
            resume=True,
            K=2,
            cb="mul1",
            num_calibration_rows=2,
            batch_size=1,
            seq_len=128,
            calibration_seed=0,
            damping_lambda=0.01,
            zero_floor=1e-7,
            skip_g_scale=True,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_matching_config_validates(self):
        cfg = self._config()
        # Should not raise
        conv._validate_resume(cfg, self.geometry, self.manifest)

    def test_rejects_mismatched_K(self):
        cfg = self._config(K=3)
        with self.assertRaises(conv.IncompatibleResumeError) as ctx:
            conv._validate_resume(cfg, self.geometry, self.manifest)
        self.assertIn("bits (K)", str(ctx.exception))

    def test_rejects_mismatched_calibration_seed(self):
        cfg = self._config(calibration_seed=42)
        with self.assertRaises(conv.IncompatibleResumeError) as ctx:
            conv._validate_resume(cfg, self.geometry, self.manifest)
        self.assertIn("calibration.seed", str(ctx.exception))

    def test_rejects_mismatched_calibration_seq_len(self):
        cfg = self._config(seq_len=256)
        with self.assertRaises(conv.IncompatibleResumeError) as ctx:
            conv._validate_resume(cfg, self.geometry, self.manifest)
        self.assertIn("calibration.seq_len", str(ctx.exception))

    def test_rejects_mismatched_geometry_num_layers(self):
        cfg = self._config()
        bad_geo = conv.ModelGeometry(
            architecture="llama", hidden_size=128, intermediate_size=256,
            num_layers=99, num_heads=2, num_kv_heads=2, head_dim=64,
            rms_norm_eps=1e-6, vocab_size=96
        )
        with self.assertRaises(conv.IncompatibleResumeError) as ctx:
            conv._validate_resume(cfg, bad_geo, self.manifest)
        self.assertIn("num_layers", str(ctx.exception))

    def test_force_resume_overrides_mismatches(self):
        cfg = self._config(K=4, calibration_seed=999, force_resume=True)
        # Does not raise when force_resume=True
        conv._validate_resume(cfg, self.geometry, self.manifest)


@unittest.skipUnless(torch.xpu.is_available(), "Intel XPU device required")
class TestFaultRecoveryAndResumption(unittest.TestCase):
    """P3-04 & P3-06: Interruption fault injection, atomic recovery, and state continuity."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("xpu:0")
        cls.model_dir = _fresh(_TEST_ROOT / "recovery_model")
        write_sharded_model(cls.model_dir, num_layers=_NUM_LAYERS, shards=_SHARDS)
        cls.out_dir = _fresh(_TEST_ROOT / "recovery_out")
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
            skip_g_scale=True,
            checkpoint_interval=1,
        )

    def test_interruption_and_safe_resumption(self):
        out_dir = _fresh(_TEST_ROOT / "interrupted_run")
        cfg = conv.ConversionConfig(
            model_dir=str(self.model_dir),
            output_dir=str(out_dir),
            K=2,
            num_calibration_rows=2,
            batch_size=1,
            seq_len=128,
            calibration_seed=0,
            skip_g_scale=True,
            checkpoint_interval=1,
        )

        quantized_layers_pass1: List[int] = []

        # 1. First run: inject a simulated failure during layer 1 before publish
        def fault_injector(stage: str, layer_idx: int):
            if stage == "before_quantization":
                quantized_layers_pass1.append(layer_idx)
            if stage == "before_publish" and layer_idx == 1:
                # Leave a stray .tmp file to test atomic publication cleanup
                stray_tmp = out_dir / "qtensors" / "model.layers.1.safetensors.tmp"
                stray_tmp.write_bytes(b"uncommitted write bytes")
                raise conv.CheckpointInterruptedError("simulated crash at layer 1 before publish")

        with self.assertRaises(conv.CheckpointInterruptedError):
            conv.convert_model(cfg, fault_injector=fault_injector)

        # Verify disk state after crash:
        # Layer 0 must be published and valid
        self.assertTrue(conv.verify_layer_checkpoint(out_dir / "qtensors" / "model.layers.0.safetensors", 0))
        # Layer 1 must NOT be published
        self.assertFalse((out_dir / "qtensors" / "model.layers.1.safetensors").exists())
        # Intermediate manifest must show layer 0 completed, status in_progress
        manifest_path = out_dir / "quantization_config.json"
        self.assertTrue(manifest_path.is_file())
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["status"], "in_progress")
        self.assertEqual(manifest["completed_layers"], [0])

        # 2. Second run: resume with resume=True
        cfg_resume = conv.ConversionConfig(
            model_dir=str(self.model_dir),
            output_dir=str(out_dir),
            K=2,
            num_calibration_rows=2,
            batch_size=1,
            seq_len=128,
            calibration_seed=0,
            skip_g_scale=True,
            checkpoint_interval=1,
            resume=True,
        )

        quantized_layers_pass2: List[int] = []

        def pass2_tracker(stage: str, layer_idx: int):
            if stage == "before_quantization":
                quantized_layers_pass2.append(layer_idx)

        final_report = conv.convert_model(cfg_resume, fault_injector=pass2_tracker)

        # Crucial check: Layer 0 was SKIPPED during pass 2 (no duplicate work)
        self.assertEqual(quantized_layers_pass2, [1, 2])
        # Total converted layers across the final report is all 3
        self.assertEqual(len(final_report.layers), 3)
        self.assertEqual([e.layer_index for e in final_report.layers], [0, 1, 2])

        # Verify final on-disk state
        for idx in range(_NUM_LAYERS):
            layer_path = out_dir / "qtensors" / f"model.layers.{idx}.safetensors"
            self.assertTrue(conv.verify_layer_checkpoint(layer_path, idx))

        # Stray .tmp files were cleaned up
        self.assertFalse(list((out_dir / "qtensors").glob("*.tmp")))

        final_manifest = json.loads(manifest_path.read_text())
        self.assertEqual(final_manifest["status"], "completed")
        self.assertEqual(final_manifest["completed_layers"], [0, 1, 2])

    def test_resume_with_deleted_calibration_state_falls_back_to_replay(self):
        """If calibration_state.pt is deleted, resume replays through published layers."""
        out_dir = _fresh(_TEST_ROOT / "replay_resume_run")
        cfg = conv.ConversionConfig(
            model_dir=str(self.model_dir),
            output_dir=str(out_dir),
            K=2,
            num_calibration_rows=2,
            batch_size=1,
            seq_len=128,
            calibration_seed=0,
            skip_g_scale=True,
            checkpoint_interval=1,
        )

        # Run layer 0, interrupt before layer 1
        def fail_at_1(stage: str, layer_idx: int):
            if stage == "before_calibration" and layer_idx == 1:
                raise conv.CheckpointInterruptedError("interrupt before layer 1")

        with self.assertRaises(conv.CheckpointInterruptedError):
            conv.convert_model(cfg, fault_injector=fail_at_1)

        # Delete calibration_state.pt to force the replay path
        cal_state = out_dir / "qtensors" / "calibration_state.pt"
        if cal_state.exists():
            cal_state.unlink()

        cfg_resume = conv.ConversionConfig(
            model_dir=str(self.model_dir),
            output_dir=str(out_dir),
            K=2,
            num_calibration_rows=2,
            batch_size=1,
            seq_len=128,
            calibration_seed=0,
            skip_g_scale=True,
            resume=True,
        )

        # Resuming without calibration_state.pt successfully replays layer 0 and finishes
        report = conv.convert_model(cfg_resume)
        self.assertEqual(len(report.layers), 3)
        self.assertTrue(conv.verify_layer_checkpoint(out_dir / "qtensors" / "model.layers.2.safetensors", 2))


class TestFailureModesAndResilience(unittest.TestCase):
    """P3-08: Resilience under disk space exhaustion, missing shards, and corrupted checkpoints."""

    def setUp(self) -> None:
        self.model_dir = _fresh(_TEST_ROOT / "fail_model")
        write_sharded_model(self.model_dir, num_layers=2, shards=2)
        self.out_dir = _fresh(_TEST_ROOT / "fail_out")

    def test_insufficient_disk_space_raises(self):
        cfg = conv.ConversionConfig(
            model_dir=str(self.model_dir),
            output_dir=str(self.out_dir),
            min_disk_free_bytes=100 * 1024 * 1024,  # 100 MiB
        )
        fake_usage = type("Usage", (), {"free": 1024, "total": 1000000, "used": 998976})()
        with unittest.mock.patch("shutil.disk_usage", return_value=fake_usage):
            with self.assertRaises(conv.DiskSpaceError):
                conv.convert_model(cfg)

    def test_missing_model_shard_raises(self):
        # Delete an indexed shard file from model directory
        shards = sorted(self.model_dir.glob("model-*.safetensors"))
        self.assertTrue(bool(shards), "expected shards in model_dir")
        shards[0].unlink()

        cfg = conv.ConversionConfig(
            model_dir=str(self.model_dir),
            output_dir=str(self.out_dir),
        )
        with self.assertRaises((FileNotFoundError, conv.UnsupportedModeError, Exception)):
            conv.convert_model(cfg)

    def test_corrupted_existing_checkpoint_halts_resume(self):
        # Create a corrupted layer 0
        qt_dir = self.out_dir / "qtensors"
        qt_dir.mkdir(parents=True, exist_ok=True)
        corrupted = qt_dir / "model.layers.0.safetensors"
        corrupted.write_bytes(b"\x00\x00\x00corrupt")

        manifest = {
            "status": "in_progress",
            "completed_layers": [0],
            "last_completed_layer": 0,
            "quant_method": "exl3",
            "bits": 2,
            "codebook": "mul1",
            "architecture": "llama",
            "num_layers": 2,
            "calibration": {"seed": 0, "seq_len": 128, "batch_size": 1, "num_rows": 2},
        }
        (self.out_dir / "quantization_config.json").write_text(json.dumps(manifest))

        cfg = conv.ConversionConfig(
            model_dir=str(self.model_dir),
            output_dir=str(self.out_dir),
            resume=True,
        )
        with self.assertRaises(conv.CorruptedCheckpointError):
            conv.convert_model(cfg)


if __name__ == "__main__":
    unittest.main()
