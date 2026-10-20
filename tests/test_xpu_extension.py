"""Unit tests for the minimal XPU extension probe (P0-18 / P0-19).

These tests build (cached at artifacts/build/probe_xpu) and exercise the SYCL
extension on the current XPU device; they are skipped when torch.xpu is not
available. The build requires icpx on PATH (see exl3_quant/xpu_probe).
"""

import unittest

from exl3_quant import xpu_probe

try:
    import torch  # noqa: F401
    XPU_AVAILABLE = torch.xpu.is_available()
except Exception:  # pragma: no cover - torch/xpu import failures
    XPU_AVAILABLE = False

_skip_reason = "torch.xpu device not available"


@unittest.skipUnless(XPU_AVAILABLE, _skip_reason)
class TestXpuExtension(unittest.TestCase):
    """P0-18: framework-integrated custom XPU extension probes."""

    @classmethod
    def setUpClass(cls):
        cls.ext = xpu_probe.build_extension()

    def test_device_selection(self):
        """Kernel runs on the tensor's device via the current torch queue."""
        result = xpu_probe.verify_device_selection(self.ext)
        self.assertEqual(result["device_count"], 1)
        self.assertTrue(result["names_agree"],
                        f"torch vs sycl device name: {result}")
        self.assertTrue(result["vec_add_correct"], f"vec_add wrong: {result}")

    def test_output_correctness_bit_exact(self):
        """Kernel adds a scalar bit-exactly against a CPU reference."""
        result = xpu_probe.verify_output_correctness(self.ext)
        self.assertTrue(result["bit_exact_vs_cpu"], f"not bit exact: {result}")
        self.assertEqual(result["max_abs_diff"], 0.0)

    def test_queue_ordering_interleaved(self):
        """Torch ops and custom kernels alternate correctly on one queue."""
        result = xpu_probe.verify_queue_ordering(self.ext)
        self.assertTrue(result["interleaved_sequence_correct"],
                        f"ordering broken: {result}")

    def test_tensor_lifetime_reuse(self):
        """Freed tensors do not corrupt later allocations via unsafe reuse."""
        result = xpu_probe.verify_tensor_lifetime(self.ext)
        self.assertTrue(result["reuse_correct"], f"reuse broken: {result}")

    def test_device_capability_capture(self):
        """P0-19: SYCL-side caps must agree with torch properties."""
        caps = xpu_probe.capture_environment()
        sycl = caps["sycl_via_probe_extension"]
        self.assertEqual(sycl["sycl_max_sub_group_size"], 32, str(sycl))
        self.assertGreaterEqual(sycl["sycl_max_work_group_size"], 128)
        self.assertEqual(caps["torch"]["gpu_subslice_count"], 8, str(sycl))
        self.assertGreater(sycl["sycl_global_mem_size"], 16 * (1 << 30))
        self.assertGreater(sycl["sycl_max_mem_alloc_size"], 16 * (1 << 30))

    def test_hfma_random_vs_exact(self):
        """O-6 (random block): device fp16 fma == exact fp16 single rounding."""
        result = xpu_probe.run_hfma_probe(self.ext)
        self.assertEqual(result["fma_vs_exact_mismatches"], 0, str(result))
        self.assertEqual(result["expr_vs_exact_mismatches"], 0, str(result))

    def test_hfma_adversarial_vs_exact(self):
        """O-6 (adversarial block): device fma must NOT match the fp32
        double-rounding emulation on the discriminating pairs."""
        result = xpu_probe.run_hfma_adversarial_probe(self.ext)
        self.assertNotIn("error", result, str(result))
        self.assertGreater(result["fp32_emul_vs_exact_mismatches"], 0,
                           "adversarial set no longer discriminates")
        self.assertEqual(result["fma_matches_exact"], result["n"], str(result))
        self.assertEqual(result["fma_matches_fp32_emul"], 0, str(result))
        self.assertEqual(
            result["conclusion"], "device fma == exact single rounding",
            str(result))


if __name__ == "__main__":
    unittest.main()
