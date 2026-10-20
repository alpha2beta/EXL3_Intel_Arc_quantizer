"""Unit tests for the configurable memory-pressure policy (V140-02 / P0-20)."""

import os
import tempfile
import unittest

from exl3_quant.memory_policy import (
    BYTES_PER_GIB,
    BYTES_PER_MIB,
    MemoryPolicy,
    MemoryPressureError,
    MemorySnapshot,
    capture_snapshot,
    format_bytes,
    read_system_memory,
)

try:
    import torch
    XPU_AVAILABLE = torch.xpu.is_available()
except Exception:  # pragma: no cover
    XPU_AVAILABLE = False


def make_snapshot(binding_bytes, gpu=True):
    """Snapshot whose device/system counters share one binding figure."""
    return MemorySnapshot(
        gpu_available=gpu,
        system_total_bytes=32 * BYTES_PER_GIB,
        system_available_bytes=binding_bytes if gpu else binding_bytes,
        system_free_bytes=binding_bytes,
        cached_bytes=0,
        swap_total_bytes=32 * BYTES_PER_GIB,
        swap_free_bytes=30 * BYTES_PER_GIB,
        device_free_bytes=binding_bytes if gpu else 0,
        device_total_bytes=30 * BYTES_PER_GIB if gpu else 0,
        device_allocated_bytes=0,
        device_reserved_bytes=0,
    )


class TestMeminfoParsing(unittest.TestCase):
    def test_parses_captured_keys_in_bytes(self):
        content = [
            "MemTotal:       32376860 kB",
            "MemFree:         9270612 kB",
            "MemAvailable:   14998932 kB",
            "Buffers:          123456 kB",
            "Cached:          5403460 kB",
            "SwapTotal:      32376828 kB",
            "SwapFree:       30893988 kB",
            "SwapCached:         3572 kB",
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".meminfo",
                                         delete=False) as handle:
            handle.write("\n".join(content) + "\n")
            path = handle.name
        try:
            parsed = read_system_memory(path)
        finally:
            os.unlink(path)

        self.assertEqual(parsed["MemTotal"], 32376860 * 1024)
        self.assertEqual(parsed["MemAvailable"], 14998932 * 1024)
        self.assertEqual(parsed["SwapFree"], 30893988 * 1024)
        self.assertEqual(parsed["SwapTotal"], 32376828 * 1024)
        self.assertNotIn("Buffers", parsed)  # uncaptured keys are ignored


class TestPolicyArithmetic(unittest.TestCase):
    def test_binding_uses_min_of_overlapping_counters(self):
        snap = MemorySnapshot(
            gpu_available=True,
            system_total_bytes=32 * BYTES_PER_GIB,
            system_available_bytes=12 * BYTES_PER_GIB,
            system_free_bytes=9 * BYTES_PER_GIB,
            cached_bytes=3 * BYTES_PER_GIB,
            swap_total_bytes=32 * BYTES_PER_GIB,
            swap_free_bytes=30 * BYTES_PER_GIB,
            device_free_bytes=8 * BYTES_PER_GIB,
            device_total_bytes=28 * BYTES_PER_GIB,
            device_allocated_bytes=0,
            device_reserved_bytes=0,
        )
        self.assertEqual(snap.binding_available_bytes, 8 * BYTES_PER_GIB)

        cpu_only = make_snapshot(10 * BYTES_PER_GIB, gpu=False)
        self.assertEqual(cpu_only.binding_available_bytes, 10 * BYTES_PER_GIB)

    def test_usable_subtracts_reserve_and_clamps_at_zero(self):
        policy = MemoryPolicy(reserve_bytes=6 * BYTES_PER_GIB)
        self.assertEqual(
            policy.usable_bytes(make_snapshot(10 * BYTES_PER_GIB)),
            4 * BYTES_PER_GIB)
        self.assertEqual(
            policy.usable_bytes(make_snapshot(3 * BYTES_PER_GIB)), 0)

    def test_admission_boundary(self):
        policy = MemoryPolicy(reserve_bytes=6 * BYTES_PER_GIB)
        snap = make_snapshot(10 * BYTES_PER_GIB)
        # Exactly the usable budget is allowed.
        self.assertEqual(policy.check(4 * BYTES_PER_GIB, snap),
                         4 * BYTES_PER_GIB)
        with self.assertRaises(MemoryPressureError) as ctx:
            policy.check(4 * BYTES_PER_GIB + 1, snap)
        err = ctx.exception
        self.assertEqual(err.need_bytes, 4 * BYTES_PER_GIB + 1)
        self.assertEqual(err.usable_bytes, 4 * BYTES_PER_GIB)
        self.assertIn("memory-pressure abort", str(err))
        self.assertIn("reserve", str(err))

    def test_outstanding_commitments_reduce_budget(self):
        policy = MemoryPolicy(reserve_bytes=6 * BYTES_PER_GIB)
        snap = make_snapshot(10 * BYTES_PER_GIB)
        outstanding = 1 * BYTES_PER_GIB
        self.assertEqual(
            policy.usable_bytes(snap, outstanding_bytes=outstanding),
            3 * BYTES_PER_GIB)
        policy.check(3 * BYTES_PER_GIB, snap, outstanding_bytes=outstanding)
        with self.assertRaises(MemoryPressureError) as ctx:
            policy.check(3 * BYTES_PER_GIB + 1, snap,
                         outstanding_bytes=outstanding)
        self.assertIn("outstanding", str(ctx.exception))
        self.assertIn("1.00 GiB", str(ctx.exception))

    def test_negative_outstanding_rejected(self):
        policy = MemoryPolicy()
        with self.assertRaises(ValueError):
            policy.usable_bytes(make_snapshot(10 * BYTES_PER_GIB),
                                outstanding_bytes=-1)

    def test_negative_need_rejected(self):
        policy = MemoryPolicy()
        with self.assertRaises(ValueError):
            policy.check(-1, make_snapshot(10 * BYTES_PER_GIB))

    def test_reserve_above_available_blocks_everything(self):
        policy = MemoryPolicy(reserve_bytes=12 * BYTES_PER_GIB)
        snap = make_snapshot(8 * BYTES_PER_GIB)
        with self.assertRaises(MemoryPressureError):
            policy.check(1, snap)

    def test_single_allocation_ceiling(self):
        policy = MemoryPolicy(reserve_bytes=0,
                              max_single_alloc_bytes=4 * BYTES_PER_GIB)
        snap = make_snapshot(30 * BYTES_PER_GIB)
        policy.check(4 * BYTES_PER_GIB, snap)  # at the ceiling
        with self.assertRaises(MemoryPressureError) as ctx:
            policy.check(4 * BYTES_PER_GIB + 1, snap)
        self.assertIn("single-allocation ceiling", str(ctx.exception))

    def test_post_commit_breach_detection(self):
        policy = MemoryPolicy(reserve_bytes=6 * BYTES_PER_GIB,
                              safety_slack_bytes=256 * BYTES_PER_MIB)
        # Within slack of the reserve is tolerated (measurement noise).
        policy.check_after_commit(
            make_snapshot(6 * BYTES_PER_GIB - 128 * BYTES_PER_MIB))
        # A real breach aborts.
        with self.assertRaises(MemoryPressureError) as ctx:
            policy.check_after_commit(
                make_snapshot(5 * BYTES_PER_GIB))
        self.assertIn("reserve breached", str(ctx.exception))

    def test_allocator_growth_assertion(self):
        """M1: the commit check verifies allocator-visible growth, not OS counters."""
        policy = MemoryPolicy(reserve_bytes=6 * BYTES_PER_GIB)
        pre = make_snapshot(10 * BYTES_PER_GIB)
        post = MemorySnapshot(
            gpu_available=True,
            system_total_bytes=pre.system_total_bytes,
            system_available_bytes=pre.system_available_bytes,
            system_free_bytes=pre.system_free_bytes,
            cached_bytes=pre.cached_bytes,
            swap_total_bytes=pre.swap_total_bytes,
            swap_free_bytes=pre.swap_free_bytes,
            device_free_bytes=pre.device_free_bytes,
            device_total_bytes=pre.device_total_bytes,
            device_allocated_bytes=pre.device_allocated_bytes + 2 * BYTES_PER_MIB,
            device_reserved_bytes=pre.device_reserved_bytes,
        )
        # Growth meets the expectation -> allowed.
        policy.check_after_commit(post, pre_snapshot=pre,
                                  expected_bytes=2 * BYTES_PER_MIB)
        # Growth falls short -> committed bytes are not actually backed.
        with self.assertRaises(MemoryPressureError) as ctx:
            policy.check_after_commit(post, pre_snapshot=pre,
                                      expected_bytes=3 * BYTES_PER_MIB)
        self.assertIn("allocator commitment failure", str(ctx.exception))
        self.assertIn("expected at least", str(ctx.exception))

    def test_error_dict_has_full_context(self):
        policy = MemoryPolicy(reserve_bytes=6 * BYTES_PER_GIB)
        try:
            policy.check(8 * BYTES_PER_GIB, make_snapshot(10 * BYTES_PER_GIB),
                         context="unit")
        except MemoryPressureError as exc:
            payload = exc.as_dict()
        else:
            self.fail("expected MemoryPressureError")
        self.assertEqual(payload["error"], "MemoryPressureError")
        self.assertEqual(payload["context"], "unit")
        self.assertIn("binding_available_bytes", payload["snapshot"])
        self.assertIn("usable_bytes", payload)

    def test_format_bytes(self):
        self.assertEqual(format_bytes(None), "n/a")
        self.assertEqual(format_bytes(1024), "1024 B")
        self.assertEqual(format_bytes(5 * BYTES_PER_MIB), "5.0 MiB")
        self.assertEqual(format_bytes(2 * BYTES_PER_GIB), "2.00 GiB")
        self.assertEqual(format_bytes(-2 * BYTES_PER_GIB), "-2.00 GiB")


class TestLiveCapture(unittest.TestCase):
    def test_capture_snapshot_on_host(self):
        snap = capture_snapshot()
        self.assertGreater(snap.system_total_bytes, 0)
        self.assertGreater(snap.system_available_bytes, 0)
        self.assertGreaterEqual(snap.swap_free_bytes, 0)
        self.assertIsInstance(snap.gpu_available, bool)

    @unittest.skipUnless(XPU_AVAILABLE, "torch.xpu device not available")
    def test_gpu_allocation_under_policy(self):
        policy = MemoryPolicy(reserve_bytes=6 * BYTES_PER_GIB)
        pre = capture_snapshot()
        self.assertTrue(pre.gpu_available)
        size = 32 * BYTES_PER_MIB
        policy.check(size, pre, context="test")
        buffer = torch.empty(size // 4, dtype=torch.float32, device="xpu")
        try:
            buffer.fill_(1.0)
            torch.xpu.synchronize()
            post = capture_snapshot()
            policy.check_after_commit(post)
            self.assertGreaterEqual(
                post.device_allocated_bytes - pre.device_allocated_bytes, size
            )
            self.assertLess(
                policy.usable_bytes(
                    post, outstanding_bytes=post.device_allocated_bytes
                ),
                policy.usable_bytes(
                    pre, outstanding_bytes=pre.device_allocated_bytes
                ),
            )
        finally:
            del buffer
            torch.xpu.empty_cache()


if __name__ == "__main__":
    unittest.main()
