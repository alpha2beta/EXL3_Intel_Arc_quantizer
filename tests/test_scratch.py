"""Unit tests for Scratch Memory Estimator.

Directly tests P1-10 regression fixtures from tasks.md lines 121-129:
    S = 65536 >> K; T = 256; B = 256
    costs = B * 2 * S * 2 bytes
    edges = B * T * S * 2 bytes

    K=1: S=32768, edges=4 GiB,   costs=32 MiB
    K=2: S=16384, edges=2 GiB,   costs=16 MiB
    K=3: S=8192,  edges=1 GiB,   costs=8 MiB
    K=4: S=4096,  edges=0.5 GiB, costs=4 MiB
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exl3_quant.scratch_estimator import (
    ScratchEstimator,
    BYTES_PER_MIB,
    BYTES_PER_GIB,
    BYTES_PER_MB,
    BYTES_PER_GB,
)


class TestScratchEstimator(unittest.TestCase):

    def test_tasks_md_regression_fixtures(self):
        """Verify the exact allocation fixtures declared in tasks.md lines 121-129."""
        fixtures = [
            # k, expected_states, expected_edges_gib, expected_costs_mib
            (1, 32768, 4.0, 32.0),
            (2, 16384, 2.0, 16.0),
            (3, 8192, 1.0, 8.0),
            (4, 4096, 0.5, 4.0),
        ]

        for k, exp_states, exp_edges_gib, exp_costs_mib in fixtures:
            with self.subTest(k=k):
                alloc = ScratchEstimator.estimate_scratch(k=k, t=256, b=256)
                
                # Check state count
                self.assertEqual(alloc.states, exp_states)
                
                # Check binary GiB / MiB exact match
                self.assertAlmostEqual(alloc.edges_gib, exp_edges_gib, places=6)
                self.assertAlmostEqual(alloc.costs_mib, exp_costs_mib, places=6)

                # Check exact byte counts
                expected_edge_bytes = int(exp_edges_gib * BYTES_PER_GIB)
                expected_cost_bytes = int(exp_costs_mib * BYTES_PER_MIB)
                self.assertEqual(alloc.edges_bytes, expected_edge_bytes)
                self.assertEqual(alloc.costs_bytes, expected_cost_bytes)

    def test_binary_vs_decimal_distinction(self):
        """P1-10: Distinguish GiB/MiB (1024) from decimal GB/MB (1000)."""
        alloc = ScratchEstimator.estimate_scratch(k=1, t=256, b=256)
        
        # 4 GiB = 4,294,967,296 bytes
        # In decimal GB, 4,294,967,296 / 10^9 = ~4.295 GB != 4.0 GB
        self.assertAlmostEqual(alloc.edges_gib, 4.0, places=4)
        self.assertAlmostEqual(alloc.edges_gb, 4.294967, places=4)
        self.assertNotEqual(alloc.edges_gib, alloc.edges_gb)

        # 32 MiB = 33,554,432 bytes
        # In decimal MB, 33,554,432 / 10^6 = ~33.554 MB != 32.0 MB
        self.assertAlmostEqual(alloc.costs_mib, 32.0, places=4)
        self.assertAlmostEqual(alloc.costs_mb, 33.554432, places=4)
        self.assertNotEqual(alloc.costs_mib, alloc.costs_mb)

    def test_adaptive_batch_calculator_arc_b580_12gb(self):
        """Arc B580 (12 GiB) with 2 GiB reserve and 4 GiB single alloc limit."""
        vram_12gib = 12 * BYTES_PER_GIB
        reserve_2gib = 2 * BYTES_PER_GIB
        single_alloc_limit = 4 * BYTES_PER_GIB

        # For K=1: 256 batch requires 4 GiB edge buffer + 32 MiB costs.
        # Single alloc limit is 4 GiB, so B=256 exactly hits the 4 GiB single buffer ceiling!
        max_b, alloc = ScratchEstimator.calculate_max_batch_size(
            k=1,
            available_vram_bytes=vram_12gib,
            vram_reserve_bytes=reserve_2gib,
            max_single_alloc_bytes=single_alloc_limit,
            t=256,
        )
        self.assertEqual(max_b, 256)
        self.assertLessEqual(alloc.edges_bytes, single_alloc_limit)
        self.assertLessEqual(alloc.total_bytes, vram_12gib - reserve_2gib)

    def test_invalid_parameters_raise(self):
        """Verify parameter bounds checking."""
        with self.assertRaises(ValueError):
            ScratchEstimator.estimate_scratch(k=0)
        with self.assertRaises(ValueError):
            ScratchEstimator.estimate_scratch(k=17)
        with self.assertRaises(ValueError):
            ScratchEstimator.estimate_scratch(k=3, b=0)
        with self.assertRaises(ValueError):
            ScratchEstimator.estimate_scratch(k=3, t=-1)


if __name__ == "__main__":
    unittest.main()
