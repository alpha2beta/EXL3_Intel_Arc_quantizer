"""Negative tests and perturbation detection harness.

Implements P0-28 requirement:
- Prove the test harness actively detects deliberately perturbed values, bit errors,
  corrupted codebook levels, illegal state transitions, and invalid datatypes.
"""

import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exl3_quant.bitpack import SubByteBitPacker
from exl3_quant.reference_encoder import (
    Mul1Codebook,
    TrellisStateSpace,
    ReferenceViterbiEncoder,
)
from exl3_quant.scratch_estimator import ScratchEstimator, BYTES_PER_MIB


class TestNegativePerturbations(unittest.TestCase):

    def test_bit_corruption_detected(self):
        """P0-28: Deliberately flip 1 bit in packed container; assert harness detects mismatch."""
        original = np.array([1, 2, 3, 0, 2, 1, 3, 2], dtype=np.int64)
        packed = SubByteBitPacker.pack(original, bits_per_value=2)

        # Deliberately flip bit 0 in the first packed uint32 word
        corrupted_packed = packed.copy()
        corrupted_packed[0] ^= 0x1

        unpacked = SubByteBitPacker.unpack(corrupted_packed, bits_per_value=2, total_elements=len(original))

        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(unpacked, original)

    def test_codebook_level_mutation_detected(self):
        """P0-28: Mutate codebook decoding parameter; assert harness detects reconstruction error."""
        codebook_normal = Mul1Codebook(bits=2, scale=1.0)
        codebook_mutated = Mul1Codebook(bits=2, scale=1.05)  # 5% scale distortion

        symbols = np.array([0, 1, 2, 3])
        levels_normal = codebook_normal.decode_array(symbols)
        levels_mutated = codebook_mutated.decode_array(symbols)

        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(levels_normal, levels_mutated, rtol=1e-5, atol=1e-5)

    def test_illegal_trellis_transition_detected(self):
        """P0-28: Deliberately swap adjacent path states; assert invalid transition detection."""
        encoder = ReferenceViterbiEncoder(num_states=16, bits_per_symbol=2, scale=1.0)
        weights = np.array([0.5, -0.2, 1.1, -1.4, 0.0])
        result = encoder.encode(weights)

        # Corrupt path_states by introducing an invalid transition
        corrupted_path = result.path_states.copy()
        # Invert top bits of state at step 2 to make transition illegal
        corrupted_path[2] ^= 0b1100

        mask = encoder.num_states - 1
        is_legal = True
        for t in range(len(weights)):
            s_prev = corrupted_path[t]
            s_next = corrupted_path[t + 1]
            sym = result.indices[t]
            expected_next = ((s_prev << encoder.bits) | sym) & mask
            if s_next != expected_next:
                is_legal = False
                break

        self.assertFalse(is_legal, "Harness must detect illegal trellis transitions")

    def test_non_integer_float_rejection(self):
        """Verify that floating-point numbers with non-zero fractional parts are rejected."""
        with self.assertRaises(ValueError):
            SubByteBitPacker.pack([2.7, 1.0, 0.0], bits_per_value=2)

    def test_insufficient_memory_rejection(self):
        """Verify that calculate_max_batch_size fails cleanly if even B=1 cannot fit."""
        # 1 MiB available, 2 MiB reserve -> usable VRAM is negative
        with self.assertRaises(ValueError):
            ScratchEstimator.calculate_max_batch_size(
                k=1,
                available_vram_bytes=1 * BYTES_PER_MIB,
                vram_reserve_bytes=2 * BYTES_PER_MIB,
            )

        # Usable VRAM is only 1 KiB, while K=1, B=1 requires ~16 MiB
        with self.assertRaises(ValueError):
            ScratchEstimator.calculate_max_batch_size(
                k=1,
                available_vram_bytes=1024 * 1024 + 1024,
                vram_reserve_bytes=1024 * 1024,
            )


if __name__ == "__main__":
    unittest.main()
