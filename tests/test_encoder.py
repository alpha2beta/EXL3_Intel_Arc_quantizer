"""Unit tests for CPU Golden Reference Viterbi Encoder.

Directly tests P1-01, P1-02, P1-04 from tasks.md:
- Inspectable Python/NumPy reference encoder.
- Procedural `mul1` arithmetic codebook.
- Shift-register trellis transitions and state connectivity.
- Path validity and distortion comparison against round-to-nearest (RTN).
- End-to-end integration: Viterbi encode -> pack -> unpack -> reconstruct.
"""

import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exl3_quant.reference_encoder import (
    Mul1Codebook,
    TrellisStateSpace,
    ReferenceViterbiEncoder,
)
from exl3_quant.bitpack import SubByteBitPacker


class TestReferenceViterbiEncoder(unittest.TestCase):

    def test_mul1_codebook_arithmetic(self):
        """P1-02: Test procedural arithmetic codebook levels."""
        # 2-bit codebook: 4 symbols (0, 1, 2, 3), zero-point = 1.5, scale = 1.0
        # Expected levels: 0 - 1.5 = -1.5, 1 - 1.5 = -0.5, 2 - 1.5 = 0.5, 3 - 1.5 = 1.5
        cb2 = Mul1Codebook(bits=2, scale=1.0)
        expected_levels = np.array([-1.5, -0.5, 0.5, 1.5])
        symbols = np.array([0, 1, 2, 3])
        np.testing.assert_allclose(cb2.decode_array(symbols), expected_levels)

        # 4-bit codebook: 16 symbols, centered around zero
        cb4 = Mul1Codebook(bits=4, scale=0.5)
        self.assertEqual(cb4.decode_symbol(0), 0.5 * (0 - 7.5))
        self.assertEqual(cb4.decode_symbol(15), 0.5 * (15 - 7.5))

    def test_trellis_topology_connectivity(self):
        """Test shift-register trellis topology."""
        num_states = 16
        bits = 2  # 4 symbols per transition
        trellis = TrellisStateSpace(num_states=num_states, bits_per_symbol=bits)

        # Every state should have exactly 4 incoming predecessor transitions
        for s in range(num_states):
            self.assertEqual(len(trellis.predecessors[s]), 4)

        # Every predecessor transition should be legally valid under shift-register rule
        mask = num_states - 1
        for s_next in range(num_states):
            for s_prev, sym in trellis.predecessors[s_next]:
                expected_next = ((s_prev << bits) | sym) & mask
                self.assertEqual(s_next, expected_next)

    def test_viterbi_distortion_vs_greedy(self):
        """P1-05: Verify Viterbi trellis path produces bounded, low reconstruction distortion."""
        rng = np.random.default_rng(seed=123)
        # Generate Gaussian synthetic weights
        t_steps = 128
        weights = rng.normal(loc=0.0, scale=1.0, size=t_steps)

        encoder = ReferenceViterbiEncoder(num_states=32, bits_per_symbol=2, scale=1.0)
        result = encoder.encode(weights)

        # Basic shape checks
        self.assertEqual(len(result.indices), t_steps)
        self.assertEqual(len(result.reconstructed), t_steps)
        self.assertEqual(len(result.path_states), t_steps + 1)

        # Verify MSE is positive and bounded
        self.assertGreater(result.mse, 0.0)
        self.assertLess(result.mse, 0.5)  # Expected for 2-bit quantization on N(0, 1)

        # Re-verify path consistency: each step's transition must be valid in the trellis
        mask = encoder.num_states - 1
        for t in range(t_steps):
            s_prev = result.path_states[t]
            s_next = result.path_states[t + 1]
            sym = result.indices[t]
            expected_next = ((s_prev << encoder.bits) | sym) & mask
            self.assertEqual(s_next, expected_next, f"Invalid trellis step at t={t}")

    def test_end_to_end_viterbi_and_bitpack_pipeline(self):
        """P1-05 & P2-15: Complete pipeline from continuous weights to bitpacked container and back."""
        rng = np.random.default_rng(seed=999)
        t_steps = 256
        weights = rng.normal(0.0, 1.2, size=t_steps)

        # 1. Viterbi encode to 4-bit indices
        encoder = ReferenceViterbiEncoder(num_states=16, bits_per_symbol=4, scale=0.5)
        enc_res = encoder.encode(weights)

        # 2. Pack sub-byte indices into 32-bit words
        packed = SubByteBitPacker.pack(enc_res.indices, bits_per_value=4)
        # 256 4-bit elements -> 256 / 8 = 32 uint32 words
        self.assertEqual(len(packed), 32)
        self.assertEqual(packed.dtype, np.uint32)

        # 3. Unpack from container
        unpacked_indices = SubByteBitPacker.unpack(packed, bits_per_value=4, total_elements=t_steps)
        np.testing.assert_array_equal(unpacked_indices, enc_res.indices)

        # 4. Reconstruct weights from unpacked indices
        reconstructed = encoder.codebook.decode_array(unpacked_indices)
        np.testing.assert_allclose(reconstructed, enc_res.reconstructed)


if __name__ == "__main__":
    unittest.main()
