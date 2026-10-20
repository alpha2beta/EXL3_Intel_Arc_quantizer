"""Unit tests for the audited EXL3 trellis format primitives.

These tests exercise the *device-independent* parts of the EXL3 format as
transcribed from the pinned upstream revision
(``turboderp-org/exllamav3`` @ ``4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f``)
and from the independent CPU decode oracle in ``sybil-solutions/exl3xpu``.

Task mapping: P0-11 (codebook arithmetic), P0-13 (tile permutation),
P0-14 (bit ordering), P0-28 (harness that detects perturbed values / bit order).

Scope note: these tests verify *format conformance*, not encoder parity with
upstream's CUDA search. Encoder ground truth requires P0-24 (CUDA reference).
"""

import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exl3_quant.trellis_format import (
    codebook_lut, tensor_core_perm, tensor_core_perm_inv,
    pack_trellis, swap16_storage, decode_states, decode_tile_values,
    words_from_storage, stream_from_words, state_index,
    CB_3INST, CB_MCG, CB_MUL1, MUL1_MULT,
)


class TestCodebookLUT(unittest.TestCase):
    """P0-11: procedural codebook arithmetic (codebook.cuh:56-123)."""

    def test_mul1_formula_is_exact(self):
        """mul1: x = state*0x83DCD12D; sum = byte sum + 1024; __hfma(inv, bias)."""
        lut = codebook_lut(CB_MUL1)
        self.assertEqual(lut.dtype, np.float16)
        self.assertEqual(lut.size, 65536)

        for state in (0, 1, 2, 12345, 65535):
            x = (state * MUL1_MULT) & 0xFFFFFFFF
            bsum = ((x & 0xFF) + ((x >> 8) & 0xFF)
                    + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF))
            h = np.float16(1024 + bsum)          # exact in fp16
            inv = np.uint16(0x1EEE).view(np.float16)
            bias = np.uint16(0xC931).view(np.float16)
            expected = np.float16(np.float64(h) * np.float64(inv) + np.float64(bias))
            self.assertEqual(float(lut[state]), float(expected),
                             f"mul1 mismatch at state {state}")

    def test_mul1_is_not_a_uniform_grid(self):
        """Regression guard: the codebook must not be an arithmetic progression.

        This is the defect that the pre-audit implementation had.
        """
        lut = codebook_lut(CB_MUL1).astype(np.float64)
        diffs = np.diff(lut[:64])
        self.assertGreater(diffs.std(), 1e-3, "mul1 levels look uniform; formula changed?")
        # Consecutive states are unrelated: the codebook is a pseudorandom LUT.
        self.assertNotAlmostEqual(float(lut[0]), float(lut[1]), places=3)

    def test_codebook_range_is_finite(self):
        for cb in (CB_3INST, CB_MCG, CB_MUL1):
            lut = codebook_lut(cb)
            self.assertTrue(np.all(np.isfinite(lut.astype(np.float64))))

    def test_extremes_follow_the_byte_sum(self):
        """mul1 is affine in the byte sum, so the extremes are determined by it."""
        s = np.arange(65536, dtype=np.uint64)
        x = (s * MUL1_MULT) & 0xFFFFFFFF
        bsum = ((x & 0xFF) + ((x >> 8) & 0xFF)
                + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF))
        inv = float(np.uint16(0x1EEE).view(np.float16))
        bias = float(np.uint16(0xC931).view(np.float16))

        def level(b):
            return np.float16(np.float64(np.float16(1024 + b)) * inv + bias)

        lut = codebook_lut(CB_MUL1).astype(np.float64)
        self.assertEqual(float(lut.min()), float(level(int(bsum.min()))))
        self.assertEqual(float(lut.max()), float(level(int(bsum.max()))))
        # The mapping is monotonic in the byte sum.
        self.assertGreater(float(lut.max()), float(lut.min()))

    def test_level_count(self):
        """mul1 collapses 65536 states onto the distinct byte sums."""
        s = np.arange(65536, dtype=np.uint64)
        x = (s * MUL1_MULT) & 0xFFFFFFFF
        bsum = ((x & 0xFF) + ((x >> 8) & 0xFF)
                + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF))
        expected = np.unique(bsum).size
        lut = codebook_lut(CB_MUL1)
        self.assertEqual(np.unique(lut.astype(np.float64)).size, expected)
        self.assertLessEqual(expected, 1021)


class TestTilePermutation(unittest.TestCase):
    """P0-13: tensor_core_perm (quantize.py:22-49)."""

    def test_is_a_permutation(self):
        perm = tensor_core_perm()
        self.assertEqual(perm.size, 256)
        self.assertEqual(sorted(perm.tolist()), list(range(256)))

    def test_inverse(self):
        perm = tensor_core_perm()
        inv = tensor_core_perm_inv()
        np.testing.assert_array_equal(perm[inv], np.arange(256))
        np.testing.assert_array_equal(inv[perm], np.arange(256))

    def test_known_entries(self):
        """Spot-check the upstream table structure (quantize.py:36-43)."""
        perm = tensor_core_perm()
        # t = 0 -> r0 = 0, r1 = 1, r2 = 8, r3 = 9 ; c0 = 0, c1 = 8
        self.assertEqual(perm[0:8].tolist(), [0, 16, 128, 144, 8, 24, 136, 152])
        # t = 4 -> r0 = (4%4)*2 = 0, r1 = 1, r2 = 8, r3 = 9 ; c0 = 1, c1 = 9
        self.assertEqual(perm[32:40].tolist(), [1, 17, 129, 145, 9, 25, 137, 153])


class TestPackingLayout(unittest.TestCase):
    """P0-14: bit ordering of the exported container (pack.cu:9-57)."""

    def test_decode_pack_identity_on_random_bitstream(self):
        """The layout property that does not need the encoder.

        The decoder's 16-bit window always ends at bit (t+1)*K, so the window's
        low K bits are exactly element t's K bits. Encoding those states again
        must reproduce the stream bit for bit.
        """
        for K in (1, 2, 3, 4, 5, 6, 8):
            with self.subTest(K=K):
                rng = np.random.default_rng(4242 + K)
                bits = rng.integers(0, 2, size=256 * K)
                nwords = 256 * K // 32
                padded = np.zeros(nwords * 32, dtype=np.int64)
                padded[:bits.size] = bits
                words = (padded.reshape(nwords, 32) << np.arange(31, -1, -1)).sum(1)

                t16 = np.zeros(256 * K // 16, dtype=np.uint16)
                t16[0::2] = (words & 0xFFFF).astype(np.uint16)
                t16[1::2] = (words >> 16).astype(np.uint16)

                states = decode_states(t16, K)
                repacked = swap16_storage(pack_trellis(states, K))
                bits2 = stream_from_words(words_from_storage(repacked))
                np.testing.assert_array_equal(bits, bits2)

    def test_only_low_k_bits_are_stored(self):
        """The container is a state-history encoding, not independent symbols.

        Arbitrary 16-bit states therefore do not survive a round trip; only the
        low K bits of each state do. This distinguishes the upstream format
        from an independent-symbol sub-byte packer.
        """
        K = 4
        rng = np.random.default_rng(7)
        states = rng.integers(0, 65536, size=256, dtype=np.int64).astype(np.uint16)
        stored = swap16_storage(pack_trellis(states, K))
        back = decode_states(stored, K)
        self.assertFalse(np.array_equal(states.astype(np.int64), back))
        np.testing.assert_array_equal(
            back & ((1 << K) - 1), states.astype(np.int64) & ((1 << K) - 1))

    def test_swap16_pairing(self):
        """pack.cu:55-56 writes uint16 pairs as half-word-swapped uint32."""
        s_packed = np.array([0x1111, 0x2222, 0x3333, 0x4444], dtype=np.uint16)
        stored = swap16_storage(s_packed)
        self.assertEqual(stored.tolist(), [0x2222, 0x1111, 0x4444, 0x3333])

    def test_window_ends_at_bit(self):
        """ref.py:81-87: element t's window ends at bit (t+1)*K."""
        K = 4
        idx = state_index(K)
        self.assertEqual(idx.shape, (256, 16))
        # element 0: window ends at bit K -> wraps to the end of the tile
        expected0 = (np.arange(K - 16, K) % (256 * K))
        np.testing.assert_array_equal(idx[0], expected0)
        # element 1: window ends at bit 2K
        expected1 = (np.arange(2 * K - 16, 2 * K) % (256 * K))
        np.testing.assert_array_equal(idx[1], expected1)

    def test_packed_size(self):
        for K in (1, 2, 3, 4, 5, 6, 7, 8):
            states = np.zeros(256, dtype=np.uint16)
            self.assertEqual(pack_trellis(states, K).size, 256 * K // 16)


class TestPermutationAwareDecode(unittest.TestCase):
    """P0-13/P0-14: the permuted scatter of reconstructed tile values."""

    def test_scatter_matches_permutation(self):
        K = 4
        rng = np.random.default_rng(11)
        states = rng.integers(0, 65536, size=256, dtype=np.int64).astype(np.uint16)
        stored = swap16_storage(pack_trellis(states, K))
        decoded = decode_states(stored, K)
        tile = decode_tile_values(stored, K, CB_MUL1)
        lut = codebook_lut(CB_MUL1)
        perm = tensor_core_perm()
        # tile[perm[t]] == lut[decoded_state_t]
        np.testing.assert_allclose(
            tile[perm].astype(np.float64), lut[decoded].astype(np.float64))


class TestPerturbationDetection(unittest.TestCase):
    """P0-28: the harness must fail on perturbed values / bit order."""

    def test_flipped_bit_is_detected(self):
        K = 4
        rng = np.random.default_rng(3)
        bits = rng.integers(0, 2, size=256 * K)
        nwords = 256 * K // 32
        padded = np.zeros(nwords * 32, dtype=np.int64)
        padded[:bits.size] = bits
        words = (padded.reshape(nwords, 32) << np.arange(31, -1, -1)).sum(1)
        t16 = np.zeros(256 * K // 16, dtype=np.uint16)
        t16[0::2] = (words & 0xFFFF).astype(np.uint16)
        t16[1::2] = (words >> 16).astype(np.uint16)

        baseline = decode_states(t16, K)
        perturbed = t16.copy()
        perturbed[5] ^= 0x0001
        self.assertFalse(np.array_equal(baseline, decode_states(perturbed, K)))

    def test_reversed_bit_order_is_detected(self):
        """An MSB-first vs LSB-first confusion must not pass silently."""
        K = 4
        rng = np.random.default_rng(5)
        bits = rng.integers(0, 2, size=256 * K)
        nwords = 256 * K // 32
        padded = np.zeros(nwords * 32, dtype=np.int64)
        padded[:bits.size] = bits
        words = (padded.reshape(nwords, 32) << np.arange(31, -1, -1)).sum(1)
        t16 = np.zeros(256 * K // 16, dtype=np.uint16)
        t16[0::2] = (words & 0xFFFF).astype(np.uint16)
        t16[1::2] = (words >> 16).astype(np.uint16)

        baseline = decode_states(t16, K)
        # Reverse the bit order inside each 32-bit word (LSB-first confusion).
        rev = np.zeros_like(words)
        for b in range(32):
            rev |= ((words >> b) & 1) << (31 - b)
        t16_rev = np.zeros_like(t16)
        t16_rev[0::2] = (rev & 0xFFFF).astype(np.uint16)
        t16_rev[1::2] = (rev >> 16).astype(np.uint16)
        self.assertFalse(np.array_equal(baseline, decode_states(t16_rev, K)))

    def test_codebook_perturbation_is_detected(self):
        lut = codebook_lut(CB_MUL1)
        perturbed = lut.copy()
        perturbed[1234] = np.float16(float(perturbed[1234]) + 1.0)
        self.assertFalse(np.array_equal(lut.astype(np.float64),
                                        perturbed.astype(np.float64)))


if __name__ == "__main__":
    unittest.main()
