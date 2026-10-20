"""Unit tests for Sub-Byte Bitpacker.

Directly tests P2-13, P2-14, P2-15 from tasks.md:
- Exact round-trip verification: unpack(pack(x)) == x across 2, 3, 4, 8 bits.
- Boundary values: zeros, max values, random seed arrays, non-multiples of word size.
- Rejection of out-of-range and negative values.
"""

import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exl3_quant.bitpack import SubByteBitPacker


class TestSubByteBitPacker(unittest.TestCase):

    def test_round_trip_all_supported_bit_widths(self):
        """P2-15: Test exact round trips for all supported bit widths across various lengths."""
        rng = np.random.default_rng(seed=42)
        test_lengths = [1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 100, 256, 1023]

        for bits in [2, 3, 4, 8]:
            max_val = (1 << bits) - 1
            for length in test_lengths:
                with self.subTest(bits=bits, length=length):
                    # Random integers in [0, 2^bits - 1]
                    original = rng.integers(0, max_val + 1, size=length, dtype=np.int64)

                    packed = SubByteBitPacker.pack(original, bits_per_value=bits)
                    self.assertEqual(packed.dtype, np.uint32)

                    unpacked = SubByteBitPacker.unpack(
                        packed, bits_per_value=bits, total_elements=length
                    )

                    np.testing.assert_array_equal(
                        unpacked,
                        original,
                        err_msg=f"Mismatch for {bits}-bit round trip with length {length}",
                    )

    def test_all_zeros_and_all_max_values(self):
        """Test boundary vectors: all zeros and all 1s (maximum discrete value)."""
        length = 64
        for bits in [2, 3, 4, 8]:
            max_val = (1 << bits) - 1

            # All zeros
            zeros = np.zeros(length, dtype=np.int64)
            packed_zeros = SubByteBitPacker.pack(zeros, bits)
            unpacked_zeros = SubByteBitPacker.unpack(packed_zeros, bits, length)
            np.testing.assert_array_equal(unpacked_zeros, zeros)

            # All max values
            all_max = np.full(length, max_val, dtype=np.int64)
            packed_max = SubByteBitPacker.pack(all_max, bits)
            unpacked_max = SubByteBitPacker.unpack(packed_max, bits, length)
            np.testing.assert_array_equal(unpacked_max, all_max)

    def test_empty_array(self):
        """Test packing and unpacking an empty array."""
        for bits in [2, 3, 4, 8]:
            empty = np.empty(0, dtype=np.int64)
            packed = SubByteBitPacker.pack(empty, bits)
            self.assertEqual(packed.size, 0)
            unpacked = SubByteBitPacker.unpack(packed, bits, 0)
            self.assertEqual(unpacked.size, 0)

    def test_out_of_bounds_rejection(self):
        """P2-15: Test that negative or out-of-range values raise ValueError."""
        # Value exceeds max for 4-bit (max is 15)
        with self.assertRaises(ValueError):
            SubByteBitPacker.pack(np.array([0, 5, 16]), bits_per_value=4)

        # Negative value
        with self.assertRaises(ValueError):
            SubByteBitPacker.pack(np.array([0, -1, 3]), bits_per_value=2)

        # Unsupported bit width
        with self.assertRaises(ValueError):
            SubByteBitPacker.pack(np.array([0, 1]), bits_per_value=5)


if __name__ == "__main__":
    unittest.main()
