"""EXL3 Intel Arc Quantizer: Sub-Byte Bitpacking and Unpacking.

Generic CPU sub-byte bitpacker (unverified reference, pending upstream format audit P0-14):
- Sub-byte integer packing (2, 3, 4, 8 bits) into contiguous 32-bit unsigned integers.
- Exact bit-level round-trip verification (unpack(pack(x)) == x).
- Bounds checking and strict non-integer input rejection.
"""

from typing import List, Sequence, Union
import numpy as np


class SubByteBitPacker:
    """Packs and unpacks sub-byte quantized integer indices into 32-bit unsigned words."""

    VALID_BIT_WIDTHS = {2, 3, 4, 8}

    @classmethod
    def validate_indices(cls, indices: np.ndarray, bits_per_value: int) -> None:
        """Validates that all input indices are integers within [0, 2^bits - 1]."""
        if bits_per_value not in cls.VALID_BIT_WIDTHS:
            raise ValueError(
                f"Unsupported bit width: {bits_per_value}. Supported: {sorted(cls.VALID_BIT_WIDTHS)}"
            )
        if indices.size == 0:
            return

        max_val = (1 << bits_per_value) - 1
        if np.any(indices < 0) or np.any(indices > max_val):
            raise ValueError(
                f"All indices for {bits_per_value}-bit packing must be in range [0, {max_val}]."
            )

    @classmethod
    def pack(cls, indices: Union[Sequence[int], np.ndarray], bits_per_value: int) -> np.ndarray:
        """Pack an array of integers into uint32 packed words.
        
        Args:
            indices: 1D array or sequence of non-negative integers.
            bits_per_value: Number of bits per quantized index (2, 3, 4, 8).
            
        Returns:
            1D numpy array of dtype np.uint32.
        """
        raw_arr = np.asarray(indices)
        if raw_arr.ndim != 1:
            raw_arr = raw_arr.ravel()

        if np.issubdtype(raw_arr.dtype, np.floating):
            if not np.all(np.equal(raw_arr, np.floor(raw_arr))):
                raise ValueError("Indices must be exact integer values, got non-integer floats.")

        arr = raw_arr.astype(np.int64)
        if arr.ndim != 1:
            arr = arr.ravel()
        cls.validate_indices(arr, bits_per_value)

        total_elements = arr.size
        if total_elements == 0:
            return np.empty(0, dtype=np.uint32)

        if bits_per_value in (2, 4, 8):
            elements_per_word = 32 // bits_per_value
            mask = (1 << bits_per_value) - 1
            num_words = (total_elements + elements_per_word - 1) // elements_per_word
            words = [0] * num_words

            for idx, val in enumerate(arr):
                word_idx = idx // elements_per_word
                sub_idx = idx % elements_per_word
                shift = sub_idx * bits_per_value
                words[word_idx] |= (int(val) & mask) << shift

            return np.array(words, dtype=np.int64).astype(np.uint32)

        elif bits_per_value == 3:
            total_bits = total_elements * 3
            num_words = (total_bits + 31) // 32
            words = [0] * num_words

            for idx, val in enumerate(arr):
                bit_pos = idx * 3
                word_idx = bit_pos // 32
                bit_offset = bit_pos % 32
                mask = 0x7

                words[word_idx] |= (int(val) & mask) << bit_offset

                if bit_offset > 29 and (word_idx + 1) < num_words:
                    bits_in_first = 32 - bit_offset
                    bits_in_second = 3 - bits_in_first
                    overflow_val = (int(val) >> bits_in_first) & ((1 << bits_in_second) - 1)
                    words[word_idx + 1] |= overflow_val

            return np.array(words, dtype=np.int64).astype(np.uint32)

        raise ValueError(f"Unhandled bit width: {bits_per_value}")

    @classmethod
    def unpack(
        cls,
        packed: np.ndarray,
        bits_per_value: int,
        total_elements: int,
    ) -> np.ndarray:
        """Unpack uint32 packed words back into original integer indices.
        
        Args:
            packed: 1D numpy array of uint32 words.
            bits_per_value: Number of bits per quantized index (2, 3, 4, 8).
            total_elements: Exact number of elements to reconstruct.
            
        Returns:
            1D numpy array of dtype np.int64 with length total_elements.
        """
        if bits_per_value not in cls.VALID_BIT_WIDTHS:
            raise ValueError(f"Unsupported bit width: {bits_per_value}")
        if total_elements == 0:
            return np.empty(0, dtype=np.int64)

        packed = np.asarray(packed, dtype=np.uint32)
        unpacked = np.zeros(total_elements, dtype=np.int64)

        if bits_per_value in (2, 4, 8):
            elements_per_word = 32 // bits_per_value
            mask = (1 << bits_per_value) - 1

            for idx in range(total_elements):
                word_idx = idx // elements_per_word
                sub_idx = idx % elements_per_word
                shift = sub_idx * bits_per_value
                unpacked[idx] = (int(packed[word_idx]) >> shift) & mask

            return unpacked

        elif bits_per_value == 3:
            mask = 0x7
            num_words = len(packed)
            for idx in range(total_elements):
                bit_pos = idx * 3
                word_idx = bit_pos // 32
                bit_offset = bit_pos % 32

                val = (int(packed[word_idx]) >> bit_offset) & mask

                if bit_offset > 29 and (word_idx + 1) < num_words:
                    bits_in_first = 32 - bit_offset
                    bits_in_second = 3 - bits_in_first
                    overflow_val = (int(packed[word_idx + 1]) & ((1 << bits_in_second) - 1)) << bits_in_first
                    val = (val & ((1 << bits_in_first) - 1)) | overflow_val

                unpacked[idx] = val

            return unpacked

        raise ValueError(f"Unhandled bit width: {bits_per_value}")
