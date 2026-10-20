"""EXL3 trellis format primitives, transcribed from pinned upstream source.

This module is the *audited* reference for the parts of the EXL3 format that are
device-independent. Every function below is a direct transcription of upstream code,
with the upstream file and line range given in the docstring.

Pinned upstream revision:
    https://github.com/turboderp-org/exllamav3 @ 4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f

Sources
-------
codebook : exllamav3/exllamav3_ext/quant/codebook.cuh:56-123
packing  : exllamav3/exllamav3_ext/quant/pack.cu:9-57, 176-201
perm     : exllamav3/modules/quant/exl3_lib/quantize.py:22-49
decode   : sybil-solutions/exl3xpu @ exl3xpu/ref.py:30-45, 52-99 (CPU decode oracle)

No torch dependency: everything is numpy with explicit fp16 semantics.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Codebook (codebook.cuh:56-123)
# ---------------------------------------------------------------------------

MUL1_MULT = 0x83DCD12D
MCG_MULT = 0xCBAC1FED
THREEINST_MULT = 89226354
THREEINST_ADD = 64248484

CB_3INST, CB_MCG, CB_MUL1 = 0, 1, 2


def _f16_from_bits(bits: np.ndarray) -> np.ndarray:
    """Reinterpret uint16 bit patterns as float16."""
    return np.asarray(bits, dtype=np.uint16).view(np.float16)


def codebook_lut(cb: int) -> np.ndarray:
    """fp16 value for every 16-bit trellis state.

    Mirrors ``decode_3inst<cb>`` (codebook.cuh:56-90) bit for bit. The fp16
    arithmetic is reproduced by computing the exact intermediate in float64 and
    rounding once to float16, which is what a fused multiply-add does.

    Args:
        cb: codebook selector (CB_3INST, CB_MCG, CB_MUL1).

    Returns:
        float16 array of length 65536, indexed by the 16-bit state.
    """
    if cb not in (CB_3INST, CB_MCG, CB_MUL1):
        raise ValueError(f"Unknown codebook {cb}")

    s = np.arange(65536, dtype=np.uint64)

    if cb == CB_MUL1:
        # codebook.cuh:76-89
        x = (s * MUL1_MULT) & 0xFFFFFFFF
        bsum = ((x & 0xFF) + ((x >> 8) & 0xFF)
                + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF))
        # The accumulator 0x6400 is the fp16 *bit pattern* for 1024.0; the fp16
        # encoding of 1024 + bsum is 0x6400 + bsum, so the value is exact.
        h = (1024 + bsum).astype(np.float16)
        inv = float(_f16_from_bits(np.array([0x1EEE], dtype=np.uint16))[0])
        bias = float(_f16_from_bits(np.array([0xC931], dtype=np.uint16))[0])
        val = (h.astype(np.float64) * inv + bias).astype(np.float16)
        return val

    if cb == CB_3INST:
        x = (s * THREEINST_MULT + THREEINST_ADD) & 0xFFFFFFFF
    else:
        x = (s * MCG_MULT) & 0xFFFFFFFF

    # codebook.cuh:69 / 73 -> lop3 immLut 0x6a == (a & b) ^ c
    x = ((x & 0x8FFF8FFF) ^ 0x3B603B60) & 0xFFFFFFFF
    lo = _f16_from_bits((x & 0xFFFF).astype(np.uint16)).astype(np.float64)
    hi = _f16_from_bits((x >> 16).astype(np.uint16)).astype(np.float64)
    return (lo + hi).astype(np.float16)


# ---------------------------------------------------------------------------
# Tile permutation (quantize.py:22-49)
# ---------------------------------------------------------------------------

def tensor_core_perm() -> np.ndarray:
    """The 16x16 tile permutation expected by the EXL3 tensor-core kernels.

    Upstream ``quantize.py:22-44``; identical in ``exl3xpu/ref.py:30-45``.
    Element ``t`` of the encoded tile is original tile position ``perm[t]``.
    """
    perm = [0] * 256
    for t in range(32):
        r0 = (t % 4) * 2
        r1, r2, r3 = r0 + 1, r0 + 8, r0 + 9
        c0 = t // 4
        c1 = c0 + 8
        perm[t * 8 + 0] = r0 * 16 + c0
        perm[t * 8 + 1] = r1 * 16 + c0
        perm[t * 8 + 2] = r2 * 16 + c0
        perm[t * 8 + 3] = r3 * 16 + c0
        perm[t * 8 + 4] = r0 * 16 + c1
        perm[t * 8 + 5] = r1 * 16 + c1
        perm[t * 8 + 6] = r2 * 16 + c1
        perm[t * 8 + 7] = r3 * 16 + c1
    return np.asarray(perm, dtype=np.int64)


def tensor_core_perm_inv() -> np.ndarray:
    """Inverse permutation (``quantize.py:47-49``)."""
    return np.argsort(tensor_core_perm())


# ---------------------------------------------------------------------------
# Packing (pack.cu:9-57)
# ---------------------------------------------------------------------------

def pack_trellis(states: np.ndarray, K: int) -> np.ndarray:
    """Pack 16-bit trellis states into the upstream bitstream words.

    Transcription of ``pack_trellis_kernel<K>`` (pack.cu:9-57): 16 spans of 16
    elements, the low ``K`` bits of each state appended MSB-first, one uint16
    word emitted whenever the staging buffer holds >= 16 bits.

    Args:
        states: (256,) array of 16-bit states (the encoder's ``output_indices``).
        K: bits per element.

    Returns:
        uint16 array of length ``256 * K // 16`` (pre-SWAP16 storage).
    """
    if not 1 <= K <= 8:
        raise ValueError(f"K must be in 1..8, got {K}")
    s_unpacked = np.asarray(states, dtype=np.uint16)
    if s_unpacked.size != 256:
        raise ValueError(f"Expected 256 states per tile, got {s_unpacked.size}")

    packed_size = 256 * K // 16
    s_packed = np.zeros(packed_size, dtype=np.uint16)

    spans, length = 16, 256 // 16
    for t in range(spans):
        i = length * t
        j = K * t
        k = 32
        buf = 0
        for _ in range(length):
            v = int(s_unpacked[i]) & ((1 << K) - 1)
            k -= K
            buf |= v << k
            if k <= 16:
                s_packed[j] = (buf >> 16) & 0xFFFF
                buf = (buf << 16) & 0xFFFFFFFF
                k += 16
                j += 1
            i += 1
    return s_packed


def swap16_storage(s_packed: np.ndarray) -> np.ndarray:
    """Apply the ``SWAP16`` half-word swap of ``pack.cu:55-56``.

    The upstream kernel writes the uint16 array as uint32 words with the halves
    swapped. This returns the int16 array exactly as a decoder reads it back.
    """
    s_packed = np.asarray(s_packed, dtype=np.uint16)
    words = s_packed[0::2].astype(np.uint32) | (s_packed[1::2].astype(np.uint32) << 16)
    words = ((words >> 16) | (words << 16)) & 0xFFFFFFFF
    out = np.zeros(words.size * 2, dtype=np.uint16)
    out[0::2] = (words & 0xFFFF).astype(np.uint16)
    out[1::2] = (words >> 16).astype(np.uint16)
    return out


# ---------------------------------------------------------------------------
# Decoding (exl3xpu/ref.py:81-99)
# ---------------------------------------------------------------------------

def words_from_storage(t16: np.ndarray) -> np.ndarray:
    """uint16 storage -> uint32 words (``ref.py:93``)."""
    t16 = np.asarray(t16, dtype=np.uint16)
    return t16[0::2].astype(np.uint64) | (t16[1::2].astype(np.uint64) << 16)


def stream_from_words(words: np.ndarray) -> np.ndarray:
    """uint32 words -> MSB-first bit stream (``ref.py:94-95``)."""
    shifts = np.arange(31, -1, -1, dtype=np.uint64)
    return ((np.asarray(words, dtype=np.uint64)[:, None] >> shifts[None, :]) & 1).reshape(-1)


def state_index(K: int) -> np.ndarray:
    """Bit positions of each element's 16-bit window (``ref.py:81-87``).

    Element ``t`` occupies the 16-bit window that *ends* at bit ``(t+1)*K``,
    circular inside the tile.
    """
    nbits = 256 * K
    t = np.arange(256)
    b0 = (t * K + K - 16 + nbits) % nbits
    return (b0[:, None] + np.arange(16)[None, :]) % nbits


def decode_states(t16: np.ndarray, K: int) -> np.ndarray:
    """Packed tile -> the 256 reconstructed 16-bit states.

    Transcription of ``exl3xpu/ref.py:90-99`` (``decode_states``).
    """
    bits = stream_from_words(words_from_storage(t16))
    win = bits[state_index(K)]
    weights = (1 << np.arange(15, -1, -1))
    return (win * weights[None, :]).sum(-1).astype(np.int64)


def decode_tile_values(t16: np.ndarray, K: int, cb: int = CB_MUL1) -> np.ndarray:
    """Packed tile -> the 256 reconstructed codebook values, in permuted order.

    Mirrors ``exl3xpu/ref.py:102-115`` (``reconstruct_inner``) without the
    Hadamard/sign-vector stages: values are looked up by state and scattered
    back through ``tensor_core_perm``.
    """
    states = decode_states(t16, K)
    vals = codebook_lut(cb)[states.astype(np.int64)]
    tile = np.empty(256, dtype=np.float16)
    tile[tensor_core_perm()] = vals
    return tile
