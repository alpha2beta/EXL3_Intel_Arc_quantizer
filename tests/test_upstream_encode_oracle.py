"""Upstream-derived encode oracle: run upstream's own CUDA test recipe on CPU.

Upstream's ``tests/test_quant_fn.py`` (pinned revision
``4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f``) encodes seeded random tiles with
the CUDA kernel and asserts two properties:

1. **Tail-biting ring invariant** — for the encoded 16-bit states,

       (states[0]   & 0xFFFF) >> K  ==  (states[255] & 0xFFFF) & ((1 << (16-K)) - 1)

   i.e. the wrap-around predecessor of the first element is the low bits of the
   last element's state.
2. **Mean squared error ceiling** — ``mse(in_tile/scale, out_tile/scale)`` must
   stay below a per-K threshold that upstream derived from the CUDA kernel
   (``max_mse_per_K``).

Those thresholds are a *quantitative* artifact of upstream's CUDA encoder, so
reproducing them on the host CPU is a real tier-4 cross-check that does not
require an NVIDIA GPU. It constrains the search quality (not the exact path),
which is exactly the acceptance posture the project specification prescribes:
path-cost parity rather than identical state indices on tied branches.

Upstream's recipe:

    torch.manual_seed(0)
    in_tile = torch.randn((batch_size, 256), device=device) * scale

    cb tuples -> (mcg, mul1, scale):
        (False, False, 1.24371088)  3INST, scale = codebook_scale
        (True,  False, 1.24371088)  mcg,   scale = codebook_scale
        (False, True,  1.0)         mul1,  scale = 1.0

The tiles are fed to ``quantize_tiles`` in the kernel's element order (upstream
applies ``tensor_core_perm`` only in the conversion path), so the reference
encoder is called with ``pre_permute=False``.
"""

import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exl3_quant.reference_encoder import ReferenceTrellisEncoder
from exl3_quant.trellis_format import CB_3INST, CB_MCG, CB_MUL1

try:
    import torch
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


# Upstream: tests/test_quant_fn.py:30-39
MAX_MSE_PER_K = {
    1: 0.3,
    2: 0.1,
    3: 0.1,
    4: 0.1,
    5: 0.1,
    6: 0.07,
    7: 0.05,
    8: 0.04,
}

# Upstream: tests/test_quant_fn.py:64 -> (mcg, mul1, scale)
CODEBOOK_CASES = (
    (CB_3INST, 1.24371088),
    (CB_MCG, 1.24371088),
    (CB_MUL1, 1.0),
)


def upstream_tiles(batch_size, scale, seed=0):
    """Reproduce upstream's seeded input tiles.

    The exact values depend on torch's RNG, so torch is required to match
    upstream bit for bit. The *distribution* is what the MSE ceiling measures.
    """
    torch.manual_seed(seed)
    return (torch.randn((batch_size, 256)) * scale).numpy().astype(np.float32)


@unittest.skipUnless(HAVE_TORCH, "torch is required to reproduce upstream's seeded tiles")
class TestUpstreamEncodeOracle(unittest.TestCase):
    """Tier-4 cross-check against upstream's CUDA-derived thresholds."""

    BATCH = 1

    def _encode_batch(self, tiles, K, cb):
        enc = ReferenceTrellisEncoder(K=K, cb=cb)
        indices = []
        reconstructed = []
        for tile in tiles:
            result = enc.encode_tile(tile, pre_permute=False)
            indices.append(np.asarray(result.output_indices, dtype=np.uint16))
            reconstructed.append(np.asarray(result.reconstructed, dtype=np.float32))
        return np.stack(indices), np.stack(reconstructed)

    def test_tail_biting_invariant(self):
        """states[0] >> K must equal the low (16-K) bits of states[255]."""
        for cb, scale in CODEBOOK_CASES:
            for K in (2, 3, 4):
                with self.subTest(cb=cb, K=K):
                    tiles = upstream_tiles(self.BATCH, scale)
                    indices, _ = self._encode_batch(tiles, K, cb)
                    first = (indices[:, 0].astype(np.int64) & 0xFFFF) >> K
                    last = indices[:, 255].astype(np.int64) & ((1 << (16 - K)) - 1)
                    np.testing.assert_array_equal(first, last)

    def test_mse_ceiling_per_K(self):
        """Reconstruction MSE must stay below upstream's per-K ceiling."""
        for cb, scale in CODEBOOK_CASES:
            for K in (2, 3, 4):
                with self.subTest(cb=cb, K=K):
                    tiles = upstream_tiles(self.BATCH, scale)
                    _, recon = self._encode_batch(tiles, K, cb)
                    mse = float(np.mean(((tiles - recon) / scale) ** 2))
                    self.assertLess(mse, MAX_MSE_PER_K[K],
                                    f"cb={cb} K={K} mse={mse:.6f} "
                                    f">= upstream ceiling {MAX_MSE_PER_K[K]}")


class TestUpstreamStateLayout(unittest.TestCase):
    """Upstream's own 'ideal encoding' construction pins the state layout.

    ``tests/test_quant_fn.py:98-112`` builds a random *valid, tail-biting*
    encoding by placing each element's K-bit symbol in the **low** K bits of
    its state and shifting the previous elements' symbols up. That is the
    layout the container relies on, and it is verifiable without a GPU.
    """

    def test_symbol_occupies_the_low_k_bits(self):
        K = 4
        rng = np.random.default_rng(0)
        symbols = rng.integers(0, 1 << K, size=256, dtype=np.int64)
        states = symbols.copy()
        for shift in range(1, int(np.ceil(16 / K))):
            prev = np.roll(symbols, shift)
            states |= (prev << (K * shift)) & 0xFFFF

        # Low K bits carry the element's own symbol ...
        np.testing.assert_array_equal(states & ((1 << K) - 1), symbols)
        # ... and the history is the preceding symbols, shifted up.
        np.testing.assert_array_equal((states >> K) & ((1 << K) - 1),
                                      np.roll(symbols, 1))

    def test_packed_container_carries_those_low_bits(self):
        """The container stores the low K bits of each state, nothing more."""
        from exl3_quant.trellis_format import (decode_states, pack_trellis,
                                                swap16_storage)
        K = 4
        rng = np.random.default_rng(1)
        states = rng.integers(0, 65536, size=256, dtype=np.uint16)
        packed = pack_trellis(states, K)
        self.assertEqual(packed.size, 256 * K // 16)
        # The container carries exactly the low K bits of each state.
        back = decode_states(swap16_storage(packed), K)
        np.testing.assert_array_equal(back & ((1 << K) - 1),
                                      states.astype(np.int64) & ((1 << K) - 1))


if __name__ == "__main__":
    unittest.main()
