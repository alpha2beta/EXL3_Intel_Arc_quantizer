"""Test bit-exact parity against the downstream XPU runtime decoder oracle (exl3xpu.ref).

Directly validates H5 from review-2026-10-20T18:30:21+02:00:
- Verifies that the transcribed format primitives in exl3_quant.trellis_format match
  the independent decoder oracle in exl3xpu.ref bit-for-bit on live execution.
- Verifies tensor_core_perm permutation array identity.
- Verifies codebook_lut(cb) identity across all 65,536 states for CB_3INST, CB_MCG, CB_MUL1.
- Verifies decode_states(packed, K) across all supported bitrates K in {1, 2, 3, 4, 5, 6, 8}.
- Verifies end-to-end decode parity: tiles encoded with ReferenceTrellisEncoder decode
  identically through exl3xpu.ref.decode_states and reconstruct_inner.
"""

import unittest
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Insert adjacent exl3xpu repository to import the live decoder oracle
EXL3XPU_PATH = REPO_ROOT.parent / "exl3xpu"
if EXL3XPU_PATH.exists() and str(EXL3XPU_PATH) not in sys.path:
    sys.path.insert(0, str(EXL3XPU_PATH))

try:
    import exl3xpu.ref as oracle_ref
    ORACLE_AVAILABLE = True
except Exception:
    oracle_ref = None
    ORACLE_AVAILABLE = False

from exl3_quant import trellis_format
from exl3_quant.reference_encoder import ReferenceTrellisEncoder


@unittest.skipUnless(ORACLE_AVAILABLE, "exl3xpu.ref oracle not available")
class TestOracleParity(unittest.TestCase):

    def test_tensor_core_perm_oracle_identity(self):
        """Verify tensor_core_perm matches exl3xpu.ref bit for bit."""
        perm_local = trellis_format.tensor_core_perm()
        perm_oracle = oracle_ref.tensor_core_perm()
        self.assertEqual(list(perm_local), perm_oracle)

    def test_codebook_lut_oracle_identity(self):
        """Verify codebook LUT values match exl3xpu.ref for all 3 codebooks across all 65536 states."""
        for cb in (oracle_ref.CB_3INST, oracle_ref.CB_MCG, oracle_ref.CB_MUL1):
            lut_local = trellis_format.codebook_lut(cb)
            lut_oracle = oracle_ref.codebook_lut(cb)
            np.testing.assert_array_equal(
                lut_local,
                lut_oracle.numpy(),
                err_msg=f"Codebook LUT mismatch against exl3xpu.ref for cb={cb}",
            )

    def test_decode_states_oracle_identity(self):
        """Verify decode_states reproduces exl3xpu.ref bit for bit across all K in {1..8}."""
        rng = np.random.default_rng(seed=777)
        for K in (1, 2, 3, 4, 5, 6, 8):
            num_words = 16 * K // 2
            words = rng.integers(0, 0xFFFFFFFF, size=num_words, dtype=np.uint32)
            t16 = np.zeros(16 * K, dtype=np.uint16)
            t16[0::2] = (words & 0xFFFF).astype(np.uint16)
            t16[1::2] = (words >> 16).astype(np.uint16)

            states_local = trellis_format.decode_states(t16, K)
            states_oracle = oracle_ref.decode_states(torch.from_numpy(t16), K).numpy()

            np.testing.assert_array_equal(
                states_local,
                states_oracle,
                err_msg=f"decode_states mismatch against exl3xpu.ref for K={K}",
            )

    def test_encoder_output_oracle_reconstruction(self):
        """Verify that ReferenceTrellisEncoder output decodes through exl3xpu.ref identically."""
        rng = np.random.default_rng(seed=888)
        weights = rng.normal(0.0, 1.2, size=256).astype(np.float32)

        for K in (2, 3, 4):
            encoder = ReferenceTrellisEncoder(K=K, cb=trellis_format.CB_MUL1)
            res = encoder.encode_tile(weights, pre_permute=True)

            # Pass packed container to exl3xpu.ref.decode_states
            torch_packed = torch.from_numpy(res.packed.view(np.int16))
            oracle_states = oracle_ref.decode_states(torch_packed, K).numpy()
            np.testing.assert_array_equal(
                oracle_states,
                res.output_indices,
                err_msg=f"Oracle decode_states failed to recover encoder indices for K={K}",
            )

            # Reconstruct through oracle codebook and tensor_core_perm
            oracle_lut = oracle_ref.codebook_lut(oracle_ref.CB_MUL1).numpy()
            oracle_vals = oracle_lut[oracle_states]
            perm = oracle_ref.tensor_core_perm()
            oracle_tile = np.zeros(256, dtype=np.float16)
            oracle_tile[perm] = oracle_vals

            np.testing.assert_array_equal(
                oracle_tile,
                res.reconstructed,
                err_msg=f"Oracle reconstructed tile does not match encoder reconstruction for K={K}",
            )


if __name__ == "__main__":
    unittest.main()
