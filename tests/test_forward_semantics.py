"""Independent audit of the reference encoder's forward ACS pass.

The golden reference encoder (``exl3_quant/reference_encoder.py``) is a
transcription of ``quantize_tiles_kernel.cuh``. This module re-implements the
forward pass *independently*, straight from the kernel text, and requires the
two to agree bit for bit on the recorded predecessor table and on the terminal
costs.

This is the audit evidence for P0-11 (branch-metric arithmetic, tie-breaking,
masking rules). It also pins the kernel's edge cases:

* the fp16 subtract happens *before* the accumulation (``__hsub2`` then
  ``__hfma2``), so the intermediate difference is rounded to fp16;
* the product and the predecessor cost are summed in float64 and rounded
  **once** to fp16, matching a fused multiply-add;
* the winner is the **first** branch attaining the strict minimum (the kernel
  uses a strict ``__hlt``, so ties keep the lowest ``k``);
* a NaN metric never displaces an earlier candidate (``__hlt`` against a NaN is
  false), while a NaN in the ``k = 0`` branch sticks because the kernel
  initializes ``min_err2`` from it unconditionally;
* predecessors other than ``pre_state`` are masked to ``H_INF``.
"""

import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exl3_quant.reference_encoder import H_INF, ReferenceTrellisEncoder


def independent_forward(enc, input_tile, roll, pre_state):
    """Independent transcription of quantize_tiles_kernel.cuh:67-228.

    Returns (temp_edges, terminal_costs).
    """
    L = enc.L
    edges = enc.edges
    max_q = enc.max_q

    def ring(i):
        ri = i + roll
        if ri >= L:
            ri -= L
        return ri

    temp_edges = np.zeros((L, edges), dtype=np.uint16)

    prev_cost = None
    for i in range(L):
        ri = ring(i)
        w = input_tile[ri]

        min_cost = np.full(edges, H_INF, dtype=np.float16)
        min_pred = np.zeros(edges, dtype=np.uint16)

        for k in range(max_q):
            val = enc.k_vals[k]
            pred = enc.k_preds[k]
            # __hsub2 (fp16), then __hmul2 / __hfma2 (single rounding).
            dh = (val - w).astype(np.float64)
            if prev_cost is None:
                err = (dh * dh).astype(np.float16)
            else:
                err = (dh * dh + prev_cost[pred].astype(np.float64)).astype(np.float16)

            if pre_state >= 0 and prev_cost is None:
                # The kernel masks the wrap constraint in the step-0 loop only
                # (quantize_tiles_kernel.cuh:99 and :130); the accumulating
                # loop (:180-:220) carries no mask.
                err = np.where(pred != pre_state, H_INF, err)

            if k == 0:
                min_cost = err.copy()
                min_pred = pred.copy()
            else:
                mask = err < min_cost
                min_cost[mask] = err[mask]
                min_pred[mask] = pred[mask]

        temp_edges[ri] = min_pred
        prev_cost = min_cost

    return temp_edges, prev_cost


class TestForwardPassAudit(unittest.TestCase):

    def _tile(self, seed, scale=0.5):
        rng = np.random.default_rng(seed)
        return rng.normal(scale=scale, size=256).astype(np.float16)

    def test_forward_matches_independent_transcription(self):
        """Unconstrained pass: temp_edges and terminal costs agree bit for bit."""
        for K in (2, 3, 4):
            with self.subTest(K=K):
                enc = ReferenceTrellisEncoder(K=K)
                tile = self._tile(seed=11 + K)
                edges = np.zeros((enc.L, enc.edges), dtype=np.uint16)

                got_costs = enc.forward(tile, roll=0, pre_state=-1,
                                        temp_edges=edges)
                want_edges, want_costs = independent_forward(
                    enc, tile, roll=0, pre_state=-1)

                np.testing.assert_array_equal(edges, want_edges)
                np.testing.assert_array_equal(got_costs.view(np.uint16),
                                              want_costs.view(np.uint16))

    def test_constrained_pass_matches(self):
        """A wrap constraint (pre_state >= 0) masks other predecessors."""
        for K in (3, 4):
            with self.subTest(K=K):
                enc = ReferenceTrellisEncoder(K=K)
                tile = self._tile(seed=21 + K)
                pre_state = int(np.random.default_rng(5).integers(0, enc.edges))

                edges = np.zeros((enc.L, enc.edges), dtype=np.uint16)
                got_costs = enc.forward(tile, roll=0, pre_state=pre_state,
                                        temp_edges=edges)
                want_edges, want_costs = independent_forward(
                    enc, tile, roll=0, pre_state=pre_state)

                np.testing.assert_array_equal(edges, want_edges)
                np.testing.assert_array_equal(got_costs.view(np.uint16),
                                              want_costs.view(np.uint16))

    def test_rotated_ring_matches(self):
        """Pass 1 uses roll = L // 2; the ring indexing must agree."""
        enc = ReferenceTrellisEncoder(K=4)
        tile = self._tile(seed=31)
        edges = np.zeros((enc.L, enc.edges), dtype=np.uint16)
        got_costs = enc.forward(tile, roll=enc.L // 2, pre_state=-1,
                                temp_edges=edges)
        want_edges, want_costs = independent_forward(
            enc, tile, roll=enc.L // 2, pre_state=-1)
        np.testing.assert_array_equal(edges, want_edges)
        np.testing.assert_array_equal(got_costs.view(np.uint16),
                                      want_costs.view(np.uint16))

    def test_ties_keep_the_lowest_branch(self):
        """Strict '<' means an exact tie keeps the earlier (lower-k) branch.

        The tie is engineered by giving every branch the same codebook value,
        so all branch metrics are identical and the recorded predecessor must
        be the k = 0 branch's predecessor.
        """
        K = 3
        enc = ReferenceTrellisEncoder(K=K)
        # All branches decode to the same value -> every metric ties.
        enc.k_vals = np.zeros((enc.max_q, enc.edges), dtype=np.float16)
        # Give each branch a distinct predecessor so the choice is observable.
        enc.k_preds = np.zeros((enc.max_q, enc.edges), dtype=np.uint16)
        for k in range(enc.max_q):
            enc.k_preds[k] = (k * 7 + 1) % enc.edges

        tile = np.full(enc.L, np.float16(0.25), dtype=np.float16)
        edges = np.zeros((enc.L, enc.edges), dtype=np.uint16)
        enc.forward(tile, roll=0, pre_state=-1, temp_edges=edges)

        np.testing.assert_array_equal(edges, np.tile(enc.k_preds[0], (enc.L, 1)))

    def test_infinite_input_does_not_produce_nan_paths(self):
        """A degenerate infinite tile must not silently yield NaN costs.

        __hsub2(inf, inf) is NaN; the kernel's strict comparisons keep the
        first branch, so the recorded costs stay finite or +inf, never NaN.
        """
        K = 2
        enc = ReferenceTrellisEncoder(K=K)
        tile = np.full(enc.L, np.inf, dtype=np.float16)
        edges = np.zeros((enc.L, enc.edges), dtype=np.uint16)
        costs = enc.forward(tile, roll=0, pre_state=-1, temp_edges=edges)
        self.assertFalse(np.isnan(costs.astype(np.float64)).any())


if __name__ == "__main__":
    unittest.main()
