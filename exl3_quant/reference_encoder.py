"""EXL3 Intel Arc Quantizer: Golden Reference Trellis Encoder.

Audited transcription of ``quantize_tiles_kernel.cuh`` from the pinned upstream
revision:
    https://github.com/turboderp-org/exllamav3 @ 4d7b69c7e177c2b0f34ed8c90b7036fe31a7281f
    File: exllamav3/exllamav3_ext/quant/quantize_tiles_kernel.cuh:23-313

Normative Contract:
- 16-bit state space: S = 65536 >> K tracked edges, Kr = 16 - K, max_q = 1 << K.
- Branch transitions: candidate state sigma = (k << Kr) | e, pred = sigma >> K.
- Procedural codebook arithmetic via ``codebook_lut(cb)`` (fp16).
- Two-pass tail-biting circular ring schedule:
    Pass 1: forward(L / 2, pre_state=-1), backward(L / 2, write=false, argmin_cost()) -> end_state
    Pass 2: forward(0, pre_state=end_state), backward(0, write=true, edge=end_state)
- fp16 metric with fused accumulation (__hfma2 emulation) and strict '<' tie-breaking.
- Tensor-core pre-permutation: input tiles are pre-permuted via ``tensor_core_perm``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np

from exl3_quant.trellis_format import (
    CB_3INST,
    CB_MCG,
    CB_MUL1,
    codebook_lut,
    tensor_core_perm,
    tensor_core_perm_inv,
    pack_trellis,
    swap16_storage,
    decode_states,
    decode_tile_values,
)

H_INF = np.uint16(0x7C00).view(np.float16)


@dataclass
class EncodingResult:
    """Result of trellis quantization encoding on a tile of weights."""
    output_indices: np.ndarray        # 16-bit states in tensor-core order (length L=256, uint16)
    reconstructed: np.ndarray         # Reconstructed float16 weights in original tile order
    reconstructed_perm: np.ndarray    # Reconstructed float16 weights in tensor-core order
    packed: np.ndarray                # Upstream-compatible packed container (int16 / uint16 with SWAP16)
    end_state: int                    # Tail-biting wrap state at ring index 0
    distortion: float                 # Total squared error sum((w - w_hat)^2)
    mse: float                        # Mean squared error
    K: int                            # Bitrate parameter K
    cb: int                           # Codebook selector


class ReferenceTrellisEncoder:
    """Golden reference Viterbi trellis encoder for EXL3 model conversion.

    Direct transcription of the upstream CUDA kernel ``quantize_tiles_kernel.cuh``
    with vectorized NumPy acceleration.
    """

    def __init__(
        self,
        K: int = 2,
        cb: int = CB_MUL1,
        tile_len: int = 256,
    ):
        """Initialize reference encoder.

        Args:
            K: Bits per weight (1..8, typically 2, 3, or 4).
            cb: Codebook selector (CB_MUL1=2, CB_MCG=1, CB_3INST=0).
            tile_len: Length of tile ring (default 256 for 16x16 EXL3 tiles).
        """
        if not 1 <= K <= 8:
            raise ValueError(f"K must be between 1 and 8, got {K}")
        if cb not in (CB_3INST, CB_MCG, CB_MUL1):
            raise ValueError(f"Unknown codebook {cb}")
        if tile_len <= 0 or tile_len % 2 != 0:
            raise ValueError(f"tile_len must be positive even integer, got {tile_len}")

        self.K = K
        self.cb = cb
        self.L = tile_len
        self.Kr = 16 - K
        self.max_q = 1 << K
        self.edges = 65536 >> K

        # Load fp16 codebook LUT
        self.lut = codebook_lut(cb)

        # Precompute candidate branch state tables for vectorized step evaluation:
        # For each k in [0, max_q):
        #   st = (k << Kr) | e
        #   pred = st >> K
        #   val = lut[st]
        e_arr = np.arange(self.edges, dtype=np.uint16)
        self.k_states = np.zeros((self.max_q, self.edges), dtype=np.uint16)
        self.k_preds = np.zeros((self.max_q, self.edges), dtype=np.uint16)
        self.k_vals = np.zeros((self.max_q, self.edges), dtype=np.float16)

        for k in range(self.max_q):
            st = ((k << self.Kr) | e_arr).astype(np.uint16)
            self.k_states[k] = st
            self.k_preds[k] = (st >> self.K).astype(np.uint16)
            self.k_vals[k] = self.lut[st]

        self.perm = tensor_core_perm()
        self.inv_perm = tensor_core_perm_inv()

    @staticmethod
    def ring(i: int, roll: int, L: int = 256) -> int:
        """ri = (i + roll) mod L, matching quantize_tiles_kernel.cuh:60-65."""
        ri = i + roll
        if ri >= L:
            ri -= L
        return ri

    def forward(
        self,
        input_tile: np.ndarray,
        roll: int,
        pre_state: int,
        temp_edges: np.ndarray,
    ) -> np.ndarray:
        """Forward Add-Compare-Select (ACS) dynamic programming pass.

        Transcription of quantize_tiles_kernel.cuh:67-228.
        Computes forward path costs and records predecessor edge pointers in
        temp_edges[ri].

        Args:
            input_tile: (L,) float16 input weights (in permuted tile order).
            roll: Ring rotation offset (L // 2 for pass 1, 0 for pass 2).
            pre_state: End state constraint for wrap-around (-1 for unconstrained).
            temp_edges: (L, edges) uint16 array to record backpointers.

        Returns:
            (edges,) float16 array of terminal edge path costs.
        """
        # Step 0: no accumulated predecessor cost (quantize_tiles_kernel.cuh:74-146)
        ri = self.ring(0, roll, self.L)
        w = input_tile[ri]

        min_cost = np.full(self.edges, H_INF, dtype=np.float16)
        min_pred = np.zeros(self.edges, dtype=np.uint16)

        for k in range(self.max_q):
            val = self.k_vals[k]
            pred = self.k_preds[k]
            # fp16 subtract and square
            dh = val.astype(np.float32) - np.float32(w)
            err = (dh * dh).astype(np.float16)

            if pre_state >= 0:
                err = np.where(pred != pre_state, H_INF, err)

            if k == 0:
                min_cost = err.copy()
                min_pred = pred.copy()
            else:
                # Strict '<' tie-breaking (first k wins ties)
                mask = err < min_cost
                min_cost[mask] = err[mask]
                min_pred[mask] = pred[mask]

        temp_edges[ri] = min_pred
        prev_cost = min_cost

        # Steps 1 .. L-1: fused multiply-add accumulation (quantize_tiles_kernel.cuh:149-228)
        for i in range(1, self.L):
            ri = self.ring(i, roll, self.L)
            w = input_tile[ri]

            for k in range(self.max_q):
                val = self.k_vals[k]
                pred = self.k_preds[k]
                dh = val.astype(np.float32) - np.float32(w)
                # Fused __hfma2(dh, dh, cost_in[pred]): single rounding to fp16
                err = (dh * dh + prev_cost[pred].astype(np.float32)).astype(np.float16)

                if k == 0:
                    min_cost = err.copy()
                    min_pred = pred.copy()
                else:
                    mask = err < min_cost
                    min_cost[mask] = err[mask]
                    min_pred[mask] = pred[mask]

            temp_edges[ri] = min_pred
            prev_cost = min_cost

        return prev_cost

    def argmin_cost(self, costs: np.ndarray) -> int:
        """Find edge with minimum accumulated cost with strict '<' tie-break.

        Matches quantize_tiles_kernel.cuh:231-284. If all costs are non-finite,
        returns 0.
        """
        if not np.any(np.isfinite(costs)) or np.all(costs == H_INF):
            return 0
        return int(np.argmin(costs))

    def backward(
        self,
        temp_edges: np.ndarray,
        roll: int,
        write: bool,
        edge: int,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Backward traceback path recovery.

        Transcription of quantize_tiles_kernel.cuh:286-307.

        Args:
            temp_edges: (L, edges) backpointer table.
            roll: Ring rotation offset.
            write: Whether to emit indices and reconstructed tile values.
            edge: Starting terminal edge index.

        Returns:
            Tuple of (end_state, output_indices, output_tile).
        """
        output_indices = np.zeros(self.L, dtype=np.uint16)
        output_tile = np.zeros(self.L, dtype=np.float16)

        for i in range(self.L - 1, -1, -1):
            ri = self.ring(i, roll, self.L)
            prev_edge = int(temp_edges[ri, edge])
            # Reconstruct 16-bit state: (prev_edge << K) | edge
            encoded = (prev_edge << self.K) | edge
            edge = prev_edge
            if write:
                output_indices[ri] = np.uint16(encoded)
                output_tile[ri] = self.lut[encoded]
            elif ri == 0:
                break

        return edge, output_indices, output_tile

    def encode_tile(
        self,
        tile: np.ndarray,
        pre_permute: bool = True,
    ) -> EncodingResult:
        """Encode a single 16x16 tile into optimal TCQ trellis states.

        Args:
            tile: (256,) or (16, 16) array of float weights.
            pre_permute: If True (default), permutes tile into tensor-core order
                         before encoding, matching upstream quantize.py:579.

        Returns:
            EncodingResult containing discrete indices, reconstructed weights,
            packed container, and distortion metrics.
        """
        raw_arr = np.asarray(tile, dtype=np.float32).reshape(-1)
        if raw_arr.size != self.L:
            raise ValueError(f"Tile must have {self.L} elements, got {raw_arr.size}")

        if pre_permute:
            input_tile = raw_arr[self.perm].astype(np.float16)
        else:
            input_tile = raw_arr.astype(np.float16)

        temp_edges = np.zeros((self.L, self.edges), dtype=np.uint16)

        # Pass 1: rotation by L/2, unconstrained (pre_state = -1)
        # quantize_tiles_kernel.cuh:309-310
        c1 = self.forward(input_tile, roll=self.L // 2, pre_state=-1, temp_edges=temp_edges)
        best_edge = self.argmin_cost(c1)
        end_state, _, _ = self.backward(temp_edges, roll=self.L // 2, write=False, edge=best_edge)

        # Pass 2: natural order (roll = 0), wrap constrained to end_state
        # quantize_tiles_kernel.cuh:311-312
        _ = self.forward(input_tile, roll=0, pre_state=end_state, temp_edges=temp_edges)
        _, output_indices, output_tile = self.backward(temp_edges, roll=0, write=True, edge=end_state)

        # Pack into bit-exact upstream container
        packed = swap16_storage(pack_trellis(output_indices, self.K))

        reconstructed_perm = output_tile
        if pre_permute:
            reconstructed = output_tile[self.inv_perm]
            orig_eval = raw_arr
        else:
            reconstructed = output_tile
            orig_eval = input_tile.astype(np.float32)

        distortion = float(np.sum((orig_eval - reconstructed.astype(np.float32)) ** 2))
        mse = distortion / float(self.L)

        return EncodingResult(
            output_indices=output_indices,
            reconstructed=reconstructed,
            reconstructed_perm=reconstructed_perm,
            packed=packed,
            end_state=end_state,
            distortion=distortion,
            mse=mse,
            K=self.K,
            cb=self.cb,
        )

    def encode_batch(
        self,
        tiles: np.ndarray,
        pre_permute: bool = True,
    ) -> list[EncodingResult]:
        """Encode a batch of tiles.

        Args:
            tiles: Array of shape (B, 256) or (B, 16, 16).
            pre_permute: Whether to apply tensor-core permutation.

        Returns:
            List of EncodingResult objects.
        """
        arr = np.asarray(tiles, dtype=np.float32)
        if arr.ndim == 2:
            batch_size = arr.shape[0]
            flat_tiles = arr
        elif arr.ndim == 3:
            batch_size = arr.shape[0]
            flat_tiles = arr.reshape(batch_size, -1)
        else:
            raise ValueError(f"Expected 2D or 3D array of tiles, got shape {arr.shape}")

        return [self.encode_tile(flat_tiles[b], pre_permute=pre_permute) for b in range(batch_size)]


# ---------------------------------------------------------------------------
# Legacy Provisional Classes (Retained for negative testing / backward compat)
# ---------------------------------------------------------------------------

@dataclass
class LegacyEncodingResult:
    indices: np.ndarray
    reconstructed: np.ndarray
    distortion: float
    mse: float
    best_final_state: int
    path_states: np.ndarray


class Mul1Codebook:
    """Provisional Mul1 codebook retained for negative perturbation testing."""
    def __init__(self, bits: int, scale: float = 1.0):
        self.bits = bits
        self.num_symbols = 1 << bits
        self.scale = scale
        self.zero_point = (self.num_symbols - 1) / 2.0

    def decode_symbol(self, symbol: int) -> float:
        return self.scale * (symbol - self.zero_point)

    def decode_array(self, symbols: np.ndarray) -> np.ndarray:
        return self.scale * (symbols.astype(np.float64) - self.zero_point)


class TrellisStateSpace:
    """Provisional shift register state space retained for legacy tests."""
    def __init__(self, num_states: int, bits_per_symbol: int):
        self.num_states = num_states
        self.bits = bits_per_symbol
        self.num_symbols = 1 << bits_per_symbol
        if (num_states & (num_states - 1)) != 0:
            raise ValueError(f"Number of states must be a power of 2, got {num_states}")
        self.state_bits = int(np.log2(num_states))
        if self.state_bits < self.bits:
            raise ValueError("State bits must be >= symbol bits")
        self.predecessors = [[] for _ in range(num_states)]
        mask = num_states - 1
        for s_prev in range(num_states):
            for symbol in range(self.num_symbols):
                s_next = ((s_prev << self.bits) | symbol) & mask
                self.predecessors[s_next].append((s_prev, symbol))


class ReferenceViterbiEncoder:
    """Provisional Viterbi encoder retained for negative test harness."""
    def __init__(self, num_states: int = 64, bits_per_symbol: int = 2, scale: float = 1.0):
        self.num_states = num_states
        self.bits = bits_per_symbol
        self.scale = scale
        self.codebook = Mul1Codebook(bits=bits_per_symbol, scale=scale)
        self.trellis = TrellisStateSpace(num_states=num_states, bits_per_symbol=bits_per_symbol)

    def encode(self, weights: np.ndarray) -> LegacyEncodingResult:
        w_arr = np.asarray(weights, dtype=np.float64)
        T = len(w_arr)
        S = self.num_states
        costs = np.full((T + 1, S), np.inf, dtype=np.float64)
        costs[0, 0] = 0.0
        backpointers = np.zeros((T, S), dtype=np.int32)
        symbols = np.zeros((T, S), dtype=np.int32)
        for t in range(T):
            w_val = w_arr[t]
            for s_next in range(S):
                for s_prev, sym in self.trellis.predecessors[s_next]:
                    level = self.codebook.decode_symbol(sym)
                    err = (w_val - level) ** 2
                    cand_cost = costs[t, s_prev] + err
                    if cand_cost < costs[t + 1, s_next]:
                        costs[t + 1, s_next] = cand_cost
                        backpointers[t, s_next] = s_prev
                        symbols[t, s_next] = sym
        best_end = int(np.argmin(costs[T]))
        opt_states = np.zeros(T + 1, dtype=np.int32)
        opt_symbols = np.zeros(T, dtype=np.int32)
        curr = best_end
        opt_states[T] = curr
        for t in range(T - 1, -1, -1):
            opt_symbols[t] = symbols[t, curr]
            curr = backpointers[t, curr]
            opt_states[t] = curr
        reconstructed = self.codebook.decode_array(opt_symbols)
        dist = float(costs[T, best_end])
        return LegacyEncodingResult(
            indices=opt_symbols,
            reconstructed=reconstructed,
            distortion=dist,
            mse=dist / T,
            best_final_state=best_end,
            path_states=opt_states,
        )

