"""EXL3 Intel Arc Quantizer: Golden Reference Trellis Encoder (Provisional Prototype).

PROVISIONAL NOTICE:
Implements an inspectable Python/NumPy reference prototype for Trellis Coded Quantization (TCQ).
Per review-2026-10-20T13:50:25+02:00 (B1):
- Uses generic shift-register state transitions: s_next = ((s_prev << bits) | sym) & (S - 1)
- Uses provisional centered mul1 zero-point (half-integer) pending upstream audit P0-11.
- Uses unweighted MSE objective (upstream uses Hessian-weighted distortion ||W - W_hat||_H).
- Uses unconstrained terminal state argmin (upstream tail closure rules pending audit P0-11).
- This is an unverified reference prototype, not audited contract-conformant code.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class EncodingResult:
    """Result of trellis quantization encoding on a sequence of weights."""
    indices: np.ndarray          # Optimal discrete codebook indices (length T)
    reconstructed: np.ndarray    # Reconstructed float weights (length T)
    distortion: float            # Total squared error sum((w - w_hat)^2)
    mse: float                   # Mean squared error
    best_final_state: int        # Terminal state with minimum accumulated cost
    path_states: np.ndarray      # States visited along the optimal path (length T + 1)


class Mul1Codebook:
    """Procedural mul1 arithmetic codebook.
    
    Decodes discrete index c into real-valued level using direct arithmetic:
        level = scale * (c - zero_point)
    Zero memory lookups required.
    """

    def __init__(self, bits: int, scale: float = 1.0):
        self.bits = bits
        self.num_symbols = 1 << bits
        self.scale = scale
        # Centered zero-point: (2^bits - 1) / 2
        self.zero_point = (self.num_symbols - 1) / 2.0

    def decode_symbol(self, symbol: int) -> float:
        """Decode a single integer symbol into a reconstructed float level."""
        return self.scale * (symbol - self.zero_point)

    def decode_array(self, symbols: np.ndarray) -> np.ndarray:
        """Decode an array of integer symbols into float levels."""
        return self.scale * (symbols.astype(np.float64) - self.zero_point)


class TrellisStateSpace:
    """Manages shift-register trellis graph topology and state transitions."""

    def __init__(self, num_states: int, bits_per_symbol: int):
        self.num_states = num_states
        self.bits = bits_per_symbol
        self.num_symbols = 1 << bits_per_symbol

        # Verify power-of-2 state count
        if (num_states & (num_states - 1)) != 0:
            raise ValueError(f"Number of states must be a power of 2, got {num_states}")

        self.state_bits = int(np.log2(num_states))
        if self.state_bits < self.bits:
            raise ValueError(
                f"State bits ({self.state_bits}) must be >= symbol bits ({self.bits})"
            )

        # Precompute predecessor transitions:
        # For each state s_next in [0, S-1], find all (s_prev, symbol) pairs that transition to s_next
        # Transition rule: s_next = ((s_prev << bits) | symbol) & (S - 1)
        self.predecessors = [[] for _ in range(num_states)]
        mask = num_states - 1

        for s_prev in range(num_states):
            for symbol in range(self.num_symbols):
                s_next = ((s_prev << self.bits) | symbol) & mask
                self.predecessors[s_next].append((s_prev, symbol))


class ReferenceViterbiEncoder:
    """Golden reference Viterbi encoder for EXL3 Trellis Coded Quantization."""

    def __init__(
        self,
        num_states: int = 64,
        bits_per_symbol: int = 2,
        scale: float = 1.0,
    ):
        """Initialize encoder.
        
        Args:
            num_states: Number of trellis states S (power of 2, e.g. 64).
            bits_per_symbol: Quantization bitrate K (e.g. 2, 3, 4).
            scale: Step size scaling factor.
        """
        self.num_states = num_states
        self.bits = bits_per_symbol
        self.codebook = Mul1Codebook(bits=bits_per_symbol, scale=scale)
        self.trellis = TrellisStateSpace(num_states, bits_per_symbol)

    def encode(self, target_weights: np.ndarray) -> EncodingResult:
        """Encode a 1D sequence of continuous weights into optimal discrete indices.
        
        Args:
            target_weights: 1D array of float weights of length T.
            
        Returns:
            EncodingResult containing indices, reconstructed weights, and distortion.
        """
        weights = np.asarray(target_weights, dtype=np.float64)
        t_steps = len(weights)
        if t_steps == 0:
            raise ValueError("Target weights sequence must not be empty.")

        # Allocate Viterbi path cost buffers: 2 ping-pong buffers
        # Initialize state costs: State 0 has 0 cost, all other states initialized to infinity
        prev_cost = np.full(self.num_states, np.inf, dtype=np.float64)
        prev_cost[0] = 0.0

        # Allocate predecessor edge backpointer table: shape (T, S)
        # and symbol table: shape (T, S)
        pred_table = np.zeros((t_steps, self.num_states), dtype=np.int32)
        symbol_table = np.zeros((t_steps, self.num_states), dtype=np.int32)

        # Precompute candidate reconstructed values for each discrete symbol
        symbol_values = np.array([
            self.codebook.decode_symbol(sym) for sym in range(self.codebook.num_symbols)
        ], dtype=np.float64)

        # -------------------------------------------------------------
        # Forward Pass: Add-Compare-Select (ACS) loop
        # -------------------------------------------------------------
        for t in range(t_steps):
            target_w = weights[t]
            curr_cost = np.full(self.num_states, np.inf, dtype=np.float64)

            # Precompute squared branch metrics for each symbol: (w - symbol_val)^2
            branch_metrics = (target_w - symbol_values) ** 2

            for s_next in range(self.num_states):
                best_cost = np.inf
                best_prev = 0
                best_sym = 0

                # Evaluate all incoming transitions to s_next
                for s_prev, sym in self.trellis.predecessors[s_next]:
                    cand_cost = prev_cost[s_prev] + branch_metrics[sym]
                    if cand_cost < best_cost:
                        best_cost = cand_cost
                        best_prev = s_prev
                        best_sym = sym

                curr_cost[s_next] = best_cost
                pred_table[t, s_next] = best_prev
                symbol_table[t, s_next] = best_sym

            prev_cost = curr_cost

        # -------------------------------------------------------------
        # Backward Pass: Traceback optimal path
        # -------------------------------------------------------------
        # Find terminal state with minimum cost
        best_final_state = int(np.argmin(prev_cost))
        total_distortion = float(prev_cost[best_final_state])

        optimal_indices = np.zeros(t_steps, dtype=np.int64)
        path_states = np.zeros(t_steps + 1, dtype=np.int32)
        path_states[-1] = best_final_state

        curr_state = best_final_state
        for t in range(t_steps - 1, -1, -1):
            optimal_indices[t] = symbol_table[t, curr_state]
            prev_s = pred_table[t, curr_state]
            path_states[t] = prev_s
            curr_state = prev_s

        reconstructed_weights = self.codebook.decode_array(optimal_indices)
        mse = total_distortion / t_steps

        return EncodingResult(
            indices=optimal_indices,
            reconstructed=reconstructed_weights,
            distortion=total_distortion,
            mse=mse,
            best_final_state=best_final_state,
            path_states=path_states,
        )
