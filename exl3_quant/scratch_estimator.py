"""EXL3 Intel Arc Quantizer: Scratch Memory Estimator and Batch Sizing.

Implements P1-09, P1-10, P1-11 from tasks.md:
- Checked scratch-size estimator using pinned allocator contract.
- Distinguishes binary (GiB/MiB) and decimal (GB/MB) units.
- Adaptive batch capacity B calculator given available VRAM and safety reserves.
- Enforces single-allocation ceilings (e.g. 4 GiB max buffer on certain Level Zero / OpenCL runtimes).
"""

from dataclasses import dataclass
from typing import Optional, Tuple


BYTES_PER_KIB = 1024
BYTES_PER_MIB = 1024 ** 2
BYTES_PER_GIB = 1024 ** 3

BYTES_PER_KB = 1000
BYTES_PER_MB = 1000 ** 2
BYTES_PER_GB = 1000 ** 3


@dataclass(frozen=True)
class ScratchAllocation:
    """Detailed memory breakdown for a specific quantizer configuration."""
    k: int
    states: int
    t: int
    b: int
    bytes_per_cost: int
    bytes_per_edge: int

    costs_bytes: int
    edges_bytes: int
    total_bytes: int

    @property
    def costs_mib(self) -> float:
        return self.costs_bytes / BYTES_PER_MIB

    @property
    def costs_mb(self) -> float:
        return self.costs_bytes / BYTES_PER_MB

    @property
    def edges_gib(self) -> float:
        return self.edges_bytes / BYTES_PER_GIB

    @property
    def edges_gb(self) -> float:
        return self.edges_bytes / BYTES_PER_GB

    @property
    def total_gib(self) -> float:
        return self.total_bytes / BYTES_PER_GIB

    @property
    def total_gb(self) -> float:
        return self.total_bytes / BYTES_PER_GB


class ScratchEstimator:
    """Estimates temporary scratch memory for the EXL3 Viterbi trellis encoder."""

    @staticmethod
    def get_state_count(k: int) -> int:
        """Calculate number of trellis states: S = 65536 >> K.
        
        Args:
            k: Trellis parameter K (typically 1 to 4).
        """
        if k < 1 or k > 16:
            raise ValueError(f"Trellis parameter K must be between 1 and 16, got {k}")
        return 65536 >> k

    @classmethod
    def estimate_scratch(
        cls,
        k: int,
        t: int = 256,
        b: int = 256,
        bytes_per_cost: int = 2,
        bytes_per_edge: int = 2,
    ) -> ScratchAllocation:
        """Estimate costs and edge scratch buffers for given parameters.
        
        Formula from upstream contract (tasks.md lines 121-124):
            S = 65536 >> K
            costs = B * 2 * S * bytes_per_cost  (2 ping-pong cost buffers)
            edges = B * T * S * bytes_per_edge  (backpointer matrix)
        """
        if b <= 0:
            raise ValueError(f"Batch size B must be positive, got {b}")
        if t <= 0:
            raise ValueError(f"Sequence length T must be positive, got {t}")

        states = cls.get_state_count(k)
        costs_bytes = b * 2 * states * bytes_per_cost
        edges_bytes = b * t * states * bytes_per_edge
        total_bytes = costs_bytes + edges_bytes

        return ScratchAllocation(
            k=k,
            states=states,
            t=t,
            b=b,
            bytes_per_cost=bytes_per_cost,
            bytes_per_edge=bytes_per_edge,
            costs_bytes=costs_bytes,
            edges_bytes=edges_bytes,
            total_bytes=total_bytes,
        )

    @classmethod
    def calculate_max_batch_size(
        cls,
        k: int,
        available_vram_bytes: int,
        vram_reserve_bytes: int = 2 * BYTES_PER_GIB,
        max_single_alloc_bytes: Optional[int] = 4 * BYTES_PER_GIB,
        t: int = 256,
        bytes_per_cost: int = 2,
        bytes_per_edge: int = 2,
    ) -> Tuple[int, ScratchAllocation]:
        """Calculates maximum valid batch size B that fits within device memory constraints.
        
        Args:
            k: Trellis parameter K.
            available_vram_bytes: Total available VRAM on device (e.g. 12 or 16 GiB).
            vram_reserve_bytes: Safety headroom for OS, drivers, PyTorch base runtime.
            max_single_alloc_bytes: Optional single-allocation buffer limit (e.g. 4 GiB driver ceiling).
            t: Sequence length (steps).
            bytes_per_cost: Cost metric storage dtype size (default FP16 = 2 bytes).
            bytes_per_edge: Predecessor index storage dtype size (default uint16 = 2 bytes).
            
        Returns:
            Tuple of (max_b, scratch_allocation).
        """
        usable_vram = available_vram_bytes - vram_reserve_bytes
        if usable_vram <= 0:
            raise ValueError(
                f"Available VRAM ({available_vram_bytes / BYTES_PER_GIB:.2f} GiB) must exceed "
                f"reserve ({vram_reserve_bytes / BYTES_PER_GIB:.2f} GiB)"
            )

        states = cls.get_state_count(k)
        bytes_per_unit_b = (2 * states * bytes_per_cost) + (t * states * bytes_per_edge)
        
        # Max B bound by total usable VRAM
        b_vram_bound = usable_vram // bytes_per_unit_b

        # Max B bound by single allocation limit (edges buffer is the largest allocation)
        if max_single_alloc_bytes is not None:
            bytes_per_edge_unit_b = t * states * bytes_per_edge
            b_alloc_bound = max_single_alloc_bytes // bytes_per_edge_unit_b
            max_b = min(b_vram_bound, b_alloc_bound)
        else:
            max_b = b_vram_bound

        if max_b < 1:
            raise ValueError(
                f"Insufficient memory: minimum batch size B=1 requires "
                f"{bytes_per_unit_b / BYTES_PER_MIB:.2f} MiB, which exceeds allowable memory budget."
            )

        allocation = cls.estimate_scratch(
            k=k,
            t=t,
            b=int(max_b),
            bytes_per_cost=bytes_per_cost,
            bytes_per_edge=bytes_per_edge,
        )
        return int(max_b), allocation
