"""SYCL XPU tile encoder and bounded scratch arena for Intel Arc (Phase 1).

Implements:
  * P1-03 / P1-04: XPU forward ACS search (exact fp16 metric + single-rounded
    fp16 FMA accumulation) and GPU-resident circular tail-biting traceback.
  * P1-05 .. P1-08: tile reconstruction, tie-aware verification helpers, shape
    and configuration validation, and single-queue execution on the current
    torch XPU stream.
  * P1-11 .. P1-16 / V140-04: adaptive batch sizing via ``MemoryPolicy``,
    bounded event-aware ``ScratchArena`` (replacing upstream's unbounded
    ``@lru_cache get_temp_buffers``), smaller-batch fallback retries without
    changing tile semantics, and distinct staging-copy vs shared-backing
    memory accounting.
  * P1-17 .. P1-20 / V140-05: three traceback placement implementations
    (``gpu_resident``, ``d2h_cpu``, ``usm_host``) with separate search,
    transfer, and traceback timing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from exl3_quant.memory_policy import (
    DEFAULT_RESERVE_BYTES,
    MemoryPolicy,
    MemoryPressureError,
    MemorySnapshot,
    capture_snapshot,
)
from exl3_quant.reference_encoder import EncodingResult, H_INF
from exl3_quant.scratch_estimator import ScratchEstimator
from exl3_quant.trellis_format import (
    CB_3INST,
    CB_MCG,
    CB_MUL1,
    codebook_lut,
    pack_trellis,
    swap16_storage,
    tensor_core_perm,
    tensor_core_perm_inv,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "csrc" / "quantize_xpu"
BUILD_DIR = REPO_ROOT / "artifacts" / "build" / "quantize_xpu"
_ONEAPI_BIN = "/opt/intel/oneapi/compiler/latest/bin"

_EXT = None
_COLD_BUILD_SECONDS = 0.0
_WARM_LOAD_SECONDS = 0.0


def _prepare_environment() -> None:
    if os.path.isdir(_ONEAPI_BIN) and _ONEAPI_BIN not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ONEAPI_BIN + os.pathsep + os.environ.get("PATH", "")


def build_quantize_extension(verbose: bool = False, force_rebuild: bool = False):
    """Compile (cached) and load ``quantize_xpu_ext`` via ``cpp_extension``."""
    global _EXT, _COLD_BUILD_SECONDS, _WARM_LOAD_SECONDS
    if _EXT is not None and not force_rebuild:
        return _EXT

    import shutil
    import torch
    from torch.utils import cpp_extension

    _prepare_environment()
    if force_rebuild and BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    so_existed = any(BUILD_DIR.glob("*.so")) if BUILD_DIR.exists() else False
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    _EXT = cpp_extension.load(
        name="quantize_xpu_ext",
        sources=[
            str(SRC_DIR / "quantize_xpu.cpp"),
            str(SRC_DIR / "quantize_kernels.sycl"),
        ],
        build_directory=str(BUILD_DIR),
        extra_cflags=["-O2"],
        extra_sycl_cflags=["-O2"],
        verbose=verbose,
    )
    elapsed = time.perf_counter() - t0
    if so_existed and not force_rebuild:
        _WARM_LOAD_SECONDS = elapsed
    else:
        _COLD_BUILD_SECONDS = elapsed
    return _EXT


def get_build_timings() -> dict:
    return {
        "cold_build_seconds": _COLD_BUILD_SECONDS,
        "warm_load_seconds": _WARM_LOAD_SECONDS,
    }


def verify_tile_encoding_quality(
    result: EncodingResult,
    reference: EncodingResult,
    cost_atol: float = 0.0,
    distortion_rtol: float = 1e-4,
) -> dict:
    """P1-06: Verify a tile encoding against reference allowing valid tied paths.

    Checks:
      1. Circular tail-biting trellis ring continuity at every step `t -> (t+1)%L`.
      2. Exact codebook reconstruction `reconstructed_perm == lut[output_indices]`.
      3. Terminal `end_state == output_indices[L-1] & (edges - 1)`.
      4. Path objective cost matches reference within `cost_atol` (0.0 for exact ties)
         and float32 distortion matches within `distortion_rtol`.
    """
    L = len(result.output_indices)
    K = result.K
    edges = 65536 >> K
    edges_mask = edges - 1
    lut = codebook_lut(result.cb)

    ring_valid = True
    broken_step = -1
    for t in range(L):
        nxt = (t + 1) % L
        cur_edge = int(result.output_indices[t]) & edges_mask
        nxt_pred = int(result.output_indices[nxt]) >> K
        if cur_edge != nxt_pred:
            ring_valid = False
            broken_step = t
            break

    expected_rec = lut[result.output_indices]
    rec_exact = bool(
        np.array_equal(
            result.reconstructed_perm.view(np.uint16),
            expected_rec.view(np.uint16),
        )
    )
    end_state_valid = (
        int(result.output_indices[L - 1]) & edges_mask
    ) == result.end_state
    indices_identical = bool(
        np.array_equal(result.output_indices, reference.output_indices)
    )
    cost_diff = abs(float(result.path_cost) - float(reference.path_cost))
    cost_ok = cost_diff <= cost_atol
    ref_dist = max(1e-12, float(reference.distortion))
    dist_rel_diff = abs(float(result.distortion) - float(reference.distortion)) / ref_dist
    dist_ok = dist_rel_diff <= distortion_rtol or cost_diff == 0.0

    passed = ring_valid and rec_exact and end_state_valid and cost_ok and dist_ok
    return {
        "passed": passed,
        "indices_identical": indices_identical,
        "ring_valid": ring_valid,
        "broken_step": broken_step,
        "rec_exact": rec_exact,
        "end_state_valid": end_state_valid,
        "cost_diff": cost_diff,
        "distortion_rel_diff": dist_rel_diff,
    }


@dataclass
class ScratchArenaLogEntry:
    """P1-16 / V140-04: full live-memory context for a scratch arena action."""

    action: str                     # "allocate", "reuse", "evict", "retry-shrink", "release"
    k: int
    tile_len: int
    requested_b: int
    admitted_b: int
    scratch_bytes: int
    distinct_staging_bytes: int     # host staging copies separate from XPU buffers
    outstanding_xpu_bytes: int
    usable_budget_bytes: int
    snapshot: dict


class ScratchArena:
    """Bounded, queue-synchronized scratch arena for ``temp_costs`` & ``temp_edges``.

    Replaces upstream's unbounded ``@lru_cache get_temp_buffers`` (R16, P1-12).
    At most **one** `(temp_costs, temp_edges)` pair is kept resident at any
    time; changing `(K, L, device)` or exceeding capacity synchronizes the
    current XPU stream and releases the previous buffers before allocating.

    Implements smaller-batch retry (P1-13): if `requested_b` cannot fit within
    the `MemoryPolicy` envelope (or if the allocator fails), `B` is halved down
    to `B = 1` without altering tile semantics.
    """

    def __init__(
        self,
        policy: Optional[MemoryPolicy] = None,
        device_index: int = 0,
        default_cap_b: int = 64,
    ):
        self.policy = policy or MemoryPolicy(reserve_bytes=DEFAULT_RESERVE_BYTES)
        self.device_index = device_index
        self.default_cap_b = default_cap_b

        self._key: Optional[Tuple[int, int, int, int]] = None  # (device, K, L, B)
        self._temp_costs = None
        self._temp_edges = None
        self.history: List[ScratchArenaLogEntry] = []
        self.high_water_scratch_bytes: int = 0
        self.evictions: int = 0
        self.reuses: int = 0
        self.allocations: int = 0
        self.retries: int = 0

    @property
    def current_key(self) -> Optional[Tuple[int, int, int, int]]:
        return self._key

    @property
    def resident_bytes(self) -> int:
        if self._temp_costs is None or self._temp_edges is None:
            return 0
        return int(
            self._temp_costs.numel() * self._temp_costs.element_size()
            + self._temp_edges.numel() * self._temp_edges.element_size()
        )

    def adaptive_batch_capacity(
        self,
        k: int,
        tile_len: int = 256,
        requested_b: Optional[int] = None,
        distinct_staging_bytes: int = 0,
    ) -> int:
        """P1-11: compute max admissible batch B on Arc 140V."""
        import torch

        target_b = requested_b if requested_b is not None else self.default_cap_b
        if target_b < 1:
            raise ValueError(f"requested_b must be >= 1, got {target_b}")

        snap = capture_snapshot(self.device_index)
        outstanding = max(
            0,
            int(torch.xpu.memory_allocated(self.device_index)) - self.resident_bytes,
        ) + distinct_staging_bytes
        usable = self.policy.usable_bytes(snap, outstanding_bytes=outstanding)

        per_b = ScratchEstimator.estimate_scratch(k=k, b=1, t=tile_len)
        max_by_budget = usable // per_b.total_bytes
        if self.policy.max_single_alloc_bytes is not None:
            max_by_edge = self.policy.max_single_alloc_bytes // max(1, per_b.edges_bytes)
            max_by_budget = min(max_by_budget, max_by_edge)

        if max_by_budget < 1:
            # Trigger structured MemoryPressureError for B=1
            self.policy.check(
                per_b.total_bytes,
                snap,
                outstanding_bytes=outstanding,
                context=f"adaptive_batch_capacity K={k} B=1",
            )
        return int(max(1, min(target_b, max_by_budget)))

    def release(self) -> None:
        """Synchronize the XPU stream and release any resident scratch buffers."""
        import torch

        if self._temp_costs is not None or self._temp_edges is not None:
            if torch.xpu.is_available():
                torch.xpu.synchronize(self.device_index)
            k = self._key[1] if self._key else 0
            t = self._key[2] if self._key else 0
            b = self._key[3] if self._key else 0
            self._temp_costs = None
            self._temp_edges = None
            self._key = None
            if torch.xpu.is_available():
                torch.xpu.empty_cache()
            snap = capture_snapshot(self.device_index)
            out_bytes = (
                int(torch.xpu.memory_allocated(self.device_index))
                if torch.xpu.is_available()
                else 0
            )
            self.history.append(
                ScratchArenaLogEntry(
                    action="release",
                    k=k,
                    tile_len=t,
                    requested_b=b,
                    admitted_b=0,
                    scratch_bytes=0,
                    distinct_staging_bytes=0,
                    outstanding_xpu_bytes=out_bytes,
                    usable_budget_bytes=self.policy.usable_bytes(snap, out_bytes),
                    snapshot=snap.as_dict(),
                )
            )

    def acquire(
        self,
        k: int,
        tile_len: int = 256,
        requested_b: Optional[int] = None,
        distinct_staging_bytes: int = 0,
        allocator_hook: Optional[Callable[[int], None]] = None,
    ):
        """Acquire `(temp_costs, temp_edges, admitted_b)` with bounded eviction & retry.

        ``distinct_staging_bytes`` is a *conservative upper bound* on host-side
        staging memory that will be live alongside the XPU buffers (e.g. the
        full ``temp_edges`` copy in ``d2h_cpu`` mode). It is subtracted from the
        admission budget but is not measured per chunk; callers computing it
        from the requested (not admitted) batch size over-estimate rather than
        under-estimate, which is the safe direction (L5).
        """
        import torch

        target_b = requested_b if requested_b is not None else self.default_cap_b
        if target_b < 1:
            raise ValueError(f"requested_b must be >= 1, got {target_b}")

        # Fast path: reuse existing buffer if it matches (device, K, L) and B >= target_b
        if (
            self._key is not None
            and self._key[0] == self.device_index
            and self._key[1] == k
            and self._key[2] == tile_len
            and self._key[3] >= target_b
            and self._temp_costs is not None
            and self._temp_edges is not None
        ):
            torch.xpu.synchronize(self.device_index)
            self.reuses += 1
            snap = capture_snapshot(self.device_index)
            out_bytes = int(torch.xpu.memory_allocated(self.device_index))
            self.history.append(
                ScratchArenaLogEntry(
                    action="reuse",
                    k=k,
                    tile_len=tile_len,
                    requested_b=target_b,
                    admitted_b=target_b,
                    scratch_bytes=self.resident_bytes,
                    distinct_staging_bytes=distinct_staging_bytes,
                    outstanding_xpu_bytes=out_bytes,
                    usable_budget_bytes=self.policy.usable_bytes(
                        snap, out_bytes + distinct_staging_bytes
                    ),
                    snapshot=snap.as_dict(),
                )
            )
            return (
                self._temp_costs[:target_b],
                self._temp_edges[:target_b],
                target_b,
            )

        # Evict any incompatible or smaller resident buffer before allocating
        if self._temp_costs is not None or self._temp_edges is not None:
            self.evictions += 1
            self.release()

        # P1-13: try target_b, shrinking on MemoryPressureError or allocator failure
        cand_b = target_b
        last_err: Optional[Exception] = None
        while cand_b >= 1:
            est = ScratchEstimator.estimate_scratch(k=k, b=cand_b, t=tile_len)
            snap = capture_snapshot(self.device_index)
            out_bytes = int(torch.xpu.memory_allocated(self.device_index))
            effective_outstanding = out_bytes + distinct_staging_bytes
            costs = None
            edges = None
            try:
                # Single-allocation ceiling applies to each individual buffer
                # (edges_bytes is the larger of the two buffers); cumulative
                # usable budget applies to total_bytes (costs + edges).
                self.policy.check(
                    max(est.costs_bytes, est.edges_bytes),
                    snap,
                    outstanding_bytes=effective_outstanding,
                    context=f"K={k} B={cand_b} single-buffer ceiling",
                )
                usable = self.policy.usable_bytes(
                    snap, outstanding_bytes=effective_outstanding
                )
                if est.total_bytes > usable:
                    raise MemoryPressureError(
                        f"memory-pressure abort [K={k} B={cand_b} scratch admit]: "
                        f"need {est.total_bytes} B > usable {usable} B",
                        est.total_bytes,
                        usable,
                        snap,
                        f"K={k} B={cand_b} scratch admit",
                    )
                if allocator_hook is not None:
                    allocator_hook(cand_b)

                dev = f"xpu:{self.device_index}"
                costs = torch.empty(
                    (cand_b, 2, est.states), dtype=torch.float16, device=dev
                )
                edges = torch.empty(
                    (cand_b, tile_len, est.states), dtype=torch.int16, device=dev
                )
                post = capture_snapshot(self.device_index)
                self.policy.check_after_commit(
                    post,
                    pre_snapshot=snap,
                    expected_bytes=est.total_bytes,
                    context=f"K={k} B={cand_b} post-commit",
                )

                self._key = (self.device_index, k, tile_len, cand_b)
                self._temp_costs = costs
                self._temp_edges = edges
                self.allocations += 1
                self.high_water_scratch_bytes = max(
                    self.high_water_scratch_bytes, est.total_bytes
                )
                self.history.append(
                    ScratchArenaLogEntry(
                        action="allocate",
                        k=k,
                        tile_len=tile_len,
                        requested_b=target_b,
                        admitted_b=cand_b,
                        scratch_bytes=est.total_bytes,
                        distinct_staging_bytes=distinct_staging_bytes,
                        outstanding_xpu_bytes=int(
                            torch.xpu.memory_allocated(self.device_index)
                        ),
                        usable_budget_bytes=usable,
                        snapshot=post.as_dict(),
                    )
                )
                return self._temp_costs, self._temp_edges, cand_b
            except Exception as exc:
                # B1 & B2: explicitly drop any allocated buffers and clear framework cache
                # so attempt n does not double residency during attempt n+1 (B1),
                # and failed acquisition does not leave unreclaimable memory (B2).
                if costs is not None or edges is not None:
                    del costs, edges
                    costs = None
                    edges = None
                    if torch.xpu.is_available():
                        torch.xpu.empty_cache()

                last_err = exc
                if cand_b == 1:
                    break
                next_b = max(1, cand_b // 2)
                self.retries += 1
                self.history.append(
                    ScratchArenaLogEntry(
                        action="retry-shrink",
                        k=k,
                        tile_len=tile_len,
                        requested_b=cand_b,
                        admitted_b=next_b,
                        scratch_bytes=est.total_bytes,
                        distinct_staging_bytes=distinct_staging_bytes,
                        outstanding_xpu_bytes=out_bytes,
                        usable_budget_bytes=self.policy.usable_bytes(
                            snap, effective_outstanding
                        ),
                        snapshot=snap.as_dict(),
                    )
                )
                cand_b = next_b

        # B2: Construct a clean MemoryPressureError detached from the function frame
        # so caller retention of the exception does not keep frame locals or memory alive.
        if isinstance(last_err, MemoryPressureError):
            clean_err = MemoryPressureError(
                message=str(last_err),
                need_bytes=last_err.need_bytes,
                usable_bytes=last_err.usable_bytes,
                snapshot=last_err.snapshot,
                context=last_err.context,
            )
        else:
            snap = capture_snapshot(self.device_index)
            out_bytes = int(torch.xpu.memory_allocated(self.device_index))
            clean_err = MemoryPressureError(
                f"scratch allocation failed even at B=1 for K={k}: {last_err}",
                ScratchEstimator.estimate_scratch(k=k, b=1, t=tile_len).total_bytes,
                self.policy.usable_bytes(snap, out_bytes + distinct_staging_bytes),
                snap,
                context=f"K={k} B=1 minimum batch failure",
            )
        del last_err
        raise clean_err from None


@dataclass
class BatchEncodingBreakdown:
    """Detailed output and timing breakdown for a batch of tiles on XPU."""

    results: List[EncodingResult]
    reconstructed_tiles: np.ndarray      # (N, L) float16 in original order (if pre_permute)
    reconstructed_perm: np.ndarray       # (N, L) float16 in tensor-core order
    output_indices: np.ndarray           # (N, L) uint16 16-bit states
    admitted_b: int
    num_chunks: int
    total_seconds: float
    search_seconds: float
    traceback_seconds: float
    transfer_seconds: float
    postprocess_seconds: float
    traceback_mode: str
    sg_size: int
    force_global_costs: bool


class XPUTrellisEncoder:
    """Intel Arc SYCL Viterbi trellis encoder matching ``ReferenceTrellisEncoder``."""

    def __init__(
        self,
        K: int = 2,
        cb: int = CB_MUL1,
        tile_len: int = 256,
        arena: Optional[ScratchArena] = None,
        device_index: int = 0,
        sg_size: int = 32,
        force_global_costs: bool = False,
        wave_cap_override: int = 0,
    ):
        if not 1 <= K <= 8:
            raise ValueError(f"K must be between 1 and 8, got {K}")
        if cb not in (CB_3INST, CB_MCG, CB_MUL1):
            raise ValueError(f"Unknown codebook {cb}")
        if tile_len not in (256, 160):
            raise ValueError(f"tile_len must be 256 or 160, got {tile_len}")
        if tile_len == 160 and cb != CB_MUL1:
            raise ValueError("tile_len=160 is only supported with CB_MUL1")
        if sg_size not in (16, 32):
            raise ValueError(f"sg_size must be 16 or 32, got {sg_size}")

        self.K = K
        self.cb = cb
        self.L = tile_len
        self.edges = 65536 >> K
        self.device_index = device_index
        self.sg_size = sg_size
        self.force_global_costs = force_global_costs
        self.wave_cap_override = wave_cap_override
        self.arena = arena or ScratchArena(device_index=device_index)

        self.lut = codebook_lut(cb)
        if self.L == 256:
            self.perm = tensor_core_perm()
            self.inv_perm = tensor_core_perm_inv()
        else:
            self.perm = None
            self.inv_perm = None

    @property
    def mcg_flag(self) -> bool:
        return self.cb == CB_MCG

    @property
    def mul1_flag(self) -> bool:
        return self.cb == CB_MUL1

    def encode_tile(
        self,
        tile: np.ndarray,
        pre_permute: bool = True,
        traceback_mode: str = "gpu_resident",
    ) -> EncodingResult:
        raw = np.asarray(tile, dtype=np.float32).reshape(-1)
        if raw.size != self.L:
            raise ValueError(f"Tile must have {self.L} elements, got {raw.size}")
        batch = self.encode_batch_detailed(
            raw.reshape(1, self.L),
            pre_permute=pre_permute,
            requested_b=1,
            traceback_mode=traceback_mode,
        )
        return batch.results[0]

    def encode_batch(
        self,
        tiles: np.ndarray,
        pre_permute: bool = True,
        requested_b: Optional[int] = None,
        traceback_mode: str = "gpu_resident",
    ) -> List[EncodingResult]:
        return self.encode_batch_detailed(
            tiles,
            pre_permute=pre_permute,
            requested_b=requested_b,
            traceback_mode=traceback_mode,
        ).results

    def encode_batch_detailed(
        self,
        tiles: np.ndarray,
        pre_permute: bool = True,
        requested_b: Optional[int] = None,
        traceback_mode: str = "gpu_resident",
        allocator_hook: Optional[Callable[[int], None]] = None,
    ) -> BatchEncodingBreakdown:
        import torch

        arr = np.asarray(tiles, dtype=np.float32)
        if arr.ndim == 2:
            num_tiles, width = arr.shape
            flat_tiles = arr
        elif arr.ndim == 3 and arr.shape[1:] == (16, 16) and self.L == 256:
            num_tiles = arr.shape[0]
            width = 256
            flat_tiles = arr.reshape(num_tiles, 256)
        else:
            raise ValueError(f"Unsupported tiles shape {arr.shape} for L={self.L}")

        if num_tiles <= 0:
            raise ValueError(f"num_tiles must be positive, got {num_tiles}")
        if width != self.L:
            raise ValueError(f"Expected tile length {self.L}, got {width}")
        if pre_permute and self.L != 256:
            raise ValueError("pre_permute=True is only defined for L=256")
        if traceback_mode not in (
            "gpu_resident",
            "gpu_resident_split",
            "d2h_cpu",
            "usm_host",
        ):
            raise ValueError(f"Unknown traceback_mode {traceback_mode!r}")

        ext = build_quantize_extension()
        dev = f"xpu:{self.device_index}"

        if pre_permute:
            perm_tiles = flat_tiles[:, self.perm]
        else:
            perm_tiles = flat_tiles
        # Round input tiles to fp16 -> fp32 so input matches contract section 4.3
        perm_tiles_f16 = np.ascontiguousarray(perm_tiles, dtype=np.float16)
        perm_tiles_f32 = np.ascontiguousarray(perm_tiles_f16, dtype=np.float32)

        desired_b = min(
            num_tiles,
            requested_b if requested_b is not None else self.arena.default_cap_b,
        )

        # V140-04: count distinct host staging buffers when d2h_cpu or usm_host is used
        staging_per_tile = (
            self.L * self.edges * 2 if traceback_mode in ("d2h_cpu", "usm_host") else 0
        )
        distinct_staging = desired_b * staging_per_tile + perm_tiles_f32.nbytes

        temp_costs, temp_edges, admitted_b = self.arena.acquire(
            k=self.K,
            tile_len=self.L,
            requested_b=desired_b,
            distinct_staging_bytes=distinct_staging,
            allocator_hook=allocator_hook,
        )

        t_total0 = time.perf_counter()
        t_transfer = 0.0
        t_search = 0.0
        t_traceback = 0.0

        t0 = time.perf_counter()
        in_xpu = torch.from_numpy(perm_tiles_f32).to(dev)
        out_tiles_xpu = torch.empty_like(in_xpu)
        out_indices_xpu = torch.empty(
            (num_tiles, self.L), dtype=torch.int16, device=dev
        )
        torch.xpu.synchronize(self.device_index)
        t_transfer += time.perf_counter() - t0

        out_tiles_np = np.empty((num_tiles, self.L), dtype=np.float32)
        out_indices_np = np.empty((num_tiles, self.L), dtype=np.uint16)

        num_chunks = 0
        for start in range(0, num_tiles, admitted_b):
            stop = min(start + admitted_b, num_tiles)
            chunk_n = stop - start
            num_chunks += 1

            c_in = in_xpu[start:stop]
            c_out_w = out_tiles_xpu[start:stop]
            c_out_i = out_indices_xpu[start:stop]
            c_costs = temp_costs[:chunk_n]
            c_edges = temp_edges[:chunk_n]

            if traceback_mode == "gpu_resident":
                ts0 = time.perf_counter()
                ext.quantize_tiles(
                    c_in,
                    c_out_w,
                    c_out_i,
                    c_costs,
                    c_edges,
                    self.K,
                    self.mcg_flag,
                    self.mul1_flag,
                    self.sg_size,
                    self.force_global_costs,
                    self.wave_cap_override,
                )
                torch.xpu.synchronize(self.device_index)
                t_search += time.perf_counter() - ts0

            elif traceback_mode == "gpu_resident_split":
                pre_states = torch.full(
                    (chunk_n,), -1, dtype=torch.int32, device=dev
                )
                end_states = torch.empty(
                    (chunk_n,), dtype=torch.int32, device=dev
                )
                torch.xpu.synchronize(self.device_index)

                # Pass 1 forward
                ts0 = time.perf_counter()
                ext.quantize_tiles_forward_pass(
                    c_in,
                    c_costs,
                    c_edges,
                    pre_states,
                    self.K,
                    self.mcg_flag,
                    self.mul1_flag,
                    self.L // 2,
                    0,
                    self.sg_size,
                    self.force_global_costs,
                )
                torch.xpu.synchronize(self.device_index)
                t_search += time.perf_counter() - ts0

                # Pass 1 backward on GPU
                tb0 = time.perf_counter()
                ext.quantize_tiles_backward_pass(
                    c_costs,
                    c_edges,
                    c_out_w,
                    c_out_i,
                    end_states,
                    chunk_n,
                    self.K,
                    self.mcg_flag,
                    self.mul1_flag,
                    self.L // 2,
                    0,
                    False,
                    self.sg_size,
                )
                torch.xpu.synchronize(self.device_index)
                t_traceback += time.perf_counter() - tb0

                # Pass 2 forward
                ts1 = time.perf_counter()
                ext.quantize_tiles_forward_pass(
                    c_in,
                    c_costs,
                    c_edges,
                    end_states,
                    self.K,
                    self.mcg_flag,
                    self.mul1_flag,
                    0,
                    1,
                    self.sg_size,
                    self.force_global_costs,
                )
                torch.xpu.synchronize(self.device_index)
                t_search += time.perf_counter() - ts1

                # Pass 2 backward on GPU
                tb1 = time.perf_counter()
                ext.quantize_tiles_backward_pass(
                    c_costs,
                    c_edges,
                    c_out_w,
                    c_out_i,
                    end_states,
                    chunk_n,
                    self.K,
                    self.mcg_flag,
                    self.mul1_flag,
                    0,
                    1,
                    True,
                    self.sg_size,
                )
                torch.xpu.synchronize(self.device_index)
                t_traceback += time.perf_counter() - tb1

            elif traceback_mode == "d2h_cpu":
                pre_states = torch.full(
                    (chunk_n,), -1, dtype=torch.int32, device=dev
                )
                torch.xpu.synchronize(self.device_index)

                # Pass 1 forward on GPU
                ts0 = time.perf_counter()
                ext.quantize_tiles_forward_pass(
                    c_in,
                    c_costs,
                    c_edges,
                    pre_states,
                    self.K,
                    self.mcg_flag,
                    self.mul1_flag,
                    self.L // 2,
                    0,
                    self.sg_size,
                    self.force_global_costs,
                )
                torch.xpu.synchronize(self.device_index)
                t_search += time.perf_counter() - ts0

                # Copy Pass 1 costs + edges to CPU
                tx0 = time.perf_counter()
                c1_cpu = c_costs[:, 0, :].cpu().numpy()
                edges1_cpu = c_edges.cpu().numpy().view(np.uint16)
                t_transfer += time.perf_counter() - tx0

                # Pass 1 traceback on CPU
                tb0 = time.perf_counter()
                end_states_np = np.zeros(chunk_n, dtype=np.int32)
                for ti in range(chunk_n):
                    costs_row = c1_cpu[ti]
                    if not np.any(np.isfinite(costs_row)) or np.all(costs_row == H_INF):
                        edge = 0
                    else:
                        edge = int(np.argmin(costs_row))
                    for step in range(self.L - 1, -1, -1):
                        ri = (step + self.L // 2) % self.L
                        edge = int(edges1_cpu[ti, ri, edge])
                        if ri == 0:
                            break
                    end_states_np[ti] = edge
                t_traceback += time.perf_counter() - tb0

                tx1 = time.perf_counter()
                end_states_xpu = torch.from_numpy(end_states_np).to(dev)
                torch.xpu.synchronize(self.device_index)
                t_transfer += time.perf_counter() - tx1

                # Pass 2 forward on GPU
                ts1 = time.perf_counter()
                ext.quantize_tiles_forward_pass(
                    c_in,
                    c_costs,
                    c_edges,
                    end_states_xpu,
                    self.K,
                    self.mcg_flag,
                    self.mul1_flag,
                    0,
                    1,
                    self.sg_size,
                    self.force_global_costs,
                )
                torch.xpu.synchronize(self.device_index)
                t_search += time.perf_counter() - ts1

                # Copy Pass 2 edges to CPU
                tx2 = time.perf_counter()
                edges2_cpu = c_edges.cpu().numpy().view(np.uint16)
                t_transfer += time.perf_counter() - tx2

                # Pass 2 traceback on CPU
                tb1 = time.perf_counter()
                for ti in range(chunk_n):
                    edge = int(end_states_np[ti])
                    for step in range(self.L - 1, -1, -1):
                        prev = int(edges2_cpu[ti, step, edge])
                        encoded = (prev << self.K) | edge
                        edge = prev
                        out_indices_np[start + ti, step] = np.uint16(encoded)
                        out_tiles_np[start + ti, step] = float(self.lut[encoded])
                t_traceback += time.perf_counter() - tb1

            elif traceback_mode == "usm_host":
                out_w_cpu = torch.empty((chunk_n, self.L), dtype=torch.float32)
                out_i_cpu = torch.empty((chunk_n, self.L), dtype=torch.int16)
                lut_cpu = torch.from_numpy(self.lut.astype(np.float32))
                timings = ext.quantize_tiles_usm_host(
                    c_in,
                    c_costs,
                    out_w_cpu,
                    out_i_cpu,
                    lut_cpu,
                    self.K,
                    self.mcg_flag,
                    self.mul1_flag,
                    self.sg_size,
                )
                t_search += float(timings[0])
                t_traceback += float(timings[1])
                out_tiles_np[start:stop] = out_w_cpu.numpy()
                out_indices_np[start:stop] = out_i_cpu.numpy().view(np.uint16)

            if traceback_mode in ("gpu_resident", "gpu_resident_split"):
                tx = time.perf_counter()
                out_tiles_np[start:stop] = c_out_w.cpu().numpy()
                out_indices_np[start:stop] = (
                    c_out_i.cpu().numpy().view(np.uint16)
                )
                t_transfer += time.perf_counter() - tx

        total_seconds = time.perf_counter() - t_total0
        del in_xpu, out_tiles_xpu, out_indices_xpu

        t_post0 = time.perf_counter()
        reconstructed_perm = out_tiles_np.astype(np.float16)
        if pre_permute:
            reconstructed_all = reconstructed_perm[:, self.inv_perm]
        else:
            reconstructed_all = reconstructed_perm

        # Compute exact fp16 accumulated path cost along the winning trajectory
        dh_all = (reconstructed_perm - perm_tiles_f16).astype(np.float64)
        with np.errstate(over="ignore"):
            acc_cost = (dh_all[:, 0] * dh_all[:, 0]).astype(np.float16)
            for step in range(1, self.L):
                acc_cost = (
                    dh_all[:, step] * dh_all[:, step] + acc_cost.astype(np.float64)
                ).astype(np.float16)
        path_costs_np = acc_cost.astype(np.float32)

        results: List[EncodingResult] = []
        edges_mask = self.edges - 1
        for idx in range(num_tiles):
            indices_row = out_indices_np[idx]
            rec_perm_row = reconstructed_perm[idx]
            rec_row = reconstructed_all[idx]
            end_state = int(indices_row[self.L - 1]) & edges_mask
            if self.L == 256:
                packed = swap16_storage(pack_trellis(indices_row, self.K))
            else:
                packed = np.array([], dtype=np.uint16)

            orig_eval = (
                flat_tiles[idx]
                if pre_permute
                else perm_tiles_f16[idx].astype(np.float32)
            )
            distortion = float(
                np.sum((orig_eval - rec_row.astype(np.float32)) ** 2)
            )
            results.append(
                EncodingResult(
                    output_indices=indices_row,
                    reconstructed=rec_row,
                    reconstructed_perm=rec_perm_row,
                    packed=packed,
                    end_state=end_state,
                    distortion=distortion,
                    mse=distortion / float(self.L),
                    path_cost=float(path_costs_np[idx]),
                    K=self.K,
                    cb=self.cb,
                )
            )
        postprocess_seconds = time.perf_counter() - t_post0

        return BatchEncodingBreakdown(
            results=results,
            reconstructed_tiles=reconstructed_all,
            reconstructed_perm=reconstructed_perm,
            output_indices=out_indices_np,
            admitted_b=admitted_b,
            num_chunks=num_chunks,
            total_seconds=total_seconds + postprocess_seconds,
            search_seconds=t_search,
            traceback_seconds=t_traceback,
            transfer_seconds=t_transfer,
            postprocess_seconds=postprocess_seconds,
            traceback_mode=traceback_mode,
            sg_size=self.sg_size,
            force_global_costs=self.force_global_costs,
        )
