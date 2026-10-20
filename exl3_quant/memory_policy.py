"""Configurable memory-pressure policy for Arc 140V scratch allocations.

Implements V140-02 / P0-20:

* An idle- and live-memory snapshot combining OS (``/proc/meminfo``) and
  framework (``torch.xpu``) accounting.
* A configurable reserve (default 6 GiB) that must remain untouched by any
  scratch test, checked *before* allocation and re-checked after commit.
* Explicit abort (``MemoryPressureError``) with full context instead of an
  opaque OOM.

On the integrated Arc 140V, XPU allocations are backed by the same physical
DRAM the OS reports. The two views therefore overlap rather than add up:
``torch.xpu.mem_get_info`` free is a *device-visible subset* of system
available memory. The policy uses ``min(device_free, system_available)`` as the
binding figure so the reserve is never double-spent, and compares deltas of
both counters during measurements (V140-04) to verify that behaviour.

**Measured caveat (V140-02/V140-04, 2026-10-20):** on this driver neither
``MemAvailable`` nor ``mem_get_info`` free falls by the number of bytes a
process commits — a touched 1 GiB XPU tensor moved them by only ~200/9 MiB
while ``torch.xpu.memory_allocated`` moved by exactly 1024 MiB. The driver
serves shared allocations from a pool that the generic counters do not
decrement. Admission checks therefore *also* subtract the caller-reported
``outstanding_bytes`` (the framework's live allocation total); relying on the
free counters alone would over-admit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

BYTES_PER_MIB = 1024 ** 2
BYTES_PER_GIB = 1024 ** 3

DEFAULT_RESERVE_BYTES = 6 * BYTES_PER_GIB
DEFAULT_SAFETY_SLACK_BYTES = 256 * BYTES_PER_MIB

# /proc/meminfo keys captured verbatim (kB units in the file).
_MEMINFO_FIELDS = (
    "MemTotal",
    "MemFree",
    "MemAvailable",
    "Cached",
    "SwapTotal",
    "SwapFree",
    "SwapCached",
)


def format_bytes(value: Optional[int]) -> str:
    """Render byte counts with binary units for messages and tables."""
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= BYTES_PER_GIB:
        return f"{sign}{value / BYTES_PER_GIB:.2f} GiB"
    if value >= BYTES_PER_MIB:
        return f"{sign}{value / BYTES_PER_MIB:.1f} MiB"
    return f"{sign}{value} B"


def read_system_memory(path: str = "/proc/meminfo") -> dict:
    """Parse ``/proc/meminfo`` into a dict of byte counts.

    Lines whose key is not captured are ignored; missing optional fields
    (e.g. swap on a swap-less host) are reported as 0.
    """
    raw: dict = {}
    with open(path, "r", encoding="ascii") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            if key in _MEMINFO_FIELDS:
                raw[key] = int(rest.split()[0]) * 1024  # kB -> bytes
    return {field: raw.get(field, 0) for field in _MEMINFO_FIELDS}


@dataclass(frozen=True)
class MemorySnapshot:
    """Point-in-time memory state from both the OS and the XPU runtime."""

    gpu_available: bool
    system_total_bytes: int
    system_available_bytes: int
    system_free_bytes: int
    cached_bytes: int
    swap_total_bytes: int
    swap_free_bytes: int
    device_free_bytes: int
    device_total_bytes: int
    device_allocated_bytes: int
    device_reserved_bytes: int

    @property
    def binding_available_bytes(self) -> int:
        """Shared-memory view: the tighter of the two overlapping counters."""
        if self.gpu_available and self.device_free_bytes:
            return min(self.device_free_bytes, self.system_available_bytes)
        return self.system_available_bytes

    def as_dict(self) -> dict:
        return {
            "gpu_available": self.gpu_available,
            "system_total_bytes": self.system_total_bytes,
            "system_available_bytes": self.system_available_bytes,
            "system_free_bytes": self.system_free_bytes,
            "cached_bytes": self.cached_bytes,
            "swap_total_bytes": self.swap_total_bytes,
            "swap_free_bytes": self.swap_free_bytes,
            "device_free_bytes": self.device_free_bytes,
            "device_total_bytes": self.device_total_bytes,
            "device_allocated_bytes": self.device_allocated_bytes,
            "device_reserved_bytes": self.device_reserved_bytes,
            "binding_available_bytes": self.binding_available_bytes,
        }


def capture_snapshot(device_index: int = 0,
                     meminfo_path: str = "/proc/meminfo") -> MemorySnapshot:
    """Capture the live OS + XPU memory state (works without an XPU device)."""
    system = read_system_memory(meminfo_path)

    gpu_available = False
    device_free = device_total = 0
    device_allocated = device_reserved = 0
    try:
        import torch
        if torch.xpu.is_available():
            free, total = torch.xpu.mem_get_info(device_index)
            device_free, device_total = int(free), int(total)
            device_allocated = int(torch.xpu.memory_allocated(device_index))
            device_reserved = int(torch.xpu.memory_reserved(device_index))
            gpu_available = True
    except Exception:  # pragma: no cover - CPU-only hosts / broken driver
        gpu_available = False

    return MemorySnapshot(
        gpu_available=gpu_available,
        system_total_bytes=system["MemTotal"],
        system_available_bytes=system["MemAvailable"],
        system_free_bytes=system["MemFree"],
        cached_bytes=system["Cached"],
        swap_total_bytes=system["SwapTotal"],
        swap_free_bytes=system["SwapFree"],
        device_free_bytes=device_free,
        device_total_bytes=device_total,
        device_allocated_bytes=device_allocated,
        device_reserved_bytes=device_reserved,
    )


class MemoryPressureError(RuntimeError):
    """Raised when an allocation would breach the configured reserve."""

    def __init__(self, message: str, need_bytes: int, usable_bytes: int,
                 snapshot: MemorySnapshot, context: str = ""):
        super().__init__(message)
        self.need_bytes = need_bytes
        self.usable_bytes = usable_bytes
        self.snapshot = snapshot
        self.context = context

    def as_dict(self) -> dict:
        return {
            "error": "MemoryPressureError",
            "context": self.context,
            "need_bytes": self.need_bytes,
            "usable_bytes": self.usable_bytes,
            "message": str(self),
            "snapshot": self.snapshot.as_dict(),
        }


@dataclass(frozen=True)
class MemoryPolicy:
    """Reserve-based admission policy for scratch tests on shared memory.

    ``reserve_bytes`` is the headroom (system RAM *and* GPU-visible memory,
    since they share the same DRAM) that must remain free at all times.
    ``max_single_alloc_bytes`` mirrors the runtime's single-allocation ceiling.
    ``safety_slack_bytes`` tolerates measurement noise when re-checking after
    a commit; it never relaxes the pre-allocation check.
    """

    reserve_bytes: int = DEFAULT_RESERVE_BYTES
    safety_slack_bytes: int = DEFAULT_SAFETY_SLACK_BYTES
    max_single_alloc_bytes: Optional[int] = None

    def usable_bytes(self, snapshot: MemorySnapshot,
                     outstanding_bytes: int = 0) -> int:
        """Bytes that may still be committed while keeping the reserve.

        ``outstanding_bytes`` is the framework's live allocation total for
        this process. It is subtracted because the platform free counters do
        not reliably decrement for XPU allocations (see module docstring).
        """
        if outstanding_bytes < 0:
            raise ValueError(
                f"outstanding_bytes must be non-negative, got {outstanding_bytes}")
        return max(0, snapshot.binding_available_bytes
                   - self.reserve_bytes - outstanding_bytes)

    def check(self, need_bytes: int, snapshot: MemorySnapshot,
              outstanding_bytes: int = 0, context: str = "") -> int:
        """Raise ``MemoryPressureError`` unless ``need_bytes`` fits the budget.

        ``need_bytes`` is the request being admitted; ``outstanding_bytes`` is
        everything already live in the process. Callers splitting scratch into
        several buffers should pass the cumulative outstanding amount so the
        reservation is not spent twice. Returns the remaining budget on
        success so callers can log it.
        """
        if need_bytes < 0:
            raise ValueError(f"need_bytes must be non-negative, got {need_bytes}")

        usable = self.usable_bytes(snapshot, outstanding_bytes)
        if self.max_single_alloc_bytes is not None and \
                need_bytes > self.max_single_alloc_bytes:
            raise MemoryPressureError(
                f"single-allocation ceiling exceeded{self._ctx(context)}: "
                f"need {format_bytes(need_bytes)} > ceiling "
                f"{format_bytes(self.max_single_alloc_bytes)}",
                need_bytes, usable, snapshot, context)

        if need_bytes > usable:
            detail = ""
            if outstanding_bytes:
                detail = (f"; outstanding {format_bytes(outstanding_bytes)} "
                          f"already committed")
            raise MemoryPressureError(
                f"memory-pressure abort{self._ctx(context)}: need "
                f"{format_bytes(need_bytes)} but only {format_bytes(usable)} "
                f"usable (binding available {format_bytes(snapshot.binding_available_bytes)}, "
                f"reserve {format_bytes(self.reserve_bytes)}{detail})",
                need_bytes, usable, snapshot, context)
        return usable

    def check_after_commit(self, snapshot: MemorySnapshot,
                           context: str = "post-commit") -> None:
        """Abort if the reserve was breached despite the pre-check."""
        floor = self.reserve_bytes - self.safety_slack_bytes
        if snapshot.binding_available_bytes < floor:
            raise MemoryPressureError(
                f"reserve breached{self._ctx(context)}: binding available "
                f"{format_bytes(snapshot.binding_available_bytes)} < floor "
                f"{format_bytes(floor)} (reserve "
                f"{format_bytes(self.reserve_bytes)}, slack "
                f"{format_bytes(self.safety_slack_bytes)})",
                0, self.usable_bytes(snapshot), snapshot, context)

    def describe(self, snapshot: MemorySnapshot) -> dict:
        return {
            "reserve_bytes": self.reserve_bytes,
            "safety_slack_bytes": self.safety_slack_bytes,
            "max_single_alloc_bytes": self.max_single_alloc_bytes,
            "usable_bytes": self.usable_bytes(snapshot),
            "snapshot": snapshot.as_dict(),
        }

    @staticmethod
    def _ctx(context: str) -> str:
        return f" [{context}]" if context else ""
