"""Configurable memory-pressure policy for Arc 140V scratch allocations.

Implements V140-02 / P0-20:

* An idle- and live-memory snapshot combining OS (``/proc/meminfo``) and
  framework (``torch.xpu``) accounting. On non-Linux hosts the OS counters come
  from the platform's native interface instead (``GlobalMemoryStatusEx`` on
  Windows), so the snapshot degrades to "counters unavailable" rather than
  raising.
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

import os
from dataclasses import dataclass, field
from typing import Optional

BYTES_PER_MIB = 1024 ** 2
BYTES_PER_GIB = 1024 ** 3

DEFAULT_RESERVE_BYTES = 6 * BYTES_PER_GIB
DEFAULT_SAFETY_SLACK_BYTES = 256 * BYTES_PER_MIB

#: Environment overrides for the admission reserve. ``..._RESERVE_BYTES`` takes
#: an integer byte count, ``..._RESERVE_GIB`` a float number of GiB (the GiB
#: form wins if both are set). Unset means :data:`DEFAULT_RESERVE_BYTES`.
#:
#: The reserve is device-side headroom: on an integrated (UMA) device the
#: scratch arena draws from the same DRAM as the host, and the *device pool*
#: — not the host's free memory — is the binding dimension. On a host whose
#: device pool is small but whose RAM has plenty of headroom, a 6 GiB device
#: reserve can leave no admissible budget at all while the system is in no
#: danger whatsoever; the override makes that trade-off explicit and
#: auditable instead of implicit.
ENV_RESERVE_BYTES = "EXL3_MEMORY_RESERVE_BYTES"
ENV_RESERVE_GIB = "EXL3_MEMORY_RESERVE_GIB"


def default_reserve_bytes() -> int:
    """The configured admission reserve in bytes (environment aware)."""
    raw_gib = os.environ.get(ENV_RESERVE_GIB)
    if raw_gib:
        try:
            return int(float(raw_gib) * BYTES_PER_GIB)
        except (TypeError, ValueError):
            pass
    raw_bytes = os.environ.get(ENV_RESERVE_BYTES)
    if raw_bytes:
        try:
            return int(raw_bytes)
        except (TypeError, ValueError):
            pass
    return DEFAULT_RESERVE_BYTES

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


def read_system_memory_windows() -> dict:
    """Read OS memory counters on Windows via ``GlobalMemoryStatusEx``.

    Field mapping (approximations, documented rather than silently zeroed):

    * ``MemTotal``    -> ``ullTotalPhys``
    * ``MemAvailable``-> ``ullAvailPhys`` (includes the standby cache, so it is
      the closest Windows analogue of Linux ``MemAvailable``)
    * ``MemFree``     -> ``ullAvailPhys`` (Windows exposes no separate free counter)
    * ``Cached``      -> 0 (no Linux-style page-cache counter is exposed)
    * ``SwapTotal``   -> ``ullTotalPageFile - ullTotalPhys``
    * ``SwapFree``    -> ``ullAvailPageFile - ullAvailPhys``
    """
    import ctypes

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = _MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
        raise OSError("GlobalMemoryStatusEx failed")

    return {
        "MemTotal": int(stat.ullTotalPhys),
        "MemFree": int(stat.ullAvailPhys),
        "MemAvailable": int(stat.ullAvailPhys),
        "Cached": 0,
        "SwapTotal": max(0, int(stat.ullTotalPageFile) - int(stat.ullTotalPhys)),
        "SwapFree": max(0, int(stat.ullAvailPageFile) - int(stat.ullAvailPhys)),
        "SwapCached": 0,
    }


def read_system_memory_platform(path: Optional[str] = None) -> dict:
    """Read OS memory counters using the host platform's native interface.

    Args:
        path: explicit ``/proc/meminfo``-style path to parse. When given, the
            file is parsed as-is (and a missing file is an error). When omitted,
            the platform default is used: ``/proc/meminfo`` on Linux,
            ``GlobalMemoryStatusEx`` on Windows.
    """
    if path is not None:
        return read_system_memory(path)
    if os.name == "nt":
        return read_system_memory_windows()
    return read_system_memory()


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
                     meminfo_path: Optional[str] = None) -> MemorySnapshot:
    """Capture the live OS + XPU memory state (works without an XPU device).

    ``meminfo_path`` defaults to the host platform's native counter source:
    ``/proc/meminfo`` on Linux, ``GlobalMemoryStatusEx`` on Windows.
    """
    system = read_system_memory_platform(meminfo_path)

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

    reserve_bytes: int = field(default_factory=default_reserve_bytes)
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

    def check_after_commit(
        self,
        snapshot: MemorySnapshot,
        pre_snapshot: Optional[MemorySnapshot] = None,
        expected_bytes: int = 0,
        context: str = "post-commit",
    ) -> None:
        """Verify allocation commitment and detect reserve breaches (M1).

        Primary enforcement:
          Verifies allocator-visible growth (`snapshot.device_allocated_bytes -
          pre_snapshot.device_allocated_bytes >= expected_bytes`) when `pre_snapshot`
          and `expected_bytes` are provided. This is the trustworthy process-level
          signal on Intel Arc unified memory.

        Advisory secondary check:
          Checks whether OS/device counters breached the reserve floor
          (`binding_available_bytes < reserve - safety_slack`). On integrated Arc
          140V (UMA), device/OS free counters do not reliably decrement for framework
          allocations due to driver-level memory pooling; this check is advisory and
          serves as an external guard against background system processes.
        """
        if pre_snapshot is not None and expected_bytes > 0:
            growth = snapshot.device_allocated_bytes - pre_snapshot.device_allocated_bytes
            if growth < expected_bytes:
                raise MemoryPressureError(
                    f"allocator commitment failure{self._ctx(context)}: expected at least "
                    f"{format_bytes(expected_bytes)} allocated growth, observed only {format_bytes(growth)}",
                    expected_bytes,
                    self.usable_bytes(snapshot),
                    snapshot,
                    context,
                )

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
