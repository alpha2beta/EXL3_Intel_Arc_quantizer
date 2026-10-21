"""Page-cache and host-heap hygiene for long streaming conversions (P3-09).

Why this exists
---------------
The P3-09 full-model qualification run streams a 19.3 GB source checkpoint
through a 30.9 GiB shared-DRAM host (Intel Arc iGPU, unified memory). Two
allocation patterns of the bounded converter pin DRAM that the conversion
never needs again:

* the **pass-through groups** (text embed + lm_head ≈ 4.1 GiB, vision
  ≈ 0.9 GiB, MTP ≈ 0.5 GiB) are materialised in host RAM before they are
  written out;
* every **layer read** of the source checkpoint leaves its pages in the
  kernel page cache, so a 32-layer run grows the resident file-backed
  footprint to the full 19.3 GiB.

On this driver the free counters do not fall for XPU allocations but they
*do* fall for host/page-cache pressure (``device_free`` dropped 2.37 GiB for a
4 GiB host allocation in the measured experiment, and 3.31 GiB in
``MemAvailable``). The admission policy of P0-20 therefore sees the real
headroom shrink as the run progresses and eventually aborts with
``MemoryPressureError`` even though the machine has ample free DRAM.

The helpers here return the memory that the conversion has finished with,
without changing any conversion semantics:

* :func:`drop_file_cache` asks the kernel to evict the *clean* page cache of
  a file (``POSIX_FADV_DONTNEED``). Data is never discarded: pages are
  re-read from the file on the next access, and dirty pages are written back
  first, so a file that was just written out stays byte-identical.
* :func:`trim_host_heap` returns freed host arenas to the OS
  (``malloc_trim(0)``), so a transient multi-GiB payload does not keep the
  process RSS high after the payload is dropped.

Both are best-effort and never raise: a conversion must not fail because
the kernel declined an advisory hint.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

__all__ = ["drop_file_cache", "trim_host_heap", "release_after_stream"]


def drop_file_cache(path: os.PathLike | str) -> bool:
    """Evict the clean page cache of ``path`` (best effort).

    Returns ``True`` when the kernel hint was issued. Missing files and
    platforms without ``posix_fadvise`` return ``False`` instead of raising:
    the hint is an optimisation, never a correctness requirement.
    """
    target = Path(path)
    if not target.is_file():
        return False
    if not hasattr(os, "posix_fadvise"):
        return False
    try:
        fd = os.open(str(target), os.O_RDONLY)
    except OSError:
        return False
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def trim_host_heap() -> bool:
    """Return freed host arenas to the OS via ``malloc_trim(0)`` (best effort)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return trim(0) == 0
    except Exception:
        return False


def release_after_stream(path: Optional[os.PathLike | str] = None) -> None:
    """Release what a finished streaming step no longer needs.

    Drops the page cache of ``path`` (when given) and trims the host heap so
    transient payloads (pass-through groups, per-layer tensors) do not keep
    DRAM committed for the rest of the conversion.
    """
    if path is not None:
        drop_file_cache(path)
    trim_host_heap()
