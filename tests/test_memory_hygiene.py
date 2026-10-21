"""Tests for the streaming memory hygiene helpers (P3-09).

The helpers exist because the P3-09 full-model qualification run streams a
19.3 GB checkpoint through a 30.9 GiB shared-DRAM host: pass-through group
payloads and the source page cache would otherwise keep DRAM committed for
the whole run and trip the P0-20 admission policy with a
``MemoryPressureError``.

These tests pin the contract that matters: the hints are advisory, they never
raise, and they never change file content.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from exl3_quant import memory_hygiene  # noqa: E402


class TestMemoryHygiene(unittest.TestCase):
    """Advisory cache/heap release helpers."""

    @unittest.skipUnless(hasattr(os, "posix_fadvise"),
                         "page-cache eviction needs posix_fadvise (POSIX only)")
    def test_drop_file_cache_preserves_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.bin"
            payload = os.urandom(1 << 20)
            path.write_bytes(payload)
            self.assertTrue(memory_hygiene.drop_file_cache(path))
            self.assertEqual(path.read_bytes(), payload)

    def test_drop_file_cache_missing_file_is_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.bin"
            self.assertFalse(memory_hygiene.drop_file_cache(missing))

    def test_drop_file_cache_directory_is_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(memory_hygiene.drop_file_cache(Path(tmp)))

    def test_trim_host_heap_is_best_effort(self) -> None:
        # Returns a bool on Linux (malloc_trim), False elsewhere; never raises.
        self.assertIn(memory_hygiene.trim_host_heap(), (True, False))

    def test_release_after_stream_accepts_none(self) -> None:
        memory_hygiene.release_after_stream(None)  # must not raise
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.bin"
            path.write_bytes(b"x")
            memory_hygiene.release_after_stream(path)  # must not raise


if __name__ == "__main__":
    unittest.main()
