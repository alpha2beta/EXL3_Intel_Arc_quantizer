#!/usr/bin/env python
"""Capture the idle Arc 140V memory baseline (V140-02).

Samples OS memory (``/proc/meminfo``) and XPU-visible memory before any large
scratch test, so later pressure measurements have a trustworthy reference.
Writes ``artifacts/environment/idle_baseline.json``.

Usage:
    PYTHONPATH=. python scripts/measure_idle_baseline.py [--samples N] [--interval S]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exl3_quant.memory_policy import (  # noqa: E402
    MemoryPolicy,
    capture_snapshot,
    format_bytes,
)

ARTIFACT = REPO_ROOT / "artifacts" / "environment" / "idle_baseline.json"


def summarize(samples: list[dict], field: str) -> dict:
    values = [s[field] for s in samples]
    return {
        "min": min(values),
        "median": int(statistics.median(values)),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10,
                        help="number of snapshots (default 10)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="seconds between snapshots (default 1.0)")
    parser.add_argument("--reserve-gib", type=float, default=6.0,
                        help="reserve held back from allocation (default 6 GiB)")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be >= 1")

    policy = MemoryPolicy(reserve_bytes=int(args.reserve_gib * 1024 ** 3))
    snapshots = []
    for index in range(args.samples):
        snap = capture_snapshot()
        snapshots.append(snap.as_dict())
        if index + 1 < args.samples:
            time.sleep(args.interval)

    fields = ("system_available_bytes", "system_free_bytes", "cached_bytes",
              "swap_free_bytes", "device_free_bytes")
    summary = {field: summarize(snapshots, field) for field in fields}

    first = capture_snapshot()
    document = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "time_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "samples": args.samples,
        "interval_seconds": args.interval,
        "device": {
            "gpu_available": first.gpu_available,
            "device_total_bytes": first.device_total_bytes,
        },
        "policy": {
            "reserve_bytes": policy.reserve_bytes,
            "reserve_gib": policy.reserve_bytes / 1024 ** 3,
            "usable_bytes_at_idle": policy.usable_bytes(first),
            "usable_source": "min(device_free, system_available) - reserve",
        },
        "summary": summary,
        "raw_samples": snapshots,
        "note": (
            "Arc 140V is an integrated GPU: XPU memory is backed by the same "
            "DRAM the OS reports, so device and system counters overlap and "
            "are not additive. MemAvailable includes reclaimable page cache, "
            "which is why it exceeds device free here."
        ),
    }

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    print(f"idle baseline over {args.samples} samples "
          f"({args.interval:g}s apart)")
    print(f"  MemTotal            {format_bytes(first.system_total_bytes)}")
    print(f"  MemAvailable        min {format_bytes(summary['system_available_bytes']['min'])} "
          f"/ med {format_bytes(summary['system_available_bytes']['median'])} "
          f"/ max {format_bytes(summary['system_available_bytes']['max'])}")
    print(f"  SwapFree            min {format_bytes(summary['swap_free_bytes']['min'])} "
          f"/ med {format_bytes(summary['swap_free_bytes']['median'])}")
    if first.gpu_available:
        print(f"  XPU total           {format_bytes(first.device_total_bytes)}")
        print(f"  XPU free            min {format_bytes(summary['device_free_bytes']['min'])} "
              f"/ med {format_bytes(summary['device_free_bytes']['median'])} "
              f"/ max {format_bytes(summary['device_free_bytes']['max'])}")
    print(f"  reserve             {format_bytes(policy.reserve_bytes)} "
          f"({policy.reserve_bytes / 1024 ** 3:.1f} GiB)")
    print(f"  usable at idle      {format_bytes(policy.usable_bytes(first))}")
    print(f"  << wrote {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
