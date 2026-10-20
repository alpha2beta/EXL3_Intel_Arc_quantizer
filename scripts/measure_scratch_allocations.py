#!/usr/bin/env python
"""Measure representative EXL3 scratch allocations through the XPU allocator.

Implements P0-20 and the small-B validation of V140-03:

* Sizes come from :class:`ScratchEstimator` for real configurations, starting
  small (B=1) and increasing (B=16, 64, 128) so nothing large is attempted
  until the small cases are proven safe.
* Every allocation is admitted by :class:`MemoryPolicy` with a configurable
  reserve (default 6 GiB); cumulative outstanding bytes count against the
  budget.
* Buffers are touched (``zero_()``) to force real commits, timed, and then
  released; deltas of device free and system MemAvailable are recorded to
  show how shared backing is counted (V140-04 evidence).
* An intentionally oversized request is refused by the policy to prove the
  abort path triggers before touching memory.

Writes ``artifacts/environment/scratch_allocations.json``.

Measured platform caveat (recorded in the artifact as ``counter_lag_*``): the
generic free counters do not fall by the committed byte count on this driver,
so admission also subtracts the framework's live allocation total.

Usage:
    PYTHONPATH=. python scripts/measure_scratch_allocations.py [--reserve-gib 6]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exl3_quant.memory_policy import (  # noqa: E402
    MemoryPolicy,
    MemoryPressureError,
    capture_snapshot,
    format_bytes,
)
from exl3_quant.scratch_estimator import ScratchEstimator  # noqa: E402

ARTIFACT = REPO_ROOT / "artifacts" / "environment" / "scratch_allocations.json"

# Ascending representative configurations: (K, B) with upstream T=256.
# B=256 (4.03 GiB) is expected to be denied at the default 6 GiB reserve; it is
# included to show the measured capacity bound and the configurable-reserve
# knob at work.
CONFIGS = [(1, 1), (1, 16), (1, 64), (1, 128), (1, 192), (1, 256)]


def measure_config(torch, policy, k: int, b: int) -> dict:
    """Allocate, touch, time, and release one estimator configuration."""
    estimate = ScratchEstimator.estimate_scratch(k=k, b=b)
    record = {
        "k": k,
        "b": b,
        "t": estimate.t,
        "costs_bytes": estimate.costs_bytes,
        "edges_bytes": estimate.edges_bytes,
        "total_bytes": estimate.total_bytes,
        "status": "pending",
    }

    pre = capture_snapshot()
    record["pre"] = pre.as_dict()

    if not pre.gpu_available:
        record["status"] = "skipped-no-xpu"
        return record

    outstanding = int(torch.xpu.memory_allocated(0))
    record["outstanding_before_bytes"] = outstanding
    try:
        # Admit the whole configuration up front (both buffers outstanding).
        usable = policy.check(estimate.total_bytes, pre,
                              outstanding_bytes=outstanding,
                              context=f"K={k} B={b} admit")
        record["usable_bytes"] = usable
    except MemoryPressureError as exc:
        record["status"] = "blocked"
        record["blocked"] = exc.as_dict()
        return record

    if b > 1:
        time.sleep(0.2)  # let the previous release settle

    costs = edges = None
    try:
        t0 = time.perf_counter()
        costs = torch.empty(estimate.costs_bytes // 2, dtype=torch.float16,
                            device="xpu")
        edges = torch.empty(estimate.edges_bytes // 2, dtype=torch.int16,
                            device="xpu")
        allocate_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        costs.zero_()
        edges.zero_()
        torch.xpu.synchronize()
        touch_seconds = time.perf_counter() - t1

        post = capture_snapshot()
        record["post_immediate"] = post.as_dict()
        time.sleep(0.5)  # let lazy driver accounting settle
        settled = capture_snapshot()
        record["post_settled"] = settled.as_dict()
        record["allocate_seconds"] = round(allocate_seconds, 4)
        record["touch_seconds"] = round(touch_seconds, 4)

        # Real commits must not eat into the reserve.
        policy.check_after_commit(settled, context=f"K={k} B={b} post-commit")

        def delta(start, end, field):
            return getattr(start, field) - getattr(end, field)

        dev_delta = delta(pre, settled, "device_free_bytes")
        sys_delta = delta(pre, settled, "system_available_bytes")
        record["accounting"] = {
            "requested_bytes": estimate.total_bytes,
            "device_free_delta_immediate_bytes":
                delta(pre, post, "device_free_bytes"),
            "device_free_delta_settled_bytes": dev_delta,
            "system_available_delta_immediate_bytes":
                delta(pre, post, "system_available_bytes"),
            "system_available_delta_settled_bytes": sys_delta,
            "swap_free_delta_bytes": delta(pre, settled, "swap_free_bytes"),
            # How much of the commitment the generic counters failed to show.
            "counter_lag_settled_bytes":
                estimate.total_bytes - max(dev_delta, sys_delta),
            "device_allocated_bytes": settled.device_allocated_bytes,
            "device_reserved_bytes": settled.device_reserved_bytes,
            "binding_after_bytes": settled.binding_available_bytes,
        }
        record["status"] = "ok"
    except MemoryPressureError as exc:
        record["status"] = "aborted-after-commit"
        record["aborted"] = exc.as_dict()
    except Exception as exc:  # allocator/driver failure
        record["status"] = "allocator-error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["post"] = capture_snapshot().as_dict()
    finally:
        costs = None
        edges = None
        torch.xpu.empty_cache()

    release = capture_snapshot()
    record["after_release"] = release.as_dict()
    return record


def probe_abort_path(policy) -> dict:
    """Prove the policy refuses an oversized request before allocation."""
    snap = capture_snapshot()
    oversize = 20 * 1024 ** 3
    try:
        policy.check(oversize, snap, context="oversize probe")
    except MemoryPressureError as exc:
        return {"status": "refused", "rejection": exc.as_dict()}
    return {"status": "unexpectedly-allowed", "requested_bytes": oversize}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve-gib", type=float, default=6.0)
    parser.add_argument("--out", default=str(ARTIFACT),
                        help="artifact path (default artifacts/environment/...)")
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"torch unavailable: {exc}")
        return 1
    if not torch.xpu.is_available():
        print("torch.xpu is not available; nothing measured")
        return 1

    policy = MemoryPolicy(reserve_bytes=int(args.reserve_gib * 1024 ** 3))
    idle = capture_snapshot()

    results = []
    for k, b in CONFIGS:
        record = measure_config(torch, policy, k, b)
        results.append(record)
        print(f"K={k} B={b:>3}: {record['status']:<22} "
              f"total {format_bytes(record['total_bytes'])}"
              + (f"  alloc {record['allocate_seconds'] * 1e3:.1f} ms, "
                 f"touch {record['touch_seconds'] * 1e3:.1f} ms"
                 if record["status"] == "ok" else ""))

    abort_probe = probe_abort_path(policy)
    print(f"oversize probe (20 GiB): {abort_probe['status']}")

    per_b_bytes = ScratchEstimator.estimate_scratch(k=1, b=1).total_bytes
    max_b = policy.usable_bytes(idle) // per_b_bytes

    lagged = [r for r in results
              if r["status"] == "ok"
              and r["accounting"]["counter_lag_settled_bytes"] > 0]
    findings = {
        "max_b_under_policy": int(max_b),
        "per_b_bytes_k1": per_b_bytes,
        "counters_under_report_commitments": bool(lagged),
        "max_counter_lag_bytes": max(
            (r["accounting"]["counter_lag_settled_bytes"] for r in lagged),
            default=0),
        "swap_unchanged": all(
            r["accounting"]["swap_free_delta_bytes"] <= 0
            for r in results if r["status"] == "ok"),
        "process_commitment_tracked_exactly": all(
            r["accounting"]["device_allocated_bytes"] == r["total_bytes"]
            for r in results if r["status"] == "ok"),
        "device_and_system_move_together": all(
            abs(r["accounting"]["device_free_delta_settled_bytes"]
                - r["accounting"]["system_available_delta_settled_bytes"])
            <= max(64 * 1024 ** 2,
                   r["accounting"]["requested_bytes"] // 4)
            for r in results if r["status"] == "ok"),
    }

    document = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "time_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "policy": {
            "reserve_bytes": policy.reserve_bytes,
            "reserve_gib": policy.reserve_bytes / 1024 ** 3,
            "safety_slack_bytes": policy.safety_slack_bytes,
            "admission": "min(device_free, system_available) "
                         "- reserve - outstanding_allocs",
        },
        "idle": idle.as_dict(),
        "configurations": results,
        "abort_probe": abort_probe,
        "findings": findings,
        "note": (
            "Arc 140V shares DRAM with the CPU: device free and system "
            "MemAvailable are overlapping counters, so deltas are compared "
            "individually rather than summed. Measured on this driver, both "
            "counters under-report process commitments (the driver pool is "
            "not decremented), so admission subtracts the framework's own "
            "live allocation total. memory_reserved reflects the torch "
            "caching allocator's retained pool, which may exceed freshly "
            "allocated bytes."
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"<< wrote {out_path}")

    ok = sum(1 for r in results if r["status"] == "ok")
    blocked = sum(1 for r in results if r["status"] == "blocked")
    print(f"summary: {ok} ok, {blocked} blocked, "
          f"{len(results) - ok - blocked} other; "
          f"max B under policy = {max_b}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
