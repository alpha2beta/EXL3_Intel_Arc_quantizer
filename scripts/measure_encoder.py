#!/usr/bin/env python3
"""Measure raw XPU tile encoder throughput and latency.

Reproduces the pinned micro-profile from B60_B580.md / progress.md Step 25.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from exl3_quant.ldlq import make_xpu_tile_encoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure XPU tile encoder throughput.")
    parser.add_argument("--device", type=int, default=0, help="XPU device index (default: 0)")
    parser.add_argument("--tiles", type=int, default=65536, help="Number of 256-element tiles (default: 65536)")
    parser.add_argument("--k", type=int, default=2, help="K bits (default: 2)")
    parser.add_argument("--warmup-tiles", type=int, default=1024, help="Warmup tiles (default: 1024)")
    parser.add_argument("--json", type=str, default="", help="Optional output JSON path")
    args = parser.parse_args()

    dev_str = f"xpu:{args.device}"
    dev_name = torch.xpu.get_device_name(args.device)
    print(f"Target device: {dev_str} ({dev_name})")
    print(f"Parameters: K={args.k}, tiles={args.tiles}")

    # Initialize encoder
    enc = make_xpu_tile_encoder(K=args.k, device_index=args.device)

    # Warm-up run
    warm_tiles = torch.randn(args.warmup_tiles, 256, device=dev_str) * 0.02
    enc(warm_tiles)
    torch.xpu.synchronize(args.device)
    del warm_tiles
    torch.xpu.empty_cache()

    # Benchmark run
    torch.manual_seed(42)
    tiles = torch.randn(args.tiles, 256, device=dev_str) * 0.02
    torch.xpu.synchronize(args.device)

    t0 = time.perf_counter()
    res = enc(tiles)
    torch.xpu.synchronize(args.device)
    dt = time.perf_counter() - t0

    throughput = args.tiles / dt
    latency_ms = (dt * 1e3) / args.tiles

    print(f"Completed {args.tiles} tiles in {dt:.3f} s")
    print(f"Throughput: {throughput:,.1f} tiles/s")
    print(f"Latency:    {latency_ms:.3f} ms/tile")

    results = {
        "device_index": args.device,
        "device_name": dev_name,
        "tiles": args.tiles,
        "k": args.k,
        "elapsed_seconds": round(dt, 4),
        "throughput_tiles_per_sec": round(throughput, 2),
        "latency_ms_per_tile": round(latency_ms, 4),
    }

    if args.json:
        out_p = Path(args.json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2))
        print(f"Wrote benchmark results to {out_p}")


if __name__ == "__main__":
    main()
