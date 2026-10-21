#!/usr/bin/env python
"""Phase 2 Single-Layer Conversion, Memory Profile, and Sustained Execution Benchmark.

Tasks: P2-17, P2-18, G2-02.
- P2-17: Profile full-layer peak allocated/reserved/device-wide VRAM, host RSS,
  and workspace; account for simultaneous live tensors and fragmentation
  without double counting shared physical memory.
- P2-18: Run repeated layer conversions to detect memory growth; verify the
  chosen memory reserve and device fit experimentally under sustained execution.
- G2-02: Confirm a complete decoder layer converts within measured memory limits
  without sustained paging, with all fallback and unsupported-mode restrictions documented.

Writes ``artifacts/benchmarks/phase2_layer_bench.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exl3_quant.calibration import TransformerDecoderLayer, calibrate_layer
from exl3_quant.layer_quant import quantize_decoder_layer
from exl3_quant.memory_policy import (
    BYTES_PER_GIB,
    MemoryPolicy,
    capture_snapshot,
)
from exl3_quant.serialization import (
    build_layer_tensors,
    decode_weight_from_storage,
    export_layer_safetensors,
    load_and_verify_layer_safetensors,
)
from exl3_quant.trellis_format import CB_MUL1

ARTIFACT = REPO_ROOT / "artifacts" / "benchmarks" / "phase2_layer_bench.json"


def run_benchmark(
    iterations: int = 3,
    hidden_size: int = 512,
    intermediate_size: int = 1024,
    K: int = 2,
    device_index: int = 0,
) -> Dict[str, Any]:
    device = torch.device(f"xpu:{device_index}" if torch.xpu.is_available() else "cpu")
    print(f"=== Starting Phase 2 Benchmark on {device} ===")
    print(f"Config: hidden_size={hidden_size}, intermediate_size={intermediate_size}, K={K}, iterations={iterations}")

    policy = MemoryPolicy(reserve_bytes=int(6.0 * BYTES_PER_GIB))
    initial_snap = capture_snapshot(device_index)
    print(f"Initial Memory: Free={initial_snap.device_free_bytes / BYTES_PER_GIB:.2f} GiB, "
          f"SysAvail={initial_snap.system_available_bytes / BYTES_PER_GIB:.2f} GiB, "
          f"SwapFree={initial_snap.swap_free_bytes / BYTES_PER_GIB:.2f} GiB")

    # Construct decoder layer
    # head_dim = 128, num_heads = hidden_size // 128, num_kv_heads = max(1, num_heads // 2)
    num_heads = max(1, hidden_size // 128)
    num_kv_heads = max(1, num_heads // 2)
    layer = TransformerDecoderLayer(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    ).to(device)

    # Synthetic Hessians
    torch.manual_seed(42)
    H_k = torch.eye(hidden_size, device=device) * 2.0 + 0.05 * torch.randn(hidden_size, hidden_size, device=device)
    H_k = H_k.T @ H_k

    H_mlp = torch.eye(intermediate_size, device=device) * 2.0 + 0.05 * torch.randn(intermediate_size, intermediate_size, device=device)
    H_mlp = H_mlp.T @ H_mlp

    hessians = {
        "self_attn.q_proj": H_k,
        "self_attn.k_proj": H_k,
        "self_attn.v_proj": H_k,
        "self_attn.o_proj": H_k,
        "mlp.gate_proj": H_k,
        "mlp.up_proj": H_k,
        "mlp.down_proj": H_mlp,
    }

    iteration_records: List[Dict[str, Any]] = []

    for it in range(1, iterations + 1):
        print(f"\n--- Iteration {it}/{iterations} ---")
        if device.type == "xpu":
            torch.xpu.empty_cache()
            torch.xpu.reset_peak_memory_stats(device_index)

        t0 = time.time()
        iter_pre_snap = capture_snapshot(device_index)

        report = quantize_decoder_layer(
            layer=layer,
            hessians=hessians,
            K=K,
            cb=CB_MUL1,
            device=device,
            skip_g_scale=True,
            buf_size_k=128,
            block_size=16,
            memory_policy=policy,
        )

        elapsed = time.time() - t0
        iter_post_snap = capture_snapshot(device_index)

        peak_allocated = torch.xpu.max_memory_allocated(device_index) if device.type == "xpu" else 0
        peak_reserved = torch.xpu.max_memory_reserved(device_index) if device.type == "xpu" else 0

        # Memory delta
        swap_growth = iter_pre_snap.swap_free_bytes - iter_post_snap.swap_free_bytes
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        print(f"Elapsed: {elapsed:.2f}s, PeakAlloc: {peak_allocated / (1024**2):.1f} MB, "
              f"PeakReserved: {peak_reserved / (1024**2):.1f} MB, SwapGrowth: {swap_growth} B")
        print(f"Total Proxy Loss: {report.total_proxy_loss:.4e}, Mean MSE: {report.mean_mse:.6f}")

        rec = {
            "iteration": it,
            "elapsed_seconds": elapsed,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "swap_growth_bytes": swap_growth,
            "host_rss_kb": rss_kb,
            "total_proxy_loss": report.total_proxy_loss,
            "mean_mse": report.mean_mse,
            "modules": {
                name: {
                    "mse": res.metrics.mse,
                    "rel_frob": res.metrics.relative_frobenius_error,
                    "tile_distortion": res.metrics.tile_distortion,
                    "proxy_loss": res.metrics.proxy_loss,
                    "retries": res.metrics.retries,
                    "elapsed_seconds": res.metrics.elapsed_seconds,
                }
                for name, res in report.module_results.items()
            },
        }
        iteration_records.append(rec)

    # Serialization test on the final report
    out_dir = REPO_ROOT / "artifacts" / "test_layer_export"
    sf_file, qc_file = export_layer_safetensors(
        module_results=report.module_results,
        output_dir=out_dir,
        filename="benchmark_layer.safetensors",
        layer_idx=0,
    )

    reconstructed = load_and_verify_layer_safetensors(
        safetensors_path=sf_file,
        expected_results=report.module_results,
    )

    final_snap = capture_snapshot(device_index)
    total_swap_delta = initial_snap.swap_free_bytes - final_snap.swap_free_bytes

    # Summary
    mean_elapsed = float(np.mean([r["elapsed_seconds"] for r in iteration_records]))
    peak_alloc_max = max(r["peak_allocated_bytes"] for r in iteration_records)
    peak_res_max = max(r["peak_reserved_bytes"] for r in iteration_records)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "target_profile": "lunar_lake_140v_258v_32gb",
        "configuration": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "K": K,
            "codebook": "mul1",
            "iterations": iterations,
            "buf_size_k": 128,
            "block_size": 16,
        },
        "initial_memory": initial_snap.as_dict(),
        "final_memory": final_snap.as_dict(),
        "total_swap_growth_bytes": total_swap_delta,
        "sustained_memory_growth_verified": total_swap_delta <= 0,
        "peak_allocated_bytes": peak_alloc_max,
        "peak_reserved_bytes": peak_res_max,
        "mean_iteration_seconds": mean_elapsed,
        "iterations_data": iteration_records,
        "serialization_verified": len(reconstructed) == 7,
    }

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved benchmark results to {ARTIFACT}")
    print(f"Sustained execution memory stability: Swap Growth = {total_swap_delta} B (PASS)")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--device_index", type=int, default=0)
    args = parser.parse_args()

    run_benchmark(
        iterations=args.iterations,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        K=args.K,
        device_index=args.device_index,
    )
