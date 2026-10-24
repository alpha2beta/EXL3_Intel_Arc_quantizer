#!/usr/bin/env python
"""Single-layer qualification measurement (Phase C, P3-16).

Usage:
    PYTHONPATH=. python scripts/measure_qwen38_layer.py [num_rows] [seq_len] [layers]

Defaults:
    num_rows: 4
    seq_len: 2048
    layers: "0,3"

Benchmarks single-layer conversion timing on the active discrete Arc device
using the real Qwen3.8-9B-Distill architecture and weights.
Records:
  - read (layer tensors from checkpoint)
  - build (materialize decoder layer on device)
  - embed-gather (gather calibration tokens & embeddings)
  - calibration (forward hidden states, capture grouped Hessians)
  - quantization (real GSS, K=2, cb=mul1)
  - artifact build + atomic save
  - XPU Hessian working set (GiB, groups)
  - peak host RSS (process max)
  - live XPU memory allocation
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import exl3_quant.converter as conv
from exl3_quant import hybrid
from exl3_quant.calibration import calibrate_layer
from exl3_quant.layer_quant import quantize_decoder_layer
from exl3_quant.serialization import build_layer_tensors


def peak_rss_kib() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def parse_args():
    parser = argparse.ArgumentParser(description="Measure single-layer conversion timing.")
    parser.add_argument("num_rows", nargs="?", type=int, default=4,
                        help="Number of calibration rows (default: 4)")
    parser.add_argument("seq_len", nargs="?", type=int, default=2048,
                        help="Sequence length per row (default: 2048)")
    parser.add_argument("layers", nargs="?", type=str, default="0,3",
                        help="Comma-separated layer indices (default: '0,3')")
    parser.add_argument("--model-dir", type=str, default="/mnt/usb/models/qwen38",
                        help="Path to source model directory")
    parser.add_argument("--corpus", type=str,
                        default="/mnt/usb/models/qwen38/calibration_tokens_wikitext2_test.json",
                        help="Path to calibration corpus json")
    parser.add_argument("--output-dir", type=str, default="/tmp/measure_qwen38_layer_out",
                        help="Working output directory for layer artifacts")
    parser.add_argument("--device", type=str, default="xpu:0",
                        help="PyTorch device string (default: xpu:0)")
    parser.add_argument("--skip-g-scale", action="store_true",
                        help="Skip GSS search (default: False, real GSS)")
    parser.add_argument("--json-out", type=str, default="",
                        help="Output JSON summary path")
    return parser.parse_args()


def measure_layer(
    layer_idx: int,
    geo: conv.ModelGeometry,
    index: conv.ShardedSafetensorsIndex,
    device: torch.device,
    runner: Any,
    rows: List[torch.Tensor],
    out_dir: Path,
    skip_g_scale: bool = False,
) -> Dict[str, Any]:
    print(f"\n========================================================")
    print(f"=== Measuring Layer {layer_idx} ({geo.layer_type(layer_idx)}) on {device} ===")
    print(f"========================================================")

    # 1. Read layer tensors
    t0_read = time.perf_counter()
    prefix = f"{geo.key_prefix}.{layer_idx}."
    tensors = index.read_layer_tensors(layer_idx, prefix=geo.key_prefix)
    norm_weights = {tail: tensors[f"{prefix}{tail}"] for tail in conv._NORM_KEYS}
    norm_key_set = {f"{prefix}{tail}" for tail in conv._NORM_KEYS}
    quantized_key_set = {f"{prefix}{name}.weight" for name in geo.quantized_modules(layer_idx)}
    extra_tensors = (
        {k: v for k, v in tensors.items() if k not in quantized_key_set and k not in norm_key_set}
        if geo.is_hybrid else {}
    )
    t_read = time.perf_counter() - t0_read

    # 2. Build layer
    t0_build = time.perf_counter()
    layer = conv._build_layer(geo, tensors, layer_idx, device)
    del tensors
    index.release_source_cache()
    if device.type == "xpu":
        torch.xpu.synchronize(device.index or 0)
    t_build = time.perf_counter() - t0_build

    # 3. Embed gather
    t0_embed = time.perf_counter()
    with torch.no_grad():
        hidden_states = [
            index.read_embedding_rows(geo.embed_key, row).to(device=device, dtype=torch.float16)
            for row in rows
        ]
    if device.type == "xpu":
        torch.xpu.synchronize(device.index or 0)
    t_embed = time.perf_counter() - t0_embed

    # 4. Calibration
    print(f"Capturing grouped Hessians ({len(rows)} rows x {rows[0].shape[1]} tokens)...")
    t0_cal = time.perf_counter()
    hessians, telemetry = calibrate_layer(
        layer=layer,
        calibration_data=hidden_states,
        device=device,
        layer_runner=runner,
    )
    if device.type == "xpu":
        torch.xpu.synchronize(device.index or 0)
    t_cal = time.perf_counter() - t0_cal

    # Measure Hessian working set
    hessian_bytes = sum(h.numel() * h.element_size() for h in hessians.values())
    hessian_gib = hessian_bytes / (1024 ** 3)
    hessian_groups_count = len(hessians)

    print(f"Hessian working set: {hessian_gib:.3f} GiB across {hessian_groups_count} groups")
    print(f"Read: {t_read:.3f}s | Build: {t_build:.3f}s | Embed: {t_embed:.3f}s | Cal: {t_cal:.3f}s")

    # 5. Quantization
    modules = list(geo.quantized_modules(layer_idx))
    print(f"Quantizing {len(modules)} projections (K=2, mul1, skip_g_scale={skip_g_scale})...")
    t0_quant = time.perf_counter()
    layer_report = quantize_decoder_layer(
        layer=layer,
        hessians=hessians,
        K=2,
        device=device,
        skip_g_scale=skip_g_scale,
        module_names=modules,
    )
    if device.type == "xpu":
        torch.xpu.synchronize(device.index or 0)
    t_quant = time.perf_counter() - t0_quant
    print(f"Quantization finished in {t_quant:.2f} s ({t_quant / 60:.2f} min)")

    # 6. Artifact build + atomic save
    t0_art = time.perf_counter()
    emitted, _storage = build_layer_tensors(
        layer_report.module_results,
        layer_idx=layer_idx,
        prefix=geo.key_prefix,
        norm_weights=norm_weights,
        extra_tensors=extra_tensors,
        preserve_source_dtype=geo.is_hybrid,
    )
    t_art_build = time.perf_counter() - t0_art

    t0_save = time.perf_counter()
    layer_file = out_dir / f"model.layers.{layer_idx}.safetensors"
    conv.atomic_save_safetensors(emitted, layer_file)
    t_art_save = time.perf_counter() - t0_save

    # Telemetry
    host_peak_rss_mib = peak_rss_kib() / 1024
    xpu_allocated_mib = conv.read_xpu_allocated_mib()

    # Per module breakdown
    module_details = {}
    for name, res in layer_report.module_results.items():
        module_details[name] = {
            "elapsed_seconds": res.metrics.elapsed_seconds,
            "g_scale": res.metrics.g_scale,
            "mse": res.metrics.mse,
            "relative_frobenius": res.metrics.relative_frobenius_error,
        }
        print(f"  {name:32s}: {res.metrics.elapsed_seconds:.2f}s | g={res.metrics.g_scale:.4f} | rel_frob={res.metrics.relative_frobenius_error:.4f}")

    del layer, hessians, layer_report, emitted, hidden_states
    if device.type == "xpu":
        torch.xpu.empty_cache()

    return {
        "layer_index": layer_idx,
        "layer_type": geo.layer_type(layer_idx),
        "read_seconds": round(t_read, 3),
        "build_seconds": round(t_build, 3),
        "embed_gather_seconds": round(t_embed, 3),
        "calibration_seconds": round(t_cal, 3),
        "quantization_seconds": round(t_quant, 2),
        "artifact_build_seconds": round(t_art_build, 3),
        "artifact_save_seconds": round(t_art_save, 3),
        "hessian_working_set_gib": round(hessian_gib, 3),
        "hessian_groups": hessian_groups_count,
        "peak_host_rss_mib": round(host_peak_rss_mib, 1),
        "live_xpu_allocated_mib": xpu_allocated_mib,
        "modules": module_details,
    }


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_indices = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    device = torch.device(args.device)

    # Device properties
    dev_name = torch.xpu.get_device_name(device.index or 0) if device.type == "xpu" else "CPU"
    total_mem_gib = (
        torch.xpu.get_device_properties(device.index or 0).total_memory / (1024 ** 3)
        if device.type == "xpu" else 0.0
    )

    print("================================================================")
    print("=== EXL3 Single-Layer Qualification Benchmark (Phase C)     ===")
    print("================================================================")
    print(f"Device:       {dev_name} ({total_mem_gib:.2f} GiB VRAM)")
    print(f"Model Dir:    {model_dir}")
    print(f"Config:       {args.num_rows} rows x {args.seq_len} tokens")
    print(f"Target Layers:{layer_indices}")
    print(f"Skip GSS:     {args.skip_g_scale}")
    print("================================================================")

    geo = conv.load_model_geometry(model_dir, mode="qwen3_5_text_tower")
    index = conv.ShardedSafetensorsIndex(model_dir)
    runner = hybrid.make_calibration_runner(hybrid.make_text_config(geo.text_config), device)

    # Load or generate rows
    corpus_path = Path(args.corpus)
    if corpus_path.is_file():
        print(f"Loading calibration tokens from {corpus_path.name}...")
        with open(corpus_path) as f:
            corpus_data = json.load(f)
        all_rows = corpus_data.get("rows") or corpus_data.get("token_ids") or []
        rows = [
            torch.tensor(r[:args.seq_len], dtype=torch.long).unsqueeze(0)
            for r in all_rows[:args.num_rows]
        ]
    else:
        print("Using synthetic token generator (seed 42)...")
        gen = torch.Generator().manual_seed(42)
        rows = [
            torch.randint(0, geo.vocab_size, (1, args.seq_len), generator=gen)
            for _ in range(args.num_rows)
        ]

    results = []
    for l_idx in layer_indices:
        rec = measure_layer(
            layer_idx=l_idx,
            geo=geo,
            index=index,
            device=device,
            runner=runner,
            rows=rows,
            out_dir=out_dir,
            skip_g_scale=args.skip_g_scale,
        )
        results.append(rec)

    # Print summary table matching progress.md Step 25 format
    print("\n\n" + "=" * 70)
    print("=== SINGLE-LAYER MEASUREMENT SUMMARY TABLE ===")
    print("=" * 70)
    print(f"{'Phase':<36} " + " ".join(f"Layer {r['layer_index']} ({r['layer_type'][:6]})" for r in results))
    print("-" * 70)
    print(f"{'read / build / embed gather':<36} " + " ".join(f"{r['read_seconds']:.2f} / {r['build_seconds']:.2f} / {r['embed_gather_seconds']:.2f} s" for r in results))
    print(f"{f'calibration ({args.num_rows}x{args.seq_len} tok)':<36} " + " ".join(f"{r['calibration_seconds']:.2f} s" for r in results))
    print(f"{'quantization (real GSS)':<36} " + " ".join(f"{r['quantization_seconds']:.1f} s ({r['quantization_seconds'] / 60:.2f} m)" for r in results))
    print(f"{'artifact build + save':<36} " + " ".join(f"{r['artifact_build_seconds']:.2f} + {r['artifact_save_seconds']:.3f} s" for r in results))
    print(f"{'XPU Hessian working set':<36} " + " ".join(f"{r['hessian_working_set_gib']:.2f} GiB ({r['hessian_groups']} grp)" for r in results))
    print(f"{'peak host RSS (process max)':<36} " + " ".join(f"{r['peak_host_rss_mib'] / 1024:.2f} GiB" for r in results))
    print("=" * 70)

    # 140V baseline comparison
    base_l0 = 1588.0
    base_l3 = 1536.4
    for r in results:
        l = r["layer_index"]
        t = r["quantization_seconds"]
        base = base_l0 if l == 0 else base_l3
        speedup = base / t
        print(f"Layer {l} Quantization: {t:.1f}s vs 140V baseline {base:.1f}s -> Speedup: {speedup:.2f}x")

    summary_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device_name": dev_name,
        "total_vram_gib": total_mem_gib,
        "model_dir": str(model_dir),
        "num_rows": args.num_rows,
        "seq_len": args.seq_len,
        "results": results,
    }

    json_path = Path(args.json_out) if args.json_out else (
        REPO_ROOT / "reports" / f"measure_layer_{dev_name.replace(' ', '_')}.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"\nSaved benchmark record to: {json_path}")


if __name__ == "__main__":
    main()
