#!/usr/bin/env python
"""Full-model conversion orchestrator driver (Phase D, P3-16 / P3-09).

Executes the declared full-model qualification run of Qwen3.8-9B-Distill
at K=2, cb=mul1, architecture_mode=qwen3_5_text_tower, real GSS (skip_g_scale=False),
num_calibration_rows=128, seq_len=2048 using the pinned wikitext-2 test corpus.

Captures:
  - environment metadata (env.json)
  - input checksums (inputs.sha256)
  - conversion logs (run.log)
  - per-layer telemetry & progress (quantization_config.json)
  - output checksums (outputs.sha256)
  - final run summary (summary.json)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def capture_environment(device_index: int = 0) -> Dict[str, Any]:
    dev_props = {}
    if torch.xpu.is_available():
        props = torch.xpu.get_device_properties(device_index)
        dev_props = {
            "name": torch.xpu.get_device_name(device_index),
            "total_memory_bytes": props.total_memory,
            "total_memory_gib": round(props.total_memory / (1024 ** 3), 2),
            "gpu_subslice_count": getattr(props, "gpu_subslice_count", None),
            "max_work_group_size": getattr(props, "max_work_group_size", None),
        }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_xpu_available": torch.xpu.is_available(),
        "device": dev_props,
        "ze_affinity_mask": os.environ.get("ZE_AFFINITY_MASK"),
        "exl3_memory_reserve_gib": os.environ.get("EXL3_MEMORY_RESERVE_GIB"),
        "exl3_memory_reflect_outstanding": os.environ.get("EXL3_MEMORY_REFLECT_OUTSTANDING"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Full-model EXL3 qualification conversion.")
    parser.add_argument("--model-dir", type=str, default="/mnt/usb/models/qwen38",
                        help="Path to source unquantized model directory")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Output directory for quantized bundle (defaults to qwen38_exl3_k<K>_<board>)")
    parser.add_argument("--corpus", type=str,
                        default="/mnt/usb/models/qwen38/calibration_tokens_wikitext2_test.json",
                        help="Path to tokenized calibration corpus")
    parser.add_argument("--evidence", type=str, default="",
                        help="Path to evidence bundle directory (defaults to evidence_k<K>_<board>)")
    parser.add_argument("--board", type=str, choices=["b580", "b60", "auto"], default="auto",
                        help="Target board (auto-detected from ZE_AFFINITY_MASK or device name)")
    parser.add_argument("-K", "--bits-per-weight", "--K", dest="K", type=int, choices=[2, 3, 4], default=2,
                        help="Trellis bits per weight (default: 2)")
    parser.add_argument("--device", type=str, default="xpu:0",
                        help="Device string (default: xpu:0)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume conversion if partial progress exists")
    parser.add_argument("--skip-g-scale", action="store_true",
                        help="Skip GSS search (default: False, real GSS)")
    parser.add_argument("--num-rows", type=int, default=128,
                        help="Number of calibration rows (default: 128)")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Sequence length per row (default: 2048)")
    return parser.parse_args()


def detect_board(args) -> str:
    if args.board != "auto":
        return args.board.lower()
    mask = os.environ.get("ZE_AFFINITY_MASK", "")
    if mask == "1":
        return "b580"
    if mask == "0":
        return "b60"
    if torch.xpu.is_available():
        name = torch.xpu.get_device_name(0).lower()
        if "b580" in name:
            return "b580"
        if "b60" in name:
            return "b60"
    return "discrete"


def main():
    args = parse_args()
    board = detect_board(args)

    # Set up memory parameters per Phase B decisions if not already set
    if board == "b580":
        os.environ.setdefault("EXL3_MEMORY_RESERVE_GIB", "4.0")
        os.environ.setdefault("EXL3_MEMORY_REFLECT_OUTSTANDING", "1")
        os.environ.setdefault("EXL3_MAX_LAYERS_PER_RUN", "1")
    elif board == "b60":
        # B60 operates with default 6.0 GiB reserve and unreflected mode
        pass

    model_dir = Path(args.model_dir).resolve()
    corpus_path = Path(args.corpus).resolve()

    if not args.output_dir:
        args.output_dir = f"/mnt/usb/models/qwen38_exl3_k{args.K}_{board}"
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.evidence:
        args.evidence = f"reports/evidence_k{args.K}_{board}"
    evidence_dir = (REPO_ROOT / args.evidence).resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================")
    print("=== EXL3 Full-Model Qualification Runner (Phase D)           ===")
    print("================================================================")
    print(f"Target Board:    {board.upper()}")
    print(f"Model Directory: {model_dir}")
    print(f"Output Directory:{out_dir}")
    print(f"Corpus File:     {corpus_path}")
    print(f"Evidence Dir:    {evidence_dir}")
    print(f"Device:          {args.device}")
    print(f"Recipe:          K={args.K}, cb=mul1, skip_g_scale={args.skip_g_scale}, {args.num_rows}x{args.seq_len}")
    print(f"Memory Reserve:  {os.environ.get('EXL3_MEMORY_RESERVE_GIB', '6.0 (default)')} GiB")
    print(f"Reflect Out:     {os.environ.get('EXL3_MEMORY_REFLECT_OUTSTANDING', '0 (default)')}")
    print("================================================================")

    # 1. Capture environment metadata
    env_info = capture_environment()
    with open(evidence_dir / "env.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # 2. Capture input checksums
    print("\nRecording input checksums...")
    input_hashes = {}
    for fname in ["config.json", "tokenizer.json", "calibration_tokens_wikitext2_test.json", "model.safetensors.index.json"]:
        p = model_dir / fname if fname != "calibration_tokens_wikitext2_test.json" else corpus_path
        if p.is_file():
            h = sha256_file(p)
            input_hashes[p.name] = h
            print(f"  {p.name}: {h}")

    model_sf = model_dir / "model.safetensors"
    if model_sf.is_file():
        # Check if pre-existing checksum file exists to avoid rehashing 19.3 GB every invocation
        sums_file = model_dir / "SHA256SUMS"
        found_hash = None
        if sums_file.is_file():
            with open(sums_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[1] == "model.safetensors":
                        found_hash = parts[0]
                        break
        if not found_hash:
            print("  Computing SHA256 for model.safetensors (19.3 GB)...")
            found_hash = sha256_file(model_sf)
        input_hashes["model.safetensors"] = found_hash
        print(f"  model.safetensors: {found_hash}")

    with open(evidence_dir / "inputs.sha256", "w") as f:
        for k, v in sorted(input_hashes.items()):
            f.write(f"{v}  {k}\n")

    # 3. Build conversion config
    resume_flag = args.resume or (out_dir / "quantization_config.json").is_file()
    cfg = conv.ConversionConfig(
        model_dir=str(model_dir),
        output_dir=str(out_dir),
        K=args.K,
        cb="mul1",
        num_calibration_rows=args.num_rows,
        batch_size=1,
        seq_len=args.seq_len,
        calibration_token_ids=str(corpus_path),
        architecture_mode="qwen3_5_text_tower",
        device=args.device,
        skip_g_scale=args.skip_g_scale,
        checkpoint_interval=1,
        resume=resume_flag,
        min_disk_free_bytes=20 * (1024 ** 3),  # 20 GiB safety reserve
    )

    # 4. Execute conversion
    print(f"\nStarting model conversion (resume={resume_flag})...")
    t0 = time.time()
    report = conv.convert_model(cfg)
    elapsed = time.time() - t0

    print(f"\n========================================================")
    print(f"Conversion complete in {elapsed:.2f} s ({elapsed / 3600:.2f} h)!")
    print(f"Layers converted: {len(report.layers)}")
    print(f"========================================================")

    # 5. Record output checksums
    print("\nRecording output checksums...")
    output_hashes = {}
    for p in sorted(out_dir.glob("qtensors/*.safetensors")):
        h = sha256_file(p)
        output_hashes[f"qtensors/{p.name}"] = h
        print(f"  qtensors/{p.name}: {h[:16]}...")

    qcfg = out_dir / "quantization_config.json"
    if qcfg.is_file():
        h = sha256_file(qcfg)
        output_hashes["quantization_config.json"] = h

    with open(evidence_dir / "outputs.sha256", "w") as f:
        for k, v in sorted(output_hashes.items()):
            f.write(f"{v}  {k}\n")

    # 6. Final summary
    summary = {
        "board": board,
        "device": env_info["device"].get("name"),
        "total_elapsed_seconds": round(elapsed, 2),
        "total_elapsed_hours": round(elapsed / 3600, 2),
        "num_layers": len(report.layers),
        "recipe": {
            "K": args.K,
            "cb": "mul1",
            "skip_g_scale": args.skip_g_scale,
            "rows": args.num_rows,
            "seq_len": args.seq_len,
        },
        "layer_summary": [
            {
                "layer_index": l.layer_index,
                "layer_type": l.layer_type,
                "elapsed_seconds": l.elapsed_seconds,
                "proxy_loss": l.proxy_loss,
                "mean_mse": l.mean_mse,
            }
            for l in report.layers
        ],
    }
    with open(evidence_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved complete evidence bundle to: {evidence_dir}")


if __name__ == "__main__":
    main()
