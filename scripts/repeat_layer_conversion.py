#!/usr/bin/env python
"""Repeat one layer conversion and compare it with the published artifact (P3-14).

Task: P3-14 — repeat conversion (or representative deterministic stages) to
characterize reproducibility and explain acceptable tied-path differences.

Method
------
1. Re-run the *same* conversion recipe (same model, same corpus file, same
   K / rows / seq_len / GSS settings) into a fresh output directory, but stop
   the driver right after the requested layer has been published (the
   converter's ``fault_injector`` hook).
2. Compare the repeated layer artifact with the published one:
   * byte level — SHA-256 of the two ``.safetensors`` files;
   * metric level — the manifest layer entry (proxy loss, mean MSE, g-scale
     per module, elapsed-independent numerics).
3. Report the outcome; identical bytes mean the stage is bit-reproducible on
   this device, and any difference is reported with its magnitude instead of
   being explained away.

Usage
-----
    python scripts/repeat_layer_conversion.py \
        --converted-dir /path/to/published \
        --model-dir /path/to/source_checkpoint \
        --corpus /path/to/calibration_tokens.json \
        --layer 0 --work-dir /tmp/repeat
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import exl3_quant.converter as conv  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--converted-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--work-dir", required=True,
                    help="fresh directory for the repeated conversion")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    published = Path(args.converted_dir)
    work = Path(args.work_dir)
    if work.exists():
        print(f"FAIL: work dir exists (refusing to reuse it): {work}")
        return 1
    work.mkdir(parents=True)

    manifest_path = published / "quantization_config.json"
    manifest = json.loads(manifest_path.read_text())
    cal = manifest["calibration"]

    class _StopAfterLayer(Exception):
        pass

    def injector(event: str, layer_idx: int) -> None:
        if event == "after_publish" and layer_idx == args.layer:
            raise _StopAfterLayer(f"stopping after publishing layer {layer_idx}")

    cfg = conv.ConversionConfig(
        model_dir=args.model_dir,
        output_dir=str(work),
        K=int(manifest["bits"]),
        cb=str(manifest["codebook"]),
        num_calibration_rows=int(cal["num_rows"]),
        batch_size=int(cal["batch_size"]),
        seq_len=int(cal["seq_len"]),
        skip_g_scale=bool(cal["skip_g_scale"]),
        architecture_mode=str(manifest["architecture_mode"]),
        calibration_token_ids=args.corpus,
        checkpoint_interval=1,
    )

    t0 = time.time()
    stopped = False
    try:
        conv.convert_model(cfg, fault_injector=injector)
    except _StopAfterLayer:
        stopped = True
    elapsed = time.time() - t0
    print(f"repeated conversion stopped after layer {args.layer}: {stopped} "
          f"({elapsed:.1f}s)")

    name = f"model.layers.{args.layer}.safetensors"
    a = published / "qtensors" / name
    b = work / "qtensors" / name
    result = {"layer": args.layer, "elapsed_seconds": elapsed,
              "published_artifact": str(a), "repeated_artifact": str(b)}
    if not a.is_file() or not b.is_file():
        result["error"] = "artifact missing on one side"
        print(json.dumps(result, indent=2))
        return 1

    sha_a = sha256_file(a)
    sha_b = sha256_file(b)
    result["published_sha256"] = sha_a
    result["repeated_sha256"] = sha_b
    result["bit_identical"] = sha_a == sha_b
    result["published_bytes"] = a.stat().st_size
    result["repeated_bytes"] = b.stat().st_size

    # metric comparison (the repeated manifest is written in_progress)
    rep_manifest_path = work / "quantization_config.json"
    if rep_manifest_path.is_file():
        rep = json.loads(rep_manifest_path.read_text())
        pub_entry = next(e for e in manifest["layers"]
                         if e["layer_index"] == args.layer)
        rep_entry = next((e for e in rep.get("layers", [])
                          if e["layer_index"] == args.layer), None)
        if rep_entry is not None:
            result["metric_identical"] = all(
                pub_entry.get(k) == rep_entry.get(k)
                for k in ("proxy_loss", "normalized_proxy_loss", "mean_mse",
                          "max_factorization_residual", "g_scales",
                          "modules", "quantized_tensors"))
            result["published_metrics"] = {
                k: pub_entry.get(k) for k in
                ("proxy_loss", "normalized_proxy_loss", "mean_mse",
                 "max_factorization_residual", "g_scales")}
            result["repeated_metrics"] = {
                k: rep_entry.get(k) for k in
                ("proxy_loss", "normalized_proxy_loss", "mean_mse",
                 "max_factorization_residual", "g_scales")}

    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("published_metrics", "repeated_metrics")},
                     indent=2))
    return 0 if result.get("bit_identical") else 2


if __name__ == "__main__":
    sys.exit(main())
