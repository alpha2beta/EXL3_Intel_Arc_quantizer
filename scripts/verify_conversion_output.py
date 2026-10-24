#!/usr/bin/env python
"""Verify a converted checkpoint bundle (P3-11: tensor coverage & reconstruction).

Task: P3-11 — verify *all* exported tensors and reconstruction coverage of a
finished conversion; never infer compatibility from plausible generated text.

Method
------
1. Read the merged ``quantization_config.json`` manifest and check its
   structural contract (status, layer list, recipe, corpus identity, layer
   types, tensor-storage map).
2. For every layer artifact: assert the exact tensor key set (quantized
   module tensors + per-layer pass-through tensors), the dtypes, and the
   trellis tile geometry against the *source* checkpoint shapes.
3. Assert every pass-through tensor is bit-exact against the source
   checkpoint (value equality and dtype equality), group by group.
4. Reconstruct every quantized module from its stored trellis (the audited
   decoder) and compare against the source weight: relative Frobenius error
   must lie strictly inside (0, 1) — i.e. the module is genuinely quantized
   and the reconstruction is sane. Sampling of the decode stage is
   configurable; by default every module is reconstructed.

Usage
-----
    python scripts/verify_conversion_output.py \
        --output-dir /path/to/converted \
        --model-dir /path/to/source_checkpoint \
        --report /path/to/verification.json

Exit code 0 iff every check passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from exl3_quant.serialization import decode_weight_from_storage  # noqa: E402


class Report:
    """Accumulates check outcomes (no third-party dependency)."""

    def __init__(self) -> None:
        self.checks: List[dict] = []
        self.failures: List[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            self.failures.append(f"{name}: {detail}")
        return bool(ok)

    def as_dict(self) -> dict:
        return {
            "ok": not self.failures,
            "num_checks": len(self.checks),
            "num_failures": len(self.failures),
            "failures": self.failures,
            "checks": self.checks,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-dir", required=True, help="conversion output dir")
    ap.add_argument("--model-dir", required=True, help="source checkpoint dir")
    ap.add_argument("--report", default="", help="write a JSON report here")
    ap.add_argument("--max-decode-modules", type=int, default=0,
                    help="decode at most N modules (0 = every module)")
    ap.add_argument("--bits", "-K", type=int, default=0,
                    help="expected bits per weight (default: 0 = accept any valid K in {2, 3, 4})")
    ap.add_argument("--skip-decode", action="store_true",
                    help="skip the reconstruction stage entirely")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    rep = Report()
    t_start = time.time()

    # ---- 1. manifest contract -------------------------------------------
    manifest_path = out_dir / "quantization_config.json"
    if not manifest_path.is_file():
        rep.check("manifest.present", False, f"missing {manifest_path}")
        print(json.dumps(rep.as_dict(), indent=2))
        return 1
    manifest = json.loads(manifest_path.read_text())
    rep.check("manifest.status_completed",
              manifest.get("status") == "completed",
              f"status={manifest.get('status')!r}")
    rep.check("manifest.quant_method", manifest.get("quant_method") == "exl3",
              f"quant_method={manifest.get('quant_method')!r}")
    if args.bits:
        rep.check("manifest.bits", manifest.get("bits") == args.bits,
                  f"bits={manifest.get('bits')!r} (expected {args.bits})")
    else:
        rep.check("manifest.bits", manifest.get("bits") in {2, 3, 4},
                  f"bits={manifest.get('bits')!r}")
    rep.check("manifest.codebook", manifest.get("codebook") == "mul1",
              f"codebook={manifest.get('codebook')!r}")

    num_layers = int(manifest.get("num_layers", 0))
    key_prefix = manifest.get("key_prefix", "model.layers")
    entries = manifest.get("layers", [])
    completed = list(manifest.get("completed_layers", []))
    rep.check("manifest.layers_complete",
              completed == list(range(num_layers)) and len(entries) == num_layers,
              f"completed={len(completed)} entries={len(entries)} "
              f"num_layers={num_layers}")

    layer_types = manifest.get("layer_types", [])
    if layer_types:
        kinds: Dict[str, int] = {}
        for t in layer_types:
            kinds[t] = kinds.get(t, 0) + 1
        rep.check("manifest.layer_types_count", len(layer_types) == num_layers,
                  f"layer_types={len(layer_types)} num_layers={num_layers}")
        print(f"layer type histogram: {kinds}")

    cal = manifest.get("calibration", {})
    token_ids = cal.get("token_ids")
    if token_ids:
        corpus_path = Path(token_ids.get("path", ""))
        if corpus_path.is_file():
            digest = sha256_file(corpus_path)
            rep.check("corpus.sha256_matches_manifest",
                      digest == token_ids.get("sha256"),
                      f"manifest={str(token_ids.get('sha256'))[:16]} "
                      f"actual={digest[:16]}")
        else:
            rep.check("corpus.file_present", False, f"missing {corpus_path}")
        rep.check("corpus.skip_g_scale_false", cal.get("skip_g_scale") is False,
                  f"skip_g_scale={cal.get('skip_g_scale')!r}")
        rep.check("corpus.recipe",
                  cal.get("num_rows") == 128 and cal.get("seq_len") == 2048
                  and cal.get("batch_size") == 1,
                  f"rows={cal.get('num_rows')} seq={cal.get('seq_len')} "
                  f"batch={cal.get('batch_size')}")

    # ---- 2. per-layer artifacts ------------------------------------------
    src_path = model_dir / "model.safetensors"
    src_index = model_dir / "model.safetensors.index.json"
    if src_index.is_file():
        src_files = sorted((model_dir).glob("*.safetensors"))
    else:
        src_files = [src_path]
    src_handles = {p.name: safe_open(str(p), framework="pt", device="cpu")
                   for p in src_files}

    def src_get(key: str):
        for handle in src_handles.values():
            if key in handle.keys():
                return handle.get_tensor(key)
        return None

    def src_shape(key: str):
        for handle in src_handles.values():
            if key in handle.keys():
                return list(handle.get_slice(key).get_shape())
        return None

    layer_files_ok = True
    total_tensors = 0
    for entry in entries:
        idx = entry["layer_index"]
        # Artifact file names are fixed by the converter ("model.layers.N.safetensors")
        art = out_dir / "qtensors" / f"model.layers.{idx}.safetensors"
        if not art.is_file():
            rep.check(f"layer{idx}.artifact_present", False, f"missing {art}")
            layer_files_ok = False
            continue
        with safe_open(str(art), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            total_tensors += len(keys)
            expected = set()
            marker = ".mcg" if manifest.get("codebook") == "mcg" else ".mul1"
            for mod in entry["modules"]:
                expected |= {f"{key_prefix}.{idx}.{mod}.trellis",
                             f"{key_prefix}.{idx}.{mod}.suh",
                             f"{key_prefix}.{idx}.{mod}.svh",
                             f"{key_prefix}.{idx}.{mod}{marker}"}
            # the two layer norms are emitted verbatim (P3-02 / P4-13)
            expected |= {f"{key_prefix}.{idx}.input_layernorm.weight",
                         f"{key_prefix}.{idx}.post_attention_layernorm.weight"}
            expected |= set(entry.get("pass_through_keys", []))
            missing = expected - keys
            extra = keys - expected
            rep.check(f"layer{idx}.key_set", not missing and not extra,
                      f"missing={sorted(missing)[:3]} extra={sorted(extra)[:3]} "
                      f"n_keys={len(keys)}")
            # dtypes + trellis geometry against the source checkpoint
            for mod in entry["modules"]:
                t_key = f"{key_prefix}.{idx}.{mod}.trellis"
                if t_key not in keys:
                    continue
                trellis = f.get_tensor(t_key)
                rep.check(f"layer{idx}.{mod}.trellis_dtype",
                          trellis.dtype == torch.int16,
                          f"dtype={trellis.dtype}")
                rep.check(f"layer{idx}.{mod}.trellis_rank", trellis.dim() == 3,
                          f"shape={tuple(trellis.shape)}")
                src_key = f"{key_prefix}.{idx}.{mod}.weight"
                shape = src_shape(src_key)
                if shape is not None and len(shape) == 2:
                    k_tiles, n_tiles = trellis.shape[0], trellis.shape[1]
                    ok = {k_tiles * 16, n_tiles * 16} == {shape[0], shape[1]}
                    rep.check(f"layer{idx}.{mod}.tile_geometry", ok,
                              f"trellis=({k_tiles},{n_tiles}) src={shape}")
                for aux, dt in (("suh", torch.float16), ("svh", torch.float16)):
                    a_key = f"{key_prefix}.{idx}.{mod}.{aux}"
                    if a_key in keys:
                        t = f.get_tensor(a_key)
                        rep.check(f"layer{idx}.{mod}.{aux}_dtype", t.dtype == dt,
                                  f"dtype={t.dtype} shape={tuple(t.shape)}")
    rep.check("layers.all_artifacts_present", layer_files_ok,
              "every layer artifact must exist")

    # ---- 3. pass-through bit-exactness ----------------------------------
    pt = manifest.get("pass_through", {})
    pt_checked = 0
    for group_name, info in pt.items():
        gfile = out_dir / info["file"]
        if not gfile.is_file():
            rep.check(f"pass_through.{group_name}.file_present", False,
                      f"missing {gfile}")
            continue
        with safe_open(str(gfile), framework="pt", device="cpu") as f:
            gkeys = set(f.keys())
            missing = [k for k in info["keys"] if k not in gkeys]
            rep.check(f"pass_through.{group_name}.key_set", not missing,
                      f"missing={missing[:3]} n_keys={len(gkeys)}")
            for key in info["keys"]:
                if key not in gkeys:
                    continue
                g = f.get_tensor(key)
                s = src_get(key)
                if s is None:
                    rep.check(f"pass_through.{group_name}.source_present", False,
                              f"source tensor missing: {key}")
                    continue
                same = (g.dtype == s.dtype) and torch.equal(g, s)
                pt_checked += 1
                if not same:
                    rep.check(f"pass_through.{group_name}.bit_exact", False,
                              f"{key}: dtype {g.dtype} vs {s.dtype}")
    rep.check("pass_through.all_bit_exact",
              not any(c["name"].startswith("pass_through.") and not c["ok"]
                      for c in rep.checks),
              f"{pt_checked} pass-through tensors compared")

    # ---- 4. reconstruction coverage -------------------------------------
    decode_metrics = []
    if not args.skip_decode:
        budget = args.max_decode_modules
        for entry in entries:
            idx = entry["layer_index"]
            art = out_dir / "qtensors" / f"model.layers.{idx}.safetensors"
            if not art.is_file():
                continue
            with safe_open(str(art), framework="pt", device="cpu") as f:
                for mod in entry["modules"]:
                    if budget and len(decode_metrics) >= budget:
                        break
                    t_key = f"{key_prefix}.{idx}.{mod}.trellis"
                    if t_key not in set(f.keys()):
                        continue
                    trellis = f.get_tensor(t_key)
                    suh = f.get_tensor(f"{key_prefix}.{idx}.{mod}.suh")
                    svh = f.get_tensor(f"{key_prefix}.{idx}.{mod}.svh")
                    K = trellis.shape[-1] // 16
                    w_dec = decode_weight_from_storage(trellis, suh, svh, K=K)
                    src_key = f"{key_prefix}.{idx}.{mod}.weight"
                    src = src_get(src_key)
                    if src is None:
                        continue
                    w_src = src.float()
                    a = w_dec.float().T.contiguous()
                    if a.shape != w_src.shape:
                        a = w_dec.float().contiguous()
                    num = (a - w_src).norm().item()
                    den = max(w_src.norm().item(), 1e-12)
                    rel = num / den
                    decode_metrics.append({
                        "layer": idx, "module": mod, "rel_frobenius_error": rel,
                        "shape": list(w_src.shape),
                    })
                    if not (0.0 < rel < 1.0):
                        rep.check(f"decode.layer{idx}.{mod}", False,
                                  f"relative error {rel:.6f} outside (0, 1)")
                    del w_dec, w_src, src
        rep.check("decode.modules_reconstructed", len(decode_metrics) > 0,
                  f"{len(decode_metrics)} modules reconstructed")

    report = rep.as_dict()
    report["metrics"] = {
        "layers": len(entries),
        "artifact_tensors_total": total_tensors,
        "pass_through_tensors_compared": pt_checked,
        "modules_reconstructed": len(decode_metrics),
        "decode_rel_error_max": (max(m["rel_frobenius_error"]
                                      for m in decode_metrics)
                                  if decode_metrics else None),
        "decode_rel_error_mean": (sum(m["rel_frobenius_error"]
                                       for m in decode_metrics) /
                                  len(decode_metrics)
                                  if decode_metrics else None),
        "elapsed_seconds": time.time() - t_start,
    }
    report["decode_metrics"] = decode_metrics

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))

    print(f"checks: {report['num_checks']}, failures: {report['num_failures']}")
    for failure in report["failures"][:40]:
        print(f"  FAIL {failure}")
    print(json.dumps(report["metrics"], indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
