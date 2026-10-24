#!/usr/bin/env python
"""Export a converted EXL3 bundle into a downstream-runtime checkpoint (P3-13).

Task: P3-13 — load and execute the converted checkpoint in the pinned
supported downstream runtime (``exl3xpu``'s vLLM plugin). The bounded
converter publishes a *working-directory* layout (``qtensors/`` + a merged
``quantization_config.json``), while the runtime expects an ordinary
HuggingFace-style checkpoint directory: the tensors themselves, a
``model.safetensors.index.json`` weight map and a ``quantization_config.json``
in the runtime's schema.

Method
------
1. Read the conversion manifest and collect every published safetensors file
   (the per-layer artifacts plus the pass-through groups).
2. Publish them as hardlinks (``--link``, the default: no extra disk, the
   files are immutable once published) or copies (``--copy``) under canonical
   shard names.
3. Build the weight map from the *actual* tensor names in each shard and
   refuse duplicate keys across shards.
4. Copy the model's config/tokenizer files so the runtime can build the
   architecture, and publish the manifest as ``quantization_config.json``
   (it already carries ``quant_method``/``bits``/``codebook``/``tensor_storage``
   in the schema the runtime reads).

Usage
-----
    python scripts/export_for_runtime.py \
        --converted-dir /path/to/converted \
        --model-dir /path/to/source_checkpoint \
        --out-dir /path/to/runtime_checkpoint

Exit code 0 iff the export is complete and consistent.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

from safetensors import safe_open

#: Files the runtime needs to build the architecture and tokenizer.
MODEL_SIDECAR_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "head.safetensors",
    "judge_config.json",
    "calibration.json",
)


def shard_key_map(path: Path) -> Dict[str, str]:
    """Map every tensor key of a safetensors file to its dtype string."""
    out: Dict[str, str] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            out[key] = str(f.get_slice(key).get_dtype())
    return out


def publish(src: Path, dst: Path, mode: str) -> int:
    """Publish ``src`` as ``dst`` (hardlink or copy); returns the byte size."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "link":
        try:
            os.link(str(src), str(dst))
            return src.stat().st_size
        except OSError:
            pass
    shutil.copy2(str(src), str(dst))
    return dst.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--converted-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of hardlinking the shards")
    args = ap.parse_args()

    converted = Path(args.converted_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    mode = "copy" if args.copy else "link"

    manifest_path = converted / "quantization_config.json"
    if not manifest_path.is_file():
        print(f"FAIL: no conversion manifest at {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text())

    qt_dir = converted / "qtensors"
    shards = sorted(qt_dir.glob("*.safetensors"))
    if not shards:
        print(f"FAIL: no published tensors under {qt_dir}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(shards)
    weight_map: Dict[str, str] = {}
    total_size = 0
    for i, shard in enumerate(shards, start=1):
        name = f"model-{i:05d}-of-{total:05d}.safetensors"
        size = publish(shard, out_dir / name, mode)
        total_size += size
        for key in shard_key_map(shard):
            if key in weight_map:
                print(f"FAIL: duplicate tensor key across shards: {key}")
                return 1
            weight_map[key] = name

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2))

    copied = []
    for name in MODEL_SIDECAR_FILES:
        src = model_dir / name
        if src.is_file():
            shutil.copy2(str(src), str(out_dir / name))
            copied.append(name)

    # The runtime reads the manifest's own schema (quant_method / bits /
    # codebook / tensor_storage); publish it verbatim so the checkpoint
    # carries its own provenance.
    (out_dir / "quantization_config.json").write_text(
        json.dumps(manifest, indent=2))

    # P3-13: vLLM resolves the quantization method from the *model config*
    # (`hf_config.quantization_config` -> `quant_cls.from_config`). A sidecar
    # file is not enough: with no embedded block, vLLM falls back to
    # instantiating the config class with no arguments, which fails for any
    # out-of-tree config with required parameters (observed with the pinned
    # exl3xpu plugin: "Exl3Config.__init__() missing 3 required positional
    # arguments"). Embed the manifest in config.json, the HF convention that
    # quantized checkpoints already follow; the sidecar copy above stays as
    # the plugin's `tensor_storage` source.
    model_config_path = out_dir / "config.json"
    if model_config_path.is_file():
        model_cfg = json.loads(model_config_path.read_text())
        model_cfg["quantization_config"] = manifest
        model_config_path.write_text(json.dumps(model_cfg, indent=2))
    else:
        print("WARN: no config.json to embed quantization_config into")

    print(f"exported {total} shards, {len(weight_map)} tensors, "
          f"{total_size / 2**30:.2f} GiB -> {out_dir}")
    print(f"sidecar files copied: {copied}")
    print(f"quantization_config.json: quant_method="
          f"{manifest.get('quant_method')!r} bits={manifest.get('bits')!r} "
          f"codebook={manifest.get('codebook')!r} "
          f"exl3_modules={sum(1 for v in manifest.get('tensor_storage', {}).values() if v.get('quant_format') == 'exl3')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
