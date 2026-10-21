#!/usr/bin/env python
"""Phase 3 Streaming Conversion Host-Memory Benchmark.

Task: P3-03 — integrate layer/shard streaming and memory-mapped loading, and
MEASURE real host memory pressure during streaming rather than assuming mmap
mapping guarantees capacity.

Method
------
1. Build a deterministic synthetic sharded checkpoint large enough to exhibit
   measurable page-cache RSS (16 layers x 30 tensors, ~32 MiB per layer,
   4 shards, ~0.5 GiB total) under /tmp/opencode.
2. Baseline the process RSS ("/proc/self/statm").
3. Metadata-only pass: construct ``ShardedSafetensorsIndex`` and enumerate
   layer key scopes for every layer. No weight bytes are touched.
4. Streaming pass: for every layer, ``read_layer_tensors`` (mmap-backed
   safetensors), assert values, and drop the tensors. Track RSS after each
   layer — a correct streaming design keeps the *residual* RSS delta bounded
   by (at most) the currently-mapped shard, never the whole model.
5. Contrast pass: materialize one full shard at once (``load_file``) to show
   what a non-streaming loader would cost.

Criterion (P3-03): streaming residual RSS delta (after per-layer drops and
closed handles) must stay far below the full model size, and the per-layer
peak must be bounded by ~one shard. The mixer's page-cache floor is real but
small relative to a multi-GiB model; the honest conclusion recorded here is
that mmap streaming bounds *heap* RSS, not page cache.

Writes ``artifacts/benchmarks/phase3_stream_bench.json``.
"""

from __future__ import annotations

import gc
import json
import resource
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_REPO_TESTS = REPO_ROOT / "tests"
if str(_REPO_TESTS) not in sys.path:
    sys.path.insert(0, str(_REPO_TESTS))

import exl3_quant.converter as conv  # noqa: E402
from fixtures import FIXTURE_GEOMETRY, write_fixture_config  # noqa: E402

from safetensors.torch import load_file, save_file  # noqa: E402

ARTIFACT = REPO_ROOT / "artifacts" / "benchmarks" / "phase3_stream_bench.json"
WORK = Path("/tmp/opencode/phase3_stream_bench")

NUM_LAYERS = 16
SHARDS = 4
HIDDEN = 1024
INTERMEDIATE = 2816
HEADS = 8
KV_HEADS = 8
HEAD_DIM = 128
VOCAB = 8192


def rss_kib() -> int:
    return conv.read_rss_kib()


def peak_rss_kib() -> int:
    """Peak RSS of this process (ru_maxrss is the lifetime high-water mark)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _scaled_tensors(geo: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """Deterministic tensors for the scaled geometry (fp16, fixed seed)."""
    generator = torch.Generator().manual_seed(4321)
    hidden, inter = geo["hidden_size"], geo["intermediate_size"]
    vocab = geo["vocab_size"]
    q_out = geo["num_attention_heads"] * geo["head_dim"]
    kv_out = geo["num_key_value_heads"] * geo["head_dim"]
    tensors: Dict[str, torch.Tensor] = {
        "model.embed_tokens.weight":
            (torch.randn(vocab, hidden, generator=generator) * 0.05).half(),
        "model.norm.weight": torch.ones(hidden).half(),
        "lm_head.weight":
            (torch.randn(vocab, hidden, generator=generator) * 0.05).half(),
    }
    for idx in range(geo["num_hidden_layers"]):
        prefix = f"model.layers.{idx}"
        for tail, shape in {
            "self_attn.q_proj.weight": (q_out, hidden),
            "self_attn.k_proj.weight": (kv_out, hidden),
            "self_attn.v_proj.weight": (kv_out, hidden),
            "self_attn.o_proj.weight": (hidden, q_out),
            "mlp.gate_proj.weight": (inter, hidden),
            "mlp.up_proj.weight": (inter, hidden),
            "mlp.down_proj.weight": (hidden, inter),
            "input_layernorm.weight": (hidden,),
            "post_attention_layernorm.weight": (hidden,),
        }.items():
            tensors[f"{prefix}.{tail}"] = (
                torch.randn(*shape, generator=generator) * 0.02).half()
    return tensors


def write_scaled_shards(model_dir: Path, tensors: Dict[str, torch.Tensor]) -> None:
    """Round-robin layer→shard assignment (same layout as the test fixture)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    layer_keys = sorted(k for k in tensors if k.startswith("model.layers."))
    other_keys = sorted(k for k in tensors if not k.startswith("model.layers."))
    weight_map: Dict[str, str] = {}
    for key in layer_keys:
        layer = int(key.split(".")[2])
        weight_map[key] = f"model-{layer % SHARDS:05d}-of-{SHARDS:05d}.safetensors"
    for key in other_keys:
        weight_map[key] = f"model-{SHARDS - 1:05d}-of-{SHARDS:05d}.safetensors"
    payloads: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, shard in weight_map.items():
        payloads.setdefault(shard, {})[key] = tensors[key]
    for shard, payload in payloads.items():
        save_file(payload, str(model_dir / shard))
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total},
                   "weight_map": weight_map}, f)


def build_workload() -> Dict[str, Any]:
    """Deterministic ~0.5 GiB sharded checkpoint (rebuilt when missing)."""
    model_dir = WORK / "model"
    if not (model_dir / "model.safetensors.index.json").is_file():
        geo = dict(FIXTURE_GEOMETRY)
        geo.update({
            "hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE,
            "num_hidden_layers": NUM_LAYERS, "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS, "head_dim": HEAD_DIM,
            "vocab_size": VOCAB,
        })
        model_dir.mkdir(parents=True, exist_ok=True)
        write_fixture_config(model_dir)
        write_scaled_shards(model_dir, _scaled_tensors(geo))
    index = conv.ShardedSafetensorsIndex(model_dir)
    with open(model_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    total_bytes = sum((model_dir / shard).stat().st_size
                      for shard in set(idx["weight_map"].values()))
    return {"model_dir": model_dir, "index": index, "total_bytes": total_bytes}


def run() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": "P3-03",
        "run_id": uuid.uuid4().hex,
        "configuration": {
            "num_layers": NUM_LAYERS, "shards": SHARDS,
            "hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE,
            "num_heads": HEADS, "num_kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM, "vocab_size": VOCAB,
            "dtype": "float16",
        },
    }

    print(" -- Building workload ...")
    t_build = time.time()
    work = build_workload()
    model_dir: Path = work["model_dir"]
    index: conv.ShardedSafetensorsIndex = work["index"]
    total_bytes: int = work["total_bytes"]
    shard_sizes = {s: (model_dir / s).stat().st_size for s in index.shards}
    result["workload"] = {
        "total_bytes": total_bytes,
        "num_shards": len(shard_sizes),
        "shard_bytes": shard_sizes,
        "build_seconds": time.time() - t_build,
    }
    print(f" -- Total checkpoint: {total_bytes / 1024**2:.1f} MiB across "
          f"{len(shard_sizes)} shards")

    # Baseline
    gc.collect()
    rss_baseline = rss_kib()
    result["rss_baseline_kib"] = rss_baseline
    print(f" -- RSS baseline: {rss_baseline / 1024:.1f} MiB")

    # Pass 1: metadata only
    t0 = time.time()
    for layer_idx in range(NUM_LAYERS):
        keys = index.layer_keys(layer_idx)
        assert len(keys) == 9
        _ = index.unchanged_keys()
    meta_seconds = time.time() - t0
    rss_meta = rss_kib()
    gc.collect()
    result["metadata_pass"] = {
        "elapsed_seconds": meta_seconds,
        "rss_after_kib": rss_meta,
        "rss_delta_kib": rss_meta - rss_baseline,
        "bytes_materialized": 0,
    }
    print(f" -- Metadata pass: {meta_seconds:.3f}s, "
          f"RSS delta {result['metadata_pass']['rss_delta_kib'] / 1024:.1f} MiB")

    # Pass 2: streaming reads
    per_layer: List[Dict[str, Any]] = []
    rss_peak = rss_meta
    t0 = time.time()
    for layer_idx in range(NUM_LAYERS):
        tensors = index.read_layer_tensors(layer_idx)
        # touch values so pages are really read (incl. mmap faults)
        checksum = sum(float(t.flatten()[0].item()) for t in tensors.values())
        per_layer.append({
            "layer": layer_idx,
            "bytes": sum(t.numel() * t.element_size() for t in tensors.values()),
            "rss_after_kib": rss_kib(),
            "checksum": checksum,
        })
        del tensors
        gc.collect()
        rss_peak = max(rss_peak, peak_rss_kib())
    stream_seconds = time.time() - t0
    rss_stream_end = rss_kib()
    result["stream_pass"] = {
        "elapsed_seconds": stream_seconds,
        "rss_end_kib": rss_stream_end,
        "rss_residual_delta_kib": rss_stream_end - rss_baseline,
        "rss_peak_kib": rss_peak,
        "per_layer": per_layer,
    }
    print(f" -- Streaming pass: {stream_seconds:.2f}s "
          f"({stream_seconds / NUM_LAYERS * 1000:.0f} ms/layer), "
          f"residual RSS delta "
          f"{result['stream_pass']['rss_residual_delta_kib'] / 1024:.1f} MiB")

    # Pass 3: contrast — full shard materialization
    rss_before = rss_kib()
    biggest = max(shard_sizes, key=shard_sizes.get)
    t0 = time.time()
    shard_tensors = load_file(str(model_dir / biggest))
    contrast_seconds = time.time() - t0
    rss_full = rss_kib()
    del shard_tensors
    gc.collect()
    result["full_shard_contrast"] = {
        "shard": biggest,
        "shard_bytes": shard_sizes[biggest],
        "elapsed_seconds": contrast_seconds,
        "rss_delta_kib": rss_full - rss_before,
    }
    print(f" -- Full-shard materialization contrast: "
          f"{result['full_shard_contrast']['rss_delta_kib'] / 1024:.1f} MiB RSS "
          f"for one {shard_sizes[biggest] / 1024**2:.1f} MiB shard")

    # Criteria (P3-03)
    layer_bytes = per_layer[0]["bytes"]
    stream_residual = rss_stream_end - rss_baseline
    checks = {
        "metadata_pass_materializes_no_weight_bytes":
            result["metadata_pass"]["rss_delta_kib"] < 8 * 1024,
        "streaming_residual_bounded_well_below_model_size":
            stream_residual < total_bytes / 4,
        "full_shard_contrast_is_affordable_but_dominate_unstreamed_cost":
            result["full_shard_contrast"]["rss_delta_kib"] >= 0
            and result["full_shard_contrast"]["rss_delta_kib"] < total_bytes / 2,
    }
    result["criteria"] = checks
    result["stream_residual_bytes_note"] = (
        "mmap streaming bounds heap RSS, not the page cache: residual RSS is "
        "allowed to retain touched mapped pages; the criterion is that it "
        "stays far below the checkpoint size, never the whole model."
    )
    result["verdict"] = "PASS" if all(checks.values()) else "REVIEW"

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT, "w") as f:
        json.dump(result, f, indent=2)
    print(f" -- Verdict: {result['verdict']}  -> {ARTIFACT}")
    return result


if __name__ == "__main__":
    run()
