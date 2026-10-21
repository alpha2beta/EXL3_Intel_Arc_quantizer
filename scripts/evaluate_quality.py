#!/usr/bin/env python
"""Teacher-forced quality evaluation of a converted checkpoint (P3-12).

Task: P3-12 — evaluate teacher-forced logits/KLD, fixed-corpus NLL/perplexity
against the floating-point baseline. (The *matched CUDA-quantized reference*
is deferred under ``skip_cuda.md`` / ``SKIP-CUDA-2026-10-21-01`` and is
explicitly not claimed here.)

Method
------
The comparison is made inside the repo's own conversion semantics, so the only
difference between the two runs is the weights:

1. **Baseline pass** — the source checkpoint is streamed layer by layer (one
   layer resident at a time) and the evaluation sequences are forwarded
   through the *unquantized* tower with the converter's own runner (same
   rope/mrope handling and attention masks as the calibration forward).
2. **Quantized pass** — the published per-layer artifacts are decoded with
   the audited decoder and the *same* sequences are forwarded through the
   quantized tower.
3. The final norm (the reference zero-centred RMSNorm) and the LM head
   (bit-exact pass-through) turn the last hidden states into logits; the
   teacher-forced NLL / perplexity are accumulated in position chunks (the
   full logit tensor is never materialized), and the KL divergence and the
   argmax agreement between the two models are measured on a seeded sample of
   positions.

The evaluation corpus must be held out from the calibration corpus.

Usage
-----
    python scripts/evaluate_quality.py \
        --converted-dir /path/to/converted \
        --model-dir /path/to/source_checkpoint \
        --eval-tokens /path/to/eval_tokens.json \
        --num-sequences 4 --out /path/to/quality.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import exl3_quant.converter as conv  # noqa: E402
import exl3_quant.hybrid as hybrid  # noqa: E402


def load_sequences(path: Path, seq_len: int, num_sequences: int) -> List[List[int]]:
    payload = json.loads(path.read_text())
    tokens = payload["tokens"] if isinstance(payload, dict) else payload
    n = len(tokens) // seq_len
    if n == 0:
        raise SystemExit(f"corpus too short for seq_len={seq_len}")
    return [tokens[i * seq_len:(i + 1) * seq_len]
            for i in range(min(n, num_sequences))]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Tower:
    """Streams a text tower (source or quantized) one layer at a time."""

    def __init__(self, converted_dir: Path, model_dir: Path, device: torch.device):
        self.manifest = json.loads(
            (converted_dir / "quantization_config.json").read_text())
        self.converted_dir = converted_dir
        self.model_dir = model_dir
        self.device = device
        self.key_prefix = self.manifest["key_prefix"]
        self.num_layers = int(self.manifest["num_layers"])
        self.K = int(self.manifest["bits"])
        self.index = conv.ShardedSafetensorsIndex(model_dir)
        self.qt_dir = converted_dir / "qtensors"
        self._group_handles: Dict[str, object] = {}
        raw_cfg = json.loads((model_dir / "config.json").read_text())
        self.text_config = hybrid.make_text_config(raw_cfg.get("text_config", {}))

    # -- pass-through groups ------------------------------------------------
    def group_tensor(self, group: str, key: str) -> torch.Tensor:
        handle = self._group_handles.get(group)
        if handle is None:
            path = self.converted_dir / self.manifest["pass_through"][group]["file"]
            handle = safe_open(str(path), framework="pt", device="cpu")
            self._group_handles[group] = handle
        return handle.get_tensor(key)

    def group_key(self, group: str, needle: str) -> str:
        for key in self.manifest["pass_through"][group]["keys"]:
            if needle in key:
                return key
        raise KeyError(f"no {needle!r} key in group {group!r}")

    # -- layers -------------------------------------------------------------
    def build_source_layer(self, layer_idx: int):
        tensors = self.index.read_layer_tensors(
            layer_idx, prefix=self.key_prefix)
        layer = hybrid.build_layer(self.text_config, layer_idx, tensors,
                                   prefix=self.key_prefix, device=self.device)
        del tensors
        return layer

    def build_quantized_layer(self, layer_idx: int):
        """Rebuild the layer from the published artifact (decoded weights)."""
        art = self.qt_dir / f"model.layers.{layer_idx}.safetensors"
        entry = self.manifest["layers"][layer_idx]
        decoded: Dict[str, torch.Tensor] = {}
        tensors: Dict[str, torch.Tensor] = {}
        with safe_open(str(art), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for mod in entry["modules"]:
                t_key = f"{self.key_prefix}.{layer_idx}.{mod}.trellis"
                if t_key not in keys:
                    continue
                w = conv.decode_weight_from_storage(
                    f.get_tensor(t_key),
                    f.get_tensor(f"{self.key_prefix}.{layer_idx}.{mod}.suh"),
                    f.get_tensor(f"{self.key_prefix}.{layer_idx}.{mod}.svh"),
                    K=self.K)
                decoded[f"{mod}.weight"] = w.T.contiguous()
            # norms and the per-layer pass-through tensors are bit-exact
            for key in keys:
                if key.endswith(".trellis") or key.endswith(".suh") \
                        or key.endswith(".svh"):
                    continue
                tensors[key] = f.get_tensor(key)
        layer = hybrid.build_layer(self.text_config, layer_idx, tensors,
                                   prefix=self.key_prefix, device=self.device,
                                   decoded=decoded)
        del tensors, decoded
        return layer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--converted-dir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--eval-tokens", required=True)
    ap.add_argument("--num-sequences", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--kl-positions", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    sequences = load_sequences(Path(args.eval_tokens), args.seq_len,
                               args.num_sequences)
    device = conv._resolve_device(args.device)
    tower = Tower(Path(args.converted_dir), Path(args.model_dir), device)
    runner = hybrid.make_calibration_runner(tower.text_config, device)

    # embedding rows for the evaluation tokens (bit-exact source table)
    embed_key = tower.group_key("text", "embed_tokens")
    weight = tower.index.read_tensor(embed_key)
    rows = torch.tensor(sequences, dtype=torch.long)
    embed = weight[rows].to(device=device, dtype=torch.float16)
    del weight
    log(f"evaluation corpus: {len(sequences)} sequences x {args.seq_len} tokens")

    states: Dict[str, List[torch.Tensor]] = {}
    for mode in ("baseline", "quantized"):
        log(f"=== {mode} pass ===")
        hidden = [h.clone() for h in embed]
        t0 = time.time()
        for layer_idx in range(tower.num_layers):
            layer = (tower.build_source_layer(layer_idx) if mode == "baseline"
                     else tower.build_quantized_layer(layer_idx))
            with torch.no_grad():
                hidden = [runner(layer, x) for x in hidden]
            del layer
            if device.type == "xpu":
                torch.xpu.synchronize(device.index or 0)
                torch.xpu.empty_cache()
            if layer_idx % 8 == 0 or layer_idx == tower.num_layers - 1:
                log(f"  layer {layer_idx}/{tower.num_layers} "
                    f"({time.time() - t0:.0f}s)")
        states[mode] = hidden

    # final norm (reference zero-centred RMSNorm) + LM head (bit-exact)
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

    norm_key = tower.group_key("text", "norm.weight")
    lm_key = tower.group_key("text", "lm_head.weight")
    norm_weight = tower.group_tensor("text", norm_key)
    final_norm = Qwen3_5RMSNorm(norm_weight.shape[0],
                                 eps=float(tower.text_config.rms_norm_eps))
    with torch.no_grad():
        final_norm.weight.copy_(norm_weight.to(torch.float32))
    final_norm = final_norm.to(device=device)
    lm_head = tower.group_tensor("text", lm_key).to(device=device,
                                                     dtype=torch.float16)

    def logits_of(hidden_row: torch.Tensor) -> torch.Tensor:
        return final_norm(hidden_row.to(torch.float16)) @ lm_head.T

    report = {"sequences": len(sequences), "seq_len": args.seq_len,
              "per_sequence": []}
    base_nll = 0.0
    quant_nll = 0.0
    total_tokens = 0
    kl_values: List[float] = []
    agree = 0
    agree_n = 0
    rng = random.Random(42)

    for si, tokens in enumerate(sequences):
        entry = {}
        for mode in ("baseline", "quantized"):
            seq = states[mode][si].reshape(args.seq_len, -1)
            nll = 0.0
            counted = 0
            with torch.no_grad():
                for start in range(0, args.seq_len - 1, 64):
                    stop = min(start + 64, args.seq_len - 1)
                    logits = logits_of(seq[start:stop]).float()
                    logp = torch.log_softmax(logits, dim=-1)
                    targets = torch.tensor(tokens[start + 1:stop + 1],
                                           device=logp.device, dtype=torch.long)
                    picked = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
                    nll += float(-picked.sum())
                    counted += int(targets.numel())
                    del logits, logp, picked
            entry[f"{mode}_nll"] = nll
            entry[f"{mode}_tokens"] = counted
            if mode == "baseline":
                base_nll += nll
            else:
                quant_nll += nll
            total_tokens = max(total_tokens, counted)
        report["per_sequence"].append(entry)

        positions = sorted(rng.sample(range(1, args.seq_len), args.kl_positions))
        hb = states["baseline"][si].reshape(args.seq_len, -1)
        hq = states["quantized"][si].reshape(args.seq_len, -1)
        with torch.no_grad():
            for pos in positions:
                lb = torch.log_softmax(logits_of(hb[pos:pos + 1]).float(), dim=-1)
                lq = torch.log_softmax(logits_of(hq[pos:pos + 1]).float(), dim=-1)
                kl_values.append(float((lb.exp() * (lb - lq)).sum()))
                agree += int(int(lb.argmax()) == int(lq.argmax()))
                agree_n += 1
                del lb, lq
        log(f"  sequence {si}: baseline nll={entry['baseline_nll']:.1f} "
            f"quantized nll={entry['quantized_nll']:.1f}")

    report["baseline_nll_sum"] = base_nll
    report["quantized_nll_sum"] = quant_nll
    report["scored_tokens"] = total_tokens
    report["baseline_mean_nll"] = base_nll / max(total_tokens, 1)
    report["quantized_mean_nll"] = quant_nll / max(total_tokens, 1)
    report["baseline_perplexity"] = math.exp(report["baseline_mean_nll"])
    report["quantized_perplexity"] = math.exp(report["quantized_mean_nll"])
    report["mean_kl_divergence"] = (sum(kl_values) / len(kl_values)
                                    if kl_values else None)
    report["kl_positions"] = len(kl_values)
    report["argmax_agreement"] = (agree / agree_n) if agree_n else None
    report["note"] = ("the matched CUDA-quantized reference is deferred under "
                      "skip_cuda.md (SKIP-CUDA-2026-10-21-01); no E4 parity is "
                      "claimed")

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_sequence"},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
