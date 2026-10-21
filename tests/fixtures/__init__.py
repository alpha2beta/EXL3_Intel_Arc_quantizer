"""Deterministic synthetic dense-checkpoint fixtures for Phase 3 converter tests.

Builds a mini Llama-style decoder-only checkpoint (config.json + sharded
safetensors) whose geometry satisfies the bounded Phase 3 contract:

* ``hidden_size`` 128 (divisible by the Hadamard block)
* ``intermediate_size`` 256
* ``num_attention_heads`` 2, ``num_key_value_heads`` 2 (head_dim 64, so both
  the q dim and the kv dim are divisible by 128)
* untied ``lm_head`` so the unchanged pass-through path is exercised

Values are generated from a fixed seed, so fixtures are bit-reproducible
across processes. No network access and no real model download is involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import save_file

FIXTURE_GEOMETRY = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "vocab_size": 96,
    "rms_norm_eps": 1e-6,
}

_FIXTURE_SEED = 1234
_LAYER_TAILS = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "input_layernorm.weight", "post_attention_layernorm.weight",
)


def write_fixture_config(model_dir: Path, overrides: Optional[dict] = None) -> dict:
    """Write a ``config.json`` for the fixture geometry (with overrides)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    raw = dict(FIXTURE_GEOMETRY)
    for key, value in (overrides or {}).items():
        if value is None:
            raw.pop(key, None)
        else:
            raw[key] = value
    with open(model_dir / "config.json", "w") as f:
        json.dump(raw, f, indent=2)
    return raw


def build_fixture_tensors(num_layers: int = 2) -> Dict[str, torch.Tensor]:
    """All checkpoint tensors, keyed like a HF Llama checkpoint (fp16)."""
    geo = FIXTURE_GEOMETRY
    hidden, inter = geo["hidden_size"], geo["intermediate_size"]
    vocab = geo["vocab_size"]
    head_dim = geo["head_dim"]
    q_out = geo["num_attention_heads"] * head_dim
    kv_out = geo["num_key_value_heads"] * head_dim

    generator = torch.Generator().manual_seed(_FIXTURE_SEED)

    def randn(*shape, scale=0.02):
        return (torch.randn(*shape, generator=generator) * scale).to(torch.float16)

    tensors: Dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": randn(vocab, hidden, scale=0.05),
        "model.norm.weight": torch.ones(hidden, dtype=torch.float16),
        "lm_head.weight": randn(vocab, hidden, scale=0.05),  # untied head
    }
    for idx in range(num_layers):
        p = f"model.layers.{idx}"
        shapes = {
            "self_attn.q_proj.weight": (q_out, hidden),
            "self_attn.k_proj.weight": (kv_out, hidden),
            "self_attn.v_proj.weight": (kv_out, hidden),
            "self_attn.o_proj.weight": (hidden, q_out),
            "mlp.gate_proj.weight": (inter, hidden),
            "mlp.up_proj.weight": (inter, hidden),
            "mlp.down_proj.weight": (hidden, inter),
            "input_layernorm.weight": (hidden,),
            "post_attention_layernorm.weight": (hidden,),
        }
        for tail, shape in shapes.items():
            tensors[f"{p}.{tail}"] = randn(*shape)
    return tensors


def write_sharded_model(
    model_dir: Path,
    num_layers: int = 2,
    shards: int = 2,
    tensor_overrides: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Write the deterministic mini checkpoint (single-file or sharded).

    Sharding: decoder layers are distributed round-robin across shards;
    non-layer tensors (embed / final norm / lm_head) go into the *last*
    shard, mirroring typical HF checkpoints.
    Returns the exact tensor dictionary written (for value assertions).
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    write_fixture_config(model_dir, overrides={"num_hidden_layers": num_layers})
    tensors = build_fixture_tensors(num_layers)
    for key, value in (tensor_overrides or {}).items():
        tensors[key] = value

    total_size = sum(t.numel() * t.element_size() for t in tensors.values())
    if shards <= 1:
        save_file(tensors, str(model_dir / "model.safetensors"))
        return tensors

    keys = sorted(k for k in tensors if k.startswith("model.layers."))
    other = sorted(k for k in tensors if not k.startswith("model.layers."))
    weight_map: Dict[str, str] = {}
    for i, key in enumerate(keys):
        weight_map[key] = f"model-{key.split('.')[2].zfill(5)}-of-{shards:05d}.safetensors"
    for key in other:
        weight_map[key] = f"model-{(shards - 1):05d}-of-{shards:05d}.safetensors"

    shard_payloads: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, shard in weight_map.items():
        shard_payloads.setdefault(shard, {})[key] = tensors[key]
    for shard, payload in shard_payloads.items():
        save_file(payload, str(model_dir / shard))
    with open(model_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total_size},
                   "weight_map": weight_map}, f, indent=2)
    return tensors


def layer_shard_file(layer_idx: int, shards: int) -> str:
    """Shard filename the round-robin assignment puts ``layer_idx`` into."""
    return f"model-{layer_idx:05d}-of-{shards:05d}.safetensors"
