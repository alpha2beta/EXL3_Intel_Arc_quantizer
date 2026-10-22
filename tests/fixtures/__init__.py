"""Deterministic synthetic checkpoint fixtures for Phase 3/4 converter tests.

Dense profile (P3): a mini Llama-style decoder-only checkpoint (config.json +
sharded safetensors) whose geometry satisfies the bounded Phase 3 contract:

* ``hidden_size`` 128 (divisible by the Hadamard block)
* ``intermediate_size`` 256
* ``num_attention_heads`` 2, ``num_key_value_heads`` 2 (head_dim 64, so both
  the q dim and the kv dim are divisible by 128)
* untied ``lm_head`` so the unchanged pass-through path is exercised

Hybrid profile (P4-12/P4-13): :func:`write_qwen35_hybrid_model` writes a
miniature ``Qwen3_5ForConditionalGeneration`` checkpoint — nested
``model.language_model`` text tower with GatedDeltaNet + full-attention
layers, vision-tower and MTP tensors, bfloat16 source dtype — for the
opt-in ``architecture_mode="qwen3_5_text_tower"`` conversion path.

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


# ---------------------------------------------------------------------------
# P4-12/P4-13: synthetic Qwen3.5 hybrid text-tower checkpoint
# ---------------------------------------------------------------------------

#: Miniature geometry of a ``Qwen3_5ForConditionalGeneration`` text tower.
#: Every width satisfies the bounded conversion contract (hidden/intermediate
#: and every quantized projection's k/n divisible by the 128-wide Hadamard
#: block; ``head_dim == hidden//heads``; kv width 128; partial rotary 0.25 of
#: 128 = 32, mrope_section [6,5,5] sums to 16 = rotary_dim//2). The two tiny
#: ``in_proj_a``/``in_proj_b`` projections (n=8) exercise the documented
#: bit-exact pass-through boundary of P4-12.
HYBRID_FIXTURE_GEOMETRY = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "image_token_id": 999,
    "video_token_id": 1000,
    "vision_start_token_id": 997,
    "vision_end_token_id": 998,
    "tie_word_embeddings": False,
    "vision_config": {"depth": 2, "hidden_size": 64,
                      "num_attention_heads": 4, "out_hidden_size": 64,
                      "patch_size": 16, "spatial_merge_size": 2},
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "vocab_size": 1024,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 512,
        "attn_output_gate": True,
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 64,
        "linear_num_value_heads": 8,
        "linear_value_head_dim": 32,
        "layer_types": ["linear_attention", "linear_attention",
                        "full_attention"],
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                            "partial_rotary_factor": 0.25,
                            "mrope_section": [6, 5, 5],
                            "mrope_interleaved": True},
        "mtp_num_hidden_layers": 1,
    },
}

HYBRID_FIXTURE_SEED = 4321

#: Layer tails of the GatedDeltaNet (``linear_attention``) variant.
_HYBRID_LINEAR_TAILS = (
    "input_layernorm.weight",
    "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
    "linear_attn.conv1d.weight", "linear_attn.A_log",
    "linear_attn.dt_bias", "linear_attn.norm.weight",
    "linear_attn.out_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)

#: Layer tails of the gated full-attention variant.
_HYBRID_FULL_TAILS = (
    "input_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)


def build_hybrid_fixture_tensors(layer_types) -> Dict[str, torch.Tensor]:
    """All checkpoint tensors of the hybrid fixture, in ``bfloat16``.

    Source dtype is bf16 (like the real checkpoint) so tests can prove the
    pass-through groups are preserved **bit-exact**, not re-cast. Weights are
    scaled small (0.02) so fp16 forwards stay finite.
    """
    geo = HYBRID_FIXTURE_GEOMETRY
    text = geo["text_config"]
    hidden = text["hidden_size"]
    inter = text["intermediate_size"]
    vocab = text["vocab_size"]
    head_dim = text["head_dim"]
    heads = text["num_attention_heads"]
    kv_heads = text["num_key_value_heads"]
    key_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
    value_dim = text["linear_num_value_heads"] * text["linear_value_head_dim"]
    value_heads = text["linear_num_value_heads"]
    conv_dim = 2 * key_dim + value_dim
    conv_k = text["linear_conv_kernel_dim"]
    vis_hidden = geo["vision_config"]["hidden_size"]

    generator = torch.Generator().manual_seed(HYBRID_FIXTURE_SEED)

    def randn(*shape, scale=0.02):
        return (torch.randn(*shape, generator=generator) * scale).to(
            torch.bfloat16)

    tensors: Dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": randn(vocab, hidden, scale=0.05),
        "model.language_model.norm.weight": randn(hidden, scale=0.05),
        "lm_head.weight": randn(vocab, hidden, scale=0.05),  # untied
        # vision tower (pass-through group "vision")
        "model.visual.patch_embed.proj.weight": randn(
            vis_hidden, 3, 2, 16, 16),
        "model.visual.blocks.0.norm1.weight": randn(vis_hidden),
        "model.visual.blocks.0.attn.qkv.weight": randn(3 * vis_hidden, vis_hidden),
        "model.visual.merger.norm.weight": randn(vis_hidden),
        # MTP head (pass-through group "mtp"; unused by text-only decode)
        "mtp.fc.weight": randn(2 * hidden, hidden),
        "mtp.norm.weight": randn(hidden),
        "mtp.layers.0.self_attn.q_proj.weight": randn(
            heads * head_dim * 2, hidden),
        "mtp.layers.0.mlp.gate_proj.weight": randn(inter, hidden),
    }
    for idx, layer_type in enumerate(layer_types):
        p = f"model.language_model.layers.{idx}"
        if layer_type == "linear_attention":
            shapes = {
                "input_layernorm.weight": (hidden,),
                "linear_attn.in_proj_qkv.weight": (2 * key_dim + value_dim, hidden),
                "linear_attn.in_proj_z.weight": (value_dim, hidden),
                "linear_attn.in_proj_a.weight": (value_heads, hidden),
                "linear_attn.in_proj_b.weight": (value_heads, hidden),
                "linear_attn.conv1d.weight": (conv_dim, 1, conv_k),
                "linear_attn.A_log": (value_heads,),
                "linear_attn.dt_bias": (value_heads,),
                "linear_attn.norm.weight": (text["linear_value_head_dim"],),
                "linear_attn.out_proj.weight": (hidden, value_dim),
                "post_attention_layernorm.weight": (hidden,),
                "mlp.gate_proj.weight": (inter, hidden),
                "mlp.up_proj.weight": (inter, hidden),
                "mlp.down_proj.weight": (hidden, inter),
            }
        else:
            shapes = {
                "input_layernorm.weight": (hidden,),
                "self_attn.q_proj.weight": (heads * head_dim * 2, hidden),
                "self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
                "self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
                "self_attn.o_proj.weight": (hidden, heads * head_dim),
                "self_attn.q_norm.weight": (head_dim,),
                "self_attn.k_norm.weight": (head_dim,),
                "post_attention_layernorm.weight": (hidden,),
                "mlp.gate_proj.weight": (inter, hidden),
                "mlp.up_proj.weight": (inter, hidden),
                "mlp.down_proj.weight": (hidden, inter),
            }
        for tail, shape in shapes.items():
            tensors[f"{p}.{tail}"] = randn(*shape)
    return tensors


def write_qwen35_hybrid_model(
    model_dir: Path,
    layer_types=None,
) -> Dict[str, torch.Tensor]:
    """Write the deterministic mini hybrid checkpoint (single safetensors).

    Mirrors the declared target's structure: nested ``model.language_model``
    text-tower prefix, ``model.visual.*`` vision tensors, ``mtp.*`` tensors,
    and a bfloat16 source dtype. Returns the exact tensor dictionary written
    (for bit-exact pass-through assertions).
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    types = list(layer_types) if layer_types is not None else list(
        HYBRID_FIXTURE_GEOMETRY["text_config"]["layer_types"])
    raw = json.loads(json.dumps(HYBRID_FIXTURE_GEOMETRY))  # deep copy
    raw["text_config"]["layer_types"] = types
    raw["text_config"]["num_hidden_layers"] = len(types)
    with open(model_dir / "config.json", "w") as f:
        json.dump(raw, f, indent=2)

    tensors = build_hybrid_fixture_tensors(types)
    save_file(tensors, str(model_dir / "model.safetensors"))
    return tensors


def write_qwen35_text_only_model(
    model_dir: Path,
    layer_types=None,
) -> Dict[str, torch.Tensor]:
    """Write the deterministic mini **text-only** qwen3_5 checkpoint (P4-14).

    Same weights as the mini hybrid fixture, re-keyed into the text-only
    container layout of e.g. ``autotrust/JEV-9B``:

    * flat ``config.json`` — no ``text_config`` block, ``model_type:
      qwen3_5_text``, ``architectures: [Qwen3_5ForCausalLM]``;
    * decoder stack at the top-level ``model.layers.`` prefix (no
      ``model.language_model.`` nesting);
    * no vision tower and no ``mtp.*`` tensors, while the config keeps the
      *stale* ``mtp_num_hidden_layers: 1`` declaration JEV-9B ships — the
      case where config-declared provenance must not invent a pass-through
      group the checkpoint does not carry.

    Returns the exact tensor dictionary written.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    types = list(layer_types) if layer_types is not None else list(
        HYBRID_FIXTURE_GEOMETRY["text_config"]["layer_types"])

    flat = json.loads(json.dumps(
        HYBRID_FIXTURE_GEOMETRY["text_config"]))  # deep copy of the text block
    flat["layer_types"] = types
    flat["num_hidden_layers"] = len(types)
    # Text-only container keys, mirroring the JEV-9B config shape.
    flat.update({
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
        "dtype": "bfloat16",
        "tie_word_embeddings": False,
        "partial_rotary_factor": 0.25,
        "full_attention_interval": 4,
        "mlp_only_layers": [],
        "mamba_ssm_dtype": "float32",
        "mtp_use_dedicated_embeddings": False,
    })
    with open(model_dir / "config.json", "w") as f:
        json.dump(flat, f, indent=2)

    tensors = {
        k.replace("model.language_model.", "model."): v
        for k, v in build_hybrid_fixture_tensors(types).items()
        if not k.startswith(("model.visual.", "mtp."))
    }
    save_file(tensors, str(model_dir / "model.safetensors"))
    return tensors
