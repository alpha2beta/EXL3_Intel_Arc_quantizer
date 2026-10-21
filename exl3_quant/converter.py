"""Bounded full-model conversion orchestrator (Phase 3.1, P3-01..P3-08).

Implements the audited upstream conversion semantics from
``convert_model.py`` (pinned @ ``4d7b69c``), restricted to the *supported
configuration space* declared in this module:

* **P3-01** — :class:`ConversionConfig` + :meth:`ConversionConfig.validate`:
  every unsupported mode (MoE / vision / embedding targets, fractional
  bitrates, non-dense architectures, non-``mul1`` codebooks, out-of-range or
  non-integer K, geometry not divisible by the Hadamard block, missing input
  artifacts) is rejected in pure-Python *before any expensive work* — no
  device is touched, no file is read except ``config.json``.
* **P3-02** — :func:`convert_model` preserves upstream **calibration
  propagation and conversion order** exactly (``convert_model.py:1095-1316``):
  for each decoder layer in ascending order,
  1. capture grouped Hessians by forwarding the *current running hidden
     states* (produced by all previously **quantized** layers) through the
     layer under conversion;
  2. quantize the layer's projections against those Hessians;
  3. reload the module with its quantized numerics (``w_hat`` into the
     layer's linear weights); and
  4. re-forward the *same pre-layer* hidden states through the now-quantized
     layer to produce the running state for the next layer.
  Unchanged components (``embed_tokens``, ``lm_head``, RMSNorm weights) are
  never quantized and are recorded as pass-through in the manifest, per the
  pinned format contract.
* **P3-03** — :class:`ShardedSafetensorsIndex` streams layer tensors from a
  HuggingFace-style sharded checkpoint through memory-mapped
  ``safetensors.safe_open`` handles: only the tensors of the layer under
  conversion are materialized, one shard at a time; host RSS is measured
  around every phase (:func:`read_rss_kib`) rather than assumed.
* **P3-04** — Working-directory layout (``qtensors/``), configurable
  checkpoint interval, per-layer checkpointing, and manifest progress
  tracking (``quantization_config.json``).
* **P3-05** — Atomic checkpoint publication via temporary files and POSIX
  replace (``atomic_save_safetensors``), preventing partial or truncated
  files from being treated as valid checkpoints; integrity verification
  (``verify_layer_checkpoint``) checking all 30 expected tensors per layer.
* **P3-06** — Interruption-safe fault recovery: tested against failures at
  calibration, quantization, reload, and publication boundaries, resuming
  cleanly without dropped or duplicated layers.
* **P3-07** — Rigorous resume argument validation (``_validate_resume``)
  ensuring configuration, bitwidth, codebook, calibration, and architecture
  invariance across restarts, with documented ``force_resume`` override.
* **P3-08** — Resilient failure handling: pre-checked disk space limits
  (``check_disk_space``), missing shard handling, and corrupted checkpoint
  detection (``CorruptedCheckpointError``).

Deliberately *not* implemented here (later tasks): real-model qualification
runs (P3-09+), fractional bitrate recipes (P4-02), MoE / embedding /
two-sided modes (P4-03..P4-05).

Hybrid extension (P4-12/P4-13): ``architecture_mode="qwen3_5_text_tower"``
enables text-tower-only conversion of the declared Active Target — a
``Qwen3_5`` hybrid checkpoint (GatedDeltaNet ``linear_attention`` +
``full_attention`` layers). Only the biases-free linear projections whose
widths are multiples of the 128-wide Hadamard block are quantized (the
exhaustive per-layer table lives in :mod:`exl3_quant.hybrid`); all other
parameters — GatedDeltaNet state, q/k norms, layer norms, the vision tower,
and the MTP head — are published bit-exact in per-group
``qtensors/untouched*.safetensors`` files recorded in the manifest. The mode
is opt-in: the default ``"dense"`` behavior (and every historical rejection)
is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from exl3_quant.calibration import (
    TransformerDecoderLayer,
    calibrate_layer,
    make_rotary_embeddings,
)
from exl3_quant import hybrid
from exl3_quant import memory_hygiene
from exl3_quant.layer_quant import (
    QuantizedModuleResult,
    quantize_decoder_layer,
)
from exl3_quant.serialization import (
    build_layer_tensors,
    decode_weight_from_storage,
)

# ---------------------------------------------------------------------------
# P3-01: bounded configuration space (reject unsupported modes early)
# ---------------------------------------------------------------------------

#: Decoder-only dense architectures covered by the audited pinned contract.
SUPPORTED_ARCHITECTURES = {"llama", "qwen2", "mistral", "gemma"}

#: Only the audited codebook (P1/P2 parity covers CB_MUL1; others are P4-01).
SUPPORTED_CODEBOOKS = {"mul1"}

MIN_K = 2
MAX_K = 4          # bounded fixed-K recipe; fractional recipes are P4-02
HAD_DIM = 128      # Hadamard block: every quantized linear dim must divide


class UnsupportedModeError(ValueError):
    """Raised for any configuration the bounded engine does not support."""


class IncompatibleResumeError(ValueError):
    """Raised when resume arguments conflict with the existing checkpoint manifest (P3-07)."""


class CorruptedCheckpointError(RuntimeError):
    """Raised when an existing checkpoint file fails integrity checks (P3-05, P3-08)."""


class CheckpointInterruptedError(RuntimeError):
    """Raised when a conversion step is explicitly interrupted at a boundary (P3-06)."""


class DiskSpaceError(OSError):
    """Raised when free disk space is insufficient for conversion artifacts (P3-08)."""


@dataclass
class ModelGeometry:
    """Validated geometry extracted from the input model's ``config.json``.

    Dense profiles (llama/qwen2/mistral/gemma) use the historical defaults.
    The Qwen3.5 hybrid text-tower extension (P4-12/P4-13) sets
    ``architecture_mode='qwen3_5_text_tower'``, nested key prefixes, per-layer
    ``layer_types``, and the pass-through/multimodal metadata recorded in the
    manifest.
    """

    architecture: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int

    # --- P4-12/P4-13 extension (defaults = dense profile) -------------------
    architecture_mode: str = "dense"
    layer_types: List[str] = field(default_factory=list)
    key_prefix: str = "model.layers"
    text_prefix: str = "model"
    has_vision: bool = False
    has_mtp: bool = False
    tie_word_embeddings: bool = False
    text_config: Optional[dict] = None
    hybrid_meta: Dict = field(default_factory=dict)

    @property
    def is_hybrid(self) -> bool:
        return self.architecture_mode == "qwen3_5_text_tower"

    def layer_type(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx] if self.layer_types else "full_attention"

    def quantized_modules(self, layer_idx: int) -> List[str]:
        if self.is_hybrid:
            return list(hybrid.quantized_module_names(self.layer_type(layer_idx)))
        return list(_QUANTIZED_MODULES)

    @property
    def embed_key(self) -> str:
        return f"{self.text_prefix}.embed_tokens.weight"

    @property
    def final_norm_key(self) -> str:
        return f"{self.text_prefix}.norm.weight"

    @property
    def lm_head_key(self) -> str:
        return "lm_head.weight"

    def layer_pass_through_keys(self, layer_idx: int) -> List[str]:
        """Full checkpoint keys of non-quantized per-layer parameters.

        For the dense profile the layer norms are the only such tensors (they
        are verified separately via :data:`_NORM_KEYS`), so this returns ``[]``
        and the historical verification is unchanged. Hybrid layers add the
        GatedDeltaNet state and q/k norms (bit-exact pass-through, P4-12).
        """
        if not self.is_hybrid:
            return []
        prefix = f"{self.key_prefix}.{layer_idx}."
        return [prefix + name
                for name in hybrid.PASS_THROUGH_MODULE_NAMES[
                    self.layer_type(layer_idx)]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UnsupportedModeError(message)


#: Architecture modes accepted by :class:`ConversionConfig`.
ARCHITECTURE_MODES = ("dense", "qwen3_5_text_tower")


def _load_qwen35_text_tower_geometry(raw: dict) -> ModelGeometry:
    """Validate a Qwen3.5 hybrid text tower for text-only conversion (P4-12/P4-13).

    The declared Active Target is a dense (non-MoE) ``qwen3_5`` checkpoint:
    GatedDeltaNet and full-attention decoder layers, a vision tower, and an MTP
    head. Only the text-tower linear projections are quantized; vision/MTP are
    bit-exact pass-through (see ``exl3_quant/hybrid.py``).
    """
    archs = [str(a) for a in (raw.get("architectures") or [])]
    model_type = str(raw.get("model_type", ""))
    supported = any(a in hybrid.TEXT_TOWER_ARCHITECTURES for a in archs) \
        or model_type in hybrid.TEXT_TOWER_MODEL_TYPES
    _require(
        supported,
        f"hybrid text-tower mode expects a Qwen3.5 checkpoint "
        f"(architectures={archs}, model_type={model_type!r}); supported: "
        f"{list(hybrid.TEXT_TOWER_ARCHITECTURES)}",
    )
    _require(
        not raw.get("quantization_config"),
        "pre-quantized checkpoints are outside the conversion contract",
    )

    text = raw.get("text_config")
    _require(isinstance(text, dict),
             "qwen3_5 checkpoints require a text_config block for the text tower")

    # MoE variants are a separately scoped extension (P4-05), not this track.
    moe_keys = ("num_experts", "num_local_experts", "num_experts_per_tok",
                "moe", "decoder_sparse_step")
    moe_present = sorted(k for k in moe_keys
                         if text.get(k) not in (None, [], {}, 0, False))
    _require(not moe_present,
             f"MoE qwen3_5 variants are not supported by the text-tower "
             f"extension (found {moe_present}; MoE is P4-05)")

    core_keys = ("num_hidden_layers", "hidden_size", "intermediate_size",
                 "num_attention_heads", "vocab_size")
    missing_core = [k for k in core_keys if k not in text]
    _require(not missing_core,
             f"text_config is missing required keys: {missing_core}")

    num_layers = int(text["num_hidden_layers"])
    # Mirror Qwen3_5TextConfig: absent layer_types are derived from
    # full_attention_interval (default 4) and legacy names are remapped
    # (mamba/conv -> linear_attention, attention -> full_attention).
    layer_types = [str(t) for t in (text.get("layer_types") or [])]
    if not layer_types:
        interval = int(text.get("full_attention_interval", 4) or 4)
        _require(interval >= 1, "full_attention_interval must be >= 1")
        layer_types = ["linear_attention" if (i + 1) % interval
                       else "full_attention" for i in range(num_layers)]
    else:
        layer_types = [hybrid.LEGACY_LAYER_TYPE_REMAP.get(t, t)
                       for t in layer_types]
    _require(len(layer_types) == num_layers,
             f"layer_types length {len(layer_types)} != num_hidden_layers "
             f"{num_layers}")
    unsupported = sorted({t for t in layer_types
                          if t not in hybrid.SUPPORTED_LAYER_TYPES})
    _require(not unsupported,
             f"unsupported hybrid layer types {unsupported}; supported: "
             f"{list(hybrid.SUPPORTED_LAYER_TYPES)}")

    hidden = int(text["hidden_size"])
    intermediate = int(text["intermediate_size"])
    heads = int(text["num_attention_heads"])
    kv_heads = int(text.get("num_key_value_heads", heads))
    head_dim = int(text.get("head_dim", hidden // heads))
    vocab = int(text["vocab_size"])
    eps = float(text.get("rms_norm_eps", 1e-6))

    _require(kv_heads <= heads, "num_key_value_heads exceeds attention heads")
    _require(heads % kv_heads == 0,
             "GQA requires num_attention_heads % num_key_value_heads == 0")
    _require(head_dim == hidden // heads,
             "head_dim differing from hidden_size // num_attention_heads is "
             "outside this text-tower contract")
    _require(head_dim % 2 == 0, f"head_dim {head_dim} must be even (rotary)")
    _require(0 < vocab < 2**24, "vocab_size out of supported range")

    # Missing linear-attention fields fall back to the Qwen3_5TextConfig
    # dataclass defaults so validation mirrors the reference implementation.
    key_heads = int(text.get("linear_num_key_heads", 16))
    value_heads = int(text.get("linear_num_value_heads", 32))
    key_head_dim = int(text.get("linear_key_head_dim", 128))
    value_head_dim = int(text.get("linear_value_head_dim", 128))
    _require(key_heads > 0 and value_heads > 0,
             "linear attention head counts must be positive")
    _require(value_heads % key_heads == 0,
             "linear_num_value_heads must be a multiple of linear_num_key_heads")
    key_dim = key_heads * key_head_dim
    value_dim = value_heads * value_head_dim
    _require(int(text.get("linear_conv_kernel_dim", 4)) >= 1,
             "linear_conv_kernel_dim must be >= 1")

    # Every quantized projection needs its input (k) and output (n) width to be
    # a multiple of the 128-wide Hadamard block. in_proj_a/in_proj_b are
    # deliberately excluded (bit-exact pass-through, see hybrid.py).
    widths = {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "attn q width (2 x heads x head_dim)": heads * head_dim * 2,
        "attn kv width (kv_heads x head_dim)": kv_heads * head_dim,
        "attn o input (heads x head_dim)": heads * head_dim,
        "linear value width (value_heads x value_head_dim)": value_dim,
        "linear qkv width (2 x key_dim + value_dim)": 2 * key_dim + value_dim,
    }
    too_narrow = {k: v for k, v in widths.items() if v % HAD_DIM}
    _require(
        not too_narrow,
        f"geometry is not divisible by the {HAD_DIM}-wide Hadamard block: "
        f"{too_narrow}",
    )

    rope = text.get("rope_parameters") or {}
    _require(isinstance(rope, dict),
             "text_config.rope_parameters must be a mapping when present")
    # Qwen3_5TextConfig.__post_init__ applies `kwargs.setdefault(
    # "partial_rotary_factor", 0.25)` for backwards compatibility, so an
    # unspecified factor behaves as 0.25 in the reference forward.
    partial = float(rope.get(
        "partial_rotary_factor", text.get("partial_rotary_factor", 0.25)))
    rotary_dim = int(head_dim * partial)
    _require(0 < partial <= 1.0 and rotary_dim % 2 == 0,
             f"invalid partial_rotary_factor {partial} for head_dim {head_dim}")

    multimodal = {
        key: raw.get(key)
        for key in ("image_token_id", "video_token_id",
                    "vision_start_token_id", "vision_end_token_id")
        if raw.get(key) is not None
    }
    hybrid_meta = {
        "multimodal": multimodal or None,
        "vision_config": raw.get("vision_config"),
        "rope": {
            "rope_parameters": rope,
            "rope_theta": rope.get("rope_theta"),
            "partial_rotary_factor": partial,
            "rotary_dim": rotary_dim,
            "mrope_section": rope.get("mrope_section"),
            "mrope_interleaved": rope.get("mrope_interleaved"),
        },
        "norm_semantics": "zero_centered_rmsnorm: y = x_norm * (1 + weight)",
        "attn_output_gate": bool(text.get("attn_output_gate")),
        "linear_attention": {
            "num_key_heads": key_heads,
            "num_value_heads": value_heads,
            "key_head_dim": key_head_dim,
            "value_head_dim": value_head_dim,
            "key_dim": key_dim,
            "value_dim": value_dim,
            "conv_kernel_dim": int(text.get("linear_conv_kernel_dim", 4)),
            "mamba_ssm_dtype": text.get("mamba_ssm_dtype"),
            "quantized": list(hybrid.QUANTIZED_MODULES["linear_attention"]),
            "pass_through": ["linear_attn.in_proj_a.weight",
                             "linear_attn.in_proj_b.weight",
                             "linear_attn.conv1d.weight",
                             "linear_attn.A_log",
                             "linear_attn.dt_bias",
                             "linear_attn.norm.weight"],
        },
        "mtp_num_hidden_layers": text.get("mtp_num_hidden_layers",
                                          raw.get("mtp_num_hidden_layers")),
    }

    return ModelGeometry(
        architecture="qwen3_5",
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_layers=num_layers,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        rms_norm_eps=eps,
        vocab_size=vocab,
        architecture_mode="qwen3_5_text_tower",
        layer_types=layer_types,
        key_prefix="model.language_model.layers",
        text_prefix="model.language_model",
        has_vision=bool(raw.get("vision_config")),
        has_mtp=bool(text.get("mtp_num_hidden_layers")
                     or raw.get("mtp_num_hidden_layers")),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        text_config=dict(text),
        hybrid_meta=hybrid_meta,
    )


def load_model_geometry(model_dir: Path, mode: str = "dense") -> ModelGeometry:
    """Read and validate ``config.json`` *without touching any model weights*."""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise UnsupportedModeError(f"model dir has no config.json: {model_dir}")
    with open(config_path) as f:
        raw = json.load(f)

    if mode == "qwen3_5_text_tower":
        return _load_qwen35_text_tower_geometry(raw)
    _require(mode == "dense",
             f"unknown architecture_mode {mode!r}; supported: "
             f"{list(ARCHITECTURE_MODES)}")

    archs = raw.get("architectures") or []
    arch = ""
    for a in archs:
        lowered = a.lower()
        for supported in SUPPORTED_ARCHITECTURES:
            if supported in lowered:
                arch = supported
                break
        if arch:
            break
    _require(
        bool(arch),
        f"unsupported architecture {archs}: only dense llama/qwen2/mistral/gemma "
        "decoder-only models are covered by the pinned contract",
    )

    # Unsupported structural modes: MoE, vision/multimodal, MTP, pre-quantized.
    # The MTP and multimodal token-id keys harden against the declared active
    # targets `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` and
    # `empero-ai/Qwen3.8-9B-Distill` (sibling qwen3_5 distills, config.json
    # verified 2026-10-21), which carry them at top level.
    forbidden_keys = (
        "num_local_experts", "num_experts", "moe", "vision_config",
        "text_config", "mtp_num_heads", "mtp_num_hidden_layers",
        "quantization_config", "image_token_id", "video_token_id",
        "vision_start_token_id", "vision_end_token_id",
    )
    present = sorted(k for k in forbidden_keys
                     if raw.get(k) not in (None, [], {}))
    _require(
        not present,
        f"unsupported model features present in config: {present}",
    )

    # Hybrid attention: every decoder layer must be standard full attention.
    # `layer_types` mixes softmax-attention with linear-attention (Mamba-style
    # SSM) layers in the declared target's architecture; only the dense full-
    # attention decoder is covered by upstream EXL3 and this port.
    layer_types = raw.get("layer_types")
    if layer_types:
        non_full = sorted({str(t) for t in layer_types
                           if t != "full_attention"})
        _require(
            not non_full,
            f"hybrid attention layer types {non_full} are outside the "
            "bounded conversion contract (only full_attention is supported)",
        )

    hidden = int(raw["hidden_size"])
    intermediate = int(raw["intermediate_size"])
    layers = int(raw["num_hidden_layers"])
    heads = int(raw["num_attention_heads"])
    kv_heads = int(raw.get("num_key_value_heads", heads))
    head_dim = int(raw.get("head_dim", hidden // heads))
    vocab = int(raw["vocab_size"])
    eps = float(raw.get("rms_norm_eps", 1e-6))

    _require(layers >= 1, "model has no decoder layers")
    _require(kv_heads <= heads, "num_key_value_heads exceeds attention heads")
    _require(heads % kv_heads == 0,
             "GQA requires num_attention_heads % num_key_value_heads == 0")
    _require(hidden % HAD_DIM == 0,
             f"hidden_size {hidden} not divisible by {HAD_DIM}")
    _require(intermediate % HAD_DIM == 0,
             f"intermediate_size {intermediate} not divisible by {HAD_DIM}")
    _require(heads * head_dim % HAD_DIM == 0,
             f"q dim ({heads} x {head_dim}) not divisible by {HAD_DIM}")
    _require(kv_heads * head_dim % HAD_DIM == 0,
             f"kv dim ({kv_heads} x {head_dim}) not divisible by {HAD_DIM}")
    _require(head_dim % 2 == 0, f"head_dim {head_dim} must be even (rotary)")
    _require(head_dim == hidden // heads,
             "head_dim differing from hidden_size // num_attention_heads is "
             "outside the bounded conversion contract")
    _require(0 < vocab < 2**24, "vocab_size out of supported range")
    # Partial / scaled rotary variants are outside the pinned contract.
    rope_pct = raw.get("partial_rotary_factor")
    if rope_pct is not None:
        _require(float(rope_pct) == 1.0,
                 f"partial rotary factor {rope_pct} unsupported (needs full rotary)")
    rope_scaling = raw.get("rope_scaling")
    _require(not rope_scaling,
             "rope_scaling is outside the pinned conversion contract")
    _require(raw.get("sliding_window") in (None, False) or
             int(raw.get("sliding_window", 0)) <= 0,
             "sliding-window attention is outside the pinned contract")

    return ModelGeometry(
        architecture=arch, hidden_size=hidden, intermediate_size=intermediate,
        num_layers=layers, num_heads=heads, num_kv_heads=kv_heads,
        head_dim=head_dim, rms_norm_eps=eps, vocab_size=vocab,
    )


@dataclass
class ConversionConfig:
    """Bounded conversion entry point (P3-01).

    Only parameters the pinned contract supports are exposed; everything else
    is an explicit rejection, not a silent default.
    """

    model_dir: str
    output_dir: str

    # Fixed-K recipe (fractional average-bitrate recipes are a Phase 4 task)
    K: int = 2
    cb: str = "mul1"

    # Calibration. The synthetic deterministic token corpus is the default;
    # the P3-09 qualification run wires a real corpus by passing a
    # pre-tokenized token-ids file (produced by
    # scripts/prepare_calibration_tokens.py) via calibration_token_ids.
    num_calibration_rows: int = 2
    batch_size: int = 1
    seq_len: int = 128
    calibration_seed: int = 0
    calibration_token_ids: Optional[str] = None

    # Numerics pass-through to the quantization chain
    damping_lambda: float = 0.01
    zero_floor: float = 1e-7
    skip_g_scale: bool = False   # upstream default: real GSS search
    device: str = "auto"

    # Checkpoint and resume semantics (P3-04, P3-07)
    resume: bool = False
    force_resume: bool = False
    checkpoint_interval: int = 1
    min_disk_free_bytes: int = 10 * 1024 * 1024  # 10 MiB safety margin for synthetic/real runs

    # Architecture mode (P4-12/P4-13). "dense" is the historical llama/qwen2
    # profile; "qwen3_5_text_tower" is the explicit opt-in for the declared
    # hybrid Active Target (GatedDeltaNet + full-attention text tower; vision
    # and MTP tensors are bit-exact pass-through, never quantized).
    architecture_mode: str = "dense"

    def __post_init__(self) -> None:
        self.model_dir = str(Path(self.model_dir))
        self.output_dir = str(Path(self.output_dir))
        if self.calibration_token_ids is not None:
            self.calibration_token_ids = str(Path(self.calibration_token_ids))

    def validate(self) -> ModelGeometry:
        """Reject all unsupported modes before any weight tool runs."""
        model_dir = Path(self.model_dir)
        _require(model_dir.is_dir(), f"model_dir does not exist: {model_dir}")

        _require(isinstance(self.K, int) and not isinstance(self.K, bool),
                 f"K must be an integer, got {self.K!r}")
        _require(MIN_K <= self.K <= MAX_K,
                 f"K must be in [{MIN_K}, {MAX_K}] for the bounded fixed-K recipe "
                 f"(got {self.K}; fractional recipes are an explicit Phase 4 task)")
        _require(self.cb in SUPPORTED_CODEBOOKS,
                 f"unsupported codebook {self.cb!r}; audited codebooks: "
                 f"{sorted(SUPPORTED_CODEBOOKS)} (alternate codebooks are P4-01)")

        _require(self.num_calibration_rows >= 1, "num_calibration_rows must be >= 1")
        _require(self.batch_size >= 1, "batch_size must be >= 1")
        _require(self.seq_len >= 16 and self.seq_len % 16 == 0,
                 "seq_len must be a positive multiple of 16")
        _require(self.damping_lambda > 0, "damping_lambda must be positive")
        _require(self.zero_floor > 0, "zero_floor must be positive")
        _require(isinstance(self.skip_g_scale, bool), "skip_g_scale must be bool")
        _require(self.checkpoint_interval >= 1, "checkpoint_interval must be >= 1")
        _require(self.min_disk_free_bytes >= 0, "min_disk_free_bytes must be >= 0")

        out_dir = Path(self.output_dir).resolve()
        _require(out_dir != model_dir.resolve(),
                 "output_dir must differ from model_dir")
        if not self.resume:
            _require(not out_dir.exists() or not any(out_dir.iterdir()),
                     f"output_dir must be empty or not exist: {out_dir}")

        _require(self.architecture_mode in ARCHITECTURE_MODES,
                 f"unknown architecture_mode {self.architecture_mode!r}; "
                 f"supported: {list(ARCHITECTURE_MODES)}")
        if self.architecture_mode == "qwen3_5_text_tower":
            # Fail before any weight byte is read if the pinned reference
            # implementation (transformers qwen3_5) is unavailable here.
            hybrid.require_support()

        geometry = load_model_geometry(model_dir, mode=self.architecture_mode)
        if self.calibration_token_ids is not None:
            # P3-09 real-corpus wiring: fully validate the pre-tokenized
            # corpus (structure, shapes, token bounds) before any weight
            # byte is read, exactly like every other recipe field.
            load_calibration_token_rows(self, geometry)
        return geometry


#: Format stamp required in pre-tokenized calibration corpus files (P3-09).
CALIBRATION_TOKEN_IDS_FORMAT = "exl3_quant.calibration_token_ids.v1"


def calibration_token_ids_sha256(path: str) -> str:
    """SHA-256 of the corpus file bytes (provenance + resume identity)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_calibration_token_rows(
    config: ConversionConfig, geometry: ModelGeometry
) -> "tuple[List[torch.Tensor], Dict]":
    """Load and validate a pre-tokenized calibration corpus (P3-09).

    The file is a JSON object stamped with
    :data:`CALIBRATION_TOKEN_IDS_FORMAT` whose ``rows`` entry is a list of
    flat token-id sequences, each exactly ``batch_size * seq_len`` integers
    in ``[0, vocab_size)``; all other entries are free-form provenance
    metadata carried into the conversion manifest. At least
    ``num_calibration_rows`` rows must be present (the first
    ``num_calibration_rows`` are used).

    Returns ``(rows, meta)`` where each row is an int64
    ``(batch_size, seq_len)`` tensor and ``meta`` is the file's metadata
    (everything except ``rows``).

    Raises :class:`UnsupportedModeError` for any structural problem, so the
    whole corpus contract is rejected in pure Python during ``validate()``.
    """
    path = Path(config.calibration_token_ids or "")
    _require(path.is_file(), f"calibration_token_ids file does not exist: {path}")
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        _require(False, f"calibration_token_ids is not valid JSON: {path} ({e})")
        raise  # unreachable; keeps type-checkers happy
    _require(isinstance(payload, dict),
             f"calibration_token_ids must be a JSON object, got {type(payload).__name__}")
    _require(payload.get("format") == CALIBRATION_TOKEN_IDS_FORMAT,
             f"calibration_token_ids format stamp must be "
             f"{CALIBRATION_TOKEN_IDS_FORMAT!r}, got {payload.get('format')!r}")

    raw_rows = payload.get("rows")
    _require(isinstance(raw_rows, list) and len(raw_rows) > 0,
             "calibration_token_ids 'rows' must be a non-empty list")
    _require(len(raw_rows) >= config.num_calibration_rows,
             f"calibration_token_ids has {len(raw_rows)} rows but "
             f"num_calibration_rows={config.num_calibration_rows}")
    if "seq_len" in payload:
        _require(payload["seq_len"] == config.seq_len,
                 f"calibration_token_ids seq_len={payload['seq_len']} does not "
                 f"match config seq_len={config.seq_len}")

    row_tokens = config.batch_size * config.seq_len
    rows: List[torch.Tensor] = []
    for i, row in enumerate(raw_rows):
        _require(isinstance(row, list) and len(row) == row_tokens,
                 f"calibration_token_ids row {i} must be a flat list of "
                 f"{row_tokens} tokens (batch_size * seq_len), got "
                 f"{len(row) if isinstance(row, list) else type(row).__name__}")
        for t in row:
            if not (isinstance(t, int) and not isinstance(t, bool)
                    and 0 <= t < geometry.vocab_size):
                _require(False,
                         f"calibration_token_ids row {i} contains a token "
                         f"outside int/[0, {geometry.vocab_size}): {t!r}")
        rows.append(torch.tensor(row, dtype=torch.int64)
                    .reshape(config.batch_size, config.seq_len))

    meta = {k: v for k, v in payload.items() if k != "rows"}
    return rows, meta


# ---------------------------------------------------------------------------
# P3-03: memory-mapped sharded checkpoint streaming
# ---------------------------------------------------------------------------

_PAGE_KIB = 4


def read_rss_kib() -> int:
    """Current resident set size of this process, KiB (``/proc/self/statm``)."""
    try:
        fields = Path("/proc/self/statm").read_text().split()
        return int(fields[1]) * _PAGE_KIB
    except (OSError, IndexError, ValueError):
        return 0


class ShardedSafetensorsIndex:
    """Lazy, memory-mapped access to a HF-style sharded checkpoint (P3-03).

    ``safe_open`` mmaps each shard; tensors are materialized only on request,
    so a streaming conversion touches at most the layer being converted. The
    weight map (from the index JSON or a single-file scan) is the only
    persistent structure — no weight bytes are retained.
    """

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        self.is_sharded = index_path.is_file()
        if self.is_sharded:
            with open(index_path) as f:
                index = json.load(f)
            self.weight_map: Dict[str, str] = dict(index["weight_map"])
        else:
            single = sorted(self.model_dir.glob("*.safetensors"))
            _require(len(single) >= 1, f"no safetensors found in {model_dir}")
            _require(len(single) == 1,
                     "multiple .safetensors files without model.safetensors.index.json")
            self.weight_map = {}
            with safe_open(single[0], framework="pt", device="cpu") as f:
                for key in f.keys():
                    self.weight_map[key] = single[0].name
        self.shards = sorted(set(self.weight_map.values()))
        #: Path of the single checkpoint file when the checkpoint is not
        #: sharded (the declared target's case); ``None`` for sharded
        #: checkpoints. The P3-09 memory hygiene evicts the page cache of the
        #: streamed source once a group or layer has been consumed.
        self.source_file: Optional[Path] = (
            None if self.is_sharded else (self.model_dir / self.shards[0]))

    @property
    def shard_paths(self) -> List[Path]:
        """Absolute paths of the checkpoint shards (one for a single file)."""
        return [self.model_dir / name for name in self.shards]

    def release_source_cache(self) -> None:
        """P3-09: evict the page cache of the streamed source shards.

        A 32-layer run reads a 19.3 GB checkpoint once; keeping every read
        page resident would pin the whole checkpoint in DRAM for the whole
        run. The hint is advisory (clean pages only, dirty pages are written
        back first), so file content is never affected.
        """
        for path in self.shard_paths:
            memory_hygiene.drop_file_cache(path)
        memory_hygiene.trim_host_heap()

    # -- key space ----------------------------------------------------------
    def layer_keys(self, layer_idx: int, prefix: str = "model.layers") -> List[str]:
        tag = f"{prefix}.{layer_idx}."
        return sorted(k for k in self.weight_map if k.startswith(tag))

    def unchanged_keys(self) -> List[str]:
        """Non-quantized components per the pinned format contract (P3-02)."""
        keep = []
        for key in sorted(self.weight_map):
            if key.startswith("model.layers."):
                parts = key.split(".", 3)
                tail = parts[3] if len(parts) == 4 else ""
                if tail in ("input_layernorm.weight",
                            "post_attention_layernorm.weight"):
                    keep.append(key)
                continue
            if key in ("model.embed_tokens.weight", "model.norm.weight",
                       "lm_head.weight"):
                keep.append(key)
        return keep

    def pass_through_groups(self, geometry: "ModelGeometry") -> Dict[str, List[str]]:
        """Partition non-layer tensors into bit-exact pass-through groups (P4-13).

        * ``text`` — text embed / final norm / lm_head (the dense profile's
          historical :meth:`unchanged_keys` content, byte-identical).
        * ``vision`` — the multimodal vision tower (``model.visual.*``),
          preserved, never quantized.
        * ``mtp`` — the multi-token-prediction head (``mtp.*``), preserved for
          completeness though unused by text-only inference.

        Each group is written to its own ``qtensors/untouched.*.safetensors``
        file so no single serialization ever holds all pass-through bytes
        (the declared target's text group alone is ~4 GiB).
        """
        if not geometry.is_hybrid:
            return {"text": self.unchanged_keys()}
        text = [k for k in (geometry.embed_key, geometry.final_norm_key)
                if k in self.weight_map]
        if geometry.lm_head_key in self.weight_map:
            text.append(geometry.lm_head_key)
        return {
            "text": text,
            "vision": sorted(k for k in self.weight_map
                             if k.startswith("model.visual.")),
            "mtp": sorted(k for k in self.weight_map if k.startswith("mtp.")),
        }

    # -- streaming ----------------------------------------------------------
    def read_tensor(self, key: str) -> torch.Tensor:
        """Materialize exactly one tensor from its shard (mmap-backed)."""
        shard = self.weight_map[key]
        with safe_open(self.model_dir / shard, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def read_embedding_rows(self, key: str, rows: torch.Tensor) -> torch.Tensor:
        """Gather embedding rows for the calibration token ids.

        Reads the table transiently and returns only the gathered rows, so the
        full embedding (1.9 GiB bf16 for the declared target) is not kept
        resident for the whole run.
        """
        weight = self.read_tensor(key)
        gathered = weight[rows.to(weight.device)]
        del weight
        return gathered

    def read_tensors(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        """Materialize only the requested tensors, grouped per shard."""
        out: Dict[str, torch.Tensor] = {}
        for shard in self.shards:
            shard_keys = [k for k in keys if self.weight_map[k] == shard]
            if not shard_keys:
                continue
            with safe_open(self.model_dir / shard, framework="pt",
                           device="cpu") as f:
                for k in shard_keys:
                    out[k] = f.get_tensor(k)
        return out

    def read_layer_tensors(self, layer_idx: int,
                           prefix: str = "model.layers") -> Dict[str, torch.Tensor]:
        """Materialize only this layer's tensors, opening each shard once."""
        keys = self.layer_keys(layer_idx, prefix=prefix)
        _require(bool(keys), f"no tensors found for layer {layer_idx}")
        return self.read_tensors(keys)


# ---------------------------------------------------------------------------
# P3-02: upstream calibration propagation & conversion order
# ---------------------------------------------------------------------------

#: Linear submodules quantized in every decoder layer, in conversion order.
_QUANTIZED_MODULES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)

_NORM_KEYS = ("input_layernorm.weight", "post_attention_layernorm.weight")


@dataclass
class LayerConversionEntry:
    """Manifest + report entry for the conversion of one decoder layer."""

    layer_index: int
    module_keys: List[str]
    quantized_tensors: int
    proxy_loss: float
    normalized_proxy_loss: float
    mean_mse: float
    max_factorization_residual: float
    g_scales: Dict[str, float]
    elapsed_seconds: float
    rss_kib_before: int
    rss_kib_after: int
    # P4-12/P4-13: hybrid layer kind + the bit-exact tensors emitted with it
    layer_type: str = "full_attention"
    pass_through_keys: List[str] = field(default_factory=list)


@dataclass
class ConversionReport:
    """Aggregate result of one bounded conversion run."""

    geometry: ModelGeometry
    layers: List[LayerConversionEntry] = field(default_factory=list)
    unchanged_keys: List[str] = field(default_factory=list)
    device: str = ""
    total_elapsed_seconds: float = 0.0
    # P4-13: group -> {file, keys, quant_format} for the pass-through files
    pass_through: Dict[str, dict] = field(default_factory=dict)
    # P3-09: real-corpus provenance (None = synthetic deterministic corpus)
    corpus_meta: Optional[Dict] = None
    corpus_sha256: Optional[str] = None


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("xpu:0" if torch.xpu.is_available() else "cpu")
    return torch.device(requested)


def _build_layer(geometry: ModelGeometry, tensors: Dict[str, torch.Tensor],
                 layer_idx: int, device: torch.device) -> torch.nn.Module:
    """Instantiate the decoder layer for this profile from streamed tensors.

    Dense profiles get the historical :class:`TransformerDecoderLayer`;
    hybrid text-tower layers are built by the pinned reference implementation
    (``transformers.models.qwen3_5``) so calibration matches the model math.
    """
    prefix = f"{geometry.key_prefix}.{layer_idx}."
    if geometry.is_hybrid:
        return hybrid.build_layer(
            hybrid.make_text_config(geometry.text_config),
            layer_idx, tensors, prefix=geometry.key_prefix, device=device)
    layer = TransformerDecoderLayer(
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        num_heads=geometry.num_heads,
        num_kv_heads=geometry.num_kv_heads,
        rms_norm_eps=geometry.rms_norm_eps,
    )
    layer.self_attn.head_dim = geometry.head_dim
    layer.to(device=device, dtype=torch.float16)
    for name, module in [(m, getattr_path(layer, m)) for m in _QUANTIZED_MODULES]:
        key = prefix + name + ".weight"
        _require(key in tensors, f"layer {layer_idx} is missing required tensor {key}")
        module.weight.data.copy_(
            tensors[key].to(device=device, dtype=torch.float16))
    for tail in _NORM_KEYS:
        key = prefix + tail
        _require(key in tensors, f"layer {layer_idx} is missing required tensor {key}")
        norm = layer.input_layernorm if tail.startswith("input") \
            else layer.post_attention_layernorm
        norm.weight.data.copy_(
            tensors[key].to(device=device, dtype=torch.float16))
    return layer


def getattr_path(root, dotted: str):
    """Resolve ``"self_attn.q_proj"`` through nested attributes."""
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _reload_quantized(layer: TransformerDecoderLayer,
                      results: Dict[str, QuantizedModuleResult]) -> None:
    """Upstream step 3: swap quantized numerics (``w_hat``) into the module.

    ``w_hat`` is stored as (k, n) = (in_features, out_features); ``nn.Linear``
    weights are (out_features, in_features), hence the transposing copy.
    """
    for name, res in results.items():
        module = getattr_path(layer, name)
        module.weight.data.copy_(
            res.w_hat.detach().T.contiguous().to(module.weight.dtype))


def _build_quantized_layer(
    geometry: ModelGeometry,
    tensors: Dict[str, torch.Tensor],
    layer_idx: int,
    device: torch.device,
    K: int,
) -> torch.nn.Module:
    """Instantiate the decoder layer from decoded quantized tensors (P3-04).

    Projections are decoded from their trellis storage; every remaining
    parameter (the two layer norms, and for hybrid layers the bit-exact
    pass-through state — GatedDeltaNet conv/A_log/dt_bias/norm or q/k norms)
    is taken verbatim from the artifact.
    """
    prefix = f"{geometry.key_prefix}.{layer_idx}."
    modules = geometry.quantized_modules(layer_idx)

    if geometry.is_hybrid:
        decoded: Dict[str, torch.Tensor] = {}
        for name in modules:
            base = prefix + name
            w = decode_weight_from_storage(
                tensors[f"{base}.trellis"], tensors[f"{base}.suh"],
                tensors[f"{base}.svh"], K=K)
            decoded[f"{name}.weight"] = w.T.contiguous()
        return hybrid.build_layer(
            hybrid.make_text_config(geometry.text_config),
            layer_idx, tensors, prefix=geometry.key_prefix,
            device=device, decoded=decoded)

    layer = TransformerDecoderLayer(
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        num_heads=geometry.num_heads,
        num_kv_heads=geometry.num_kv_heads,
        rms_norm_eps=geometry.rms_norm_eps,
    )
    layer.self_attn.head_dim = geometry.head_dim
    layer.to(device=device, dtype=torch.float16)
    for name in modules:
        base = prefix + name
        trellis = tensors[f"{base}.trellis"]
        suh = tensors[f"{base}.suh"]
        svh = tensors[f"{base}.svh"]
        w = decode_weight_from_storage(trellis, suh, svh, K=K)
        module = getattr_path(layer, name)
        module.weight.data.copy_(w.T.contiguous().to(device=device, dtype=torch.float16))
    for tail in _NORM_KEYS:
        key = prefix + tail
        norm = layer.input_layernorm if tail.startswith("input") \
            else layer.post_attention_layernorm
        norm.weight.data.copy_(tensors[key].to(device=device, dtype=torch.float16))
    return layer


# ---------------------------------------------------------------------------
# P3-04..P3-08: Checkpointing, atomic publication, integrity & fault recovery
# ---------------------------------------------------------------------------

def check_disk_space(path: Path, min_bytes: int) -> None:
    """Verify free disk space at path exceeds min_bytes (P3-08)."""
    if min_bytes <= 0:
        return
    check_dir = path
    while not check_dir.exists() and check_dir.parent != check_dir:
        check_dir = check_dir.parent
    try:
        usage = shutil.disk_usage(check_dir)
        if usage.free < min_bytes:
            raise DiskSpaceError(
                f"insufficient disk space at {path}: {usage.free} bytes free, "
                f"requires at least {min_bytes} bytes"
            )
    except OSError as e:
        if isinstance(e, DiskSpaceError):
            raise


def atomic_save_safetensors(tensors: Dict[str, torch.Tensor], target_path: Path) -> None:
    """Atomically save safetensors file via temporary file and rename (P3-05)."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")

    # Save to tmp
    save_file(tensors, str(tmp_path))

    # Verify integrity of written tmp file before publication
    try:
        with safe_open(str(tmp_path), framework="pt", device="cpu") as f:
            keys = f.keys()
            for k in tensors:
                if k not in keys:
                    raise CorruptedCheckpointError(f"written checkpoint missing key: {k}")
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise CorruptedCheckpointError(f"failed to verify checkpoint before publication: {e}") from e

    # Atomic rename (POSIX replace)
    os.replace(str(tmp_path), str(target_path))


def verify_layer_checkpoint(layer_path: Path, layer_idx: int,
                            expected_modules: Optional[List[str]] = None,
                            key_prefix: str = "model.layers",
                            expected_extra_keys: Optional[List[str]] = None) -> bool:
    """Verify that a layer's .safetensors artifact is complete and valid (P3-05).

    Returns True if valid. Raises CorruptedCheckpointError if file exists but is corrupted/truncated.
    Returns False if file does not exist.

    ``expected_modules``/``key_prefix``/``expected_extra_keys`` generalize the
    check to hybrid layers (P4-12), whose projection table differs and whose
    artifact also carries the bit-exact pass-through tensors; the defaults are
    exactly the dense-profile expectations.
    """
    modules = _QUANTIZED_MODULES if expected_modules is None else expected_modules
    layer_path = Path(layer_path)
    if not layer_path.is_file():
        return False
    if layer_path.stat().st_size == 0:
        raise CorruptedCheckpointError(f"layer {layer_idx} checkpoint is empty: {layer_path}")
    try:
        with safe_open(str(layer_path), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            expected_suffixes = (".trellis", ".suh", ".svh", ".mul1")
            for mod in modules:
                base = f"{key_prefix}.{layer_idx}.{mod}"
                for sfx in expected_suffixes:
                    if f"{base}{sfx}" not in keys:
                        raise CorruptedCheckpointError(
                            f"layer {layer_idx} checkpoint missing tensor: {base}{sfx}"
                        )
            for tail in _NORM_KEYS:
                norm_key = f"{key_prefix}.{layer_idx}.{tail}"
                if norm_key not in keys:
                    raise CorruptedCheckpointError(
                        f"layer {layer_idx} checkpoint missing norm tensor: {norm_key}"
                    )
            for extra in expected_extra_keys or ():
                if extra not in keys:
                    raise CorruptedCheckpointError(
                        f"layer {layer_idx} checkpoint missing pass-through "
                        f"tensor: {extra}"
                    )
            return True
    except Exception as e:
        if isinstance(e, CorruptedCheckpointError):
            raise
        raise CorruptedCheckpointError(f"layer {layer_idx} checkpoint failed safe_open: {e}") from e


def _validate_resume(config: ConversionConfig, geometry: ModelGeometry,
                     manifest: dict) -> None:
    """Verify that current config matches the checkpoint manifest (P3-07)."""
    if config.force_resume:
        return

    mismatches = []
    if manifest.get("quant_method") != "exl3":
        mismatches.append(f"quant_method: expected 'exl3', got {manifest.get('quant_method')!r}")
    if manifest.get("bits") != config.K:
        mismatches.append(f"bits (K): config has {config.K}, manifest has {manifest.get('bits')}")
    if manifest.get("codebook") != config.cb:
        mismatches.append(f"codebook: config has {config.cb!r}, manifest has {manifest.get('codebook')!r}")

    cal = manifest.get("calibration", {})
    if cal.get("seed") != config.calibration_seed:
        mismatches.append(f"calibration.seed: config has {config.calibration_seed}, manifest has {cal.get('seed')}")
    if cal.get("seq_len") != config.seq_len:
        mismatches.append(f"calibration.seq_len: config has {config.seq_len}, manifest has {cal.get('seq_len')}")
    if cal.get("batch_size") != config.batch_size:
        mismatches.append(f"calibration.batch_size: config has {config.batch_size}, manifest has {cal.get('batch_size')}")
    if cal.get("num_rows") != config.num_calibration_rows:
        mismatches.append(f"calibration.num_rows: config has {config.num_calibration_rows}, manifest has {cal.get('num_rows')}")

    # Corpus identity (P3-09): the pre-tokenized file's content hash, or the
    # synthetic corpus. Presence must agree, and file content must be stable.
    want_sha = (calibration_token_ids_sha256(config.calibration_token_ids)
                if config.calibration_token_ids else None)
    have_ids = cal.get("token_ids")
    if want_sha is None and have_ids is not None:
        mismatches.append("calibration.token_ids: config uses the synthetic "
                          "corpus, manifest records a token-ids file")
    elif want_sha is not None and have_ids is None:
        mismatches.append("calibration.token_ids: config uses a token-ids "
                          f"file ({Path(config.calibration_token_ids).name}), "
                          "manifest records the synthetic corpus")
    elif want_sha is not None and have_ids.get("sha256") != want_sha:
        mismatches.append("calibration.token_ids.sha256: corpus file content "
                          f"differs from the manifest ({want_sha[:12]} vs "
                          f"{str(have_ids.get('sha256'))[:12]})")

    if manifest.get("architecture") != geometry.architecture:
        mismatches.append(f"architecture: config has {geometry.architecture!r}, manifest has {manifest.get('architecture')!r}")
    if manifest.get("num_layers") != geometry.num_layers:
        mismatches.append(f"num_layers: config has {geometry.num_layers}, manifest has {manifest.get('num_layers')}")
    manifest_mode = manifest.get("architecture_mode", "dense")
    if manifest_mode != config.architecture_mode:
        mismatches.append(
            f"architecture_mode: config has {config.architecture_mode!r}, "
            f"manifest has {manifest_mode!r}")
    if geometry.is_hybrid and manifest.get("layer_types") != geometry.layer_types:
        mismatches.append(
            "layer_types: config text_config and manifest disagree "
            f"(manifest has {manifest.get('layer_types')!r})")

    if mismatches:
        raise IncompatibleResumeError(
            f"incompatible resume arguments with existing checkpoint in {config.output_dir}:\n" +
            "\n".join(f"  - {m}" for m in mismatches) +
            "\n(use force_resume=True to override)"
        )


def convert_model(
    config: ConversionConfig,
    geometry: Optional[ModelGeometry] = None,
    progress: Optional[Callable[[str], None]] = None,
    fault_injector: Optional[Callable[[str, int], None]] = None,
) -> ConversionReport:
    """Bounded conversion driver with atomic publication and resumability (P3-01..P3-08).

    Executes the audited P3-02 loop over the streamed P3-03 checkpoint.
    Supports atomic per-layer checkpointing, explicit resumability (P3-04),
    integrity verification (P3-05), fault-injection safe restart (P3-06),
    resume argument validation (P3-07), and disk-space pre-checking (P3-08).
    """
    t_start = time.time()
    if geometry is None:
        geometry = config.validate()

    model_dir = Path(config.model_dir)
    out_dir = Path(config.output_dir)
    check_disk_space(out_dir, config.min_disk_free_bytes)
    out_dir.mkdir(parents=True, exist_ok=True)
    qt_dir = out_dir / "qtensors"
    qt_dir.mkdir(parents=True, exist_ok=True)

    # Clean up any partial or stray .tmp files from an interrupted prior write
    for tmp_file in qt_dir.glob("*.tmp"):
        try:
            tmp_file.unlink()
        except OSError:
            pass

    device = _resolve_device(config.device)
    is_xpu = device.type == "xpu"

    def say(msg: str) -> None:
        if progress is not None:
            progress(msg)

    index = ShardedSafetensorsIndex(model_dir)
    if index.is_sharded:
        say(f"streaming sharded checkpoint: {len(index.shards)} shards")

    # ---- calibration corpus: pre-tokenized file (P3-09) or synthetic -------
    if config.calibration_token_ids is not None:
        rows, corpus_meta = load_calibration_token_rows(config, geometry)
        corpus_sha = calibration_token_ids_sha256(config.calibration_token_ids)
        say(f"calibration corpus: {len(rows)} rows from "
            f"{Path(config.calibration_token_ids).name} (sha256 {corpus_sha[:12]})")
    else:
        generator = torch.Generator().manual_seed(config.calibration_seed)
        rows = [
            torch.randint(0, geometry.vocab_size,
                          (config.batch_size, config.seq_len), generator=generator)
            for _ in range(config.num_calibration_rows)
        ]
        corpus_meta = None
        corpus_sha = None

    # ---- pass-through groups: bit-exact, atomically published (P4-13) ------
    # Dense profile: one group ("text") identical to the historical
    # unchanged_keys() content. Hybrid: text + vision + MTP, each in its own
    # file so no serialization ever holds all pass-through bytes at once.
    groups = index.pass_through_groups(geometry)
    group_files = {"text": "untouched.safetensors",
                   "vision": "untouched.vision.safetensors",
                   "mtp": "untouched.mtp.safetensors"}
    say("pass-through groups: " + ", ".join(
        f"{name}={len(keys)}" for name, keys in groups.items() if keys))

    report = ConversionReport(geometry=geometry,
                              unchanged_keys=list(groups["text"]),
                              device=str(device),
                              corpus_meta=corpus_meta,
                              corpus_sha256=corpus_sha)
    rss_start = read_rss_kib()
    for group_name, keys in groups.items():
        if not keys:
            continue
        target = qt_dir / group_files[group_name]
        if not (config.resume and target.is_file()):
            payload = index.read_tensors(keys)
            atomic_save_safetensors(payload, target)
            del payload
            # P3-09: the group payload (up to ~4.1 GiB for the text group) is
            # finished with; give the DRAM back instead of pinning it for the
            # whole 32-layer run (the admission policy sees host pressure).
            index.release_source_cache()
        report.pass_through[group_name] = {
            "file": f"qtensors/{group_files[group_name]}",
            "keys": list(keys),
            "quant_format": "source-dtype",
            "unchanged": True,
        }
    say(f"unchanged (pass-through) components: {len(report.unchanged_keys)}")

    embed_key = geometry.embed_key
    _require(embed_key in index.weight_map,
             f"checkpoint lacks {embed_key}")

    # -- forward runner: dense profile keeps the pinned rotary invocation,
    #    hybrid layers use the reference qwen3_5 forward (partial mrope + causal
    #    mask on full-attention blocks, padding-mask slot on linear blocks).
    if geometry.is_hybrid:
        pos_emb = None
        runner = hybrid.make_calibration_runner(
            hybrid.make_text_config(geometry.text_config), device)
    else:
        pos_emb = make_rotary_embeddings(seq_len=config.seq_len,
                                         head_dim=geometry.head_dim,
                                         device=device, dtype=torch.float16)

        def runner(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            return layer(x, position_embeddings=pos_emb)

    def embed_rows() -> List[torch.Tensor]:
        """Gather the calibration token rows only (table read transiently)."""
        with torch.no_grad():
            return [index.read_embedding_rows(embed_key, row)
                    .to(device=device, dtype=torch.float16) for row in rows]

    # ---- resume detection & validation (P3-04, P3-07) -----------------------
    completed_layers: List[int] = []
    resume_from_layer = 0
    existing_manifest = None
    manifest_path = out_dir / "quantization_config.json"

    if config.resume:
        if manifest_path.is_file():
            with open(manifest_path) as f:
                existing_manifest = json.load(f)
            _validate_resume(config, geometry, existing_manifest)

            # Check consecutive completed layers on disk
            for idx in range(geometry.num_layers):
                layer_file = qt_dir / f"model.layers.{idx}.safetensors"
                if verify_layer_checkpoint(
                        layer_file, idx,
                        expected_modules=geometry.quantized_modules(idx),
                        key_prefix=geometry.key_prefix,
                        expected_extra_keys=geometry.layer_pass_through_keys(idx)):
                    completed_layers.append(idx)
                else:
                    break

            if len(completed_layers) == geometry.num_layers and existing_manifest.get("status") == "completed":
                say("all layers already completed; verifying and returning existing report")
                entries = []
                for entry_data in existing_manifest.get("layers", []):
                    entries.append(LayerConversionEntry(
                        layer_index=entry_data["layer_index"],
                        module_keys=entry_data["modules"],
                        quantized_tensors=entry_data["quantized_tensors"],
                        proxy_loss=entry_data["proxy_loss"],
                        normalized_proxy_loss=entry_data["normalized_proxy_loss"],
                        mean_mse=entry_data["mean_mse"],
                        max_factorization_residual=entry_data["max_factorization_residual"],
                        g_scales=entry_data["g_scales"],
                        elapsed_seconds=entry_data["elapsed_seconds"],
                        rss_kib_before=entry_data["rss_kib_before"],
                        rss_kib_after=entry_data["rss_kib_after"],
                        layer_type=entry_data.get("layer_type", "full_attention"),
                        pass_through_keys=list(entry_data.get("pass_through_keys", [])),
                    ))
                report.layers = entries
                report.total_elapsed_seconds = existing_manifest.get("total_elapsed_seconds", 0.0)
                return report

            resume_from_layer = len(completed_layers)
            say(f"resuming conversion from layer {resume_from_layer} "
                f"({len(completed_layers)} layers already complete)")

            # Populate historical layer entries from manifest
            if existing_manifest and "layers" in existing_manifest:
                for entry_data in existing_manifest["layers"][:resume_from_layer]:
                    report.layers.append(LayerConversionEntry(
                        layer_index=entry_data["layer_index"],
                        module_keys=entry_data["modules"],
                        quantized_tensors=entry_data["quantized_tensors"],
                        proxy_loss=entry_data["proxy_loss"],
                        normalized_proxy_loss=entry_data["normalized_proxy_loss"],
                        mean_mse=entry_data["mean_mse"],
                        max_factorization_residual=entry_data["max_factorization_residual"],
                        g_scales=entry_data["g_scales"],
                        elapsed_seconds=entry_data["elapsed_seconds"],
                        rss_kib_before=entry_data["rss_kib_before"],
                        rss_kib_after=entry_data["rss_kib_after"],
                        layer_type=entry_data.get("layer_type", "full_attention"),
                        pass_through_keys=list(entry_data.get("pass_through_keys", [])),
                    ))

    # ---- running calibration state restoration or replay --------------------
    state_file = qt_dir / "calibration_state.pt"
    if resume_from_layer == 0:
        hidden_states = embed_rows()
    else:
        loaded_state = False
        if state_file.is_file():
            try:
                state_data = torch.load(str(state_file), map_location="cpu", weights_only=False)
                if (state_data.get("layer_index") == resume_from_layer - 1 and
                    state_data.get("seed") == config.calibration_seed and
                    state_data.get("corpus_sha256") == corpus_sha):
                    hidden_states = [h.to(device=device, dtype=torch.float16)
                                     for h in state_data["hidden_states"]]
                    loaded_state = True
                    say(f"loaded calibration state from checkpoint for layer {resume_from_layer - 1}")
            except Exception as e:
                say(f"could not load calibration_state.pt ({e}); falling back to replay")

        if not loaded_state:
            say(f"replaying calibration state through layers 0..{resume_from_layer - 1}")
            with torch.no_grad():
                hidden_states = embed_rows()
                for l_idx in range(resume_from_layer):
                    tensors = load_file(str(qt_dir / f"model.layers.{l_idx}.safetensors"))
                    prev_layer = _build_quantized_layer(geometry, tensors, l_idx, device, K=config.K)
                    hidden_states = [runner(prev_layer, x) for x in hidden_states]
                    del tensors, prev_layer

    for layer_idx in range(resume_from_layer, geometry.num_layers):
        check_disk_space(out_dir, config.min_disk_free_bytes)
        layer_rss_before = read_rss_kib()
        t_layer = time.time()
        say(f"[layer {layer_idx}] streaming unquantized module (mmap)")

        if fault_injector is not None:
            fault_injector("before_calibration", layer_idx)

        # P3-03: only this layer's tensors are materialized
        prefix = f"{geometry.key_prefix}.{layer_idx}."
        tensors = index.read_layer_tensors(layer_idx, prefix=geometry.key_prefix)
        norm_weights = {tail: tensors[f"{prefix}{tail}"]
                        for tail in _NORM_KEYS}
        # P4-12/P4-13: parameters that are neither quantized projections nor
        # the two layer norms (GatedDeltaNet state, q/k norms, ...) are
        # emitted bit-exact alongside the quantized tensors. The dense profile
        # keeps its historical key set exactly (nothing else is emitted).
        norm_key_set = {f"{prefix}{tail}" for tail in _NORM_KEYS}
        quantized_key_set = {f"{prefix}{name}.weight"
                             for name in geometry.quantized_modules(layer_idx)}
        extra_tensors = (
            {k: v for k, v in tensors.items()
             if k not in quantized_key_set and k not in norm_key_set}
            if geometry.is_hybrid else {})
        layer = _build_layer(geometry, tensors, layer_idx, device)
        del tensors
        # P3-09: this layer's source pages are consumed; evict them so a
        # 32-layer run does not grow the resident page cache to the full
        # checkpoint size (the admission policy reads host pressure).
        index.release_source_cache()

        # -- step 1+2: capture Hessians on the current state, then quantize --
        say(f"[layer {layer_idx}] capturing grouped Hessians")
        hessians, _telemetry = calibrate_layer(
            layer=layer,
            calibration_data=hidden_states,
            device=device,
            damping_lambda=config.damping_lambda,
            zero_floor=config.zero_floor,
            layer_runner=runner,
        )

        if fault_injector is not None:
            fault_injector("after_calibration", layer_idx)

        if fault_injector is not None:
            fault_injector("before_quantization", layer_idx)

        say(f"[layer {layer_idx}] quantizing projections (K={config.K}, "
            f"cb={config.cb})")
        layer_report = quantize_decoder_layer(
            layer=layer,
            hessians=hessians,
            K=config.K,
            device=device,
            skip_g_scale=config.skip_g_scale,
            module_names=list(geometry.quantized_modules(layer_idx)),
        )
        del hessians

        if fault_injector is not None:
            fault_injector("after_quantization", layer_idx)

        results = layer_report.module_results
        say(f"[layer {layer_idx}] reloading module with quantized numerics")
        _reload_quantized(layer, results)

        # -- step 4: advance state through the QUANTIZED module --------------
        # Re-forwards the *same pre-layer* hidden states (upstream semantics).
        with torch.no_grad():
            hidden_states = [runner(layer, x) for x in hidden_states]

        if fault_injector is not None:
            fault_injector("before_publish", layer_idx)

        # -- per-layer working-dir artifact (upstream qtensors layout) -------
        emitted, _storage = build_layer_tensors(
            results, layer_idx=layer_idx, prefix=geometry.key_prefix,
            norm_weights=norm_weights, extra_tensors=extra_tensors,
            preserve_source_dtype=geometry.is_hybrid)
        layer_file = qt_dir / f"model.layers.{layer_idx}.safetensors"
        atomic_save_safetensors(emitted, layer_file)
        num_emitted = len(emitted)
        del emitted
        # P3-09: the published artifact is on disk; do not keep its pages
        # resident for the rest of the run.
        memory_hygiene.release_after_stream(layer_file)
        verify_layer_checkpoint(
            layer_file, layer_idx,
            expected_modules=geometry.quantized_modules(layer_idx),
            key_prefix=geometry.key_prefix,
            expected_extra_keys=sorted(extra_tensors))

        # Save calibration state atomically
        state_payload = {
            "layer_index": layer_idx,
            "seed": config.calibration_seed,
            "corpus_sha256": corpus_sha,
            "hidden_states": [h.detach().cpu().half() for h in hidden_states],
        }
        state_tmp = qt_dir / "calibration_state.pt.tmp"
        torch.save(state_payload, str(state_tmp))
        os.replace(str(state_tmp), str(state_file))
        del state_payload
        # P3-09: the state file (hidden states, ~2.1 GiB at the pinned recipe)
        # is on disk; do not keep its pages resident.
        memory_hygiene.release_after_stream(state_file)

        if fault_injector is not None:
            fault_injector("after_publish", layer_idx)

        metrics = {k: v.metrics for k, v in results.items()}
        entry = LayerConversionEntry(
            layer_index=layer_idx,
            module_keys=list(results.keys()),
            quantized_tensors=num_emitted,
            proxy_loss=sum(m.proxy_loss for m in metrics.values()) / len(metrics),
            normalized_proxy_loss=sum(m.normalized_proxy_loss
                                      for m in metrics.values()) / len(metrics),
            mean_mse=sum(m.mse for m in metrics.values()) / len(metrics),
            max_factorization_residual=max(
                m.factorization_residual for m in metrics.values()),
            g_scales={k: m.g_scale for k, m in metrics.items()},
            elapsed_seconds=time.time() - t_layer,
            rss_kib_before=layer_rss_before,
            rss_kib_after=read_rss_kib(),
            layer_type=geometry.layer_type(layer_idx),
            pass_through_keys=sorted(extra_tensors),
        )
        report.layers.append(entry)
        say(f"[layer {layer_idx}] done: proxy_loss={entry.proxy_loss:.4e} "
            f"({entry.elapsed_seconds:.1f}s)")

        if is_xpu:
            torch.xpu.synchronize(device.index or 0)
            torch.xpu.empty_cache()

        # Configurable checkpoint interval
        if (layer_idx + 1) % config.checkpoint_interval == 0 or layer_idx == geometry.num_layers - 1:
            report.total_elapsed_seconds = time.time() - t_start
            _write_manifest(out_dir, report, config, status="in_progress")

    say("writing final manifest")
    report.total_elapsed_seconds = time.time() - t_start
    _write_manifest(out_dir, report, config, status="completed")
    say(f"conversion complete: {len(report.layers)} layers "
        f"(stream RSS {rss_start} -> {read_rss_kib()} KiB)")
    return report


def _write_manifest(out_dir: Path, report: ConversionReport,
                    config: ConversionConfig,
                    status: str = "completed") -> None:
    """Merged ``quantization_config.json`` + unchanged-component pass-through (atomic)."""
    geometry = report.geometry
    tensor_storage: Dict[str, dict] = {}
    for entry in report.layers:
        for module_key in entry.module_keys:
            tensor_storage[f"{geometry.key_prefix}.{entry.layer_index}.{module_key}"] = {
                "quant_format": "exl3",
                "bits_per_weight": config.K,
                "codebook": config.cb,
            }
    if geometry.is_hybrid:
        # Every pass-through group is recorded with its file and source dtype;
        # per-layer pass-through keys are recorded on their layer entry.
        for group_name, info in report.pass_through.items():
            for key in info["keys"]:
                tensor_storage[key] = {
                    "quant_format": "source-dtype",
                    "unchanged": True,
                    "pass_through_group": group_name,
                }
    else:
        for key in report.unchanged_keys:
            tensor_storage[key] = {"quant_format": "fp16", "unchanged": True}
    manifest = {
        "status": status,
        "completed_layers": [e.layer_index for e in report.layers],
        "last_completed_layer": report.layers[-1].layer_index if report.layers else None,
        "quant_method": "exl3",
        "bits": config.K,
        "codebook": config.cb,
        "architecture": geometry.architecture,
        "architecture_mode": config.architecture_mode,
        "key_prefix": geometry.key_prefix,
        "num_layers": geometry.num_layers,
        "layer_types": geometry.layer_types,
        "device": report.device,
        "calibration": {
            "num_rows": config.num_calibration_rows,
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "seed": config.calibration_seed,
            "damping_lambda": config.damping_lambda,
            "zero_floor": config.zero_floor,
            "skip_g_scale": config.skip_g_scale,
            "corpus": ("synthetic-deterministic"
                       if report.corpus_meta is None else
                       str(report.corpus_meta.get(
                           "corpus",
                           Path(config.calibration_token_ids).name))),
            "token_ids": (None if report.corpus_meta is None else {
                "path": config.calibration_token_ids,
                "sha256": report.corpus_sha256,
                "metadata": report.corpus_meta,
            }),
        },
        "pass_through": report.pass_through,
        "hybrid": (None if not geometry.is_hybrid else {
            "text_tower_only": True,
            "multimodal": geometry.hybrid_meta.get("multimodal"),
            "vision_config": geometry.hybrid_meta.get("vision_config"),
            "mtp_num_hidden_layers": geometry.hybrid_meta.get("mtp_num_hidden_layers"),
            "rope": geometry.hybrid_meta.get("rope"),
            "norm_semantics": geometry.hybrid_meta.get("norm_semantics"),
            "attn_output_gate": geometry.hybrid_meta.get("attn_output_gate"),
            "linear_attention": geometry.hybrid_meta.get("linear_attention"),
            "reference_forward": hybrid.environment(),
        }),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "tensor_storage": tensor_storage,
        "layers": [
            {
                "layer_index": e.layer_index,
                "layer_type": e.layer_type,
                "modules": e.module_keys,
                "pass_through_keys": e.pass_through_keys,
                "quantized_tensors": e.quantized_tensors,
                "proxy_loss": e.proxy_loss,
                "normalized_proxy_loss": e.normalized_proxy_loss,
                "mean_mse": e.mean_mse,
                "max_factorization_residual": e.max_factorization_residual,
                "g_scales": e.g_scales,
                "elapsed_seconds": e.elapsed_seconds,
                "rss_kib_before": e.rss_kib_before,
                "rss_kib_after": e.rss_kib_after,
            }
            for e in report.layers
        ],
        "total_elapsed_seconds": report.total_elapsed_seconds,
    }
    tmp_path = out_dir / "quantization_config.json.tmp"
    final_path = out_dir / "quantization_config.json"
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(str(tmp_path), str(final_path))

