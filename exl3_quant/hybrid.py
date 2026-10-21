"""Qwen3.5 hybrid text-tower support (P4-12 / P4-13).

The declared Active Target ``empero-ai/Qwen3.8-9B-Distill`` (and its sibling
distill ``XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B``) is a
``Qwen3_5ForConditionalGeneration`` checkpoint: a 32-layer hybrid text tower
(24 GatedDeltaNet ``linear_attention`` layers + 8 ``full_attention`` layers,
3:1), a 27-layer vision tower, and a 1-layer MTP head.

Quantization contract of this extension (see ``tasks.md`` P4-12/P4-13):

* **Linear projections of both layer kinds are quantized** with the unchanged
  per-linear EXL3 machinery (Hessian -> regularize -> LDLQ -> trellis). The
  module table below is exhaustive; every other decoder-layer parameter is
  preserved bit-exact.
* ``linear_attn.in_proj_a`` / ``in_proj_b`` are **pass-through**: their output
  width is ``linear_num_value_heads`` (32), not a multiple of the 128-wide
  Hadamard block that ``regularize``/``ldlq``/``pack_trellis`` require. They
  are 0.26 M parameters per layer (0.09 % of the text tower) — quantizing them
  would require a format extension, which is out of scope.
* ``linear_attn.{conv1d.weight,A_log,dt_bias,norm.weight}``,
  ``self_attn.{q_norm,k_norm}.weight`` and both layer norms are non-linear
  parameters: bit-exact pass-through.
* Vision tower (``model.visual.*``) and MTP head (``mtp.*``) are bit-exact
  pass-through; they are recorded, never quantized (P4-13 / P4-04).

The layer forward used for calibration is the **reference implementation**
from the pinned environment (``transformers.models.qwen3_5``), not a
re-implementation: GatedDeltaNet (chunked gated delta rule, causal conv1d,
gated RMSNorm) and gated full attention with partial/interleaved mrope are
easy to get subtly wrong, and calibration Hessians depend on exactness. The
manifest records the reference package version and whether the reference
PyTorch fallbacks (``causal_conv1d`` / ``flash-linear-attention`` not
installed) were used.
"""

from __future__ import annotations

import importlib.util
from typing import Callable, Dict, List, Optional, Tuple

import torch

try:  # pragma: no cover - exercised through the converter tests on this host
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5TextRotaryEmbedding,
    )
    _TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - only on hosts without qwen3_5
    Qwen3_5TextConfig = None  # type: ignore[assignment]
    Qwen3_5DecoderLayer = None  # type: ignore[assignment]
    Qwen3_5TextRotaryEmbedding = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc


#: Architecture strings accepted in the hybrid text-tower mode.
TEXT_TOWER_ARCHITECTURES = ("Qwen3_5ForConditionalGeneration",)
TEXT_TOWER_MODEL_TYPES = ("qwen3_5",)

#: Decoder layer kinds covered by this extension.
SUPPORTED_LAYER_TYPES = ("linear_attention", "full_attention")

#: Legacy -> current layer-type names, mirroring
#: ``transformers.configuration_utils._LEGACY_LAYER_TYPE_REMAP`` so that
#: checkpoints written with older names still validate like the reference.
LEGACY_LAYER_TYPE_REMAP = {
    "conv": "linear_attention",   # only in LFMv2
    "mamba": "linear_attention",
    "attention": "full_attention",
}

#: Exhaustive quantized projection table per layer kind, in conversion order.
QUANTIZED_MODULES: Dict[str, Tuple[str, ...]] = {
    "linear_attention": (
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ),
    "full_attention": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ),
}

#: Per-layer parameters preserved bit-exact per layer kind (P4-12):
#: everything that is not a quantized projection and not one of the two layer
#: norms (which have their own well-known slots). ``in_proj_a``/``in_proj_b``
#: appear here because their output width is not a multiple of the 128-wide
#: Hadamard block the EXL3 encoder requires.
PASS_THROUGH_MODULE_NAMES: Dict[str, Tuple[str, ...]] = {
    "linear_attention": (
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.conv1d.weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.norm.weight",
    ),
    "full_attention": (
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ),
}


class HybridSupportUnavailable(RuntimeError):
    """Raised when the hybrid reference implementation is unavailable."""


def require_support() -> None:
    """Fail loudly (and actionably) if transformers' qwen3_5 is missing."""
    if Qwen3_5DecoderLayer is None:
        raise HybridSupportUnavailable(
            "the Qwen3.5 hybrid text-tower mode needs transformers with "
            "Qwen3_5 support (>= 5.17); import failed: "
            f"{_TRANSFORMERS_IMPORT_ERROR!r}"
        )


def quantized_module_names(layer_type: str) -> Tuple[str, ...]:
    """Projection table for one decoder layer kind."""
    if layer_type not in QUANTIZED_MODULES:
        raise HybridSupportUnavailable(
            f"unsupported hybrid layer type {layer_type!r}; supported: "
            f"{sorted(SUPPORTED_LAYER_TYPES)}")
    return QUANTIZED_MODULES[layer_type]


def make_text_config(text_cfg: dict):
    """Build the reference ``Qwen3_5TextConfig`` from a ``config.json`` block."""
    require_support()
    config = Qwen3_5TextConfig(**dict(text_cfg))
    # Force the portable eager attention path: the converter must not depend on
    # a device-specific fused kernel for its calibration semantics.
    config._attn_implementation = "eager"
    return config


def build_layer(
    text_config,
    layer_idx: int,
    tensors: Dict[str, torch.Tensor],
    prefix: str,
    device: torch.device,
    decoded: Optional[Dict[str, torch.Tensor]] = None,
) -> "Qwen3_5DecoderLayer":
    """Instantiate one reference decoder layer and load its exact tensors.

    ``decoded`` maps dotted parameter names (e.g.
    ``"mlp.down_proj.weight"``) to already-decoded ``(out, in)`` weight
    tensors; those parameters bypass ``tensors`` — used when replaying a
    published quantized artifact whose projections live in trellis form.

    ``prefix`` is the checkpoint's decoder-layer prefix (e.g.
    ``model.language_model.layers``); tensors are keyed
    ``{prefix}.{layer_idx}.{name}``.

    ``strict=True`` load guarantees the parameter set matches the reference
    module tree exactly; a missing or unexpected tensor is an explicit
    failure rather than a silently mis-calibrated layer.
    """
    require_support()
    layer = Qwen3_5DecoderLayer(text_config, layer_idx)
    layer.to(device=device, dtype=torch.float16)
    decoded = decoded or {}
    state: Dict[str, torch.Tensor] = {}
    for name, _param in layer.named_parameters():
        if name in decoded:
            state[name] = decoded[name].to(device=device, dtype=torch.float16)
            continue
        key = f"{prefix}.{layer_idx}.{name}"
        if key not in tensors:
            raise HybridSupportUnavailable(
                f"layer {layer_idx} is missing required tensor {key}")
        state[name] = tensors[key].to(device=device, dtype=torch.float16)
    layer.load_state_dict(state, strict=True)
    return layer


def make_calibration_runner(
    text_config,
    device: torch.device,
) -> Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]:
    """Return ``runner(layer, hidden_states)`` matching the reference forward.

    Text-only inputs use identical positions on all three mrope grids, so the
    reference rotary recomposition reduces to the usual partial rotary on the
    first ``rotary_dim`` channels (the reference implementation is still used
    verbatim). Full-attention layers get a causal additive mask; linear-
    attention layers take the 2-D padding mask position (``None`` for the
    unpadded calibration corpus) exactly like the reference text model.
    """
    require_support()
    rotary = Qwen3_5TextRotaryEmbedding(text_config).to(device)
    cache: dict = {}

    def runner(layer: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        block_type = getattr(layer, "block_type", "full_attention")
        key = (block_type, hidden_states.shape[0], hidden_states.shape[1],
               hidden_states.dtype)
        entry = cache.get(key)
        if entry is None:
            batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
            position_ids = torch.arange(seq_len, device=hidden_states.device)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch, -1)
            cos, sin = rotary(hidden_states, position_ids)
            attention_mask = None
            if block_type == "full_attention":
                attention_mask = torch.full(
                    (1, 1, seq_len, seq_len),
                    torch.finfo(hidden_states.dtype).min,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                ).triu(1)
            entry = (cos, sin, attention_mask)
            cache[key] = entry
        cos, sin, attention_mask = entry
        return layer(
            hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=attention_mask,
        )

    return runner


def environment() -> Dict[str, object]:
    """Reproducibility metadata for the reference forward and its kernels."""
    import transformers

    def _present(module: str) -> bool:
        return importlib.util.find_spec(module) is not None

    return {
        "reference": "transformers.models.qwen3_5",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "fast_kernels": {
            "causal_conv1d": _present("causal_conv1d"),
            "flash_linear_attention": _present("fla"),
        },
        "fallbacks": (
            "reference PyTorch implementations for causal_conv1d and "
            "chunk_gated_delta_rule (semantically exact, slower)"
        ),
    }
