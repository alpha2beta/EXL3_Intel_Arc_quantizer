"""Bounded full-model conversion orchestrator (Phase 3.1, P3-01/P3-02/P3-03).

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

Deliberately *not* implemented here (later tasks): checkpoint/resume
semantics (P3-04..P3-08), real-model qualification runs (P3-09+), fractional
bitrate recipes (P4-02), MoE / embedding / two-sided modes (P4-03..P4-05).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from exl3_quant.calibration import (
    TransformerDecoderLayer,
    calibrate_layer,
    make_rotary_embeddings,
)
from exl3_quant.layer_quant import (
    QuantizedModuleResult,
    quantize_decoder_layer,
)
from exl3_quant.serialization import build_layer_tensors

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


@dataclass
class ModelGeometry:
    """Validated geometry extracted from the input model's ``config.json``."""

    architecture: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UnsupportedModeError(message)


def load_model_geometry(model_dir: Path) -> ModelGeometry:
    """Read and validate ``config.json`` *without touching any model weights*."""
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise UnsupportedModeError(f"model dir has no config.json: {model_dir}")
    with open(config_path) as f:
        raw = json.load(f)

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

    # Unsupported structural modes: MoE, vision/multimodal, MTP, pre-quantized
    forbidden_keys = (
        "num_local_experts", "num_experts", "moe", "vision_config",
        "text_config", "mtp_num_heads", "quantization_config",
    )
    present = sorted(k for k in forbidden_keys
                     if raw.get(k) not in (None, [], {}))
    _require(
        not present,
        f"unsupported model features present in config: {present}",
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

    # Calibration. The synthetic deterministic token corpus is generated here;
    # wiring a real corpus / tokenizer belongs to the P3-09 qualification run.
    num_calibration_rows: int = 2
    batch_size: int = 1
    seq_len: int = 128
    calibration_seed: int = 0

    # Numerics pass-through to the quantization chain
    damping_lambda: float = 0.01
    zero_floor: float = 1e-7
    skip_g_scale: bool = False   # upstream default: real GSS search
    device: str = "auto"

    def __post_init__(self) -> None:
        self.model_dir = str(Path(self.model_dir))
        self.output_dir = str(Path(self.output_dir))

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

        out_dir = Path(self.output_dir).resolve()
        _require(out_dir != model_dir.resolve(),
                 "output_dir must differ from model_dir")
        _require(not out_dir.exists() or not any(out_dir.iterdir()),
                 f"output_dir must be empty or not exist: {out_dir}")

        return load_model_geometry(model_dir)


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

    # -- streaming ----------------------------------------------------------
    def read_tensor(self, key: str) -> torch.Tensor:
        """Materialize exactly one tensor from its shard (mmap-backed)."""
        shard = self.weight_map[key]
        with safe_open(self.model_dir / shard, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

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

    def read_layer_tensors(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """Materialize only this layer's tensors, opening each shard once."""
        keys = self.layer_keys(layer_idx)
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


@dataclass
class ConversionReport:
    """Aggregate result of one bounded conversion run."""

    geometry: ModelGeometry
    layers: List[LayerConversionEntry] = field(default_factory=list)
    unchanged_keys: List[str] = field(default_factory=list)
    device: str = ""
    total_elapsed_seconds: float = 0.0


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("xpu:0" if torch.xpu.is_available() else "cpu")
    return torch.device(requested)


def _build_layer(geometry: ModelGeometry, tensors: Dict[str, torch.Tensor],
                 layer_idx: int, device: torch.device) -> TransformerDecoderLayer:
    """Instantiate a :class:`TransformerDecoderLayer` from streamed tensors."""
    layer = TransformerDecoderLayer(
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        num_heads=geometry.num_heads,
        num_kv_heads=geometry.num_kv_heads,
        rms_norm_eps=geometry.rms_norm_eps,
    )
    layer.self_attn.head_dim = geometry.head_dim
    layer.to(device=device, dtype=torch.float16)
    prefix = f"model.layers.{layer_idx}."
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


def convert_model(
    config: ConversionConfig,
    geometry: Optional[ModelGeometry] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> ConversionReport:
    """Bounded conversion driver (P3-01 entry point).

    Executes the audited P3-02 loop over the streamed P3-03 checkpoint.
    Does **not** yet checkpoint/resume (P3-04..P3-08) and does **not** run a
    real-model qualification (P3-09) — those remain open tasks.
    """
    t_start = time.time()
    if geometry is None:
        geometry = config.validate()

    model_dir = Path(config.model_dir)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(config.device)
    is_xpu = device.type == "xpu"

    def say(msg: str) -> None:
        if progress is not None:
            progress(msg)

    index = ShardedSafetensorsIndex(model_dir)
    if index.is_sharded:
        say(f"streaming sharded checkpoint: {len(index.shards)} shards")

    # ---- deterministic synthetic calibration corpus ------------------------
    generator = torch.Generator().manual_seed(config.calibration_seed)
    rows = [
        torch.randint(0, geometry.vocab_size,
                      (config.batch_size, config.seq_len), generator=generator)
        for _ in range(config.num_calibration_rows)
    ]

    # ---- unchanged components: fp16 pass-through (format contract) ---------
    unchanged_keys = index.unchanged_keys()
    say(f"unchanged (pass-through) components: {len(unchanged_keys)}")

    embed_key = "model.embed_tokens.weight"
    _require(embed_key in index.weight_map,
             "checkpoint lacks model.embed_tokens.weight")
    embed_weight = index.read_tensor(embed_key)
    embed = torch.nn.Embedding(embed_weight.shape[0], embed_weight.shape[1])
    embed.weight.data.copy_(embed_weight)
    del embed_weight
    embed.to(device=device, dtype=torch.float16)

    pos_emb = make_rotary_embeddings(seq_len=config.seq_len,
                                     head_dim=geometry.head_dim,
                                     device=device, dtype=torch.float16)

    # ---- running calibration state (upstream `state`) ----------------------
    with torch.no_grad():
        hidden_states = [embed(row.to(device)) for row in rows]

    report = ConversionReport(geometry=geometry,
                              unchanged_keys=unchanged_keys,
                              device=str(device))
    rss_start = read_rss_kib()

    # Self-contained output: copy unchanged pass-through tensors verbatim
    # (source dtype preserved), per the pinned format contract.
    qt_dir = out_dir / "qtensors"
    qt_dir.mkdir(parents=True, exist_ok=True)
    unchanged_tensors = index.read_tensors(unchanged_keys)
    save_file(unchanged_tensors, str(qt_dir / "untouched.safetensors"))
    del unchanged_tensors

    for layer_idx in range(geometry.num_layers):
        layer_rss_before = read_rss_kib()
        t_layer = time.time()
        say(f"[layer {layer_idx}] streaming unquantized module (mmap)")

        # P3-03: only this layer's tensors are materialized
        tensors = index.read_layer_tensors(layer_idx)
        norm_weights = {tail: tensors[f"model.layers.{layer_idx}.{tail}"]
                        for tail in _NORM_KEYS}
        layer = _build_layer(geometry, tensors, layer_idx, device)
        del tensors

        # -- step 1+2: capture Hessians on the current state, then quantize --
        say(f"[layer {layer_idx}] capturing grouped Hessians")
        hessians, _telemetry = calibrate_layer(
            layer=layer,
            calibration_data=hidden_states,
            device=device,
            damping_lambda=config.damping_lambda,
            zero_floor=config.zero_floor,
        )
        say(f"[layer {layer_idx}] quantizing projections (K={config.K}, "
            f"cb={config.cb})")
        layer_report = quantize_decoder_layer(
            layer=layer,
            hessians=hessians,
            K=config.K,
            device=device,
            skip_g_scale=config.skip_g_scale,
        )
        del hessians

        results = layer_report.module_results
        say(f"[layer {layer_idx}] reloading module with quantized numerics")
        _reload_quantized(layer, results)

        # -- step 4: advance state through the QUANTIZED module --------------
        # Re-forwards the *same pre-layer* hidden states (upstream semantics).
        with torch.no_grad():
            hidden_states = [layer(x, position_embeddings=pos_emb)
                             for x in hidden_states]

        # -- per-layer working-dir artifact (upstream qtensors layout) -------
        emitted, _storage = build_layer_tensors(
            results, layer_idx=layer_idx, norm_weights=norm_weights)
        save_file(emitted, str(qt_dir / f"model.layers.{layer_idx}.safetensors"))

        metrics = {k: v.metrics for k, v in results.items()}
        entry = LayerConversionEntry(
            layer_index=layer_idx,
            module_keys=list(results.keys()),
            quantized_tensors=len(emitted),
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
        )
        report.layers.append(entry)
        say(f"[layer {layer_idx}] done: proxy_loss={entry.proxy_loss:.4e} "
            f"({entry.elapsed_seconds:.1f}s)")

        if is_xpu:
            torch.xpu.synchronize(device.index or 0)
            torch.xpu.empty_cache()

    say("writing merged manifest")
    _write_manifest(out_dir, report, config)
    report.total_elapsed_seconds = time.time() - t_start
    say(f"conversion complete: {len(report.layers)} layers "
        f"(stream RSS {rss_start} -> {read_rss_kib()} KiB)")
    return report


def _write_manifest(out_dir: Path, report: ConversionReport,
                    config: ConversionConfig) -> None:
    """Merged ``quantization_config.json`` + unchanged-component pass-through."""
    tensor_storage: Dict[str, dict] = {}
    for entry in report.layers:
        for module_key in entry.module_keys:
            tensor_storage[f"model.layers.{entry.layer_index}.{module_key}"] = {
                "quant_format": "exl3",
                "bits_per_weight": config.K,
                "codebook": config.cb,
            }
    unchanged = {k: {"quant_format": "fp16", "unchanged": True}
                 for k in report.unchanged_keys}
    manifest = {
        "quant_method": "exl3",
        "bits": config.K,
        "codebook": config.cb,
        "architecture": report.geometry.architecture,
        "num_layers": report.geometry.num_layers,
        "device": report.device,
        "calibration": {
            "num_rows": config.num_calibration_rows,
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "seed": config.calibration_seed,
            "damping_lambda": config.damping_lambda,
            "zero_floor": config.zero_floor,
            "skip_g_scale": config.skip_g_scale,
            "corpus": "synthetic-deterministic (real corpus wiring is P3-09)",
        },
        "tensor_storage": {**tensor_storage, **unchanged},
        "layers": [
            {
                "layer_index": e.layer_index,
                "modules": e.module_keys,
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
    with open(out_dir / "quantization_config.json", "w") as f:
        json.dump(manifest, f, indent=2)
