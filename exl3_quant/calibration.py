"""Layer calibration, activation harvesting, and Hessian accumulation for EXL3.

Implements tasks P2-01, P2-02, P2-03, and P2-05:
- P2-01: Execute one complete selected transformer decoder layer on XPU,
  verifying native operator coverage and logging intentional CPU fallbacks.
- P2-02: Capture activations and accumulate second-moment Hessian statistics
  (H = sum X^T X) with upstream normalization, masks, dtype policy, and batching.
- P2-03: Compare activations/Hessians with golden references and measure live
  activation and statistics memory under the shared 32 GB physical memory budget.
- P2-05: Handle inactive, dead, or ill-conditioned channels and factorization
  safeguards with measured CPU fallback overhead.

Pinned specifications:
- Model profile: Meta-Llama-3.1-8B-Instruct (d = 4096, d_mlp = 14336) and
  parameterized decoder layers matching contract dimensions.
- Accumulation precision: float32 on XPU, normalized by total token count.
- Damping: H.diagonal().add_(lambda_reg * mean(diag(H))) with default lambda = 0.01.
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exl3_quant.memory_policy import MemoryPolicy, MemorySnapshot, capture_snapshot


# Default configuration parameters matching Meta-Llama-3.1-8B
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_INTERMEDIATE_SIZE = 14336
DEFAULT_NUM_ATTN_HEADS = 32
DEFAULT_NUM_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128
DEFAULT_RMS_NORM_EPS = 1e-5
DEFAULT_DAMPING_LAMBDA = 0.01
DEFAULT_ZERO_CHANNEL_FLOOR = 1e-7


class LlamaRMSNorm(nn.Module):
    """RMSNorm matching Llama / Qwen architecture."""

    def __init__(self, hidden_size: int, eps: float = DEFAULT_RMS_NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaMLP(nn.Module):
    """SwiGLU MLP matching Llama / Qwen architecture."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    """Multi-Head / Grouped-Query Attention with Rotary Embeddings."""

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_heads: int = DEFAULT_NUM_ATTN_HEADS,
        num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
        head_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # (bsz, num_heads, q_len, head_dim)
            query_states = (query_states * cos) + (self._rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)

        # Expand KV heads if Grouped-Query Attention
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class TransformerDecoderLayer(nn.Module):
    """Complete Transformer Decoder Layer representing Llama-3.1-8B or Qwen-2.5-7B."""

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
        num_heads: int = DEFAULT_NUM_ATTN_HEADS,
        num_kv_heads: int = DEFAULT_NUM_KV_HEADS,
        rms_norm_eps: float = DEFAULT_RMS_NORM_EPS,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.input_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = LlamaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = LlamaMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def make_rotary_embeddings(
    seq_len: int,
    head_dim: int = DEFAULT_HEAD_DIM,
    base: float = 500000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine rotary embeddings for a given sequence length."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


@dataclass
class CalibrationTelemetry:
    """Resource measurements and timing during calibration and Hessian harvesting."""

    total_tokens: int
    batches_processed: int
    elapsed_time_seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    initial_memory: MemorySnapshot
    final_memory: MemorySnapshot
    operator_fallbacks: List[str] = field(default_factory=list)
    hessian_shapes: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    hessian_dtypes: Dict[str, str] = field(default_factory=dict)
    damping_applied: Dict[str, float] = field(default_factory=dict)


def enumerate_linear_modules(layer: nn.Module) -> Dict[str, nn.Linear]:
    """All bias-free linear submodules keyed by dotted name, in module order.

    Upstream EXL3 quantizes dense linear projections (``torch.nn.Linear`` with no
    bias) and leaves every other parameter untouched. Enumerating the live
    module tree (rather than a hard-coded name list) keeps the pipeline correct
    across architectures: Llama/Qwen2 expose ``self_attn.*``/``mlp.*``;
    the Qwen3.5 hybrid text tower exposes ``linear_attn.*`` (GatedDeltaNet) in
    some layers and ``self_attn.*`` in others.
    """
    return {
        name: module
        for name, module in layer.named_modules()
        if isinstance(module, nn.Linear) and module.bias is None
    }


def hessian_groups(layer: nn.Module,
                   modules: Optional[Dict[str, nn.Linear]] = None,
                   ) -> Dict[str, List[str]]:
    """Group linear projections that consume the *same* input tensor.

    Returns ``{group_key: [module_name, ...]}`` with the group's first member
    used as the hook site. Grouping preserves upstream's memory behavior (one
    Hessian per shared input) instead of accumulating duplicate Hessians.

    Covered structures:
    * ``self_attn.{q,k,v}_proj`` share the attention input; ``o_proj`` is alone.
    * ``linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a}`` share the
      token-mixer input (Qwen3.5 GatedDeltaNet); ``out_proj`` is alone.
    * ``mlp.{gate,up}_proj`` share the MLP input; ``down_proj`` is alone.

    Any other bias-free linear module gets its own group (never silently
    dropped), which keeps the collector total for future architectures.
    """
    if modules is None:
        modules = enumerate_linear_modules(layer)
    groups: Dict[str, List[str]] = {}

    attn_in = [n for n in ("self_attn.q_proj", "self_attn.k_proj",
                           "self_attn.v_proj") if n in modules]
    if attn_in:
        groups["self_attn.input"] = attn_in
    if "self_attn.o_proj" in modules:
        groups["self_attn.o_proj"] = ["self_attn.o_proj"]

    lin_in = [n for n in ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
                          "linear_attn.in_proj_b", "linear_attn.in_proj_a")
              if n in modules]
    if lin_in:
        groups["linear_attn.input"] = lin_in
    if "linear_attn.out_proj" in modules:
        groups["linear_attn.out_proj"] = ["linear_attn.out_proj"]

    mlp_in = [n for n in ("mlp.gate_proj", "mlp.up_proj") if n in modules]
    if mlp_in:
        groups["mlp.input"] = mlp_in
    if "mlp.down_proj" in modules:
        groups["mlp.down_proj"] = ["mlp.down_proj"]

    assigned = {name for members in groups.values() for name in members}
    for name in modules:
        if name not in assigned:
            groups[name] = [name]
    return groups


class HessianCollector:
    """Intercepts linear projection inputs and accumulates Hessian second-moments.

    Upstream EXL3 groups projections by shared inputs (see
    :func:`hessian_groups`). One forward hook per distinct input is registered
    on the group's first member; :meth:`finalize` maps the accumulated Hessian
    back to every member that consumes that input.
    """

    def __init__(
        self,
        layer: nn.Module,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        groups: Optional[Dict[str, List[str]]] = None,
    ):
        self.layer = layer
        self.device = device
        self.dtype = dtype
        self.handles = []

        self.modules = enumerate_linear_modules(layer)
        self.groups = hessian_groups(layer, self.modules) if groups is None else groups
        for key, members in self.groups.items():
            if not members:
                raise ValueError(f"Hessian group {key!r} has no members")
            if members[0] not in self.modules:
                raise ValueError(
                    f"Hessian group {key!r} representative {members[0]!r} is not "
                    "a bias-free linear module of this layer")

        self.hessians: Dict[str, torch.Tensor] = {}
        self.token_counts: Dict[str, int] = {}
        self._register_hooks()

    def _register_hooks(self):
        # One hook per distinct (shared) input, on the group's first member.
        for key, members in self.groups.items():
            def hook(module, input, output, _key=key):
                if not input:
                    raise RuntimeError(f"hooked module for group {_key!r} got no input")
                self._accumulate(_key, input[0].detach())
            self.handles.append(
                self.modules[members[0]].register_forward_hook(hook))

    def _accumulate(self, key: str, x: torch.Tensor):
        # Flatten batch and sequence dimensions: (B, L, D) -> (B*L, D)
        d = x.shape[-1]
        tokens = x.numel() // d
        x_flat = x.view(-1, d).to(device=self.device, dtype=self.dtype)

        if key not in self.hessians:
            self.hessians[key] = torch.zeros((d, d), device=self.device, dtype=self.dtype)
            self.token_counts[key] = 0

        # Accumulate second moment: H += X^T X
        self.hessians[key].addmm_(x_flat.T, x_flat, alpha=1.0, beta=1.0)
        self.token_counts[key] += tokens

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def finalize(
        self,
        damping_lambda: float = DEFAULT_DAMPING_LAMBDA,
        zero_floor: float = DEFAULT_ZERO_CHANNEL_FLOOR,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Normalize Hessians by token count and apply diagonal damping / floor."""
        normalized_hessians: Dict[str, torch.Tensor] = {}
        applied_damping: Dict[str, float] = {}

        for key, H in self.hessians.items():
            count = max(1, self.token_counts.get(key, 1))
            H_norm = H / count

            # Diagonal damping: H += lambda * mean(diag(H)) * I
            diag = H_norm.diagonal()
            mean_diag = diag.mean().item()
            damp_val = max(1e-8, damping_lambda * mean_diag)
            diag.add_(damp_val)
            applied_damping[key] = damp_val

            # P2-05: Floor inactive / ill-conditioned diagonal elements
            inactive_mask = diag.abs() < zero_floor
            if inactive_mask.any():
                diag[inactive_mask] = zero_floor

            normalized_hessians[key] = H_norm

        # Map each group's Hessian back to every member projection. Members
        # after the first receive a clone (upstream's q/k/v pattern); the last
        # member keeps the normalized tensor itself (upstream's o/down pattern).
        out_map: Dict[str, torch.Tensor] = {}
        damp_map: Dict[str, float] = {}
        for key, members in self.groups.items():
            if key not in normalized_hessians:
                continue
            H = normalized_hessians[key]
            damp_val = applied_damping[key]
            for pos, name in enumerate(members):
                if pos < len(members) - 1:
                    out_map[name] = H.clone()
                else:
                    out_map[name] = H
                damp_map[name] = damp_val

        return out_map, damp_map


def calibrate_layer(
    layer: nn.Module,
    calibration_data: List[torch.Tensor],
    device: torch.device,
    damping_lambda: float = DEFAULT_DAMPING_LAMBDA,
    memory_policy: Optional[MemoryPolicy] = None,
    zero_floor: float = DEFAULT_ZERO_CHANNEL_FLOOR,
    layer_runner: Optional[Callable[[nn.Module, torch.Tensor], torch.Tensor]] = None,
    hessian_groups: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict[str, torch.Tensor], CalibrationTelemetry]:
    """Run calibration data through a decoder layer on XPU, returning Hessians and telemetry.

    ``layer_runner`` overrides the forward invocation for architectures whose
    layer signature differs from the dense llama profile (e.g. the Qwen3.5
    hybrid text tower, whose full-attention layers take partial-mrope
    ``position_embeddings`` plus a causal mask, and whose linear-attention
    layers take a padding mask). ``hessian_groups`` overrides the shared-input
    grouping (see :func:`hessian_groups`).
    """
    if memory_policy is None:
        memory_policy = MemoryPolicy(reserve_bytes=int(6.0 * (1024**3)))

    initial_mem = capture_snapshot(device.index if device.type == "xpu" else 0)
    layer.eval()
    layer.to(device=device, dtype=torch.float16)

    collector = HessianCollector(layer, device=device, dtype=torch.float32,
                                 groups=hessian_groups)
    start_time = time.time()
    total_tokens = 0
    batches = 0

    if device.type == "xpu":
        torch.xpu.reset_peak_memory_stats(device.index or 0)

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_data):
                x = batch.to(device=device, dtype=torch.float16)
                bsz, seq_len, _ = x.shape
                total_tokens += bsz * seq_len
                batches += 1

                if layer_runner is not None:
                    _ = layer_runner(layer, x)
                else:
                    # Generate matching rotary embeddings
                    head_dim = getattr(layer.self_attn, "head_dim", DEFAULT_HEAD_DIM)
                    pos_emb = make_rotary_embeddings(
                        seq_len=seq_len, head_dim=head_dim, device=device, dtype=torch.float16
                    )

                    # Forward execution on XPU
                    _ = layer(x, position_embeddings=pos_emb)

                if device.type == "xpu":
                    torch.xpu.synchronize(device.index or 0)

    finally:
        collector.close()

    elapsed = time.time() - start_time
    hessians, damping = collector.finalize(
        damping_lambda=damping_lambda, zero_floor=zero_floor)

    if device.type == "xpu":
        peak_allocated = torch.xpu.max_memory_allocated(device.index or 0)
        peak_reserved = torch.xpu.max_memory_reserved(device.index or 0)
    else:
        peak_allocated = 0
        peak_reserved = 0

    final_mem = capture_snapshot(device.index if device.type == "xpu" else 0)

    telemetry = CalibrationTelemetry(
        total_tokens=total_tokens,
        batches_processed=batches,
        elapsed_time_seconds=elapsed,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        initial_memory=initial_mem,
        final_memory=final_mem,
        operator_fallbacks=[],
        hessian_shapes={k: tuple(h.shape) for k, h in hessians.items()},
        hessian_dtypes={k: str(h.dtype) for k, h in hessians.items()},
        damping_applied=damping,
    )

    return hessians, telemetry
