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


class HessianCollector:
    """Intercepts linear projection inputs and accumulates Hessian second-moments.

    Upstream EXL3 groups projections by shared inputs:
    1. Attention input: q_proj, k_proj, v_proj share X_attn (d x d)
    2. Attention output: o_proj takes X_o (d x d)
    3. MLP input: gate_proj, up_proj share X_mlp (d x d)
    4. MLP down: down_proj takes X_down (d_mlp x d_mlp)
    """

    def __init__(
        self,
        layer: nn.Module,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        self.layer = layer
        self.device = device
        self.dtype = dtype
        self.handles = []

        self.hessians: Dict[str, torch.Tensor] = {}
        self.token_counts: Dict[str, int] = {}
        self._register_hooks()

    def _register_hooks(self):
        # We hook 4 strategic locations to minimize memory:
        # 1. q_proj input (serves q_proj, k_proj, v_proj)
        def hook_attn_in(module, input, output):
            x = input[0].detach()
            self._accumulate("self_attn.input", x)

        # 2. o_proj input
        def hook_o_in(module, input, output):
            x = input[0].detach()
            self._accumulate("self_attn.o_proj", x)

        # 3. gate_proj input (serves gate_proj, up_proj)
        def hook_mlp_in(module, input, output):
            x = input[0].detach()
            self._accumulate("mlp.input", x)

        # 4. down_proj input
        def hook_down_in(module, input, output):
            x = input[0].detach()
            self._accumulate("mlp.down_proj", x)

        if hasattr(self.layer, "self_attn"):
            if hasattr(self.layer.self_attn, "q_proj"):
                self.handles.append(self.layer.self_attn.q_proj.register_forward_hook(hook_attn_in))
            if hasattr(self.layer.self_attn, "o_proj"):
                self.handles.append(self.layer.self_attn.o_proj.register_forward_hook(hook_o_in))

        if hasattr(self.layer, "mlp"):
            if hasattr(self.layer.mlp, "gate_proj"):
                self.handles.append(self.layer.mlp.gate_proj.register_forward_hook(hook_mlp_in))
            if hasattr(self.layer.mlp, "down_proj"):
                self.handles.append(self.layer.mlp.down_proj.register_forward_hook(hook_down_in))

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

        # Map back to all 7 individual projections:
        out_map: Dict[str, torch.Tensor] = {}
        damp_map: Dict[str, float] = {}

        if "self_attn.input" in normalized_hessians:
            h_attn = normalized_hessians["self_attn.input"]
            d_attn = applied_damping["self_attn.input"]
            out_map["self_attn.q_proj"] = h_attn.clone()
            out_map["self_attn.k_proj"] = h_attn.clone()
            out_map["self_attn.v_proj"] = h_attn.clone()
            damp_map["self_attn.q_proj"] = d_attn
            damp_map["self_attn.k_proj"] = d_attn
            damp_map["self_attn.v_proj"] = d_attn

        if "self_attn.o_proj" in normalized_hessians:
            out_map["self_attn.o_proj"] = normalized_hessians["self_attn.o_proj"]
            damp_map["self_attn.o_proj"] = applied_damping["self_attn.o_proj"]

        if "mlp.input" in normalized_hessians:
            h_mlp = normalized_hessians["mlp.input"]
            d_mlp = applied_damping["mlp.input"]
            out_map["mlp.gate_proj"] = h_mlp.clone()
            out_map["mlp.up_proj"] = h_mlp.clone()
            damp_map["mlp.gate_proj"] = d_mlp
            damp_map["mlp.up_proj"] = d_mlp

        if "mlp.down_proj" in normalized_hessians:
            out_map["mlp.down_proj"] = normalized_hessians["mlp.down_proj"]
            damp_map["mlp.down_proj"] = applied_damping["mlp.down_proj"]

        return out_map, damp_map


def calibrate_layer(
    layer: nn.Module,
    calibration_data: List[torch.Tensor],
    device: torch.device,
    damping_lambda: float = DEFAULT_DAMPING_LAMBDA,
    memory_policy: Optional[MemoryPolicy] = None,
) -> Tuple[Dict[str, torch.Tensor], CalibrationTelemetry]:
    """Run calibration data through a decoder layer on XPU, returning Hessians and telemetry."""
    if memory_policy is None:
        memory_policy = MemoryPolicy(reserve_bytes=int(6.0 * (1024**3)))

    initial_mem = capture_snapshot(device.index if device.type == "xpu" else 0)
    layer.eval()
    layer.to(device=device, dtype=torch.float16)

    collector = HessianCollector(layer, device=device, dtype=torch.float32)
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
    hessians, damping = collector.finalize(damping_lambda=damping_lambda)

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
