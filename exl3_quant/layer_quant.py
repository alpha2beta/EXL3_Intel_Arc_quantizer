"""End-to-end single-layer quantization pipeline and distortion metrics.

Implements tasks P2-11 and P2-12:
- P2-11: Integrate scratch reuse with layer/module lifetimes and library
  workspaces; verify stream synchronization before reuse or eviction.
- P2-12: Measure factorization residuals, reconstruction error, tile distortion,
  and layerwise proxy loss using the pinned objective definition:
  Proxy Loss = tr(E^T H E) where E = W - W_hat.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from exl3_quant.gss import g_scale_fn_from_encoder
from exl3_quant.hadamard import (
    HAD_DIM,
    blockwise_preapply_had_l_,
    blockwise_preapply_had_r_,
    weight_from_quantized,
)
from exl3_quant.ldlq import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_BUF_SIZE_K,
    BlockLDLResult,
    LDLQResult,
    TileEncoder,
    block_ldl,
    ldlq,
    make_reference_tile_encoder,
    make_xpu_tile_encoder,
)
from exl3_quant.memory_policy import MemoryPolicy, capture_snapshot
from exl3_quant.regularize import (
    RegularizeResult,
    regularize,
)
from exl3_quant.xpu_quantizer import CB_MUL1, ScratchArena, XPUTrellisEncoder


@dataclass
class ModuleMetrics:
    """Numerical distortion and proxy loss metrics for a quantized linear module (P2-12)."""

    mse: float
    relative_frobenius_error: float
    tile_distortion: float
    proxy_loss: float
    normalized_proxy_loss: float
    factorization_residual: float
    retries: int
    g_scale: float
    elapsed_seconds: float


@dataclass
class QuantizedModuleResult:
    """Outputs and artifacts of a quantized linear projection."""

    name: str
    K: int
    cb: int
    original_shape: Tuple[int, int]    # (in_features, out_features)
    weight_orig: torch.Tensor          # original W in (k, n)
    w_hat: torch.Tensor                # reconstructed W_hat in (k, n)
    encoded: torch.Tensor              # trellis states (tiles_k, tiles_n, 256) int16
    su: torch.Tensor                   # (k, 1) scale/sign vector
    sv: torch.Tensor                   # (1, n) scale/sign vector
    metrics: ModuleMetrics
    regularize_result: RegularizeResult
    ldl_result: BlockLDLResult


@dataclass
class LayerQuantizationReport:
    """Complete report for single-layer conversion."""

    K: int
    codebook: str
    total_elapsed_seconds: float
    module_results: Dict[str, QuantizedModuleResult]
    total_proxy_loss: float
    mean_mse: float
    peak_device_memory_bytes: int
    scratch_arena_reused: bool
    device_index: int


def compute_factorization_residual(H: torch.Tensor, chol: torch.Tensor) -> float:
    """Compute relative Frobenius error ||H - L L^T||_F / ||H||_F."""
    reconstructed = chol @ chol.T
    diff = H - reconstructed
    frob_diff = torch.linalg.norm(diff).item()
    frob_H = max(1e-12, torch.linalg.norm(H).item())
    return float(frob_diff / frob_H)


def compute_proxy_loss(
    W: torch.Tensor,
    W_hat: torch.Tensor,
    H: torch.Tensor,
) -> Tuple[float, float]:
    """Compute layerwise Hessian-weighted proxy loss tr(E^T H E) (contract §4 / §9.3).

    Uses (H @ E * E).sum() to avoid O(N^2) intermediates:
    tr(E^T H E) = sum_{j} (E^T H E)_{jj} = sum_{i,j} (H @ E)_{ij} * E_{ij}.
    """
    E = (W - W_hat).to(dtype=torch.float32, device=H.device)
    H_f32 = H.to(dtype=torch.float32)

    # (H @ E) is (k, n); elementwise multiply by E and sum
    HE = H_f32 @ E
    proxy_loss = float((HE * E).sum().item())

    # Normalization by original energy tr(W^T H W)
    W_f32 = W.to(dtype=torch.float32, device=H.device)
    HW = H_f32 @ W_f32
    norm_energy = max(1e-12, float((HW * W_f32).sum().item()))
    normalized_loss = float(proxy_loss / norm_energy)

    return proxy_loss, normalized_loss


def quantize_linear_module(
    name: str,
    weight: torch.Tensor,
    H: torch.Tensor,
    K: int = 2,
    cb: int = CB_MUL1,
    tile_encoder: Optional[TileEncoder] = None,
    skip_g_scale: bool = True,
    buf_size_k: int = DEFAULT_BUF_SIZE_K,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> QuantizedModuleResult:
    """Quantize one linear weight matrix using LDLQ error-feedback loop.

    Args:
        name: Name of the projection (e.g. "self_attn.q_proj").
        weight: (out_features, in_features) or (in_features, out_features) float32 tensor.
        H: (in_features, in_features) float32 Hessian matrix.
        K: Bits per weight (2, 3, or 4).
        cb: Codebook selector (CB_MUL1 default).
        tile_encoder: Callback for encoding tile batches.
        skip_g_scale: Whether to bypass GSS global scale search.
        buf_size_k: Span size for LDLQ traversal.
        block_size: Block size for block-LDL inversion.

    Returns:
        :class:`QuantizedModuleResult` with reconstructed weight, trellis, and metrics.
    """
    t0 = time.time()
    # Normalize weight orientation to (k=in_features, n=out_features)
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")

    k_dim = H.shape[0]
    if weight.shape[1] == k_dim:
        # Standard nn.Linear weight of shape (out_features, in_features) -> transpose to (k, n)
        W = weight.T.clone().to(torch.float32)
    elif weight.shape[0] == k_dim:
        # Already in (k, n) orientation
        W = weight.clone().to(torch.float32)
    else:
        raise ValueError(
            f"Weight shape {tuple(weight.shape)} does not match Hessian dim {k_dim}"
        )

    k, n = W.shape
    device = W.device

    # Initialize sign vectors (su has shape (k, 1), sv has shape (1, n))
    su = torch.ones((k, 1), dtype=torch.float32, device=device)
    sv = torch.ones((1, n), dtype=torch.float32, device=device)

    # Setup default tile encoder if none provided
    if tile_encoder is None:
        if device.type == "xpu":
            tile_encoder = make_xpu_tile_encoder(K=K, device_index=device.index or 0)
        else:
            tile_encoder = make_reference_tile_encoder(K=K)

    # Optional GSS function
    g_scale_fn = None
    if not skip_g_scale:
        g_scale_fn = g_scale_fn_from_encoder(tile_encoder)

    # 1. Weight regularization (P2-04)
    reg = regularize(
        weight=W,
        su=su,
        sv=sv,
        H_diag=H.diagonal(),
        skip_g_scale=skip_g_scale,
        g_scale_fn=g_scale_fn,
        had_dim=HAD_DIM,
    )

    # 2. Blockwise Hadamard transforms on Hessian (contract §8 / §9.1)
    H_work = H.clone().to(torch.float32)
    blockwise_preapply_had_l_(H_work, HAD_DIM)
    blockwise_preapply_had_r_(H_work, HAD_DIM)

    # 3. Block LDL Factorization (P0-22)
    ldl_res = block_ldl(H_work, block_size=block_size)
    fact_residual = compute_factorization_residual(H_work, ldl_res.cholesky)

    # 4. LDLQ error-feedback loop (P2-08)
    ldlq_res = ldlq(
        weight=reg.weight,
        L=ldl_res.transformed,
        encode_tiles=tile_encoder,
        buf_size_k=buf_size_k,
    )

    # 5. Original-basis weight reconstruction (contract §10)
    w_hat = weight_from_quantized(ldlq_res.weight_q, reg.su, reg.sv)

    # 6. Objective metrics computation (P2-12)
    diff = W - w_hat
    mse = float(diff.pow(2).mean().item())
    rel_frob = float(torch.linalg.norm(diff).item() / max(1e-12, torch.linalg.norm(W).item()))
    tile_dist = float((reg.weight - ldlq_res.weight_q).pow(2).mean().item())
    proxy_loss, norm_proxy_loss = compute_proxy_loss(W, w_hat, H)

    elapsed = time.time() - t0
    metrics = ModuleMetrics(
        mse=mse,
        relative_frobenius_error=rel_frob,
        tile_distortion=tile_dist,
        proxy_loss=proxy_loss,
        normalized_proxy_loss=norm_proxy_loss,
        factorization_residual=fact_residual,
        retries=ldl_res.retries,
        g_scale=reg.g_scale,
        elapsed_seconds=elapsed,
    )

    return QuantizedModuleResult(
        name=name,
        K=K,
        cb=cb,
        original_shape=(k, n),
        weight_orig=W,
        w_hat=w_hat,
        encoded=ldlq_res.encoded,
        su=reg.su,
        sv=reg.sv,
        metrics=metrics,
        regularize_result=reg,
        ldl_result=ldl_res,
    )


def quantize_decoder_layer(
    layer: nn.Module,
    hessians: Dict[str, torch.Tensor],
    K: int = 2,
    cb: int = CB_MUL1,
    device: Optional[torch.device] = None,
    skip_g_scale: bool = True,
    buf_size_k: int = DEFAULT_BUF_SIZE_K,
    block_size: int = DEFAULT_BLOCK_SIZE,
    memory_policy: Optional[MemoryPolicy] = None,
) -> LayerQuantizationReport:
    """Quantize an entire transformer decoder layer with bounded scratch reuse (P2-11).

    Reuses a single `ScratchArena` across all linear projections in the layer,
    synchronizing on the current stream between projections.
    """
    t_start = time.time()
    if device is None:
        device = torch.device("xpu:0" if torch.xpu.is_available() else "cpu")

    if memory_policy is None:
        memory_policy = MemoryPolicy()

    # Create one shared ScratchArena and encoder for the entire layer (P2-11)
    arena = None
    if device.type == "xpu":
        arena = ScratchArena(
            device_index=device.index or 0,
            policy=memory_policy,
        )
        tile_encoder = make_xpu_tile_encoder(
            K=K,
            device_index=device.index or 0,
            arena=arena,
        )
    else:
        tile_encoder = make_reference_tile_encoder(K=K)

    # Map of linear projections to quantize
    module_weights: Dict[str, torch.Tensor] = {}
    if hasattr(layer, "self_attn"):
        attn = layer.self_attn
        if hasattr(attn, "q_proj"):
            module_weights["self_attn.q_proj"] = attn.q_proj.weight.data
        if hasattr(attn, "k_proj"):
            module_weights["self_attn.k_proj"] = attn.k_proj.weight.data
        if hasattr(attn, "v_proj"):
            module_weights["self_attn.v_proj"] = attn.v_proj.weight.data
        if hasattr(attn, "o_proj"):
            module_weights["self_attn.o_proj"] = attn.o_proj.weight.data

    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        if hasattr(mlp, "gate_proj"):
            module_weights["mlp.gate_proj"] = mlp.gate_proj.weight.data
        if hasattr(mlp, "up_proj"):
            module_weights["mlp.up_proj"] = mlp.up_proj.weight.data
        if hasattr(mlp, "down_proj"):
            module_weights["mlp.down_proj"] = mlp.down_proj.weight.data

    results: Dict[str, QuantizedModuleResult] = {}
    try:
        for name, W in module_weights.items():
            if name not in hessians:
                raise KeyError(f"Hessian missing for module {name}")

            H = hessians[name]
            res = quantize_linear_module(
                name=name,
                weight=W,
                H=H,
                K=K,
                cb=cb,
                tile_encoder=tile_encoder,
                skip_g_scale=skip_g_scale,
                buf_size_k=buf_size_k,
                block_size=block_size,
            )
            results[name] = res

            # Stream synchronization before reusing the arena on the next module (P2-11)
            if device.type == "xpu":
                torch.xpu.synchronize(device.index or 0)

    finally:
        if arena is not None:
            arena.release()

    total_elapsed = time.time() - t_start
    total_proxy_loss = sum(r.metrics.proxy_loss for r in results.values())
    mean_mse = float(np.mean([r.metrics.mse for r in results.values()]))

    peak_mem = 0
    if device.type == "xpu":
        peak_mem = torch.xpu.max_memory_allocated(device.index or 0)

    return LayerQuantizationReport(
        K=K,
        codebook="mul1" if cb == CB_MUL1 else str(cb),
        total_elapsed_seconds=total_elapsed,
        module_results=results,
        total_proxy_loss=total_proxy_loss,
        mean_mse=mean_mse,
        peak_device_memory_bytes=peak_mem,
        scratch_arena_reused=True,
        device_index=device.index or 0 if device.type == "xpu" else -1,
    )
