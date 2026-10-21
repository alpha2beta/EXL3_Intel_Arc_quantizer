"""Upstream EXL3 weight regularization (P2-04, contract §9.3).

Transcribed from the pinned upstream ``exllamav3/modules/quant/exl3_lib/quantize.py``:

* ``block_rms`` — `.py:1079-1092`
* ``regularize`` — `.py:1130-1236`
* the output-scale skew heuristic — `.py:1164-1178`
* the zero-channel handling — `.py:1184-1207`

The routine is applied **once per linear layer** before Hessian factorization and
encoding. It (a) optionally normalizes output channels, (b) folds the ``su``/``sv``
sign vectors and the per-channel RMS scales into the matrix, (c) applies the
blockwise Hadamard transforms, and (d) applies an optional global scale found by
sampling quantization error.

Order is normative and must not be rearranged: the encode path is
``÷sv → H_r → ÷su → H_l`` and decoding reverses it with
:func:`exl3_quant.hadamard.weight_from_quantized`
(``H_l → ×su → H_r → ×sv``).

The upstream global-scale search (``g_scale_gss``, `.py:1050-1076`) is implemented
in :mod:`exl3_quant.gss`. This module accepts it via the pluggable ``g_scale_fn``
hook (defaulting to ``skip_g_scale=True``, upstream's own escape hatch with ``g_scale = 1``).
When ``skip_g_scale=False``, callers supply ``g_scale_fn_from_encoder(...)`` to run
the full upstream search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from exl3_quant.hadamard import (
    HAD_DIM,
    blockwise_preapply_had_l_,
    blockwise_preapply_had_r_,
)

# quantize.py:16-19
CODEBOOK_SCALE = 1.24371088
OUT_SCALE_SKEW_CUTOFF = 0.15   # .py:1173
SKEW_FRACTION = 50             # .py:1167  (top 2% of diagonal)
ZERO_FLOOR = 0.1               # .py:1196, .py:1214
EPS = 1e-10                    # .py:1197, .py:1215


def block_rms(x: torch.Tensor, dim: int, keepdim: bool = False,
              blocksize: int = 32) -> torch.Tensor:
    """``sqrt(x.square().mean(dim))`` accumulated in 32-slices (`.py:1079-1092`).

    The blockwise accumulation order is preserved verbatim: it rounds
    differently from a single ``x.square().mean(dim)`` and upstream uses this
    exact order.
    """
    n = x.size(dim)
    sq = None
    for block in torch.split(x, blocksize, dim=dim):
        block_sq = block.square().sum(dim=dim, keepdim=keepdim)
        if sq is None:
            sq = block_sq
        else:
            sq += block_sq
    return (sq / n).sqrt()


@dataclass
class RegularizeResult:
    """Outputs of :func:`regularize`."""

    apply_out_scales: bool
    weight: torch.Tensor            # transformed weight (float32)
    g_scale: float                  # global scale applied to the weight
    su: torch.Tensor                # (k, 1) input sign/scale vector
    sv: torch.Tensor                # (1, n) output sign/scale vector
    out_channel_scales: torch.Tensor
    in_channel_scales: torch.Tensor
    zeros: bool                     # upstream quant_args["zeros"]
    skew_factor: Optional[float]    # None when H_diag was not supplied


def regularize(
    weight: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    quant_args: Optional[dict] = None,
    H_diag: Optional[torch.Tensor] = None,
    skip_g_scale: bool = True,
    q_fallback: bool = False,
    g_scale_fn: Optional[Callable[[torch.Tensor], float]] = None,
    had_dim: int = HAD_DIM,
) -> RegularizeResult:
    """Transform a layer weight into the distribution expected by EXL3.

    Args:
        weight: ``(k, n)`` float32 layer weight (not modified in place).
        su: ``(k, 1)`` input sign vector (upstream takes it from the captured
            Hessian); entries are typically ``±1``.
        sv: ``(1, n)`` output sign vector (upstream initializes random ``±1``).
        quant_args: upstream-style dict; ``apply_out_scales`` may be
            ``True``/``False``/``None`` (auto). ``zeros`` is written back.
        H_diag: diagonal of the Hessian, used only for the skew heuristic.
        skip_g_scale: when ``True`` (default here) ``g_scale = 1`` exactly as
            upstream's own ``skip_g_scale`` switch. Set ``False`` and provide
            ``g_scale_fn`` to run a real search.
        q_fallback: mirrors upstream's fallback mode (forces the caller's
            ``apply_out_scales`` value).
        g_scale_fn: callable ``weight -> float`` implementing the global-scale
            search (upstream ``g_scale_gss``). Required when
            ``skip_g_scale=False``.
        had_dim: Hadamard block size (128 for EXL3).

    Returns:
        :class:`RegularizeResult`.
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D, got shape {tuple(weight.shape)}")
    if weight.dtype != torch.float32:
        raise ValueError(f"weight must be float32, got {weight.dtype}")
    k, n = weight.shape
    if su.numel() != k or sv.numel() != n:
        raise ValueError(
            f"su must have {k} elements and sv {n}, got {su.numel()} / {sv.numel()}"
        )
    if k % had_dim != 0 or n % had_dim != 0:
        raise ValueError(
            f"both dimensions must be divisible by {had_dim}, got {(k, n)}"
        )

    args = {} if quant_args is None else quant_args
    force_out_scales = args.get("apply_out_scales")

    # ---- output-scale decision (upstream .py:1164-1181) -------------------
    skew_factor: Optional[float] = None
    if not q_fallback and H_diag is not None:
        diag = H_diag.sqrt()
        diag, _ = torch.sort(diag, descending=True)
        cutoff = diag.shape[0] // SKEW_FRACTION
        skew_factor = float(diag[:cutoff].sum() / diag.sum())
        if force_out_scales is None:
            apply_out_scales = skew_factor < OUT_SCALE_SKEW_CUTOFF
        else:
            apply_out_scales = bool(force_out_scales)
    else:
        apply_out_scales = True if force_out_scales is None else bool(force_out_scales)

    if q_fallback:
        apply_out_scales = bool(force_out_scales)

    # ---- output channel scales (upstream .py:1184-1196) -------------------
    work = weight.clone()
    out_channel_scales = block_rms(work, dim=0, keepdim=True)
    mean = out_channel_scales.mean().item()
    if mean > 1e-30:
        out_channel_scales /= mean
        zeros = False
    else:
        zeros = True
        if force_out_scales is not None:
            apply_out_scales = True
    zero_out_scales = out_channel_scales.abs() < 1e-30

    su = su.to(torch.float32).clone()
    sv = sv.to(torch.float32).clone()

    if apply_out_scales:
        out_channel_scales[zero_out_scales] = ZERO_FLOOR
        sv = (sv * out_channel_scales + EPS).float()

    # ---- fold sv, then right Hadamard (upstream .py:1204-1210) -----------
    work /= sv
    sv[zero_out_scales] = 0.0
    blockwise_preapply_had_r_(work, had_dim)

    # ---- input channel scales + su + left Hadamard (.py:1213-1217) -------
    in_channel_scales = block_rms(work, dim=1, keepdim=True)
    in_channel_scales[in_channel_scales.abs() < 1e-30] = ZERO_FLOOR
    su = (su * in_channel_scales / (-CODEBOOK_SCALE) + EPS).float()
    work /= su
    blockwise_preapply_had_l_(work, had_dim)

    # ---- global scale (.py:1219-1225) ------------------------------------
    if skip_g_scale:
        g_scale = 1.0
    else:
        if g_scale_fn is None:
            raise ValueError(
                "skip_g_scale=False requires g_scale_fn (upstream g_scale_gss "
                "test-quantizes sample tiles and needs the encoder)"
            )
        g_scale = float(g_scale_fn(work))
        if not (g_scale > 0.0):
            raise ValueError(f"g_scale_fn returned a non-positive scale: {g_scale}")
    work *= g_scale
    su /= g_scale

    if "zeros" in args or quant_args is not None:
        args["zeros"] = zeros

    return RegularizeResult(
        apply_out_scales=bool(apply_out_scales),
        weight=work,
        g_scale=g_scale,
        su=su,
        sv=sv,
        out_channel_scales=out_channel_scales,
        in_channel_scales=in_channel_scales,
        zeros=zeros,
        skew_factor=skew_factor,
    )
