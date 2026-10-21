"""Sylvester Hadamard transforms for EXL3 (P2-06, contract §8).

Transcribed from the pinned upstream `exllamav3/util/hadamard.py:34-42, 107-137`
and cross-checked against the independent decode oracle
`exl3xpu/ref.py:118-123` (`hadamard128` + `had_rows`).

Normative properties (contract §8):

* ``H_128`` is the **Sylvester** construction, built by the natural recursion
  ``[[h, h], [h, -h]]`` (no sequency/bit-reversal permutation).
* It is applied **orthonormally** — scaled by ``1/sqrt(128)`` — so
  ``H Hᵀ = I`` and ``H`` is symmetric (``H = Hᵀ``), hence its own inverse.
* ``preapply_had_l`` acts on the **input/row** dimension (``H @ x``);
  ``preapply_had_r`` acts on the **output/column** dimension (``x @ H``).
* Encoding applies the transform **before** quantization; decoding applies the
  same (self-inverse) transform to the reconstruction, then the ``su``/``sv``
  scales. See :func:`weight_from_quantized`.

The transform is blockwise over 128-element groups, matching upstream
``blockwise_preapply_had_l_`` / ``blockwise_preapply_had_r_``
(`quantize.py:365-391`). It is **not** a full-matrix Kronecker transform.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

HAD_DIM = 128
_HAD_CACHE: dict = {}


def sylvester_hadamard(n: int) -> torch.Tensor:
    """Unnormalized Sylvester matrix of order ``n`` (``±1``, ``float64``).

    ``n`` must be a power of two (the only case used by EXL3, where
    ``n = 128``). Raises ``ValueError`` for any other size rather than
    silently falling back to a Paley construction (out of scope, contract §8).
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(
            f"Sylvester construction requires a power-of-two size, got {n}"
        )
    h = torch.ones((1, 1), dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat(
            [torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0
        )
    return h


def hadamard(n: int = HAD_DIM, dtype: torch.dtype = torch.float32,
             scale: Optional[float] = None,
             device: Optional[torch.device] = None) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix (``scale`` defaults to ``1/sqrt(n)``)."""
    if scale is None:
        scale = 1.0 / math.sqrt(n)
    dev_str = str(device) if device is not None else "cpu"
    key = (n, dtype, float(scale), dev_str)
    cached = _HAD_CACHE.get(key)
    if cached is not None:
        return cached
    matrix = (sylvester_hadamard(n) * scale).to(dtype=dtype, device=device)
    _HAD_CACHE[key] = matrix
    return matrix


def preapply_had_l(x: torch.Tensor, had_dim: int = HAD_DIM) -> torch.Tensor:
    """``H @ x`` on 128-row blocks (upstream ``preapply_had_l``, `.py:345-352`)."""
    k, n = x.shape
    if k % had_dim != 0:
        raise ValueError(f"rows ({k}) must be divisible by had_dim ({had_dim})")
    original_dtype = x.dtype
    work = x.to(torch.float32).contiguous()
    had = hadamard(had_dim, torch.float32, device=x.device)
    out = (had @ work.view(-1, had_dim, n)).view(k, n)
    return out.to(original_dtype)


def preapply_had_r(x: torch.Tensor, had_dim: int = HAD_DIM) -> torch.Tensor:
    """``x @ H`` on 128-column blocks (upstream ``preapply_had_r``, `.py:355-362`)."""
    k, n = x.shape
    if n % had_dim != 0:
        raise ValueError(f"columns ({n}) must be divisible by had_dim ({had_dim})")
    original_dtype = x.dtype
    work = x.to(torch.float32).contiguous()
    had = hadamard(had_dim, torch.float32, device=x.device)
    out = (work.view(k, -1, had_dim) @ had).view(k, n)
    return out.to(original_dtype)


def blockwise_preapply_had_l_(x: torch.Tensor,
                              had_dim: int = HAD_DIM) -> torch.Tensor:
    """In-place ``preapply_had_l``; requires contiguous float32 (upstream asserts this)."""
    if x.dtype != torch.float32:
        raise ValueError(f"in-place Hadamard requires float32, got {x.dtype}")
    k, n = x.shape
    if k % had_dim != 0:
        raise ValueError(f"rows ({k}) must be divisible by had_dim ({had_dim})")
    had = hadamard(had_dim, torch.float32, device=x.device)
    for start in range(0, k, had_dim):
        x[start:start + had_dim, :] = had @ x[start:start + had_dim, :]
    return x


def blockwise_preapply_had_r_(x: torch.Tensor,
                              had_dim: int = HAD_DIM) -> torch.Tensor:
    """In-place ``preapply_had_r``; requires contiguous float32 (upstream asserts this)."""
    if x.dtype != torch.float32:
        raise ValueError(f"in-place Hadamard requires float32, got {x.dtype}")
    k, n = x.shape
    if n % had_dim != 0:
        raise ValueError(f"columns ({n}) must be divisible by had_dim ({had_dim})")
    had = hadamard(had_dim, torch.float32, device=x.device)
    for start in range(0, n, had_dim):
        x[:, start:start + had_dim] = x[:, start:start + had_dim] @ had
    return x


def weight_from_quantized(w_hat: torch.Tensor, su: torch.Tensor,
                          sv: torch.Tensor) -> torch.Tensor:
    """Reconstruct the original-basis weight (decode order of `exl3xpu.ref.weight_orig`).

    ``w_hat`` is the quantized/reconstructed weight in the transformed basis,
    ``su`` has shape ``(k, 1)`` and ``sv`` ``(1, n)``. The sequence is:

    ``H @ w`` (rows) → ``* su`` → ``w @ H`` (cols) → ``* sv``
    """
    k, n = w_hat.shape
    if su.numel() != k or sv.numel() != n:
        raise ValueError(
            f"su must have {k} elements and sv {n}, got {su.numel()} / {sv.numel()}"
        )
    work = preapply_had_l(w_hat.to(torch.float32), HAD_DIM)
    work = work * su.to(torch.float32).reshape(k, 1)
    work = preapply_had_r(work, HAD_DIM)
    work = work * sv.to(torch.float32).reshape(1, n)
    return work
