"""One-sided LDLQ preparation, transcribed from upstream EXL3 (P0-22).

Implements the audited contract in ``docs/algorithm_contract.md`` §9.1
(upstream ``quantize.py:416-490``):

1. ``L = cholesky(H)`` — on failure, damp the diagonal and retry, up to 10
   times; the first attempt uses the undamped matrix.
2. Take the diagonal ``block_size`` blocks of ``L`` and invert them.
3. Fold each inverse into its block-column of ``L``.
4. Replace the diagonal blocks with identity matrices.

The result is *not* an ``L D Lᵀ`` factorization: it is the Cholesky factor with
block-diagonal inverses folded in, as the upstream ``ldlq`` error-feedback loop
consumes it. The retry schedule is part of the algorithm; adding damping
pre-emptively would change it.

Device policy (P0-21/P0-22, measured on Arc 140V): ``torch.linalg.cholesky``
falls back to CPU for fp32 and is unimplemented for fp16/bf16, so this module
accepts fp32 only and runs on any torch device. ``torch.linalg.inv`` and
``matmul`` remain native on XPU, so only the factorization itself crosses the
bus.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEFAULT_BLOCK_SIZE = 16
DEFAULT_SIGMA_REG = 0.025
DEFAULT_MAX_RETRIES = 10


@dataclass(frozen=True)
class BlockLDLResult:
    """Transformed factor plus the intermediates the tests inspect."""

    transformed: torch.Tensor      # L with folded block inverses, identity blocks
    cholesky: torch.Tensor         # plain Cholesky factor of the (possibly damped) H
    block_inverses: torch.Tensor   # (m, b, b) inverses of the diagonal blocks
    retries: int                   # 0 = first attempt succeeded
    damping_applied: bool
    n: int
    block_size: int


def _validate(H: torch.Tensor, block_size: int) -> None:
    if H.dim() != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be a square 2-D tensor, got shape {tuple(H.shape)}")
    if H.dtype != torch.float32:
        raise ValueError(
            f"block_ldl requires float32 (the audited contract uses fp32); "
            f"got {H.dtype}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_size > H.shape[0]:
        raise ValueError(
            f"block_size {block_size} exceeds matrix size {H.shape[0]}")
    if H.shape[0] % block_size != 0:
        raise ValueError(
            f"matrix size {H.shape[0]} is not divisible by block_size {block_size}")


def block_ldl(H: torch.Tensor,
              block_size: int = DEFAULT_BLOCK_SIZE,
              sigma_reg: float = DEFAULT_SIGMA_REG,
              max_retries: int = DEFAULT_MAX_RETRIES) -> BlockLDLResult:
    """Compute the block-folded Cholesky factor consumed by ``ldlq``.

    ``H`` is not modified; damping is applied to an internal copy exactly as
    upstream does (``H.diagonal().add_(2.0 * sigma_reg * H.diagonal().mean())``)
    only after a failed factorization attempt.

    Raises ``torch.linalg.LinAlgError`` if factorization still fails after
    ``max_retries`` damping steps.
    """
    _validate(H, block_size)
    if max_retries < 0:
        raise ValueError(f"max_retries must be non-negative, got {max_retries}")

    n = H.shape[0]
    work = H.clone()
    chol = None
    retries = 0
    for attempt in range(max_retries + 1):
        try:
            chol = torch.linalg.cholesky(work)
            retries = attempt
            break
        except torch.linalg.LinAlgError:
            if attempt == max_retries:
                raise
            # audited retry schedule; no pre-emptive damping
            work.diagonal().add_(2.0 * sigma_reg * work.diagonal().mean())

    m = n // block_size
    block_inverses = torch.empty((m, block_size, block_size),
                                 dtype=H.dtype, device=H.device)
    transformed = chol.clone()
    for block in range(m):
        start = block * block_size
        stop = start + block_size
        inverse = torch.linalg.inv(chol[start:stop, start:stop])
        block_inverses[block] = inverse
        # fold the inverse into this block-column: L[:, block] @ inv(L_bb)
        transformed[:, start:stop] = chol[:, start:stop] @ inverse

    identity = torch.eye(block_size, dtype=H.dtype, device=H.device)
    for block in range(m):
        start = block * block_size
        transformed[start:start + block_size, start:start + block_size] = identity

    return BlockLDLResult(
        transformed=transformed,
        cholesky=chol,
        block_inverses=block_inverses,
        retries=retries,
        damping_applied=retries > 0,
        n=n,
        block_size=block_size,
    )
