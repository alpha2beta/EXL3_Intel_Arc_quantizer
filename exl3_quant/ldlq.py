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
from typing import Callable, Optional

import numpy as np
import torch

from exl3_quant.trellis_format import tensor_core_perm, tensor_core_perm_inv

DEFAULT_BLOCK_SIZE = 16
DEFAULT_SIGMA_REG = 0.025
DEFAULT_MAX_RETRIES = 10
DEFAULT_BUF_SIZE_K = 128

# A tile encoder takes a ``(tiles_n, 256)`` float32 batch already in tensor-core
# order and returns ``(reconstructed, indices)``, both ``(tiles_n, 256)`` with the
# reconstruction in *tensor-core* order and the indices as 16-bit states.
TileEncoder = Callable[[torch.Tensor], "tuple[torch.Tensor, torch.Tensor]"]


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


# ---------------------------------------------------------------------------
# P2-08: one-sided LDLQ error-feedback traversal (quantize.py:493-609)
# ---------------------------------------------------------------------------


@dataclass
class LDLQResult:
    """Quantized layer plus the encoded trellis states."""

    weight_q: torch.Tensor   # (k, n) float32, reconstructed weights
    encoded: torch.Tensor    # (k // 16, n // 16, 256) int16 states, TC order
    tiles_k: int
    tiles_n: int
    buf_size_k: int


def _as_int16_states(indices: torch.Tensor) -> torch.Tensor:
    """Reinterpret 16-bit states as the signed container upstream stores.

    Accepts either ``uint16``/``int32`` values in ``[0, 65536)`` or an already
    signed ``int16`` tensor. The bit pattern is preserved; only the container
    type is signed (upstream uses ``torch.short`` for the encoded tensor).
    """
    values = indices.to(torch.int64) & 0xFFFF
    values = torch.where(values >= 0x8000, values - 0x10000, values)
    return values.to(torch.int16)


def ldlq(
    weight: torch.Tensor,
    L: torch.Tensor,
    encode_tiles: TileEncoder,
    buf_size_k: int = DEFAULT_BUF_SIZE_K,
) -> LDLQResult:
    """Quantize ``weight`` with bottom-up one-sided LDLQ error feedback.

    Faithful transcription of upstream ``quantize.py:493-609``:

    * spans of ``buf_size_k`` rows are traversed **bottom-up**;
    * within a span, 16-row blocks are traversed **bottom-up**;
    * the target rows are corrected by the *addition* of the accumulated
      compensation term (``rows = b_weight + compensation``), computed from the
      rows already quantized below and the matching strictly-lower slice of
      ``L``;
    * after a span, its full residual is propagated into ``prod_cache`` through
      the span's ``L`` rows, which is what carries error feedback into the next
      (higher) span;
    * tiles are pre-permuted into tensor-core order before encoding and the
      reconstruction is un-permuted, while the **indices stay in tensor-core
      order**, exactly as upstream.

    Args:
        weight: ``(k, n)`` float32 regularized weight (in the transformed
            basis). Not modified.
        L: ``(k, k)`` float32 output of :func:`block_ldl` (identity blocks on
            the diagonal). The inner-loop slices are strictly lower; the
            trailing span update does touch the span's own rows through the
            identity blocks, but those rows are never read again, so upstream's
            ``L[dr, dr] = 0`` is unobservable. Not modified.
        encode_tiles: callable ``tiles -> (reconstructed, indices)``; both
            outputs are ``(tiles_n, 256)`` and already in tensor-core order.
            See :func:`make_reference_tile_encoder` /
            :func:`make_xpu_tile_encoder`.
        buf_size_k: LDLQ span height; must be a multiple of 16 and divide both
            ``k`` and ``n`` (upstream asserts the ``n`` divisibility and assumes
            the ``k`` one when it steps ``range(k, 0, -buf_size_k)``).

    Returns:
        :class:`LDLQResult`.

    Note:
        Upstream's in-place ``compensation_term.addmm_`` aliases ``prod_cache``.
        The aliasing is unobservable: written rows belong to the span being
        processed and are never read again (later spans only read rows *above*
        them), so :func:`ldlq` performs the same in-place update without relying
        on aliasing.
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2-D, got shape {tuple(weight.shape)}")
    if weight.dtype != torch.float32:
        raise ValueError(f"weight must be float32, got {weight.dtype}")
    k, n = weight.shape
    if k % 16 != 0:
        raise ValueError(f"rows ({k}) must be a multiple of 16")
    if n % 128 != 0:
        raise ValueError(f"columns ({n}) must be a multiple of 128")
    if L.dim() != 2 or L.shape != (k, k):
        raise ValueError(f"L must have shape {(k, k)}, got {tuple(L.shape)}")
    if L.dtype != torch.float32:
        raise ValueError(f"L must be float32, got {L.dtype}")
    if L.device != weight.device:
        raise ValueError(
            f"L and weight must be on the same device, got {L.device} / {weight.device}"
        )

    buf_size_k = int(buf_size_k)
    if buf_size_k < 16 or buf_size_k % 16 != 0:
        raise ValueError(f"buf_size_k must be a positive multiple of 16, got {buf_size_k}")
    if n % buf_size_k != 0:
        raise ValueError(f"columns ({n}) must be divisible by buf_size_k ({buf_size_k})")
    if k % buf_size_k != 0:
        raise ValueError(
            f"rows ({k}) must be divisible by buf_size_k ({buf_size_k}); "
            "upstream's range(k, 0, -buf_size_k) assumes this"
        )

    device = weight.device
    tiles_k = k // 16
    tiles_n = n // 16
    perm = torch.as_tensor(tensor_core_perm(), dtype=torch.long, device=device)
    inv_perm = torch.as_tensor(tensor_core_perm_inv(), dtype=torch.long, device=device)

    prod_cache = torch.zeros((k, n), dtype=torch.float32, device=device)
    weight_q = torch.zeros((k, n), dtype=torch.float32, device=device)
    encoded = torch.zeros((tiles_k, tiles_n, 256), dtype=torch.int16,
                          device=device)

    for j in range(k, 0, -buf_size_k):
        i = j - buf_size_k

        b_weight = weight[i:j]
        b_weight_q = weight_q[i:j]
        b_encoded = encoded[i // 16:j // 16]
        b_prod_cache = prod_cache[i:j]
        b_L = L[i:j]

        for bj in range(buf_size_k, 0, -16):
            bi = bj - 16

            # Error of the rows already quantized within this span.
            bb_err = b_weight[bj:] - b_weight_q[bj:]
            # Matching strictly-lower slice of the block-folded Cholesky factor.
            # ``b_L`` slices rows only, so the columns are offset by the span
            # start: exactly ``L[i+bj:j, i+bi:i+bj]`` upstream.
            bb_L = b_L[bj:, i + bi:i + bj]

            compensation_term = b_prod_cache[bi:bj]
            if bb_err.shape[0] > 0:
                compensation_term.addmm_(bb_L.T, bb_err, alpha=1.0, beta=1.0)
            rows = b_weight[bi:bj] + compensation_term

            tiles = (
                rows.reshape(16, tiles_n, 16)
                .permute(1, 0, 2)
                .reshape(tiles_n, 256)
            )
            tiles = tiles[:, perm]

            quant_w, quant_i = encode_tiles(tiles)
            if quant_w.shape != (tiles_n, 256) or quant_i.shape != (tiles_n, 256):
                raise ValueError(
                    "encode_tiles must return two (tiles_n, 256) tensors, got "
                    f"{tuple(quant_w.shape)} / {tuple(quant_i.shape)}"
                )
            quant_w = quant_w.to(device=device, dtype=torch.float32)[:, inv_perm]
            quant_i = _as_int16_states(quant_i.to(device=device))

            b_weight_q[bi:bj] = (
                quant_w.reshape(tiles_n, 16, 16)
                .permute(1, 0, 2)
                .reshape(16, n)
            )
            b_encoded[bi // 16:bj // 16] = quant_i.unsqueeze(0)

        # Trailing-weight feedback for the next (higher) span.
        b_err = b_weight - b_weight_q
        prod_cache.addmm_(b_L.T, b_err, alpha=1.0, beta=1.0)

    return LDLQResult(
        weight_q=weight_q,
        encoded=encoded,
        tiles_k=tiles_k,
        tiles_n=tiles_n,
        buf_size_k=buf_size_k,
    )


# ---------------------------------------------------------------------------
# Encoder adapters
# ---------------------------------------------------------------------------


def make_reference_tile_encoder(K: int = 2) -> TileEncoder:
    """CPU golden-reference tile encoder (``ReferenceTrellisEncoder``)."""
    from exl3_quant.reference_encoder import CB_MUL1, ReferenceTrellisEncoder

    encoder = ReferenceTrellisEncoder(K=K, cb=CB_MUL1)

    def encode(tiles: torch.Tensor):
        arr = np.ascontiguousarray(tiles.detach().cpu().numpy(), dtype=np.float32)
        reconstructed = np.empty_like(arr)
        indices = np.empty(arr.shape, dtype=np.uint16)
        for t in range(arr.shape[0]):
            result = encoder.encode_tile(arr[t], pre_permute=False)
            reconstructed[t] = result.reconstructed
            indices[t] = result.output_indices
        return (
            torch.from_numpy(reconstructed).to(tiles.device),
            torch.from_numpy(indices.view(np.int16)).to(tiles.device),
        )

    # Expose the encoder's bit width so callers can validate it against the
    # requested K before the container is packed (see quantize_linear_module).
    encode.K = K
    encode.cb = CB_MUL1

    return encode


def make_xpu_tile_encoder(
    K: int = 2,
    device_index: int = 0,
    arena=None,
    sg_size: int = 32,
    force_global_costs: bool = False,
    traceback_mode: str = "gpu_resident",
) -> TileEncoder:
    """Intel Arc tile encoder (``XPUTrellisEncoder``) as an LDLQ callback.

    The encoder instance (and therefore its :class:`ScratchArena`) is reused for
    every call, which is the lifetime contract P2-11 requires. Each call still
    performs one H2D/D2H round trip per 16-row block; batching across blocks is
    a P2-07 optimisation and must not change the arithmetic.
    """
    from exl3_quant.xpu_quantizer import CB_MUL1, XPUTrellisEncoder

    encoder = XPUTrellisEncoder(
        K=K,
        cb=CB_MUL1,
        arena=arena,
        device_index=device_index,
        sg_size=sg_size,
        force_global_costs=force_global_costs,
    )

    def encode(tiles: torch.Tensor):
        arr = np.ascontiguousarray(tiles.detach().cpu().numpy(), dtype=np.float32)
        results = encoder.encode_batch(
            arr, pre_permute=False, traceback_mode=traceback_mode
        )
        reconstructed = np.stack([r.reconstructed for r in results]).astype(np.float32)
        indices = np.stack([r.output_indices for r in results]).astype(np.uint16)
        return (
            torch.from_numpy(reconstructed).to(tiles.device),
            torch.from_numpy(indices.view(np.int16)).to(tiles.device),
        )

    # Expose the encoder's bit width so callers can validate it against the
    # requested K before the container is packed (see quantize_linear_module).
    encode.K = K
    encode.cb = CB_MUL1

    return encode
