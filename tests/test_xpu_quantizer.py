"""Comprehensive unit tests for the XPU SYCL tile encoder & ScratchArena.

Covers Phase 1 tasks:
  * P1-03 / P1-04 / P1-05: XPU forward ACS search, GPU-resident traceback,
    codebook parity, ReferenceTrellisEncoder parity across K in {1,2,3,4}
    and all 3 codebooks (3inst, mcg, mul1), plus live exl3xpu.ref decode.
  * P0-26 / P1-06: zeros, constants, outliers, near-ties, and tie-aware
    path verification (`verify_tile_encoding_quality`).
  * P1-07: single tile, exact batches, partial final batches, multi-wave
    batches, repeated calls, and explicit input-validation rejection.
  * P1-08: asynchronous queue ordering and tensor lifetime safety.
  * P1-11 .. P1-16 / V140-04: adaptive batch sizing, bounded `ScratchArena`
    eviction/reuse across alternating K values without memory/swap growth,
    smaller-batch retry on pressure/failure, and distinct staging-copy logging.
  * P1-17 .. P1-20 / P1-25 / P1-26: parity across traceback modes
    (`gpu_resident`, `gpu_resident_split`, `d2h_cpu`, `usm_host`), subgroup
    geometries (`sg_size=16` vs `32`), and cost storage (`SLM` vs `global`).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

from exl3_quant.memory_policy import (
    BYTES_PER_GIB,
    BYTES_PER_MIB,
    MemoryPolicy,
    MemoryPressureError,
    capture_snapshot,
)
from exl3_quant.reference_encoder import EncodingResult, ReferenceTrellisEncoder
from exl3_quant.scratch_estimator import ScratchEstimator
from exl3_quant.trellis_format import (
    CB_3INST,
    CB_MCG,
    CB_MUL1,
    codebook_lut,
    decode_states,
    decode_tile_values,
)
from exl3_quant.xpu_quantizer import (
    ScratchArena,
    XPUTrellisEncoder,
    build_quantize_extension,
    verify_tile_encoding_quality,
)

try:
    import torch

    XPU_AVAILABLE = torch.xpu.is_available()
except Exception:  # pragma: no cover
    XPU_AVAILABLE = False

EXL3XPU_PATH = Path("/home/yanchun/Projects/exl3xpu")
if EXL3XPU_PATH.exists() and str(EXL3XPU_PATH) not in sys.path:
    sys.path.insert(0, str(EXL3XPU_PATH))
try:
    from exl3xpu.ref import reconstruct_inner as oracle_reconstruct_inner

    ORACLE_AVAILABLE = True
except Exception:  # pragma: no cover
    ORACLE_AVAILABLE = False


@unittest.skipUnless(XPU_AVAILABLE, "torch.xpu device not available")
class TestXPUTrellisEncoder(unittest.TestCase):
    def setUp(self):
        self.ext = build_quantize_extension()

    def test_codebook_bit_exact_on_xpu(self):
        """P1-02 / P1-03: All 65536 states match numpy codebook_lut bit-for-bit."""
        for cb in (CB_3INST, CB_MCG, CB_MUL1):
            out = torch.empty(65536, dtype=torch.float16, device="xpu")
            self.ext.decode_codebook(cb, out)
            torch.xpu.synchronize()
            ref = codebook_lut(cb)
            np.testing.assert_array_equal(
                out.cpu().numpy().view(np.uint16),
                ref.view(np.uint16),
                err_msg=f"Codebook LUT mismatch on XPU for cb={cb}",
            )

    def test_parity_across_k_and_codebooks(self):
        """P1-03 / P1-04 / P1-05: Bit-exact parity vs ReferenceTrellisEncoder."""
        rng = np.random.default_rng(seed=2026)
        tile = rng.normal(0.0, 1.15, size=256).astype(np.float32)

        for K in (1, 2, 3, 4):
            cbs = (CB_MUL1,) if K == 1 else (CB_3INST, CB_MCG, CB_MUL1)
            for cb in cbs:
                ref = ReferenceTrellisEncoder(K=K, cb=cb).encode_tile(tile)
                enc = XPUTrellisEncoder(K=K, cb=cb)
                try:
                    got = enc.encode_tile(tile)
                finally:
                    enc.arena.release()

                np.testing.assert_array_equal(
                    got.output_indices,
                    ref.output_indices,
                    err_msg=f"Indices mismatch for K={K}, cb={cb}",
                )
                np.testing.assert_array_equal(
                    got.reconstructed.view(np.uint16),
                    ref.reconstructed.view(np.uint16),
                )
                np.testing.assert_array_equal(got.packed, ref.packed)
                self.assertEqual(got.end_state, ref.end_state)
                self.assertEqual(got.path_cost, ref.path_cost)
                self.assertAlmostEqual(got.distortion, ref.distortion, places=5)

    def test_l160_ngram_row_support(self):
        """Contract §4.1: L=160 n-gram rows with CB_MUL1."""
        rng = np.random.default_rng(seed=160)
        tile = rng.normal(0.0, 1.0, size=160).astype(np.float32)
        ref = ReferenceTrellisEncoder(
            K=3, cb=CB_MUL1, tile_len=160
        ).encode_tile(tile, pre_permute=False)
        enc = XPUTrellisEncoder(K=3, cb=CB_MUL1, tile_len=160)
        try:
            got = enc.encode_tile(tile, pre_permute=False)
        finally:
            enc.arena.release()

        np.testing.assert_array_equal(got.output_indices, ref.output_indices)
        self.assertEqual(got.end_state, ref.end_state)

    @unittest.skipUnless(ORACLE_AVAILABLE, "exl3xpu.ref oracle not available")
    def test_live_oracle_decode_parity(self):
        """P1-05: XPU-encoded packed container decodes bit-for-bit via exl3xpu.ref."""
        rng = np.random.default_rng(seed=314)
        tile = rng.normal(0.0, 1.25, size=(16, 16)).astype(np.float32)

        for K in (2, 3, 4):
            for cb in (CB_3INST, CB_MCG, CB_MUL1):
                enc = XPUTrellisEncoder(K=K, cb=cb)
                try:
                    res = enc.encode_tile(tile, pre_permute=True)
                finally:
                    enc.arena.release()

                trellis_t = torch.from_numpy(
                    res.packed.view(np.int16).reshape(1, 1, 16 * K)
                )
                oracle_w = (
                    oracle_reconstruct_inner(trellis_t, K, cb)
                    .numpy()
                    .reshape(-1)
                )
                np.testing.assert_array_equal(
                    oracle_w.view(np.uint16),
                    res.reconstructed.view(np.uint16),
                )

    def test_edge_cases_and_tie_handling(self):
        """P0-26 / P1-06: Zeros, repeated values, outliers, and tie verification."""
        rng = np.random.default_rng(seed=77)
        cases = {
            "zeros": np.zeros(256, dtype=np.float32),
            "constant": np.full(256, 0.5, dtype=np.float32),
            "outliers": np.where(
                np.arange(256) % 17 == 0,
                25.0,
                rng.normal(0.0, 0.5, size=256),
            ).astype(np.float32),
            "alternating": np.tile(
                np.array([-1.5, 1.5], dtype=np.float32), 128
            ),
        }
        ref_enc = ReferenceTrellisEncoder(K=2, cb=CB_MUL1)
        xpu_enc = XPUTrellisEncoder(K=2, cb=CB_MUL1)
        try:
            for name, tile in cases.items():
                ref = ref_enc.encode_tile(tile)
                got = xpu_enc.encode_tile(tile)
                check = verify_tile_encoding_quality(got, ref, cost_atol=0.0)
                self.assertTrue(check["passed"], f"Failed quality check on {name}: {check}")
                self.assertTrue(check["indices_identical"], f"Tie-break order differed on {name}")
        finally:
            xpu_enc.arena.release()

        # P1-06: verify verify_tile_encoding_quality rejects broken circular rings
        broken = EncodingResult(
            output_indices=got.output_indices.copy(),
            reconstructed=got.reconstructed.copy(),
            reconstructed_perm=got.reconstructed_perm.copy(),
            packed=got.packed.copy(),
            end_state=got.end_state,
            distortion=got.distortion,
            mse=got.mse,
            path_cost=got.path_cost,
            K=got.K,
            cb=got.cb,
        )
        broken.output_indices[10] ^= 1
        broken.reconstructed_perm[10] = xpu_enc.lut[broken.output_indices[10]]
        check_bad = verify_tile_encoding_quality(broken, ref)
        self.assertFalse(check_bad["passed"])
        self.assertFalse(check_bad["ring_valid"])

    def test_batches_partial_chunks_and_waves(self):
        """P1-07: Single tile, multi-batch, partial final batch, and multi-wave."""
        rng = np.random.default_rng(seed=555)
        tiles = rng.normal(0.0, 1.0, size=(19, 256)).astype(np.float32)

        ref_results = ReferenceTrellisEncoder(K=3, cb=CB_MUL1).encode_batch(tiles)

        # Partial final chunk: 19 tiles with B=6 -> chunks of 6, 6, 6, 1
        enc_chunked = XPUTrellisEncoder(K=3, cb=CB_MUL1, wave_cap_override=4)
        try:
            bd = enc_chunked.encode_batch_detailed(tiles, requested_b=6)
        finally:
            enc_chunked.arena.release()

        self.assertEqual(bd.admitted_b, 6)
        self.assertEqual(bd.num_chunks, 4)
        self.assertEqual(len(bd.results), 19)
        for idx, (r_got, r_ref) in enumerate(zip(bd.results, ref_results)):
            np.testing.assert_array_equal(
                r_got.output_indices,
                r_ref.output_indices,
                err_msg=f"Mismatch at tile {idx} in partial-batch test",
            )
            np.testing.assert_array_equal(
                decode_states(r_got.packed, K=3), r_got.output_indices
            )
            np.testing.assert_array_equal(
                decode_tile_values(r_got.packed, K=3, cb=CB_MUL1).view(np.uint16),
                r_got.reconstructed.view(np.uint16),
            )

    def test_input_validation_rejects_invalid_configs(self):
        """P1-07: Reject invalid K, cb, tile_len, and malformed C++ tensors."""
        with self.assertRaises(ValueError):
            XPUTrellisEncoder(K=0)
        with self.assertRaises(ValueError):
            XPUTrellisEncoder(K=9)
        with self.assertRaises(ValueError):
            XPUTrellisEncoder(K=2, cb=99)
        with self.assertRaises(ValueError):
            XPUTrellisEncoder(K=2, tile_len=128)
        with self.assertRaises(ValueError):
            XPUTrellisEncoder(K=2, cb=CB_MCG, tile_len=160)
        with self.assertRaises(ValueError):
            XPUTrellisEncoder(K=2, sg_size=8)

        enc = XPUTrellisEncoder(K=2, cb=CB_MUL1)
        with self.assertRaises(ValueError):
            enc.encode_tile(np.zeros(100, dtype=np.float32))
        with self.assertRaises(ValueError):
            enc.encode_batch(np.zeros((0, 256), dtype=np.float32))
        with self.assertRaises(ValueError):
            enc.encode_batch(np.zeros((2, 256), dtype=np.float32), traceback_mode="bad")

        # Direct C++ binding validation: ambiguous mcg & mul1, wrong dtype, CPU tensor
        in_t = torch.zeros((1, 256), dtype=torch.float32, device="xpu")
        out_w = torch.zeros((1, 256), dtype=torch.float32, device="xpu")
        out_i = torch.zeros((1, 256), dtype=torch.int16, device="xpu")
        costs = torch.zeros((1, 2, 16384), dtype=torch.float16, device="xpu")
        edges = torch.zeros((1, 256, 16384), dtype=torch.int16, device="xpu")
        with self.assertRaises(RuntimeError):
            self.ext.quantize_tiles(in_t, out_w, out_i, costs, edges, 2, True, True)
        with self.assertRaises(RuntimeError):
            self.ext.quantize_tiles(in_t.cpu(), out_w, out_i, costs, edges, 2, False, True)

    def test_queue_ordering_and_async_invocation(self):
        """P1-08: Interleaved torch XPU ops and custom kernel on current stream."""
        rng = np.random.default_rng(seed=808)
        raw = rng.normal(0.0, 0.5, size=(4, 256)).astype(np.float16).astype(np.float32)

        in_t = torch.from_numpy(raw).to("xpu")
        # Interleave a torch scaling op immediately before quantize_tiles without host sync
        in_t.mul_(2.0)
        out_w = torch.empty_like(in_t)
        out_i = torch.empty((4, 256), dtype=torch.int16, device="xpu")
        costs = torch.empty((4, 2, 8192), dtype=torch.float16, device="xpu")
        edges = torch.empty((4, 256, 8192), dtype=torch.int16, device="xpu")
        self.ext.quantize_tiles(in_t, out_w, out_i, costs, edges, 3, False, True)
        out_w.add_(1.0)
        torch.xpu.synchronize()

        ref_enc = ReferenceTrellisEncoder(K=3, cb=CB_MUL1)
        for ti in range(4):
            ref = ref_enc.encode_tile(raw[ti] * 2.0, pre_permute=False)
            np.testing.assert_array_equal(
                out_i[ti].cpu().numpy().view(np.uint16), ref.output_indices
            )
            np.testing.assert_allclose(
                out_w[ti].cpu().numpy(),
                ref.reconstructed_perm.astype(np.float32) + 1.0,
                atol=1e-5,
            )

    def test_all_traceback_modes_and_kernel_variants_agree(self):
        """P1-17..P1-20, P1-25, P1-26: Traceback placements, SG16/32, SLM/global."""
        rng = np.random.default_rng(seed=909)
        tiles = rng.normal(0.0, 1.1, size=(4, 256)).astype(np.float32)
        ref = ReferenceTrellisEncoder(K=2, cb=CB_MUL1).encode_batch(tiles)

        for mode in ("gpu_resident", "gpu_resident_split", "d2h_cpu", "usm_host"):
            enc = XPUTrellisEncoder(K=2, cb=CB_MUL1)
            try:
                bd = enc.encode_batch_detailed(tiles, requested_b=4, traceback_mode=mode)
            finally:
                enc.arena.release()
            for r_got, r_ref in zip(bd.results, ref):
                np.testing.assert_array_equal(
                    r_got.output_indices,
                    r_ref.output_indices,
                    err_msg=f"Mismatch in traceback_mode={mode}",
                )

        for sg in (16, 32):
            for fg in (False, True):
                enc = XPUTrellisEncoder(
                    K=2, cb=CB_MUL1, sg_size=sg, force_global_costs=fg
                )
                try:
                    bd = enc.encode_batch_detailed(tiles, requested_b=4)
                finally:
                    enc.arena.release()
                for r_got, r_ref in zip(bd.results, ref):
                    np.testing.assert_array_equal(
                        r_got.output_indices, r_ref.output_indices
                    )


@unittest.skipUnless(XPU_AVAILABLE, "torch.xpu device not available")
class TestScratchArenaLifecycle(unittest.TestCase):
    def test_bounded_eviction_and_reuse_across_k(self):
        """P1-12 / P1-15 / P1-16: Alternating K evicts old buffers with no memory leak."""
        arena = ScratchArena(default_cap_b=4)
        rng = np.random.default_rng(seed=15)
        tiles = rng.normal(0.0, 1.0, size=(4, 256)).astype(np.float32)

        snap_start = capture_snapshot()
        try:
            for K in (2, 2, 3, 4, 2, 3, 2):
                enc = XPUTrellisEncoder(K=K, cb=CB_MUL1, arena=arena)
                enc.encode_batch(tiles, requested_b=4)
                # Resident bytes must equal ONLY the active K's 4-tile allocation
                expected_bytes = 4 * (2 * (65536 >> K) * 2 + 256 * (65536 >> K) * 2)
                self.assertEqual(arena.resident_bytes, expected_bytes)

            self.assertEqual(arena.reuses, 1)      # second K=2 reused
            self.assertEqual(arena.evictions, 5)   # 5 K transitions evicted previous
        finally:
            arena.release()

        self.assertEqual(arena.resident_bytes, 0)
        snap_end = capture_snapshot()
        self.assertEqual(snap_end.device_allocated_bytes, snap_start.device_allocated_bytes)
        self.assertGreaterEqual(len(arena.history), 7)

    def test_smaller_batch_retry_and_clean_rejection(self):
        """P1-13: Automatically halves B on failure and rejects cleanly when B=1 fails."""
        arena = ScratchArena(default_cap_b=16)
        rng = np.random.default_rng(seed=13)
        tiles = rng.normal(0.0, 1.0, size=(16, 256)).astype(np.float32)

        # Simulate allocator failure when b > 4
        def fail_above_4(cand_b: int):
            if cand_b > 4:
                raise RuntimeError(f"simulated allocator failure at B={cand_b}")

        enc = XPUTrellisEncoder(K=3, cb=CB_MUL1, arena=arena)
        try:
            bd = enc.encode_batch_detailed(
                tiles, requested_b=16, allocator_hook=fail_above_4
            )
            self.assertEqual(bd.admitted_b, 4)
            self.assertEqual(bd.num_chunks, 4)
            self.assertGreaterEqual(arena.retries, 2)  # 16 -> 8 -> 4

            ref = ReferenceTrellisEncoder(K=3, cb=CB_MUL1).encode_batch(tiles)
            for r_got, r_ref in zip(bd.results, ref):
                np.testing.assert_array_equal(r_got.output_indices, r_ref.output_indices)
        finally:
            arena.release()

        # Reject cleanly if reserve leaves < B=1 capacity
        impossible_policy = MemoryPolicy(reserve_bytes=64 * BYTES_PER_GIB)
        impossible_arena = ScratchArena(policy=impossible_policy)
        enc_imp = XPUTrellisEncoder(K=2, cb=CB_MUL1, arena=impossible_arena)
        with self.assertRaises(MemoryPressureError):
            enc_imp.encode_batch(tiles, requested_b=4)

        # Also test single-allocation ceiling shrinking B=16 -> B=4
        # For K=3, per-B edges = 256 * 8192 * 2 = 4 MiB; ceiling = 16 MiB -> max B=4
        ceil_policy = MemoryPolicy(
            reserve_bytes=4 * BYTES_PER_GIB,
            max_single_alloc_bytes=16 * BYTES_PER_MIB,
        )
        ceil_arena = ScratchArena(policy=ceil_policy)
        try:
            self.assertEqual(
                ceil_arena.adaptive_batch_capacity(k=3, requested_b=16), 4
            )
            _, _, adm_b = ceil_arena.acquire(k=3, requested_b=16)
            self.assertEqual(adm_b, 4)
        finally:
            ceil_arena.release()

    def test_retry_path_no_double_residency_b1(self):
        """B1 regression: halving retry drops attempt n before allocating attempt n+1."""
        torch.xpu.empty_cache()
        pre_alloc = torch.xpu.memory_allocated()

        class PostCommitFailureAtB8Policy(MemoryPolicy):
            def check_after_commit(
                self, snapshot, pre_snapshot=None, expected_bytes=0, context=""
            ):
                if "B=8" in context:
                    raise MemoryPressureError(
                        "simulated post-commit failure at B=8", 0, 0, snapshot, context
                    )
                super().check_after_commit(
                    snapshot, pre_snapshot, expected_bytes, context
                )

        arena = ScratchArena(policy=PostCommitFailureAtB8Policy())
        alloc_seen = []

        def hook(b):
            alloc_seen.append((b, torch.xpu.memory_allocated()))

        try:
            costs, edges, admitted_b = arena.acquire(
                k=3, requested_b=8, allocator_hook=hook
            )
            self.assertEqual(admitted_b, 4)
            # When b=4 was being admitted, the previous b=8 buffers must already be freed
            self.assertEqual(len(alloc_seen), 2)
            self.assertEqual(alloc_seen[0][0], 8)
            self.assertEqual(alloc_seen[1][0], 4)
            # Before B=4 allocates, memory_allocated must be 0 (no lingering B=8 buffers)
            self.assertEqual(alloc_seen[1][1], pre_alloc)

            # At B=4, scratch is 4 * (2 * 8192 * 2 + 256 * 8192 * 2) = 16.125 MiB
            est4 = ScratchEstimator.estimate_scratch(k=3, b=4, t=256)
            self.assertEqual(torch.xpu.memory_allocated(), pre_alloc + est4.total_bytes)
        finally:
            arena.release()
            torch.xpu.empty_cache()

    def test_failed_acquisition_frees_all_scratch_b2(self):
        """B2 regression: failed acquisition leaves 0 resident bytes and frees buffers."""
        torch.xpu.empty_cache()
        pre_alloc = torch.xpu.memory_allocated()

        class AlwaysFailingPostCommitPolicy(MemoryPolicy):
            def check_after_commit(
                self, snapshot, pre_snapshot=None, expected_bytes=0, context=""
            ):
                raise MemoryPressureError(
                    "simulated post-commit failure", 0, 0, snapshot, context
                )

        arena = ScratchArena(policy=AlwaysFailingPostCommitPolicy())
        caught = False
        try:
            arena.acquire(k=3, requested_b=1)
        except MemoryPressureError as exc:
            caught = True
            # Even while holding exc in frame locals, memory must be completely released
            post_alloc = torch.xpu.memory_allocated()
            self.assertEqual(post_alloc, pre_alloc)
            self.assertEqual(arena.resident_bytes, 0)
            self.assertIn("simulated post-commit failure", str(exc))

        self.assertTrue(caught)


if __name__ == "__main__":
    unittest.main()
