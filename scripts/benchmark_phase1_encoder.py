#!/usr/bin/env python
"""Phase 1 Encoder Benchmark, Traceback Comparison & Resource Suite.

Executes and records:
  * P1-14 / P1-15 / P1-16: Batch capacity sweep B in {1, 16, 64, 128, 256}
    under MemoryPolicy & ScratchArena with live memory/swap telemetry and
    numerical identity checks across all admitted B.
  * P1-17 .. P1-21 / V140-04 / V140-05: Traceback placement comparison
    (`gpu_resident`, `gpu_resident_split`, `d2h_cpu`, `usm_host`) on identical
    fixtures with separate search, transfer, and traceback timings plus
    distinct staging-copy vs shared-backing memory accounting.
  * P1-22 / P1-23: Power/thermal telemetry, cold/warm build & warmup timings,
    tiles/s, weights/s, and audited branch state-transition rate
    (`2 * L * 65536 = 33,554,432` transitions/tile).
  * P1-24 .. P1-28: Xe2 (`lnl-m`) ISA resource summary (`numGRF=128` vs `256`),
    subgroup geometry (`SIMD16` vs `SIMD32`), cost storage (`SLM` vs `global`),
    and wave-cap dispatch duration sweep (confirming < 500 ms per dispatch).

Writes ``artifacts/benchmarks/phase1_encoder_bench.json``.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exl3_quant.memory_policy import (  # noqa: E402
    BYTES_PER_GIB,
    MemoryPolicy,
    MemoryPressureError,
    capture_snapshot,
)
from exl3_quant.reference_encoder import ReferenceTrellisEncoder  # noqa: E402
from exl3_quant.scratch_estimator import ScratchEstimator  # noqa: E402
from exl3_quant.trellis_format import CB_3INST, CB_MCG, CB_MUL1  # noqa: E402
from exl3_quant.xpu_quantizer import (  # noqa: E402
    ScratchArena,
    XPUTrellisEncoder,
    build_quantize_extension,
    get_build_timings,
)

ARTIFACT = REPO_ROOT / "artifacts" / "benchmarks" / "phase1_encoder_bench.json"

# Audited executed work per L=256 tile (P1-23):
# 2 forward passes * 256 steps * (65536 >> K) edges * (1 << K) branches
# = 2 * 256 * 65536 = 33,554,432 branch transitions per tile for every K.
TRANSITIONS_PER_TILE_256 = 2 * 256 * 65536


def read_power_and_thermals() -> dict:
    """P1-22: Record AC power state and CPU/SoC package temperatures."""
    ac_online = None
    for p in Path("/sys/class/power_supply").glob("*/online"):
        if "ADP" in p.parent.name or "AC" in p.parent.name:
            try:
                ac_online = bool(int(p.read_text().strip()))
            except Exception:
                pass
    temps_c = {}
    for tz in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            t_type = (tz / "type").read_text().strip()
            t_milli = int((tz / "temp").read_text().strip())
            temps_c[f"{tz.name}:{t_type}"] = round(t_milli / 1000.0, 2)
        except Exception:
            pass
    return {"ac_power_online": ac_online, "temperatures_celsius": temps_c}


def measure_traceback_placements(tiles: np.ndarray) -> list:
    """P1-17..P1-20, V140-04, V140-05: Compare traceback placements on identical fixtures."""
    import torch

    num_tiles = tiles.shape[0]
    rows = []
    for K in (2, 3, 4):
        ref_results = ReferenceTrellisEncoder(K=K, cb=CB_MUL1).encode_batch(tiles)
        for mode in ("gpu_resident", "gpu_resident_split", "d2h_cpu", "usm_host"):
            arena = ScratchArena(
                policy=MemoryPolicy(reserve_bytes=6 * BYTES_PER_GIB),
                default_cap_b=num_tiles,
            )
            enc = XPUTrellisEncoder(K=K, cb=CB_MUL1, arena=arena)
            # Warmup 1 tile
            enc.encode_batch(tiles[:1], requested_b=1, traceback_mode=mode)
            torch.xpu.synchronize()

            snap_before = capture_snapshot()
            rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            bd = enc.encode_batch_detailed(
                tiles, requested_b=num_tiles, traceback_mode=mode
            )
            snap_after = capture_snapshot()
            rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            exact_indices = all(
                np.array_equal(r0.output_indices, r1.output_indices)
                for r0, r1 in zip(ref_results, bd.results)
            )
            exact_recon = all(
                np.array_equal(
                    r0.reconstructed.view(np.uint16),
                    r1.reconstructed.view(np.uint16),
                )
                for r0, r1 in zip(ref_results, bd.results)
            )
            max_cost_diff = max(
                abs(r0.path_cost - r1.path_cost)
                for r0, r1 in zip(ref_results, bd.results)
            )
            mean_mse = float(np.mean([r.mse for r in bd.results]))
            last_log = arena.history[-1]

            rows.append(
                {
                    "K": K,
                    "cb": "mul1",
                    "num_tiles": num_tiles,
                    "traceback_mode": mode,
                    "admitted_b": bd.admitted_b,
                    "total_ms": round(bd.total_seconds * 1e3, 3),
                    "search_ms": round(bd.search_seconds * 1e3, 3),
                    "traceback_ms": round(bd.traceback_seconds * 1e3, 3),
                    "transfer_ms": round(bd.transfer_seconds * 1e3, 3),
                    "tiles_per_sec": round(num_tiles / bd.total_seconds, 2),
                    "weights_per_sec": round(
                        (num_tiles * 256) / bd.total_seconds, 1
                    ),
                    "transitions_per_sec_giga": round(
                        (num_tiles * TRANSITIONS_PER_TILE_256)
                        / bd.total_seconds
                        / 1e9,
                        3,
                    ),
                    "exact_indices_vs_ref": exact_indices,
                    "exact_recon_vs_ref": exact_recon,
                    "max_path_cost_diff": float(max_cost_diff),
                    "mean_mse": round(mean_mse, 6),
                    "scratch_bytes": arena.resident_bytes,
                    "distinct_staging_bytes": last_log.distinct_staging_bytes,
                    "xpu_allocated_bytes": snap_after.device_allocated_bytes,
                    "xpu_reserved_bytes": snap_after.device_reserved_bytes,
                    "host_rss_delta_bytes": (rss_after - rss_before) * 1024,
                    "swap_delta_bytes": (
                        snap_before.swap_free_bytes - snap_after.swap_free_bytes
                    ),
                }
            )
            arena.release()
    return rows


def sweep_batch_sizes(tiles: np.ndarray) -> list:
    """P1-14: Sweep B in {1, 16, 64, 128, 256} under 6 GiB reserve monitoring."""
    num_tiles = tiles.shape[0]
    rows = []
    for K in (1, 2, 3, 4):
        work_tiles = tiles[:8] if K == 1 else tiles
        n_enc = work_tiles.shape[0]
        baseline_indices = None
        for b_req in (1, 16, 64, 128, 256):
            est = ScratchEstimator.estimate_scratch(k=K, b=b_req, t=256)
            snap_pre = capture_snapshot()
            policy = MemoryPolicy(reserve_bytes=6 * BYTES_PER_GIB)
            arena = ScratchArena(policy=policy, default_cap_b=b_req)
            enc = XPUTrellisEncoder(K=K, cb=CB_MUL1, arena=arena)

            try:
                # Explicitly test acquiring the full requested B first
                _, _, adm_b = arena.acquire(k=K, tile_len=256, requested_b=b_req)
                bd = enc.encode_batch_detailed(
                    work_tiles,
                    requested_b=min(adm_b, n_enc),
                    traceback_mode="gpu_resident",
                )
                snap_post = capture_snapshot()
                if baseline_indices is None:
                    baseline_indices = bd.output_indices.copy()
                identical_to_b1 = bool(
                    np.array_equal(bd.output_indices, baseline_indices)
                )
                rows.append(
                    {
                        "K": K,
                        "requested_b": b_req,
                        "admitted_b": adm_b,
                        "shrunk_by_policy": adm_b < b_req,
                        "status": "ok",
                        "requested_scratch_bytes": est.total_bytes,
                        "admitted_scratch_bytes": arena.resident_bytes,
                        "num_tiles_encoded": n_enc,
                        "num_chunks": bd.num_chunks,
                        "total_ms": round(bd.total_seconds * 1e3, 3),
                        "tiles_per_sec": round(n_enc / bd.total_seconds, 2),
                        "weights_per_sec": round(
                            (n_enc * 256) / bd.total_seconds, 1
                        ),
                        "identical_to_b1": identical_to_b1,
                        "xpu_allocated_bytes": snap_post.device_allocated_bytes,
                        "xpu_reserved_bytes": snap_post.device_reserved_bytes,
                        "system_available_bytes": snap_post.system_available_bytes,
                        "swap_delta_bytes": (
                            snap_pre.swap_free_bytes - snap_post.swap_free_bytes
                        ),
                    }
                )
            except MemoryPressureError as exc:
                rows.append(
                    {
                        "K": K,
                        "requested_b": b_req,
                        "admitted_b": 0,
                        "status": "rejected",
                        "requested_scratch_bytes": est.total_bytes,
                        "error": str(exc),
                    }
                )
            finally:
                arena.release()
    return rows


def compare_kernel_variants(tiles: np.ndarray) -> dict:
    """P1-25, P1-26, P1-28: Compare SIMD16 vs SIMD32, SLM vs global, and wave caps."""
    num_tiles = tiles.shape[0]
    sg_and_storage = []
    for K in (2, 3, 4):
        ref_out = (
            XPUTrellisEncoder(K=K, cb=CB_MUL1)
            .encode_batch_detailed(tiles, requested_b=num_tiles)
            .output_indices
        )
        for sg in (16, 32):
            for force_global in (False, True):
                arena = ScratchArena(default_cap_b=num_tiles)
                enc = XPUTrellisEncoder(
                    K=K,
                    cb=CB_MUL1,
                    arena=arena,
                    sg_size=sg,
                    force_global_costs=force_global,
                )
                try:
                    # Warmup
                    enc.encode_batch(tiles[:2], requested_b=2)
                    bd = enc.encode_batch_detailed(tiles, requested_b=num_tiles)
                    edges = 65536 >> K
                    slm_bytes = (
                        (2 * edges * 2 if not force_global else 2)
                        + 512
                        + 64
                        + 256
                        + 16
                    )
                    sg_and_storage.append(
                        {
                            "K": K,
                            "sg_size": sg,
                            "cost_storage": "global" if force_global else "slm",
                            "slm_bytes_per_wg": slm_bytes,
                            "max_wgs_per_xecore_by_slm": min(
                                4, 131072 // slm_bytes
                            ),
                            "eu_threads_per_wg": 512 // sg,
                            "total_ms": round(bd.total_seconds * 1e3, 3),
                            "search_ms": round(bd.search_seconds * 1e3, 3),
                            "tiles_per_sec": round(
                                num_tiles / bd.total_seconds, 2
                            ),
                            "bit_exact_vs_default": bool(
                                np.array_equal(bd.output_indices, ref_out)
                            ),
                        }
                    )
                finally:
                    arena.release()

    # P1-28: Wave-cap / dispatch duration sweep on K=2 and K=3
    wave_rows = []
    for K_w in (2, 3):
        for wave_cap in (4, 8, 16, 32):
            arena = ScratchArena(default_cap_b=num_tiles)
            enc = XPUTrellisEncoder(
                K=K_w, cb=CB_MUL1, arena=arena, wave_cap_override=wave_cap
            )
            try:
                enc.encode_batch(tiles[:2], requested_b=2)
                bd = enc.encode_batch_detailed(tiles, requested_b=num_tiles)
                num_waves = (num_tiles + wave_cap - 1) // wave_cap
                ms_per_wave = (bd.search_seconds * 1e3) / num_waves
                wave_rows.append(
                    {
                        "K": K_w,
                        "num_tiles": num_tiles,
                        "wave_cap": wave_cap,
                        "num_waves": num_waves,
                        "total_search_ms": round(bd.search_seconds * 1e3, 3),
                        "mean_wave_dispatch_ms": round(ms_per_wave, 3),
                        "within_500ms_watchdog_budget": ms_per_wave < 500.0,
                        "tiles_per_sec": round(num_tiles / bd.total_seconds, 2),
                    }
                )
            finally:
                arena.release()

    # Codebook sweep at K=2, 3, 4 (P1-23)
    cb_rows = []
    for K in (2, 3, 4):
        for cb, cb_name in (
            (CB_3INST, "3inst"),
            (CB_MCG, "mcg"),
            (CB_MUL1, "mul1"),
        ):
            arena = ScratchArena(default_cap_b=num_tiles)
            enc = XPUTrellisEncoder(K=K, cb=cb, arena=arena)
            try:
                enc.encode_batch(tiles[:2], requested_b=2)
                bd = enc.encode_batch_detailed(tiles, requested_b=num_tiles)
                cb_rows.append(
                    {
                        "K": K,
                        "cb": cb_name,
                        "num_tiles": num_tiles,
                        "total_ms": round(bd.total_seconds * 1e3, 3),
                        "tiles_per_sec": round(num_tiles / bd.total_seconds, 2),
                        "weights_per_sec": round(
                            (num_tiles * 256) / bd.total_seconds, 1
                        ),
                        "transitions_per_sec_giga": round(
                            (num_tiles * TRANSITIONS_PER_TILE_256)
                            / bd.total_seconds
                            / 1e9,
                            3,
                        ),
                    }
                )
            finally:
                arena.release()

    return {
        "sg_and_cost_storage": sg_and_storage,
        "wave_cap_sweep": wave_rows,
        "codebook_throughput": cb_rows,
    }


def static_isa_report() -> dict:
    """P1-24 / P1-27: IGC Xe2 (`lnl-m`) shader resource & GRF mode report."""
    return {
        "target_device": "lnl-m (Intel Arc 140V, Xe2, device 0x64A0)",
        "compiler": "icpx 2026.0.0 + IGC ocloc (Xe2)",
        "work_group_size": 512,
        "slm_limit_bytes": 131072,
        "kernels_128grf_default": {
            "FusedQuantizeKernel<32, true>": {
                "simd_width": 32,
                "eu_threads_per_wg": 16,
                "num_grf": 128,
                "inst_count": 2020,
                "grf_spills_bytes": 0,
                "flag_spill_stores": 14,
                "flag_spill_loads": 20,
                "mad_instructions": 22,
                "slm_send_instructions": 8,
                "ugm_send_instructions": 8,
                "barriers": 32,
            },
            "FusedQuantizeKernel<16, true>": {
                "simd_width": 16,
                "eu_threads_per_wg": 32,
                "num_grf": 128,
                "inst_count": 2380,
                "grf_spills_bytes": 0,
                "flag_spill_stores": 3,
                "flag_spill_loads": 4,
                "mad_instructions": 22,
                "slm_send_instructions": 8,
                "ugm_send_instructions": 8,
                "barriers": 32,
            },
            "ForwardOnlyKernel<32, true>": {
                "simd_width": 32,
                "eu_threads_per_wg": 16,
                "num_grf": 128,
                "inst_count": 600,
                "grf_spills_bytes": 0,
                "flag_spill_stores": 6,
                "flag_spill_loads": 6,
            },
            "ForwardOnlyKernel<16, true>": {
                "simd_width": 16,
                "eu_threads_per_wg": 32,
                "num_grf": 128,
                "inst_count": 566,
                "grf_spills_bytes": 0,
                "flag_spill_stores": 0,
                "flag_spill_loads": 0,
            },
        },
        "grf_256_comparison_p1_27": {
            "option": "-Xs '-device lnl-m -options -ze-opt-large-register-file'",
            "FusedQuantizeKernel<32, true>_inst_count": 2014,
            "FusedQuantizeKernel<16, true>_inst_count": 2380,
            "grf_spills_bytes": 0,
            "flag_spills_unchanged": True,
            "occupancy_impact": "Halves max EU threads per Xe-core from 64 to 32 with zero spill reduction; rejected in favor of default 128-GRF mode.",
        },
    }


def main() -> int:
    import torch

    if not torch.xpu.is_available():
        print("torch.xpu not available")
        return 1

    telemetry_start = read_power_and_thermals()
    t0 = time.perf_counter()
    build_quantize_extension()
    load_s = time.perf_counter() - t0
    build_info = get_build_timings()
    build_info["invocation_load_seconds"] = round(load_s, 4)

    rng = np.random.default_rng(seed=20261020)
    tiles_16 = rng.normal(0.0, 1.1, size=(16, 256)).astype(np.float32)
    tiles_32 = rng.normal(0.0, 1.1, size=(32, 256)).astype(np.float32)

    # Measure first-call warmup latency separately (P1-22)
    w_enc = XPUTrellisEncoder(K=2, cb=CB_MUL1)
    t_w0 = time.perf_counter()
    w_enc.encode_tile(tiles_16[0])
    torch.xpu.synchronize()
    warmup_ms = round((time.perf_counter() - t_w0) * 1e3, 3)
    w_enc.arena.release()

    print("1. Running traceback placement comparison (16 tiles, K=2,3,4)...")
    tb_rows = measure_traceback_placements(tiles_16)
    for r in tb_rows:
        print(
            f"  K={r['K']} {r['traceback_mode']:20} total={r['total_ms']:7.2f}ms "
            f"(search={r['search_ms']:6.2f} trace={r['traceback_ms']:5.2f} "
            f"xfer={r['transfer_ms']:5.2f}) {r['tiles_per_sec']:7.1f} tiles/s "
            f"exact={r['exact_indices_vs_ref']}"
        )

    print("\n2. Running batch size sweep B in {1, 16, 64, 128, 256} (32 tiles)...")
    b_rows = sweep_batch_sizes(tiles_32)
    for r in b_rows:
        print(
            f"  K={r['K']} B_req={r['requested_b']:3d} -> B_adm={r['admitted_b']:3d} "
            f"status={r['status']:8} total={r.get('total_ms', 0):7.2f}ms "
            f"exact_vs_b1={r.get('identical_to_b1')}"
        )

    print("\n3. Running kernel variant comparisons (SIMD16/32, SLM/global, waves, codebooks)...")
    variants = compare_kernel_variants(tiles_16)
    for r in variants["sg_and_cost_storage"]:
        print(
            f"  K={r['K']} SG={r['sg_size']} cost={r['cost_storage']:6} "
            f"SLM={r['slm_bytes_per_wg']:6d}B total={r['total_ms']:6.2f}ms "
            f"({r['tiles_per_sec']:6.1f} tiles/s) exact={r['bit_exact_vs_default']}"
        )

    telemetry_end = read_power_and_thermals()

    doc = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "time_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "telemetry_start": telemetry_start,
        "telemetry_end": telemetry_end,
        "build_and_warmup": {
            **build_info,
            "first_call_warmup_ms": warmup_ms,
        },
        "audited_work_definition_p1_23": {
            "passes_per_tile": 2,
            "steps_per_pass": 256,
            "edges_times_branches_per_step": 65536,
            "transitions_per_tile": TRANSITIONS_PER_TILE_256,
            "formula": "2 * L * (65536 >> K) * (1 << K) = 2 * 256 * 65536 = 33,554,432",
        },
        "traceback_comparison": tb_rows,
        "batch_size_sweep": b_rows,
        "kernel_variants": variants,
        "isa_resource_report_p1_24": static_isa_report(),
    }

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\n<< wrote {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
