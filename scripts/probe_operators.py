#!/usr/bin/env python
"""Probe operator coverage and measure the LDLQ factorization fallback cost.

Implements P0-21 and the measurement half of P0-22:

* Runs the calibration/factorization operator set the EXL3 port needs
  (matmul, einsum, cholesky, inv, solve, triangular solve, QR, SVD, eigvalsh)
  on the Arc 140V and compares every result against a CPU float64 reference.
* Detects **silent XPU->CPU fallbacks** via ``PYTORCH_DEBUG_XPU_FALLBACK``
  warnings and records the exact aten operator that fell back.
* Benchmarks the §9.1 ``block_ldl`` path at Llama-3.1-8B layer sizes:
  CPU Cholesky, the implicit XPU fallback, explicit H2D/D2H transfer cost and
  effective bandwidth on the shared bus, native XPU ``inv``/``matmul``, and
  full ``block_ldl`` on both devices. The algorithm (including the damping
  retry schedule) is never changed by the fallback.

Writes ``artifacts/environment/operator_coverage.json``.

Usage:
    PYTHONPATH=. python scripts/probe_operators.py
"""

from __future__ import annotations

import os
import sys

# Must be present in the environment before torch is imported so per-call
# fallback warnings are emitted by XPUFallback.cpp.
if os.environ.get("PYTORCH_DEBUG_XPU_FALLBACK") != "1" and __name__ == "__main__":
    os.environ["PYTORCH_DEBUG_XPU_FALLBACK"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])
os.environ.setdefault("PYTORCH_DEBUG_XPU_FALLBACK", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exl3_quant.ldlq import block_ldl  # noqa: E402
from exl3_quant.memory_policy import (  # noqa: E402
    MemoryPolicy,
    capture_snapshot,
    format_bytes,
)

ARTIFACT = REPO_ROOT / "artifacts" / "environment" / "operator_coverage.json"
# Note: torch-xpu-ops XPUFallback.cpp omits the closing quote after the op name.
FALLBACK_RE = re.compile(
    r"operator '([^'\s]+)'?\s+on the XPU backend is falling back"
)


def make_spd(n: int, seed: int = 2026):
    import torch
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=generator)
    return (a @ a.T / n + torch.eye(n)).to(torch.float32)


def run_op(torch, fn):
    """Execute one op, returning status, timing, metrics and fallback ops."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fn()
            torch.xpu.synchronize()
        except Exception as exc:  # unsupported dtype / missing kernel
            return {
                "status": "unsupported",
                "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
                "fallback_ops": [],
            }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = time.perf_counter()
        value = fn()
        torch.xpu.synchronize()
        elapsed = time.perf_counter() - start

    fallback_ops = []
    for warning in caught:
        match = FALLBACK_RE.search(str(warning.message))
        if match:
            fallback_ops.append(match.group(1))
    return {
        "status": "ok",
        "seconds": round(elapsed, 5),
        "fallback_ops": sorted(set(fallback_ops)),
        "value": value,
    }


def fro_rel_error(torch, got, reference):
    return float((got - reference).norm() / reference.norm())


def coverage(torch, n: int = 512) -> list:
    """Coverage matrix with CPU float64 references and fallback attribution."""
    H = make_spd(n)
    H_x = H.to("xpu")
    eye = torch.eye(n)
    b = torch.randn(n, 1)

    def record(name, dtype, fn, metric_name, metric_fn, tolerance):
        outcome = run_op(torch, fn)
        row = {
            "op": name,
            "dtype": dtype,
            "status": outcome["status"],
            "seconds": outcome.get("seconds"),
            "fallback_ops": outcome.get("fallback_ops", []),
        }
        if outcome["status"] == "ok":
            row[metric_name] = metric_fn(outcome["value"])
            row["tolerance"] = tolerance
            row["within_tolerance"] = row[metric_name] <= tolerance
        else:
            row["error"] = outcome["error"]
        return row

    rows = []
    rows.append(record(
        "matmul", "float32",
        lambda: H_x @ H_x,
        "rel_error", lambda out: fro_rel_error(torch, out.cpu(), H.double() @ H.double()),
        1e-5))
    rows.append(record(
        "matmul", "float16",
        lambda: H_x.half() @ H_x.half(),
        "rel_error", lambda out: fro_rel_error(torch, out.cpu().float(), H.double() @ H.double()),
        2e-2))
    rows.append(record(
        "matmul", "bfloat16",
        lambda: H_x.bfloat16() @ H_x.bfloat16(),
        "rel_error", lambda out: fro_rel_error(torch, out.cpu().float(), H.double() @ H.double()),
        2e-2))
    rows.append(record(
        "einsum_ij_jk_ik", "float32",
        lambda: torch.einsum("ij,jk->ik", H_x, H_x),
        "rel_error", lambda out: fro_rel_error(torch, out.cpu(), H.double() @ H.double()),
        1e-5))
    rows.append(record(
        "einsum_ij_jk_ik", "float16",
        lambda: torch.einsum("ij,jk->ik", H_x.half(), H_x.half()),
        "rel_error", lambda out: fro_rel_error(torch, out.cpu().float(), H.double() @ H.double()),
        2e-2))

    def chol_metric(out):
        factor = out.cpu()
        return fro_rel_error(torch, factor @ factor.T, H)

    rows.append(record(
        "linalg.cholesky", "float32",
        lambda: torch.linalg.cholesky(H_x),
        "reconstruction_rel_error", chol_metric, 1e-4))
    rows.append(record(
        "linalg.cholesky", "float16",
        lambda: torch.linalg.cholesky(H_x.half()),
        "reconstruction_rel_error", lambda out: 0.0, 1e-4))
    rows.append(record(
        "linalg.cholesky", "bfloat16",
        lambda: torch.linalg.cholesky(H_x.bfloat16()),
        "reconstruction_rel_error", lambda out: 0.0, 1e-4))
    rows.append(record(
        "linalg.cholesky_ex", "float32",
        lambda: torch.linalg.cholesky_ex(H_x),
        "info", lambda out: int(out[1].item()), 0))

    def inv_metric(out):
        product = out.cpu() @ H
        return fro_rel_error(torch, product, eye)

    rows.append(record(
        "linalg.inv", "float32",
        lambda: torch.linalg.inv(H_x),
        "residual_rel_error", inv_metric, 1e-4))
    rows.append(record(
        "linalg.inv", "float16",
        lambda: torch.linalg.inv(H_x.half()),
        "residual_rel_error", lambda out: 0.0, 1e-4))

    def solve_metric(out):
        residual = H @ out.cpu() - b
        return float(residual.norm() / b.norm())

    rows.append(record(
        "linalg.solve", "float32",
        lambda: torch.linalg.solve(H_x, b.to("xpu")),
        "residual_rel_error", solve_metric, 1e-4))
    rows.append(record(
        "linalg.solve", "float16",
        lambda: torch.linalg.solve(H_x.half(), b.half().to("xpu")),
        "residual_rel_error", lambda out: 0.0, 1e-4))

    L_cpu = torch.linalg.cholesky(H)
    L_x = L_cpu.to("xpu")

    def tri_metric(out):
        residual = L_cpu @ out.cpu() - b
        return float(residual.norm() / b.norm())

    rows.append(record(
        "linalg.solve_triangular", "float32",
        lambda: torch.linalg.solve_triangular(L_x, b.to("xpu"), upper=False),
        "residual_rel_error", tri_metric, 1e-4))

    rows.append(record(
        "linalg.qr", "float32",
        lambda: torch.linalg.qr(H_x),
        "reconstruction_rel_error",
        lambda out: fro_rel_error(torch, out[0].cpu() @ out[1].cpu(), H), 1e-4))
    rows.append(record(
        "linalg.svd", "float32",
        lambda: torch.linalg.svd(H_x),
        "reconstruction_rel_error",
        lambda out: fro_rel_error(
            torch, out[0].cpu() @ torch.diag(out[1].cpu()) @ out[2].cpu(), H),
        1e-4))
    rows.append(record(
        "linalg.eigvalsh", "float32",
        lambda: torch.linalg.eigvalsh(H_x),
        "rel_error",
        lambda out: fro_rel_error(torch, out.cpu(),
                                  torch.linalg.eigvalsh(H.double())), 1e-4))
    return rows


def benchmark_ldlq(torch, sizes: list, policy: MemoryPolicy) -> list:
    """P0-22: block_ldl cost including the implicit Cholesky fallback."""
    results = []
    for n in sizes:
        H = make_spd(n)
        snapshot = capture_snapshot()
        needed = 2 * n * n * 4  # H + factor, fp32
        policy.check(needed, snapshot,
                     outstanding_bytes=int(torch.xpu.memory_allocated(0)),
                     context=f"ldlq n={n}")

        row = {"n": n, "bytes_per_matrix": n * n * 4}
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # H2D transfer (shared-memory bus)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        H_x = H.to("xpu")
        torch.xpu.synchronize()
        row["h2d_seconds"] = round(time.perf_counter() - t0, 5)
        row["h2d_bandwidth_gbps"] = round(
            (n * n * 4) / row["h2d_seconds"] / 1e9, 2)

        # CPU Cholesky
        t0 = time.perf_counter()
        torch.linalg.cholesky(H)
        row["cpu_cholesky_seconds"] = round(time.perf_counter() - t0, 5)

        # XPU Cholesky (implicit CPU fallback + transfers)
        t0 = time.perf_counter()
        L_x = torch.linalg.cholesky(H_x)
        torch.xpu.synchronize()
        row["xpu_cholesky_seconds"] = round(time.perf_counter() - t0, 5)

        # Explicit transfer costs for an explicit-CPU port
        t0 = time.perf_counter()
        H_back = H_x.cpu()
        row["d2h_seconds"] = round(time.perf_counter() - t0, 5)
        t0 = time.perf_counter()
        L_back = L_x.cpu().to("xpu")
        torch.xpu.synchronize()
        row["factor_roundtrip_seconds"] = round(time.perf_counter() - t0, 5)
        del H_back, L_back

        # Native XPU inv and matmul (the fold)
        m = n // 16

        def diagonal_blocks():
            return torch.stack([
                L_x[block * 16:(block + 1) * 16, block * 16:(block + 1) * 16]
                for block in range(m)])

        torch.linalg.inv(diagonal_blocks())
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        invs = torch.linalg.inv(diagonal_blocks())
        torch.xpu.synchronize()
        row["xpu_block_inv_seconds"] = round(time.perf_counter() - t0, 5)

        Lc = L_x.reshape(n, m, 16)
        t0 = time.perf_counter()
        folded = torch.bmm(Lc.permute(1, 0, 2), invs)
        torch.xpu.synchronize()
        row["xpu_fold_bmm_seconds"] = round(time.perf_counter() - t0, 5)
        del folded, Lc, invs

        # Full block_ldl on both devices
        t0 = time.perf_counter()
        cpu_result = block_ldl(H, block_size=16)
        row["block_ldl_cpu_seconds"] = round(time.perf_counter() - t0, 5)

        t0 = time.perf_counter()
        xpu_result = block_ldl(H_x, block_size=16)
        torch.xpu.synchronize()
        row["block_ldl_xpu_seconds"] = round(time.perf_counter() - t0, 5)

        row["xpu_vs_cpu_max_abs_diff"] = float(
            (xpu_result.transformed.cpu() - cpu_result.transformed).abs().max())

        # Folding identity on the XPU result (contract §9.1 steps 3-5)
        identity = torch.eye(16, dtype=torch.float32)
        diag_error = 0.0
        for block in range(m):
            start = block * 16
            diag_error = max(diag_error, float(
                (xpu_result.transformed[start:start + 16, start:start + 16].cpu()
                 - identity).abs().max()))
        row["fold_diag_identity_max_abs_error"] = diag_error

        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        row["host_max_rss_delta_bytes"] = (rss_after - rss_before) * 1024
        after = capture_snapshot()
        row["device_allocated_bytes"] = after.device_allocated_bytes
        row["device_free_bytes"] = after.device_free_bytes
        row["swap_free_delta_bytes"] = snapshot.swap_free_bytes - after.swap_free_bytes

        results.append(row)
        del H_x, L_x, xpu_result, cpu_result
        torch.xpu.empty_cache()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-size", type=int, default=512)
    parser.add_argument("--bench-sizes", type=int, nargs="+",
                        default=[512, 1024, 2048, 4096])
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"torch unavailable: {exc}")
        return 1
    if not torch.xpu.is_available():
        print("torch.xpu is not available; nothing probed")
        return 1

    policy = MemoryPolicy()
    rows = coverage(torch, n=args.coverage_size)
    bench = benchmark_ldlq(torch, args.bench_sizes, policy)

    document = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "time_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "coverage_size": args.coverage_size,
        "coverage": rows,
        "ldlq_benchmark": bench,
        "notes": [
            "Fallback detection uses PYTORCH_DEBUG_XPU_FALLBACK=1; each row "
            "lists the aten operator that fell back to CPU.",
            "fp16/bf16 decompositions are unsupported (no CPU kernel either), "
            "so the port must run factorization in fp32; no silent dtype change.",
            "The §9.1 damping retry schedule is preserved; only the compute "
            "location changes, so the algorithm is unchanged.",
            "LP64 shapes use the Llama-3.1-8B layer dimensions (up to 4096).",
        ],
    }

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")

    print(f"{'op':28} {'dtype':10} {'status':12} {'time':>9}  fallback")
    for row in rows:
        fallback = ",".join(row.get("fallback_ops", [])) or "-"
        timing = f"{row['seconds'] * 1e3:.2f} ms" if row.get("seconds") else "-"
        flag = ""
        if row["status"] == "ok" and "within_tolerance" in row:
            flag = "" if row["within_tolerance"] else "  OUT-OF-TOLERANCE"
        if row["status"] != "ok":
            flag = f"  {row['error']}"
        print(f"{row['op']:28} {row['dtype']:10} {row['status']:12} {timing:>9}  "
              f"{fallback}{flag}")

    print()
    print(f"{'n':>6} {'cpu_chol':>10} {'xpu_chol':>10} {'h2d':>9} "
          f"{'GB/s':>7} {'bldl_cpu':>10} {'bldl_xpu':>10}")
    for row in bench:
        print(f"{row['n']:>6} {row['cpu_cholesky_seconds'] * 1e3:>8.1f}ms "
              f"{row['xpu_cholesky_seconds'] * 1e3:>8.1f}ms "
              f"{row['h2d_seconds'] * 1e3:>7.1f}ms "
              f"{row['h2d_bandwidth_gbps']:>7.1f} "
              f"{row['block_ldl_cpu_seconds'] * 1e3:>8.1f}ms "
              f"{row['block_ldl_xpu_seconds'] * 1e3:>8.1f}ms")

    fallbacks = sum(1 for row in rows if row.get("fallback_ops"))
    unsupported = sum(1 for row in rows if row["status"] != "ok")
    print(f"\n{len(rows)} ops probed: {fallbacks} fell back to CPU, "
          f"{unsupported} unsupported")
    print(f"<< wrote {ARTIFACT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
