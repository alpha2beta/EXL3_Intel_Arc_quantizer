"""Build and exercise the minimal custom XPU extension probe (P0-18, P0-19).

The extension is built through the intended framework integration
(`torch.utils.cpp_extension.load` with `sycl_sources`), verified on the Arc
140V, and used to collect the O-6 fp16-FMA parity evidence that the kernel port
depends on.

Usage:
    PYTHONPATH=. python scripts/probe_xpu_extension.py   # runs everything
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "csrc" / "probe_xpu"
BUILD_DIR = REPO_ROOT / "artifacts" / "build" / "probe_xpu"

_ONEAPI_BIN = "/opt/intel/oneapi/compiler/latest/bin"

_EXT = None
_BUILD_SECONDS = 0.0


def _prepare_environment() -> None:
    """Make sure `icpx` is discoverable so cpp_extension can locate SYCL_HOME."""
    if os.path.isdir(_ONEAPI_BIN) and _ONEAPI_BIN not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ONEAPI_BIN + os.pathsep + os.environ.get("PATH", "")


def build_extension(verbose: bool = False):
    """Compile (cached) and import the probe extension on xpu:0."""
    global _EXT, _BUILD_SECONDS
    if _EXT is not None:
        return _EXT
    import torch
    from torch.utils import cpp_extension

    _prepare_environment()
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    _EXT = cpp_extension.load(
        name="probe_xpu_ext",
        # .sycl files alongside .cpp sources trigger the mixed SYCL build;
        # alternatively use load_inline(sycl_sources=...).
        sources=[str(SRC_DIR / "probe_xpu.cpp"), str(SRC_DIR / "probe_kernels.sycl")],
        build_directory=str(BUILD_DIR),
        extra_cflags=["-O2"],
        extra_sycl_cflags=["-O2"],
        verbose=verbose,
    )
    _BUILD_SECONDS = time.perf_counter() - t0
    return _EXT


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def verify_device_selection(ext) -> dict:
    """P0-18: the kernel must run on the tensor's device, through the current queue."""
    import torch

    torch.xpu.set_device(0)
    n = 1 << 12
    src = torch.arange(n, device="xpu", dtype=torch.float32)
    dst = torch.empty_like(src)
    ext.probe_vec_add(src, dst, 0.5)
    expected = (torch.arange(n, dtype=torch.float32) + 0.5)
    ok = bool(torch.equal(dst.cpu(), expected))
    return {
        "device_count": int(ext.probe_device_count()),
        "torch_device_name": torch.xpu.get_device_name(0),
        "sycl_queue_device_name": ext.probe_queue_device_name(0),
        "names_agree": ext.probe_queue_device_name(0) == torch.xpu.get_device_name(0),
        "vec_add_correct": ok,
    }


def verify_output_correctness(ext) -> dict:
    """P0-18: kernel results must match a torch/CPU reference bit for bit."""
    import torch

    rng = np.random.default_rng(seed=2026)
    n = 1 << 16
    src_np = rng.normal(0.0, 4.0, size=n).astype(np.float32)
    addend = 1.25
    src = torch.from_numpy(src_np).to("xpu")
    dst = torch.empty_like(src)
    ext.probe_vec_add(src, dst, addend)
    cpu_ref = (torch.from_numpy(src_np) + addend).numpy()
    dst_np = dst.cpu().numpy()
    exact = bool(np.array_equal(dst_np, cpu_ref))
    return {
        "n": n,
        "bit_exact_vs_cpu": exact,
        "max_abs_diff": float(np.max(np.abs(dst_np - cpu_ref))) if not exact else 0.0,
    }


def verify_queue_ordering(ext) -> dict:
    """P0-18: interleaved torch ops and custom kernels must observe queue order."""
    import torch

    n = 1 << 10
    t = torch.zeros(n, device="xpu", dtype=torch.float32)
    ext.probe_vec_add(t, t, 1.0)          # t = 1
    t.fill_(100.0)                        # torch op on the same queue -> 100
    ext.probe_vec_add(t, t, 5.0)          # kernel after torch op -> 105
    t.mul_(2.0)                           # torch op after kernel -> 210
    torch.xpu.synchronize()
    ok = bool(torch.all(t == 210.0).item())
    return {
        "interleaved_sequence_correct": ok,
        "restriction": "current torch XPU queue only; no secondary queues (P1-08)",
    }


def verify_tensor_lifetime(ext) -> dict:
    """P0-18: released tensors must not corrupt later allocations (no unsafe reuse)."""
    import torch

    free0, total = torch.xpu.mem_get_info(0)
    n = 1 << 22  # 64 MiB
    a = torch.arange(n, device="xpu", dtype=torch.float32)
    out = torch.empty_like(a)
    ext.probe_vec_add(a, out, 1.0)
    assert bool(torch.equal(out, a + 1.0))
    del a, out
    torch.xpu.empty_cache()
    b = torch.full((n,), 2.0, device="xpu", dtype=torch.float32)
    out2 = torch.empty_like(b)
    ext.probe_vec_add(b, out2, 1.0)
    ok = bool(torch.all(out2 == 3.0).item())
    del b, out2
    torch.xpu.empty_cache()
    free1, _ = torch.xpu.mem_get_info(0)
    return {
        "reuse_correct": ok,
        "mem_free_before_bytes": int(free0),
        "mem_free_after_bytes": int(free1),
        "total_bytes": int(total),
    }


def run_hfma_probe(ext, n: int = 1 << 16, seed: int = 2026) -> dict:
    """O-6: measure whether the XPU fp16 FMA matches exact-fp16 single rounding.

    Compares three computations on identical fp16 inputs:
      * device `sycl::fma(a, a, c)`                 (mode 0)
      * device plain expression `a * a + c`          (mode 1)
      * host emulation with an exact float64 FMA, rounded once to fp16
      * host emulation with the float32 double rounding (the old reference
        encoder behaviour)
    """
    import torch

    rng = np.random.default_rng(seed=seed)
    dh = rng.normal(0.0, 1.0, size=n).astype(np.float16)
    cost = (rng.random(n) * 8.0).astype(np.float16)

    dh_t = torch.from_numpy(dh).to("xpu")
    cost_t = torch.from_numpy(cost).to("xpu")
    out_fma = torch.empty_like(dh_t)
    out_expr = torch.empty_like(dh_t)
    ext.probe_hfma(dh_t, cost_t, out_fma, 0)
    ext.probe_hfma(dh_t, cost_t, out_expr, 1)

    dev_fma = out_fma.cpu().numpy()
    dev_expr = out_expr.cpu().numpy()

    exact = (dh.astype(np.float64) * dh.astype(np.float64)
             + cost.astype(np.float64)).astype(np.float16)
    v32 = dh.astype(np.float32) ** 2
    v32 = v32 + cost.astype(np.float32)
    emul32 = v32.astype(np.float16)

    mism_fma = int(np.count_nonzero(dev_fma != exact))
    mism_expr = int(np.count_nonzero(dev_expr != exact))
    mism_emul = int(np.count_nonzero(emul32 != exact))
    fma_vs_expr = int(np.count_nonzero(dev_fma != dev_expr))

    def first_mismatch(dev, ref):
        idx = np.flatnonzero(dev != ref)
        if not idx.size:
            return None
        i = int(idx[0])
        return {"index": i, "dh": float(dh[i]), "cost": float(cost[i]),
                "device": float(dev[i]), "exact": float(exact[i])}

    return {
        "n": n,
        "seed": seed,
        "fma_vs_exact_mismatches": mism_fma,
        "expr_vs_exact_mismatches": mism_expr,
        "fma_vs_expr_mismatches": fma_vs_expr,
        "host_fp32_emul_vs_exact_mismatches": mism_emul,
        "first_fma_mismatch": first_mismatch(dev_fma, exact),
        "first_expr_mismatch": first_mismatch(dev_expr, exact),
        "first_emul_mismatch": first_mismatch(emul32, exact),
    }


def run_hfma_adversarial_probe(ext) -> dict:
    """O-6: adversarial pairs where fp32 double rounding != exact fp16 rounding.

    Discovered by an exhaustive structured grid search (dh mantissa patterns
    0x3C00..0x3CFF, tiny fp16 costs spanning 2^-25..2^-11). Random sampling
    does not hit these cases, so they are the only inputs that actually
    discriminate fused-single-rounding from fp32-double-rounding semantics.
    """
    import torch

    dh_bits = [0x3C16, 0x3C27, 0x3C32, 0x3C5D, 0x3C71, 0x3C9B,
               0x3CC9, 0x3CCE, 0x3CD3, 0x3CE1, 0x3CEA]
    cost_bits = [0x01C1, 0x00EF, 0x03C1, 0x0371, 0x01F1, 0x026F,
                 0x02EF, 0x03BF, 0x016F, 0x03EF, 0x01BF]
    dh = np.array(dh_bits, dtype=np.uint16).view(np.float16)
    cost = np.array(cost_bits, dtype=np.uint16).view(np.float16)

    exact = (dh.astype(np.float64) ** 2 + cost.astype(np.float64)
             ).astype(np.float16)
    emul32 = (dh.astype(np.float32) ** 2
              + cost.astype(np.float32)).astype(np.float16)
    emul_differs = int(np.count_nonzero(emul32 != exact))
    if emul_differs == 0:
        # The adversarial set must discriminate; regenerate from the grid
        # search if the hard-coded table ever goes stale.
        return {"error": "adversarial set no longer discriminates"}

    dh_t = torch.from_numpy(dh).to("xpu")
    cost_t = torch.from_numpy(cost).to("xpu")
    out_fma = torch.empty_like(dh_t)
    out_expr = torch.empty_like(dh_t)
    ext.probe_hfma(dh_t, cost_t, out_fma, 0)
    ext.probe_hfma(dh_t, cost_t, out_expr, 1)
    dev_fma = out_fma.cpu().numpy()
    dev_expr = out_expr.cpu().numpy()

    fma_matches_exact = int(np.count_nonzero(dev_fma == exact))
    expr_matches_exact = int(np.count_nonzero(dev_expr == exact))
    fma_matches_emul = int(np.count_nonzero(dev_fma == emul32))
    cases = []
    for i in range(dh.size):
        cases.append({
            "dh_bits": int(dh_bits[i]), "cost_bits": int(cost_bits[i]),
            "dh": float(dh[i]), "cost": float(cost[i]),
            "exact": float(exact[i]), "fp32_emul": float(emul32[i]),
            "device_fma": float(dev_fma[i]), "device_expr": float(dev_expr[i]),
        })
    return {
        "n": int(dh.size),
        "fp32_emul_vs_exact_mismatches": emul_differs,
        "fma_matches_exact": fma_matches_exact,
        "fma_matches_fp32_emul": fma_matches_emul,
        "expr_matches_exact": expr_matches_exact,
        "expr_matches_fp32_emul": int(np.count_nonzero(dev_expr == emul32)),
        "conclusion": (
            "device fma == exact single rounding"
            if fma_matches_exact == emul_differs and fma_matches_emul == 0
            else "device fma == fp32 double rounding"
            if fma_matches_emul == emul_differs and fma_matches_exact == 0
            else "mixed/other semantics"),
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# P0-19 capability capture
# ---------------------------------------------------------------------------


def capture_environment() -> dict:
    """Save torch.xpu and SYCL-side device capabilities (P0-19)."""
    import torch

    ext = _EXT
    props = torch.xpu.get_device_properties(0)
    free, total = torch.xpu.mem_get_info(0)
    caps = {
        "torch": {
            "version": torch.__version__,
            "device_name": props.name,
            "platform_name": props.platform_name,
            "driver_version": props.driver_version,
            "max_compute_units": props.max_compute_units,
            "gpu_eu_count": props.gpu_eu_count,
            "gpu_subslice_count": props.gpu_subslice_count,
            "sub_group_sizes": list(props.sub_group_sizes),
            "max_work_group_size": props.max_work_group_size,
            "max_num_sub_groups": props.max_num_sub_groups,
            "local_mem_size": props.local_mem_size,
            "last_level_cache_size": props.last_level_cache_size,
            "total_memory": props.total_memory,
            "mem_get_info_free": int(free),
            "mem_get_info_total": int(total),
            "is_integrated_gpu": bool(props.is_integrated_gpu),
            "has_fp16": bool(props.has_fp16),
            "has_fp64": bool(props.has_fp64),
            "has_atomic64": bool(props.has_atomic64),
            "has_subgroup_matrix_multiply_accumulate":
                bool(props.has_subgroup_matrix_multiply_accumulate),
            "memory_bus_width": props.memory_bus_width,
            "memory_clock_rate": props.memory_clock_rate,
            "runtime_version": props.version,
        },
        "sycl_via_probe_extension": dict(ext.probe_device_caps(0)),
        "note": "multi_processor_count does not exist on torch.xpu; SM analogue "
                "candidate is gpu_subslice_count (decision R11)",
    }
    return caps


def run_all(out_path: Path | None = None) -> dict:
    """Run every probe and return the results dict (optionally as JSON)."""
    results = {}
    torch_ok = False
    try:
        import torch
        torch_ok = torch.xpu.is_available()
    except Exception:
        pass
    results["xpu_available"] = torch_ok
    if not torch_ok:
        return results

    ext = build_extension()
    results["build_seconds"] = round(_BUILD_SECONDS, 2)
    results["device_selection"] = verify_device_selection(ext)
    results["output_correctness"] = verify_output_correctness(ext)
    results["queue_ordering"] = verify_queue_ordering(ext)
    results["tensor_lifetime"] = verify_tensor_lifetime(ext)
    results["hfma_o6_probe"] = run_hfma_probe(ext)
    results["hfma_o6_adversarial"] = run_hfma_adversarial_probe(ext)
    results["capabilities"] = capture_environment()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    return results
