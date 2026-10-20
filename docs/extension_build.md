# Building and Testing the XPU Extension (P0-23)

Reproducible procedure for the custom SYCL extension in `csrc/probe_xpu/`
(`P0-18`), plus the memory-probe scripts (`V140-02`, `P0-20`).

## 1. Prerequisites

- Intel oneAPI DPC++ compiler (`icpx`), verified with 2026.0.0.
- The project's PyTorch XPU build (verified with `2.14.0+xpu`).
- `icpx` on `PATH` — `torch.utils.cpp_extension` locates `SYCL_HOME` from it:

```bash
export PATH=/opt/intel/oneapi/compiler/latest/bin:$PATH
```

Using `source /opt/intel/oneapi/setvars.sh` also works but is not required.

## 2. Build and run the extension probe

```bash
PYTHONPATH=. python scripts/probe_xpu_extension.py
```

The first run compiles the mixed C++/SYCL module (about 10 s) into
`artifacts/build/probe_xpu/` (git-ignored). Subsequent runs reuse the cached
build. The script writes:

- `artifacts/environment/device_capabilities.json` (P0-19)
- `artifacts/environment/extension_probe.json` (P0-18 + O-6 evidence)

### Clean rebuild (verified)

```bash
rm -rf artifacts/build/probe_xpu
PYTHONPATH=. python scripts/probe_xpu_extension.py
```

## 3. Environment smoke test

```bash
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py"
```

`tests/test_xpu_extension.py` skips cleanly when `torch.xpu` is unavailable and
otherwise verifies device selection, bit-exact output, queue ordering, tensor
lifetime, capability capture, and both O-6 fp16-FMA probes.

## 4. Memory baseline and scratch allocations

```bash
PYTHONPATH=. python scripts/measure_idle_baseline.py --samples 10 --interval 1.0
PYTHONPATH=. python scripts/measure_scratch_allocations.py --reserve-gib 6
```

Outputs `artifacts/environment/idle_baseline.json` and
`artifacts/environment/scratch_allocations.json`. Every allocation is admitted
by `exl3_quant/memory_policy.py`, which keeps the configured reserve intact and
refuses requests that would breach it (abort path is exercised by the
oversized-request probe).

## 5. Known build constraints

- **Named kernel classes are mandatory.** torch's extension build adds
  `-fsycl-host-compiler`, which conflicts with `-fsycl-unnamed-lambda`.
  Kernels therefore use `h.parallel_for<class ProbeVecAdd>(...)`.
- **Include `<c10/util/Half.h>` explicitly** in the `.sycl` translation unit;
  the auto-inserted `torch/types.h` does not expose `c10::Half`.
- **Use `torch::xpu::device_count()`**, not `c10::xpu::device_count()`, in the
  C++ binding file (the `torch/xpu.h` facade is what the extension includes).
- **Single queue:** all probe kernels run on the current torch XPU stream
  (`c10::xpu::getCurrentXPUStream(device).queue()`); no secondary queues are
  created (P1-08 restriction).

## 6. Artifact inventory

| File | Purpose |
|---|---|
| `artifacts/environment/device_capabilities.json` | torch + SYCL device properties, memory limits |
| `artifacts/environment/extension_probe.json` | P0-18 verification results, O-6 random + adversarial fp16-FMA measurements |
| `artifacts/environment/idle_baseline.json` | OS + XPU idle memory baseline samples |
| `artifacts/environment/scratch_allocations.json` | allocator tests at the 6 GiB reserve |
| `artifacts/environment/scratch_allocations_reserve4.json` | capacity increase at a 4 GiB reserve |
