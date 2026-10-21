# Building and Testing the XPU Extensions (P0-23)

Reproducible procedures for the two custom SYCL modules:

- `csrc/probe_xpu/` — environment/parity probe (P0-18, O-6).
- `csrc/quantize_xpu/` — production trellis tile quantizer (Phase 1).

plus the memory-probe scripts (`V140-02`, `P0-20`).

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

## 3. Production Tile Quantizer Extension (`csrc/quantize_xpu/`)

The primary production extension implementing Phase 1 Trellis Viterbi search and circular traceback:

### 3.1 Source Components & Kernel Inventory

- [`csrc/quantize_xpu/quantize_xpu.cpp`](../csrc/quantize_xpu/quantize_xpu.cpp): PyBind11 C++ entry points, parameter validation, queue bindings, and tensor lifetime management.
- [`csrc/quantize_xpu/quantize_kernels.sycl`](../csrc/quantize_xpu/quantize_kernels.sycl): Native SYCL 2020 kernel classes templated on subgroup size (`SG ∈ {16, 32}`) and cost storage mode (`UseSLM ∈ {true, false}`):
  - `FusedQuantizeKernel<SG, UseSLM>`: Two-pass tail-biting forward ACS dynamic programming and circular traceback fused into a single submission.
  - `ForwardOnlyKernel<SG, UseSLM>`: Standalone forward ACS pass for split and CPU/USM traceback pipelines.
  - `BackwardOnlyKernel`: Standalone GPU-resident circular traceback pass.
  - `DecodeCodebookKernel`: Direct GPU procedural codebook lookup table generation.

### 3.2 Compilation and Build Lifecycle

Compiled via `torch.utils.cpp_extension.load` encapsulated in [`exl3_quant.xpu_quantizer.build_quantize_extension`](../exl3_quant/xpu_quantizer.py):

```python
from exl3_quant.xpu_quantizer import build_quantize_extension

# Force a clean cold rebuild (~13.8 s) into artifacts/build/quantize_xpu/
ext = build_quantize_extension(verbose=True, force_rebuild=True)
```

- **Cold Build Duration:** ~13.8 s (measured; re-measure with `scripts/benchmark_phase1_encoder.py --cold-build`).
- **Warm Cached Load:** ~21 ms.
- **Target Architecture:** Intel Xe2-LPG (`lnl-m`, Device ID `0x64A0`).
- **Compiler Flags:** flags are injected by torch's SYCL build; the generated
  `artifacts/build/quantize_xpu/build.ninja` shows
  `-O2 -fsycl -fsycl-targets=spir64_gen,spir64` for compilation and
  `-fsycl-link --offload-compress -Xs "-device pvc,bmg,arl-h,mtl-h,lnl-m,ptl-h,ptl-u"`
  for the device-link step. Do not assume a single `-device lnl-m` flag.

### 3.3 Shader Dump and ISA Inspection Procedure

To inspect register allocation (128 GRF vs 256 GRF), spills, and SLM usage on Lunar Lake:

```bash
mkdir -p artifacts/isa_dump
IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=artifacts/isa_dump \
  python -c "from exl3_quant.xpu_quantizer import build_quantize_extension; build_quantize_extension(force_rebuild=True)"
```

IGC writes one `OCL_asm*_simd{16,32}_entry_*.asm` file per kernel instantiation.
These are already disassembled and carry the resource headers used by the
benchmark (`//.thread_config numGRF=..., numAcc=...`,
`//.instCount`, `//.spill flag store/load`). Parse those headers directly;
no separate `ocloc disasm` invocation is needed.

> The SPIR-V is bundled inside `quantize_kernels.sycl.o`; there is no
> `quantize_kernels.spv` in the build directory, so `ocloc disasm -file
> artifacts/build/quantize_xpu/quantize_kernels.spv` does **not** work as
> written. Use the IGC dump above.

Inspected IGC metrics for `FusedQuantizeKernel<32, true>`:
- 128 GRF mode, 0 bytes of GRF spill (abort-on-spill passed), 16 EU threads per workgroup.
- SLM allocation: 66.4 KiB ($K=2$), 33.6 KiB ($K=3$), 17.2 KiB ($K=4$).

## 4. Environment smoke test

```bash
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py"
```

`tests/test_xpu_extension.py` skips cleanly when `torch.xpu` is unavailable and
otherwise verifies device selection, bit-exact output, queue ordering, tensor
lifetime, capability capture, and both O-6 fp16-FMA probes.

### CPU-only hosts

`tests/test_gss.py` contains the global-scale search tests. Each search evaluates
`COARSE_POINTS` scales over every third tile and `FINE_POINTS` scales over every
tile, i.e. `O(15 * tiles)` full trellis encodes. The CPU reference encoder needs
roughly 0.8 s per tile, so at the original sample sizes the module needs about
5.5 h on a host without an XPU device.

On such a host that module is therefore **skipped** by default and its sample
sizes are reduced, so the suite stays usable:

```bash
# default on a CPU-only host: test_gss reports a module-level skip
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py"

# force it to run at the reduced sizes (~40-60 min on a CPU-only host)
EXL3_RUN_SLOW_TESTS=1 PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py"
```

With an XPU device present the module always runs, at the original sample sizes.

## 5. Memory baseline and scratch allocations

```bash
PYTHONPATH=. python scripts/measure_idle_baseline.py --samples 10 --interval 1.0
PYTHONPATH=. python scripts/measure_scratch_allocations.py --reserve-gib 6
```

Outputs `artifacts/environment/idle_baseline.json` and
`artifacts/environment/scratch_allocations.json`. Every allocation is admitted
by `exl3_quant/memory_policy.py` (pre-allocation reserve check plus
allocator-growth verification after commit) and refuses requests that would
breach the reserve.

## 6. Known build constraints

- **Named kernel classes are mandatory.** torch's extension build adds
  `-fsycl-host-compiler`, which conflicts with `-fsycl-unnamed-lambda`.
  Kernels therefore use `h.parallel_for<class ProbeVecAdd>(...)`.
- **Include `<c10/util/Half.h>` explicitly** in the `.sycl` translation unit;
  the auto-inserted `torch/types.h` does not expose `c10::Half`.
- **Use `torch::xpu::device_count()`**, not `c10::xpu::device_count()`, in the
  C++ binding file (the `torch/xpu.h` facade is what the extension includes).
- **Single queue:** all kernels run on the current torch XPU stream
  (`c10::xpu::getCurrentXPUStream(device).queue()`); no secondary queues are
  created (P1-08 restriction).

## 7. Artifact inventory

| File | Purpose |
|---|---|
| `artifacts/environment/device_capabilities.json` | torch + SYCL device properties, memory limits |
| `artifacts/environment/extension_probe.json` | P0-18 verification results, O-6 random + adversarial fp16-FMA measurements |
| `artifacts/environment/idle_baseline.json` | OS + XPU idle memory baseline samples |
| `artifacts/environment/scratch_allocations.json` | allocator tests at the 6 GiB reserve |
| `artifacts/environment/scratch_allocations_reserve4.json` | capacity increase at a 4 GiB reserve |
| `artifacts/benchmarks/phase1_encoder_bench.json` | Phase 1 encoder, traceback, batch, ISA and telemetry results |
