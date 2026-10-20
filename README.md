# EXL3 Intel Arc Quantizer

An implementation and tooling suite for porting the **EXL3 quantization and model conversion pipeline** from NVIDIA CUDA to Intel Arc GPUs (Xe-HPG / Xe2-HPG / Xe-HPC) using SYCL, oneAPI, and PyTorch XPU.

## Project Scope

This project focuses on the **offline quantization search and model conversion**:
1. **Activation Calibration & Hessian Accumulation:** Harvesting layer-wise covariance matrices ($H = \sum X^T X$) on Intel Arc via native PyTorch XPU.
2. **Numerical Preprocessing:** Damping and Cholesky decomposition via `oneapi::mkl::lapack::potrf`.
3. **Incoherence Processing:** Fast Walsh-Hadamard Transform (FWHT) custom SYCL kernel.
4. **Trellis Coded Quantization (TCQ):** SIMD16-adapted fused Viterbi search kernel for Add-Compare-Select (ACS) optimization.
5. **Predecessor & Scratch Management:** Column-chunked batch tiling to fit within 12 GB / 16 GB VRAM budgets.
6. **Checkpoint Export:** Packing into bit-exact `.safetensors` compatible with both upstream ExLlamaV3 and downstream Intel Arc inference runtimes like `exl3xpu`.

## Architecture & Feasibility Documentation

For the full technical evaluation, capacity calculations, kernel breakdown, and 14-week engineering roadmap, refer to:
- [`EXL3_Intel_Arc_quantizer.md`](./EXL3_Intel_Arc_quantizer.md)
