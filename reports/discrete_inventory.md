# Discrete-Arc Environment and Device Inventory (Phase A / P3-15)

> **Document ID:** `reports/discrete_inventory.md`  
> **Target Boards:** Intel Arc Pro B60 (24 GB GDDR6) & Intel Arc B580 (12 GB GDDR6)  
> **Timestamp:** `2026-10-22 10:55:00 +02:00`  
> **Gate A Status:** PASSED (Inventory recorded, devices enumerated, memory semantics proven, PCIe baseline established)

---

## 1. Host System and Toolchain Specification

| Component | Queried Specification | Query Provenance |
|---|---|---|
| **Host System** | `UideaCentre` | `uname -n` |
| **Kernel** | Linux `7.0.0-38-generic #38-Ubuntu SMP PREEMPT_DYNAMIC` x86_64 | `uname -a` |
| **Operating System** | Ubuntu 26.04.1 LTS (Resolute Raccoon) | `/etc/os-release` |
| **CPU Model** | 12th Gen Intel(R) Core(TM) i5-12400F | `lscpu` (6 cores / 12 threads) |
| **System Memory** | 45 GiB total RAM, ~20.5 GiB available, 4.0 GiB swap | `free -h` / `/proc/meminfo` |
| **Kernel Driver** | `xe` (Intel Xe DRM kernel driver) | `lspci -k -d ::0300` |
| **User-mode Runtime** | Intel Level Zero Driver `1.15.39122+11`, OpenCL NEO `26.05.037020` | `xpu-smi` / `sycl-ls` |
| **DPC++/SYCL Compiler** | Intel oneAPI DPC++/C++ Compiler `2026.2.0` (`/opt/intel/oneapi/compiler/2026.2/bin/icpx`) & `2026.1.1` | `icpx --version` |
| **PyTorch XPU Stack** | PyTorch `2.14.1+xpu` / `2.13.0+xpu`, Triton `3.8.0+xpu` | `torch.xpu.is_available() == True` |

---

## 2. Discrete Device Inventory

### Device 0: Intel Arc Pro B60

| Attribute | Probed Value | Source |
|---|---|---|
| **Marketing Name** | Intel(R) Arc(TM) Pro B60 Graphics | `torch.xpu.get_device_name(0)` / `xpu-smi` |
| **PCI BDF & Device ID** | `0000:03:00.0` [8086:e211] (rev 03) | `lspci -nn` |
| **Subsystem** | Sparkle Computer Co., Ltd. [172f:0300] | `lspci -nn` |
| **Total VRAM** | 25,669,140,480 B (~24,480 MiB / 23.91 GiB) | `torch.xpu.get_device_properties(0).total_memory` |
| **Memory Bus & Clock** | 64-bit bus, 2400 MHz clock rate | `get_device_properties(0).memory_clock_rate` |
| **Execution Units (EUs)** | 160 Vector Engines / 20 Xe-cores (subslices) | `get_device_properties(0).gpu_eu_count` |
| **Local Memory (SLM)** | 131,072 B (128 KiB) per work-group | `get_device_properties(0).local_mem_size` |
| **Last-Level Cache (L3)** | 18,874,368 B (18 MiB) | `get_device_properties(0).last_level_cache_size` |
| **Subgroup Sizes** | SIMD16, SIMD32 (`[16, 32]`) | `get_device_properties(0).sub_group_sizes` |
| **Max Work-Group Size** | 1024 work-items | `get_device_properties(0).max_work_group_size` |
| **Device Classification** | `is_integrated_gpu=False` (Dedicated VRAM) | `get_device_properties(0).is_integrated_gpu` |
| **DRM Character Nodes** | `/dev/dri/card0`, `/dev/dri/renderD128` | Sysfs / `xpu-smi` |
| **Idling Temperature** | GPU Core: 41.0 °C, Memory: 42.0 °C | `xpu-smi dump --device 0` |
| **Idling Power Draw** | 32.18 W | `xpu-smi dump --device 0` |
| **Max Graphics Clock** | 2400 MHz (current idle: 1200 MHz) | `xpu-smi dump --device 0` |
| **RAS / Error Status** | 0 resets, 0 driver errors, 0 cache errors | `xpu-smi dump --device 0` |

### Device 1: Intel Arc B580

| Attribute | Probed Value | Source |
|---|---|---|
| **Marketing Name** | Intel(R) Arc(TM) B580 Graphics | `torch.xpu.get_device_name(1)` / `xpu-smi` |
| **PCI BDF & Device ID** | `0000:0b:00.0` [8086:e20b] (rev 00) | `lspci -nn` |
| **Subsystem** | Sparkle Computer Co., Ltd. [172f:4273] | `lspci -nn` |
| **Total VRAM** | 12,809,404,416 B (~12,216 MiB / 11.93 GiB) | `torch.xpu.get_device_properties(1).total_memory` |
| **Memory Bus & Clock** | 64-bit bus, 2850 MHz clock rate | `get_device_properties(1).memory_clock_rate` |
| **Execution Units (EUs)** | 160 Vector Engines / 20 Xe-cores (subslices) | `get_device_properties(1).gpu_eu_count` |
| **Local Memory (SLM)** | 131,072 B (128 KiB) per work-group | `get_device_properties(1).local_mem_size` |
| **Last-Level Cache (L3)** | 18,874,368 B (18 MiB) | `get_device_properties(1).last_level_cache_size` |
| **Subgroup Sizes** | SIMD16, SIMD32 (`[16, 32]`) | `get_device_properties(1).sub_group_sizes` |
| **Max Work-Group Size** | 1024 work-items | `get_device_properties(1).max_work_group_size` |
| **Device Classification** | `is_integrated_gpu=False` (Dedicated VRAM) | `get_device_properties(1).is_integrated_gpu` |
| **DRM Character Nodes** | `/dev/dri/card1`, `/dev/dri/renderD129` | Sysfs / `xpu-smi` |
| **Idling Temperature** | GPU Core: 46.0 °C, Memory: 66.0 °C | `xpu-smi dump --device 1` |
| **Idling Power Draw** | 33.77 W | `xpu-smi dump --device 1` |
| **Max Graphics Clock** | 2850 MHz (current idle: 1200 MHz) | `xpu-smi dump --device 1` |
| **RAS / Error Status** | 0 resets, 0 driver errors, 0 cache errors | `xpu-smi dump --device 1` |

---

## 3. PCIe Topology and Health Baseline

- **B60 PCIe Link:** `2.5 GT/s PCIe x 1` (IOMMU group 19)
- **B580 PCIe Link:** `2.5 GT/s PCIe x 1` (IOMMU group 23)
- **Error Counters Baseline:**
  - `ras.reset`: 0
  - `ras.programming.errors`: 0
  - `ras.driver.errors`: 0
  - `ras.cache.errors.correctable`: 0
  - `ras.cache.errors.uncorrectable`: 0
  - AER counters: no uncorrected errors reported.
- **Topology Isolation Note:**
  The kernel log reports:
  ```text
  xe 0000:03:00.0: cannot be used for peer-to-peer DMA as the client and provider (0000:0b:00.0) do not share an upstream bridge or whitelisted host bridge
  ```
  Peer-to-peer (P2P) transfers between the two cards are physically non-functional on this motherboard topology. As learned during earlier GPTQ runs, attempting multi-card concurrent probes triggers Level Zero P2P enumeration faults and GuC timeouts. **Each qualification phase MUST target cards individually via device masking (`ZE_AFFINITY_MASK` / `device` argument) with the other card idle.**

---

## 4. Empirical Memory Dynamics and Counter Regime Decision

To resolve Open Question 2 from `B60_B580.md`, 4.0 GiB allocations were executed on each discrete board:

| Metric | Device 0: Arc Pro B60 (24 GB) | Device 1: Arc B580 (12 GB) |
|---|---|---|
| `dev_total` | 23.91 GiB | 11.93 GiB |
| `dev_free` (before allocation) | 23.629 GiB | 11.737 GiB |
| `sys_avail` (before allocation) | 20.541 GiB | 20.495 GiB |
| **4.0 GiB XPU Allocated** | | |
| `dev_free` (after allocation) | **19.629 GiB (-4.000 GiB)** | **7.737 GiB (-4.000 GiB)** |
| `sys_avail` (after allocation) | **20.569 GiB (delta: +0.028 GiB)** | **20.487 GiB (delta: -0.008 GiB)** |
| **Freed + `empty_cache()`** | | |
| `dev_free` (restored) | 23.629 GiB | 11.737 GiB |

### Conclusions:
1. **True Discrete VRAM Independence:** Unlike the Arc 140V UMA architecture where the host page cache starved `dev_free`, discrete VRAM allocations are completely isolated from host memory (`sys_avail`). The host cache watchdog is therefore **not needed** on these discrete boards.
2. **Counter Regime Selection:** Because `dev_free` drops immediately and bit-for-bit by 4.000 GiB when 4.0 GiB is allocated, the Level Zero driver counters **do reflect live allocations**. Therefore, `EXL3_MEMORY_REFLECT_OUTSTANDING=1` must be used so that `MemoryPolicy` does not double-count outstanding commitments.
3. **Admission Reserve Policy:**
   - On **Arc B580 (12 GB)**: With `dev_total = 11.93 GiB` and working set ~2.5–4.6 GiB, a reserve of **4.0 GiB** guarantees safety without starving the scratch arena.
   - On **Arc Pro B60 (24 GB)**: With `dev_total = 23.91 GiB`, the default **6.0 GiB** reserve has immense headroom (>17 GiB free).
