// Minimal custom XPU extension probe (P0-18).
//
// Built and loaded through the intended framework integration:
//   torch.utils.cpp_extension.load(sycl_sources=[<probe_kernels.sycl text>],
//                                  sources=[probe_xpu.cpp])
//
// Scope: device selection, output correctness, queue ordering, and tensor
// lifetime verification on the Arc 140V, plus an fp16 FMA parity probe for
// open item O-6 and SYCL-side device capability capture for P0-19.
//
// Deliberate restriction (P1-08): every probe runs on the *current* torch XPU
// stream/queue. No secondary queue is created anywhere in this file.

#include <torch/extension.h>

#include <string>
#include <vector>

void probe_vec_add_impl(const float* src, float* dst, double addend, int64_t n,
                        int64_t device_index);
void probe_hfma_impl(const at::Half* dh, const at::Half* cost, at::Half* out,
                     int64_t mode, int64_t n, int64_t device_index);
std::string probe_queue_device_name_impl(int64_t device_index);
std::vector<int64_t> probe_device_caps_impl(int64_t device_index);

namespace {

int64_t xpu_device_index(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_xpu(), name, " must be an XPU tensor");
  TORCH_CHECK(t.device().is_xpu() && t.device().index() >= 0, name,
              " must carry a concrete XPU device index");
  return t.device().index();
}

}  // namespace

void probe_vec_add(at::Tensor src, at::Tensor dst, double addend) {
  const int64_t dev = xpu_device_index(src, "src");
  TORCH_CHECK(xpu_device_index(dst, "dst") == dev, "src and dst must share a device");
  TORCH_CHECK(src.scalar_type() == at::kFloat, "src must be float32");
  TORCH_CHECK(dst.scalar_type() == at::kFloat, "dst must be float32");
  TORCH_CHECK(src.numel() == dst.numel(), "src and dst must have equal numel");
  probe_vec_add_impl(src.const_data_ptr<float>(), dst.mutable_data_ptr<float>(),
                     addend, src.numel(), dev);
}

void probe_hfma(at::Tensor dh, at::Tensor cost, at::Tensor out, int64_t mode) {
  const int64_t dev = xpu_device_index(dh, "dh");
  TORCH_CHECK(xpu_device_index(cost, "cost") == dev,
              "dh and cost must share a device");
  TORCH_CHECK(xpu_device_index(out, "out") == dev,
              "dh and out must share a device");
  TORCH_CHECK(dh.scalar_type() == at::kHalf, "dh must be float16");
  TORCH_CHECK(cost.scalar_type() == at::kHalf, "cost must be float16");
  TORCH_CHECK(out.scalar_type() == at::kHalf, "out must be float16");
  TORCH_CHECK(dh.numel() == cost.numel() && dh.numel() == out.numel(),
              "probe_hfma tensors must have equal numel");
  TORCH_CHECK(mode == 0 || mode == 1,
              "mode must be 0 (sycl::fma) or 1 (plain a*a+c expression)");
  probe_hfma_impl(dh.const_data_ptr<at::Half>(), cost.const_data_ptr<at::Half>(),
                  out.mutable_data_ptr<at::Half>(), mode, dh.numel(), dev);
}

std::string probe_queue_device_name(int64_t device_index) {
  return probe_queue_device_name_impl(device_index);
}

int64_t probe_device_count() {
  return static_cast<int64_t>(torch::xpu::device_count());
}

py::dict probe_device_caps(int64_t device_index) {
  const std::vector<int64_t> caps = probe_device_caps_impl(device_index);
  TORCH_CHECK(caps.size() == 5, "probe_device_caps_impl returned wrong arity");
  py::dict out;
  out["sycl_global_mem_size"] = caps[0];
  out["sycl_max_mem_alloc_size"] = caps[1];
  out["sycl_local_mem_size"] = caps[2];
  out["sycl_max_work_group_size"] = caps[3];
  out["sycl_max_sub_group_size"] = caps[4];
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("probe_vec_add", &probe_vec_add,
        "dst[i] = src[i] + addend on the current torch XPU queue");
  m.def("probe_hfma", &probe_hfma,
        "out[i] = fma(dh[i], dh[i], cost[i]) in fp16; mode 0 = sycl::fma, "
        "mode 1 = plain expression");
  m.def("probe_queue_device_name", &probe_queue_device_name,
        "SYCL device name of the current torch XPU queue");
  m.def("probe_device_count", &probe_device_count,
        "number of XPU devices visible to the torch XPU context");
  m.def("probe_device_caps", &probe_device_caps,
        "SYCL-side device capabilities (P0-19)");
}
