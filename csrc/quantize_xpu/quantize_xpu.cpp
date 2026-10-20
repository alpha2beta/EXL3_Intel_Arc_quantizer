// C++ / pybind11 entry points for EXL3 XPU tile quantization
// (P1-03..P1-08, P1-17..P1-19).

#include <torch/extension.h>

#include <cstdint>
#include <vector>

void decode_codebook_impl(int64_t cb, at::Half* out, int64_t device_index);

void quantize_tiles_fused_impl(
    const float* input_tiles,
    float* output_tiles,
    int16_t* output_indices,
    at::Half* temp_costs,
    int16_t* temp_edges,
    int64_t num_tiles,
    int64_t batch_cap,
    int64_t K,
    int64_t cb,
    int64_t L,
    int64_t sg_size,
    bool force_global_costs,
    int64_t wave_cap_override,
    int64_t device_index);

void quantize_tiles_forward_impl(
    const float* input_tiles,
    at::Half* temp_costs,
    int16_t* temp_edges,
    const int32_t* pre_states,
    int64_t num_tiles,
    int64_t K,
    int64_t cb,
    int64_t L,
    int64_t roll,
    int64_t cost_slot,
    int64_t sg_size,
    bool force_global_costs,
    int64_t device_index);

void quantize_tiles_backward_impl(
    const at::Half* temp_costs,
    const int16_t* temp_edges,
    float* output_tiles,
    int16_t* output_indices,
    int32_t* end_states,
    int64_t num_tiles,
    int64_t K,
    int64_t cb,
    int64_t L,
    int64_t roll,
    int64_t cost_slot,
    bool write_output,
    int64_t sg_size,
    int64_t device_index);

std::vector<double> quantize_tiles_usm_host_impl(
    const float* input_tiles,
    at::Half* temp_costs,
    float* output_tiles_host,
    int16_t* output_indices_host,
    const float* lut_host,
    int64_t num_tiles,
    int64_t K,
    int64_t cb,
    int64_t L,
    int64_t sg_size,
    int64_t device_index);

namespace {

int64_t check_xpu_tensor(const at::Tensor& t, const char* name,
                         at::ScalarType expected_dtype, int64_t expected_ndim) {
  TORCH_CHECK(t.is_xpu(), name, " must be an XPU tensor");
  TORCH_CHECK(t.device().is_xpu() && t.device().index() >= 0, name,
              " must have a valid XPU device index");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == expected_dtype, name, " has wrong dtype");
  TORCH_CHECK(t.dim() == expected_ndim, name, " must be ", expected_ndim,
              "-D, got ", t.dim(), "-D");
  return t.device().index();
}

int64_t resolve_cb(bool mcg, bool mul1) {
  TORCH_CHECK(!(mcg && mul1),
              "mcg and mul1 cannot both be true (ambiguous codebook)");
  if (mul1) return 2;
  if (mcg) return 1;
  return 0;
}

void validate_common_params(int64_t K, int64_t cb, int64_t L, int64_t sg_size) {
  TORCH_CHECK(K >= 1 && K <= 8, "K must be in 1..8, got ", K);
  TORCH_CHECK(L == 256 || L == 160,
              "tile length L must be 256 or 160, got ", L);
  TORCH_CHECK(L != 160 || cb == 2,
              "tile length L=160 is only supported with mul1 (cb=2)");
  TORCH_CHECK(sg_size == 16 || sg_size == 32,
              "sg_size must be 16 or 32, got ", sg_size);
}

}  // namespace

void decode_codebook(int64_t cb, at::Tensor out_lut) {
  TORCH_CHECK(cb >= 0 && cb <= 2, "cb must be 0, 1, or 2");
  const int64_t dev = check_xpu_tensor(out_lut, "out_lut", at::kHalf, 1);
  TORCH_CHECK(out_lut.size(0) == 65536, "out_lut must have length 65536");
  decode_codebook_impl(cb, out_lut.mutable_data_ptr<at::Half>(), dev);
}

void quantize_tiles(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    int64_t K,
    bool mcg,
    bool mul1,
    int64_t sg_size = 32,
    bool force_global_costs = false,
    int64_t wave_cap_override = 0) {
  const int64_t cb = resolve_cb(mcg, mul1);
  const int64_t dev =
      check_xpu_tensor(input_tiles, "input_tiles", at::kFloat, 2);
  TORCH_CHECK(
      check_xpu_tensor(output_tiles, "output_tiles", at::kFloat, 2) == dev,
      "output_tiles must be on the same XPU device as input_tiles");
  TORCH_CHECK(
      check_xpu_tensor(output_indices, "output_indices", at::kShort, 2) == dev,
      "output_indices must be on the same XPU device as input_tiles");
  TORCH_CHECK(
      check_xpu_tensor(temp_costs, "temp_costs", at::kHalf, 3) == dev,
      "temp_costs must be on the same XPU device as input_tiles");
  TORCH_CHECK(
      check_xpu_tensor(temp_edges, "temp_edges", at::kShort, 3) == dev,
      "temp_edges must be on the same XPU device as input_tiles");

  const int64_t num_tiles = input_tiles.size(0);
  const int64_t L = input_tiles.size(1);
  TORCH_CHECK(num_tiles > 0, "num_tiles must be positive, got ", num_tiles);
  validate_common_params(K, cb, L, sg_size);

  const int64_t edges = 65536 >> K;
  const int64_t batch_cap = temp_costs.size(0);
  TORCH_CHECK(batch_cap >= 1, "temp_costs batch dimension B must be >= 1");
  TORCH_CHECK(output_tiles.size(0) == num_tiles && output_tiles.size(1) == L,
              "output_tiles shape must match input_tiles");
  TORCH_CHECK(
      output_indices.size(0) == num_tiles && output_indices.size(1) == L,
      "output_indices shape must match input_tiles");
  TORCH_CHECK(temp_costs.size(1) == 2 && temp_costs.size(2) == edges,
              "temp_costs must have shape (B, 2, 65536 >> K)");
  TORCH_CHECK(temp_edges.size(0) == batch_cap && temp_edges.size(1) == L &&
                  temp_edges.size(2) == edges,
              "temp_edges must have shape (B, L, 65536 >> K)");

  quantize_tiles_fused_impl(
      input_tiles.const_data_ptr<float>(),
      output_tiles.mutable_data_ptr<float>(),
      output_indices.mutable_data_ptr<int16_t>(),
      temp_costs.mutable_data_ptr<at::Half>(),
      temp_edges.mutable_data_ptr<int16_t>(),
      num_tiles, batch_cap, K, cb, L, sg_size, force_global_costs,
      wave_cap_override, dev);
}

void quantize_tiles_forward_pass(
    at::Tensor input_tiles,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    at::Tensor pre_states,
    int64_t K,
    bool mcg,
    bool mul1,
    int64_t roll,
    int64_t cost_slot = 0,
    int64_t sg_size = 32,
    bool force_global_costs = false) {
  const int64_t cb = resolve_cb(mcg, mul1);
  const int64_t dev =
      check_xpu_tensor(input_tiles, "input_tiles", at::kFloat, 2);
  TORCH_CHECK(
      check_xpu_tensor(temp_costs, "temp_costs", at::kHalf, 3) == dev,
      "temp_costs must be on the same XPU device");
  TORCH_CHECK(
      check_xpu_tensor(temp_edges, "temp_edges", at::kShort, 3) == dev,
      "temp_edges must be on the same XPU device");
  TORCH_CHECK(
      check_xpu_tensor(pre_states, "pre_states", at::kInt, 1) == dev,
      "pre_states must be an int32 XPU tensor");

  const int64_t num_tiles = input_tiles.size(0);
  const int64_t L = input_tiles.size(1);
  validate_common_params(K, cb, L, sg_size);
  const int64_t edges = 65536 >> K;
  TORCH_CHECK(num_tiles > 0 && num_tiles <= temp_costs.size(0),
              "num_tiles must be in 1..B");
  TORCH_CHECK(pre_states.size(0) >= num_tiles,
              "pre_states length must be >= num_tiles");
  TORCH_CHECK(temp_costs.size(1) == 2 && temp_costs.size(2) == edges,
              "temp_costs must have shape (B, 2, 65536 >> K)");
  TORCH_CHECK(temp_edges.size(0) >= num_tiles && temp_edges.size(1) == L &&
                  temp_edges.size(2) == edges,
              "temp_edges must have shape (B, L, 65536 >> K)");
  TORCH_CHECK(cost_slot == 0 || cost_slot == 1, "cost_slot must be 0 or 1");

  quantize_tiles_forward_impl(
      input_tiles.const_data_ptr<float>(),
      temp_costs.mutable_data_ptr<at::Half>(),
      temp_edges.mutable_data_ptr<int16_t>(),
      pre_states.const_data_ptr<int32_t>(),
      num_tiles, K, cb, L, roll, cost_slot, sg_size, force_global_costs, dev);
}

void quantize_tiles_backward_pass(
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor end_states,
    int64_t num_tiles,
    int64_t K,
    bool mcg,
    bool mul1,
    int64_t roll,
    int64_t cost_slot,
    bool write_output,
    int64_t sg_size = 32) {
  const int64_t cb = resolve_cb(mcg, mul1);
  const int64_t dev =
      check_xpu_tensor(temp_costs, "temp_costs", at::kHalf, 3);
  TORCH_CHECK(
      check_xpu_tensor(temp_edges, "temp_edges", at::kShort, 3) == dev,
      "temp_edges must be on the same XPU device");
  TORCH_CHECK(
      check_xpu_tensor(output_tiles, "output_tiles", at::kFloat, 2) == dev,
      "output_tiles must be on the same XPU device");
  TORCH_CHECK(
      check_xpu_tensor(output_indices, "output_indices", at::kShort, 2) == dev,
      "output_indices must be on the same XPU device");
  TORCH_CHECK(
      check_xpu_tensor(end_states, "end_states", at::kInt, 1) == dev,
      "end_states must be an int32 XPU tensor");

  const int64_t L = temp_edges.size(1);
  validate_common_params(K, cb, L, sg_size);
  TORCH_CHECK(num_tiles > 0 && num_tiles <= temp_costs.size(0),
              "num_tiles must be in 1..B");

  quantize_tiles_backward_impl(
      temp_costs.const_data_ptr<at::Half>(),
      temp_edges.const_data_ptr<int16_t>(),
      output_tiles.mutable_data_ptr<float>(),
      output_indices.mutable_data_ptr<int16_t>(),
      end_states.mutable_data_ptr<int32_t>(),
      num_tiles, K, cb, L, roll, cost_slot, write_output, sg_size, dev);
}

std::vector<double> quantize_tiles_usm_host(
    at::Tensor input_tiles,
    at::Tensor temp_costs,
    at::Tensor output_tiles_cpu,
    at::Tensor output_indices_cpu,
    at::Tensor lut_cpu,
    int64_t K,
    bool mcg,
    bool mul1,
    int64_t sg_size = 32) {
  const int64_t cb = resolve_cb(mcg, mul1);
  const int64_t dev =
      check_xpu_tensor(input_tiles, "input_tiles", at::kFloat, 2);
  TORCH_CHECK(
      check_xpu_tensor(temp_costs, "temp_costs", at::kHalf, 3) == dev,
      "temp_costs must be on the same XPU device");
  TORCH_CHECK(output_tiles_cpu.device().is_cpu() &&
                  output_tiles_cpu.is_contiguous() &&
                  output_tiles_cpu.scalar_type() == at::kFloat,
              "output_tiles_cpu must be a contiguous float32 CPU tensor");
  TORCH_CHECK(output_indices_cpu.device().is_cpu() &&
                  output_indices_cpu.is_contiguous() &&
                  output_indices_cpu.scalar_type() == at::kShort,
              "output_indices_cpu must be a contiguous int16 CPU tensor");
  TORCH_CHECK(lut_cpu.device().is_cpu() && lut_cpu.is_contiguous() &&
                  lut_cpu.scalar_type() == at::kFloat &&
                  lut_cpu.numel() == 65536,
              "lut_cpu must be a 65536-element float32 CPU tensor");

  const int64_t num_tiles = input_tiles.size(0);
  const int64_t L = input_tiles.size(1);
  validate_common_params(K, cb, L, sg_size);
  return quantize_tiles_usm_host_impl(
      input_tiles.const_data_ptr<float>(),
      temp_costs.mutable_data_ptr<at::Half>(),
      output_tiles_cpu.mutable_data_ptr<float>(),
      output_indices_cpu.mutable_data_ptr<int16_t>(),
      lut_cpu.const_data_ptr<float>(),
      num_tiles, K, cb, L, sg_size, dev);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_codebook", &decode_codebook,
        "Decode all 65536 states of codebook cb into an fp16 XPU tensor");
  m.def("quantize_tiles", &quantize_tiles,
        py::arg("input_tiles"),
        py::arg("output_tiles"),
        py::arg("output_indices"),
        py::arg("temp_costs"),
        py::arg("temp_edges"),
        py::arg("K"),
        py::arg("mcg"),
        py::arg("mul1"),
        py::arg("sg_size") = 32,
        py::arg("force_global_costs") = false,
        py::arg("wave_cap_override") = 0,
        "Fused 2-pass circular Viterbi tile quantizer on XPU");
  m.def("quantize_tiles_forward_pass", &quantize_tiles_forward_pass,
        py::arg("input_tiles"),
        py::arg("temp_costs"),
        py::arg("temp_edges"),
        py::arg("pre_states"),
        py::arg("K"),
        py::arg("mcg"),
        py::arg("mul1"),
        py::arg("roll"),
        py::arg("cost_slot") = 0,
        py::arg("sg_size") = 32,
        py::arg("force_global_costs") = false,
        "Single forward ACS pass on XPU");
  m.def("quantize_tiles_backward_pass", &quantize_tiles_backward_pass,
        py::arg("temp_costs"),
        py::arg("temp_edges"),
        py::arg("output_tiles"),
        py::arg("output_indices"),
        py::arg("end_states"),
        py::arg("num_tiles"),
        py::arg("K"),
        py::arg("mcg"),
        py::arg("mul1"),
        py::arg("roll"),
        py::arg("cost_slot"),
        py::arg("write_output"),
        py::arg("sg_size") = 32,
        "Single backward traceback pass on XPU");
  m.def("quantize_tiles_usm_host", &quantize_tiles_usm_host,
        py::arg("input_tiles"),
        py::arg("temp_costs"),
        py::arg("output_tiles_cpu"),
        py::arg("output_indices_cpu"),
        py::arg("lut_cpu"),
        py::arg("K"),
        py::arg("mcg"),
        py::arg("mul1"),
        py::arg("sg_size") = 32,
        "P1-19 USM host-accessible edge buffer + CPU traceback");
}
