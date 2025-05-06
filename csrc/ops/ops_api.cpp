
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <torch/python.h>


at::Tensor  fast_hadamard_transform(at::Tensor &x, float scale, std::optional<at::ScalarType>& out_type_);
std::vector<at::Tensor> gelu_fast_hadamard_transform_cvt(at::Tensor &x, float scale, std::optional<at::ScalarType>& out_type_);
std::vector<at::Tensor> norm_fma_hadamard_cvt(at::Tensor &x, c10::optional<at::Tensor>& weights_, at::Tensor &y_scale, at::Tensor &z_shift, float scale, bool add_one, std::optional<at::ScalarType>& out_type_);
at::Tensor rms_norm_rope(at::Tensor &x, c10::optional<at::Tensor>& weights_, at::Tensor &cos_freqs, at::Tensor &sin_freqs,  bool out_16bit);

std::vector<at::Tensor> quant_fast_hadamard_transform_cvt(at::Tensor &x, float scale, std::optional<at::ScalarType>& out_type_);
std::vector<at::Tensor> quant_fast_hadamard_transform_12N_cvt(at::Tensor &x, float scale, std::optional<at::ScalarType>& out_type_);

at::Tensor dequant_fast_hadamard_transform(at::Tensor &x, at::Tensor &row_scales, float scale);
at::Tensor dequant_fast_hadamard_transform_12N(at::Tensor &x, at::Tensor &row_scales, float scale);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("norm_fma_hadamard_cvt", &norm_fma_hadamard_cvt,
          "fused rms norm + fma + hadamard cvt to fp8");
    m.def("fast_hadamard_transform", &fast_hadamard_transform, 
          "Fast Hadamard transform");
    m.def("gelu_fast_hadamard_transform_cvt", &gelu_fast_hadamard_transform_cvt,
          "fused gelu+ hadamard cvt to fp8");
    m.def("rms_norm_rope", &rms_norm_rope,
          "fused  norm + rope + cvt");
    m.def("quant_fast_hadamard_transform_cvt", &quant_fast_hadamard_transform_cvt,
          "quantize");
    m.def("quant_fast_hadamard_transform_12N_cvt", &quant_fast_hadamard_transform_12N_cvt,
          "quantize");
    m.def("dequant_fast_hadamard_transform_cvt", &dequant_fast_hadamard_transform,
          "dequantize");
    m.def("dequant_fast_hadamard_transform_12N_cvt", &dequant_fast_hadamard_transform_12N,
          "dequantize 12N");

}
