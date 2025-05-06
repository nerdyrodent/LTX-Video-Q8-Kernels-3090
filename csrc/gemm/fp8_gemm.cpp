#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>
#include <stdio.h>

template<bool use_fast_accum>
void fp8_gemm_cuda(void* Aptr, void* Bptr, void* out, int M, int N, int K, cudaStream_t stream);
template<bool use_fast_accum>
void fp8_bias_gemm_cuda(void* Aptr, void* Bptr, void* bias_ptr, void* out, int M, int N, int K, cudaStream_t stream);


torch::Tensor fp8_gemm(torch::Tensor &a, torch::Tensor &b, c10::optional<at::Tensor>& bias_, bool use_fast_accum){
    int m, n, k;
    m = a.size(0);
    n = b.size(0);
    k = a.size(1);
    auto out = at::empty({m, n}, a.options().dtype(at::kBFloat16));
    at::cuda::CUDAGuard device_guard{(char)a.get_device()};
    at::Tensor bias;
    bool has_bias = false;
    if(bias_.has_value()){
        bias = bias_.value();
        has_bias = true;
    }
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if(has_bias){
        if (use_fast_accum){
            fp8_bias_gemm_cuda<true>(a.data_ptr(), b.data_ptr(), bias.data_ptr(), out.data_ptr(), m, n, k, stream);
        } else {
            fp8_bias_gemm_cuda<false>(a.data_ptr(), b.data_ptr(), bias.data_ptr(), out.data_ptr(), m, n, k, stream);
        }
    } else {
        if (use_fast_accum){
            fp8_gemm_cuda<true>(a.data_ptr(), b.data_ptr(), out.data_ptr(), m, n, k, stream);
        } else {
            fp8_gemm_cuda<false>(a.data_ptr(), b.data_ptr(), out.data_ptr(), m, n, k, stream);
        }
    }
    return out;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
    m.def("fp8_gemm", &fp8_gemm,
          "fp8 matmul");
      
}