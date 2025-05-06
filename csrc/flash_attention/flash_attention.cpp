#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>
#include <stdio.h>

template<bool use_fast_accum> 
void flash_attention_fp8_cuda(void* q_ptr, void* k_ptr, void* v_ptr, void* o_ptr, int batch_size, int num_heads, int num_q_tokens, int num_k_tokens, int head_dim, float softmax_scale, cudaStream_t stream);

torch::Tensor flash_attention_fp8(torch::Tensor &q, torch::Tensor &k, torch::Tensor v, float softmax_scale, bool use_fast_accum){
    
    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int num_q_tokens = q.size(2);
    int num_k_tokens = k.size(2);
    int head_dim = q.size(3);
    auto out = at::empty({batch_size, num_heads, num_q_tokens, head_dim}, q.options().dtype(at::kBFloat16));
    softmax_scale = softmax_scale*1.44269504088896340736f;
    
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (use_fast_accum){
        flash_attention_fp8_cuda<true>(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), batch_size, num_heads, num_q_tokens, num_k_tokens, head_dim, softmax_scale, stream);
    } else {
        flash_attention_fp8_cuda<false>(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), batch_size, num_heads, num_q_tokens, num_k_tokens, head_dim, softmax_scale, stream);
    }
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("package_name", &function_name, "function_docstring"")
    m.def("flash_attention_fp8", &flash_attention_fp8,
          "flash_attention_fp8");
}