/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

/*
KONAKONA666: added fp8 support
*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "fast_hadamard_transform.h"
#include "static_switch.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")


#define HINPUT_TYPE_SWITCH(ITYPE, ...)      \
    if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__;                                                              \
    } else if(ITYPE == at::ScalarType::Float8_e4m3fn) {                             \
        using input_t = at::Float8_e4m3fn;                                          \
        __VA_ARGS__;                                                              \
    }                                                                               

#define HOTYPE_SWITCH(OTYPE, ...)      \
    if (OTYPE == at::ScalarType::BFloat16) {                                 \
        using output_t = at::BFloat16;                                               \
        __VA_ARGS__;                                                              \
    } else if(OTYPE == at::ScalarType::Float8_e4m3fn) {                             \
        using output_t = at::Float8_e4m3fn;                                          \
        __VA_ARGS__;                                                              \
    }                                                                               \



#define DISPATCH_OTYPE_INT8_AND_FP8(OTYPE, NAME, ...)                                \
    if (OTYPE == at::ScalarType::Char) {                                            \
        using output_t = int8_t;                                                   \
        __VA_ARGS__();                                                              \
    } else if (OTYPE == at::ScalarType::Float8_e4m3fn) {                            \
        using output_t = at::Float8_e4m3fn;                                          \
        __VA_ARGS__();                                                              \
    }                                                                               \
    else {                                                                          \
        AT_ERROR(#NAME, " not implemented for output type '", toString(OTYPE), "'");  \
    }                                                                               \


template<typename input_t, typename output_t>
void fast_hadamard_transform_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t, typename output_t,  
         bool fuse_gelu_,
         bool fuse_rms_norm_, bool norm_affine_, 
         bool fuse_rope_, 
         bool fuse_multiply_add_, bool add_one_>
void fused_fast_hadamard_transform_cuda(UnifiedHadamardParamsBase &params, cudaStream_t stream);

template<typename input_t, typename output_t,  
         bool fuse_gelu_,
         bool fuse_rms_norm_, bool norm_affine_, 
         bool fuse_rope_, 
         bool fuse_multiply_add_, bool add_one_>
void fused_fast_hadamard_transform_12N_cuda(UnifiedHadamardParamsBase &params, cudaStream_t stream);


template<typename input_t, typename output_t, bool norm_affine>
void rms_norm_rope_cuda(NormRopeHadamardParamsBase &params, cudaStream_t stream) ;


template<typename input_t, typename output_t>
void dequant_fast_hadamard_transform_cuda(DequantHadamardParamsBase &params, cudaStream_t stream);

template<typename input_t, typename output_t>
void dequant_fast_hadamard_transform_12N_cuda(DequantHadamardParamsBase &params, cudaStream_t stream);

void set_hadamard_params(HadamardParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t multiple,
                         // device pointers
                         const at::Tensor x,
                         const at::Tensor out,
                         float scale
                         ) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim / multiple)));

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    // All stride are in elements, not bytes.
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);

    params.scale = scale;
}

void set_dequant_hadamard_params(DequantHadamardParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t multiple,
                         // device pointers
                         const at::Tensor x,
                         const at::Tensor out,
                         const at::Tensor row_scales,
                         float scale
                         ) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim / multiple)));

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    params.scales_ptr = row_scales.data_ptr();
    // All stride are in elements, not bytes.
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);

    params.scale = scale;
}

void set_fused_hadamard_params(UnifiedHadamardParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t multiple,
                         int seqlen,
                         int fma_seqlen,
                         float scale,

                         
                         int64_t x_batch_stride,
                         int64_t out_batch_stride,
                         int64_t fma_batch_stride,
                         int64_t cos_freq_batch_stride,
                         int64_t sin_freq_batch_stride,
                         
                         void * x_ptr,
                         void * out_ptr,
                         void * out_scales_ptr,

                         void * y_scale_ptr,
                         void * z_shift_ptr,

                         void * weights_ptr,

                         void * cos_freq_ptr,
                         void * sin_freq_ptr
                         ) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim / multiple)));

    params.x_ptr = x_ptr;
    params.out_ptr = out_ptr;
    params.out_scales_ptr = out_scales_ptr;

    params.y_scale_ptr = y_scale_ptr;
    params.z_shift_ptr = z_shift_ptr;

    params.weights_ptr = weights_ptr;
    params.cos_freq_ptr = cos_freq_ptr;
    params.sin_freq_ptr = sin_freq_ptr;

    
    params.x_batch_stride = x_batch_stride;
    params.out_batch_stride = out_batch_stride;
    params.fma_batch_stride = fma_batch_stride;
    params.cos_freq_batch_stride = cos_freq_batch_stride;
    params.sin_freq_batch_stride = sin_freq_batch_stride;

    params.batch_fma_change = seqlen/fma_seqlen;
    params.scale = scale;
}

void set_norm_rope_hadamard_params(NormRopeHadamardParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t multiple,
                         // device pointers
                         const at::Tensor x,
                         const at::Tensor cos_freqs,
                         const at::Tensor sin_freqs,
                         const at::Tensor weights,
                         const at::Tensor out,
                         
                         bool norm_affine,
                         float scale
                         ) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim / multiple)));

    // Set the pointers and strides.
    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    params.cos_freq_ptr = cos_freqs.data_ptr();
    params.sin_freq_ptr = sin_freqs.data_ptr();
    if (norm_affine){
        params.weights_ptr = weights.data_ptr();
    } else {
        params.weights_ptr = nullptr;
    }
    // All stride are in elements, not bytes.
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);
    params.cos_freq_batch_stride = cos_freqs.stride(0);
    params.sin_freq_batch_stride = sin_freqs.stride(0);
    
    params.scale = scale;
    
}


at::Tensor
fast_hadamard_transform(at::Tensor &x, float scale, std::optional<at::ScalarType>& out_type_) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::BFloat16 || input_type == at::ScalarType::Float8_e4m3fn);
    TORCH_CHECK(x.is_cuda());
    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);
    if (dim_og % 8 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
    }
    const int dim = x.size(1);
    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");
    at::ScalarType out_type;
    if (out_type_.has_value()){
        out_type = out_type_.value();
    } else {
        out_type = x.scalar_type();
    }
    at::Tensor out = torch::empty(x.sizes(), x.options().dtype(out_type));
    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 1, x, out, scale);
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    HOTYPE_SWITCH(out_type, 
        HINPUT_TYPE_SWITCH(x.scalar_type(), 
            fast_hadamard_transform_cuda<input_t, output_t>(params, stream);
        );
    );    
    if (dim_og % 8 != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

std::vector<at::Tensor>
gelu_fast_hadamard_transform_cvt(at::Tensor &x, float scale, std::optional<at::ScalarType>& out_type_) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda());
    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    const int seqlen = x.size(-2);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);
    if (dim_og % 8 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
    }
    const int dim = x.size(1);
    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");
    at::ScalarType out_type;
    if (out_type_.has_value()){
        out_type = out_type_.value();
    } else {
        out_type = torch::kInt8;
    }
    at::Tensor out = torch::empty(x.sizes(), x.options().dtype(out_type));
    at::Tensor out_scales = torch::empty({batch_size}, x.options().dtype(torch::kFloat32));
    UnifiedHadamardParamsBase params;
    set_fused_hadamard_params(params, batch_size, dim, 1, seqlen, 1, scale, 
                              x.stride(0), out.stride(0), 1, 1, 1,
                              x.data_ptr(), out.data_ptr(), out_scales.data_ptr(), nullptr, nullptr, nullptr, nullptr, nullptr);

    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_OTYPE_INT8_AND_FP8(out.scalar_type(), "gelu_hadamard", [&] {
        fused_fast_hadamard_transform_cuda<at::BFloat16, output_t, true, false, false, false, false, false>(params, stream);
    });

    if (dim_og % 8 != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return {out.reshape(shapes_og), out_scales.reshape(torch::IntArrayRef(shapes_og.begin(), shapes_og.size()-1))};
}

std::vector<at::Tensor>
norm_fma_hadamard_cvt(at::Tensor &x, c10::optional<at::Tensor>& weights_, at::Tensor &y_scale, at::Tensor &z_shift, float scale, bool add_one, std::optional<at::ScalarType>& out_type_) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda());
    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    const int seqlen = x.size(-2);
    const int fma_seqlen = z_shift.size(-2);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    y_scale = y_scale.reshape({-1, dim_og});
    z_shift = z_shift.reshape({-1, dim_og});
    at::Tensor weights;
    bool norm_affine = false;
    if(weights_.has_value()){
        weights = weights_.value();
        norm_affine = true;
    }
    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);
    if (dim_og % 8 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
    }
    const int dim = x.size(1);
    at::ScalarType out_type;
    if (out_type_.has_value()){
        out_type = out_type_.value();
    } else {
        out_type = torch::kInt8;
    }
    at::Tensor out = torch::empty(x.sizes(), x.options().dtype(out_type));
    at::Tensor out_scales = torch::empty({batch_size}, x.options().dtype(torch::kFloat32));
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    UnifiedHadamardParamsBase params;
    set_fused_hadamard_params(params, batch_size, dim, 1, seqlen, fma_seqlen, scale, 
                              x.stride(0), out.stride(0), y_scale.stride(0), 1, 1,
                              x.data_ptr(), out.data_ptr(), out_scales.data_ptr(), y_scale.data_ptr(), z_shift.data_ptr(), nullptr, nullptr, nullptr);
    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");
    BOOL_SWITCH(
        add_one, add_one_scale, 
        BOOL_SWITCH(norm_affine, norm_affine_t, 
           DISPATCH_OTYPE_INT8_AND_FP8(out.scalar_type(), "gelu_hadamard", [&] {
                fused_fast_hadamard_transform_cuda<at::BFloat16, output_t, false, true, norm_affine_t, false, true, add_one_scale>(params, stream);
            });
    ));
    return {out.reshape(shapes_og), out_scales.reshape(torch::IntArrayRef(shapes_og.begin(), shapes_og.size()-1))};
}


std::vector<at::Tensor>
quant_fast_hadamard_transform_cvt(at::Tensor &x, float scale, std::optional<at::ScalarType>& out_type_) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda());
    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    const int seqlen = x.size(-2);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);
    if (dim_og % 8 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
    }
    const int dim = x.size(1);
    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");
    at::ScalarType out_type;
    if (out_type_.has_value()){
        out_type = out_type_.value();
    } else {
        out_type = torch::kInt8;
    }
    at::Tensor out = torch::empty(x.sizes(), x.options().dtype(out_type));
    at::Tensor out_scales = torch::empty({batch_size}, x.options().dtype(torch::kFloat32));
    
    UnifiedHadamardParamsBase params;
    set_fused_hadamard_params(params, batch_size, dim, 1, seqlen, 1, scale, 
                              x.stride(0), out.stride(0), 1, 1, 1,
                              x.data_ptr(), out.data_ptr(), out_scales.data_ptr(), nullptr, nullptr, nullptr, nullptr, nullptr);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_OTYPE_INT8_AND_FP8(out.scalar_type(), "quant", [&] {
        fused_fast_hadamard_transform_cuda<at::BFloat16, output_t, false, false, false, false, false, false>(params, stream);
    });

    if (dim_og % 8 != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return {out.reshape(shapes_og), out_scales.reshape(torch::IntArrayRef(shapes_og.begin(), shapes_og.size()-1))};
}


at::Tensor
dequant_fast_hadamard_transform(at::Tensor &x, at::Tensor &row_scales, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Char);
    TORCH_CHECK(x.is_cuda());

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % 8 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
    }
    const int dim = x.size(1);
    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");
    at::Tensor out = torch::empty(x.sizes(), x.options().dtype(at::ScalarType::BFloat16));
    DequantHadamardParamsBase params;
    set_dequant_hadamard_params(params, batch_size, dim, 1, x, out, row_scales, scale);
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dequant_fast_hadamard_transform_cuda<int8_t, at::BFloat16>(params, stream);
    if (dim_og % 8 != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}


std::vector<at::Tensor>
quant_fast_hadamard_transform_12N_cvt(at::Tensor &x, float scale, std::optional<at::ScalarType>& out_type_) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda());
    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    const int seqlen = x.size(-2);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);
    if (dim_og % 8 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
    }
    const int dim = x.size(1);
    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");
    at::ScalarType out_type;
    if (out_type_.has_value()){
        out_type = out_type_.value();
    } else {
        out_type = torch::kInt8;
    }
    at::Tensor out = torch::empty(x.sizes(), x.options().dtype(out_type));
    at::Tensor out_scales = torch::empty({batch_size}, x.options().dtype(torch::kFloat32));
    UnifiedHadamardParamsBase params;
    set_fused_hadamard_params(params, batch_size, dim, 12, seqlen, 1, scale, 
                              x.stride(0), out.stride(0), 1, 1, 1,
                              x.data_ptr(), out.data_ptr(), out_scales.data_ptr(), nullptr, nullptr, nullptr, nullptr, nullptr);
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_OTYPE_INT8_AND_FP8(out.scalar_type(), "quant", [&] {
        fused_fast_hadamard_transform_12N_cuda<at::BFloat16, output_t, false, false, false, false, false, false>(params, stream);
    });
    if (dim_og % 8 != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return {out.reshape(shapes_og), out_scales.reshape(torch::IntArrayRef(shapes_og.begin(), shapes_og.size()-1))};
}


at::Tensor
dequant_fast_hadamard_transform_12N(at::Tensor &x, at::Tensor &row_scales, float scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Char);
    TORCH_CHECK(x.is_cuda());
    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);
    const int dim = x.size(1);
    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");
    at::Tensor out = torch::empty(x.sizes(), x.options().dtype(at::ScalarType::BFloat16));
    DequantHadamardParamsBase params;
    set_dequant_hadamard_params(params, batch_size, dim, 12, x, out, row_scales, scale);
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dequant_fast_hadamard_transform_12N_cuda<int8_t, at::BFloat16>(params, stream);
    if (dim_og % 8 != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

at::Tensor rms_norm_rope(at::Tensor &x, c10::optional<at::Tensor>& weights_, at::Tensor &cos_freqs, at::Tensor &sin_freqs, bool out_16bit) {
    auto input_type = x.scalar_type();
    float scale = 1.0f; // :D
    TORCH_CHECK(input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda());
    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];
    cos_freqs = cos_freqs.reshape({-1, dim_og});
    sin_freqs = sin_freqs.reshape({-1, dim_og});
    at::Tensor weights;
    bool norm_affine = false;
    if(weights_.has_value()){
        weights = weights_.value();
        norm_affine = true;
    }
    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);
    if (dim_og % 8 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
    }
    const int dim = x.size(1);
    at::Tensor out;
    if (out_16bit){
        out = torch::empty(x.sizes(), x.options().dtype(torch::kBFloat16));
    } else {
        out = torch::empty(x.sizes(), x.options().dtype(torch::kFloat8_e4m3fn));
    }
    at::cuda::CUDAGuard device_guard{(char)x.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    NormRopeHadamardParamsBase params;
    set_norm_rope_hadamard_params(params, batch_size, dim, 1, x, cos_freqs, sin_freqs, weights, out, norm_affine, scale);
    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");
    if (norm_affine){
        if (out_16bit){
            rms_norm_rope_cuda<at::BFloat16, at::BFloat16, true>(params, stream);
        } else {
            rms_norm_rope_cuda<at::BFloat16, at::Float8_e4m3fn, true>(params, stream);
        }
        
    } else {
        if (out_16bit){
            rms_norm_rope_cuda<at::BFloat16, at::BFloat16, false>(params, stream);
        } else {
            rms_norm_rope_cuda<at::BFloat16, at::Float8_e4m3fn, false>(params, stream);
        }
    }
    return out.reshape(shapes_og);
}