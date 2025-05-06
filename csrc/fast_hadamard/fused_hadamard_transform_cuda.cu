/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// #pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "fast_hadamard_transform.h"
#include "fast_hadamard_transform_common.h"
#include "fast_hadamard_transform_special.h"
#include "static_switch.h"


template<int kNThreads_, int kLogN_, typename input_t_, typename output_t_, 
         bool fuse_gelu_,
         bool fuse_rms_norm_, bool norm_affine_, 
         bool fuse_rope_, 
         bool fuse_multiply_add_, bool add_one_>
struct fused_fast_hadamard_transform_kernel_traits {
    using input_t = input_t_;
    using output_t = output_t_;

    static constexpr bool fuse_rms_norm = fuse_rms_norm_;
    static constexpr bool norm_affine = norm_affine_;
    static constexpr bool fuse_gelu = fuse_gelu_;
    static constexpr bool fuse_rope = fuse_rope_;
    static constexpr bool fuse_multiply_add = fuse_multiply_add_;
    static constexpr bool add_one = add_one_;
    
    static_assert((add_one && fuse_multiply_add)|| !add_one);
    static_assert((norm_affine && fuse_rms_norm)|| !norm_affine);
    
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = 1 << kLogN;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 1 || kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : kNBytes == 2 ? 8 : 8;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    // We don't want to use more than 32 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 32 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

template<int kNThreads_, int kLogN_, typename input_t_, typename output_t_, 
         bool fuse_gelu_,
         bool fuse_rms_norm_, bool norm_affine_, 
         bool fuse_rope_, 
         bool fuse_multiply_add_, bool add_one_>
struct fused_fast_hadamard_transform_12N_kernel_traits {
    using input_t = input_t_;
    using output_t = output_t_;

    static constexpr bool fuse_rms_norm = fuse_rms_norm_;
    static constexpr bool norm_affine = norm_affine_;
    static constexpr bool fuse_gelu = fuse_gelu_;
    static constexpr bool fuse_rope = fuse_rope_;
    static constexpr bool fuse_multiply_add = fuse_multiply_add_;
    static constexpr bool add_one = add_one_;
    
    static_assert((add_one && fuse_multiply_add)|| !add_one);
    static_assert((norm_affine && fuse_rms_norm)|| !norm_affine);
    
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = (1 << kLogN) * 12;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 1 || kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = 4;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    // We don't want to use more than 32 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 24 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
};

template <int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread_chunk_12(float x[kNChunks][12]) {
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_12(x[c]); }
}


template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void fused_fast_hadamard_transform_kernel(UnifiedHadamardParamsBase params) {
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
    constexpr int kNExchangeRounds = Ktraits::kNExchangeRounds;
    constexpr int kNChunks = Ktraits::kNChunks;
    constexpr bool fuse_gelu = Ktraits::fuse_gelu;
    constexpr bool fuse_rms_norm = Ktraits::fuse_rms_norm;
    constexpr bool norm_affine = Ktraits::norm_affine;
    constexpr bool fuse_rope = Ktraits::fuse_rope;
    constexpr bool fuse_multiply_add = Ktraits::fuse_multiply_add;
    constexpr bool add_one = Ktraits::add_one;
    

    using input_t = typename Ktraits::input_t;
    using output_t = typename Ktraits::output_t;
    using vec_t = typename Ktraits::vec_t;

    using weights_t = typename Ktraits::input_t;
    using freqs_t = typename Ktraits::input_t;
    using fma_yz_t = typename Ktraits::input_t;

    constexpr int kLogNElts = cilog2(Ktraits::kNElts);
    static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");
    constexpr int kWarpSize = std::min(kNThreads, 32);
    constexpr int kLogWarpSize = cilog2(kWarpSize);
    static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int kLogNWarps = cilog2(kNWarps);
    static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");
    constexpr int kLoadsPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNThreads);
    static_assert(kLoadsPerExchange * sizeof(vec_t) * kNThreads == Ktraits::kSmemExchangeSize, "kSmemExchangeSize should be a power of 2");
    static_assert(kNExchangeRounds * kLoadsPerExchange * sizeof(vec_t) == kNChunks * kNElts * sizeof(float));

    constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
    static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec * kNThreads == Ktraits::kSmemExchangeSize);
    constexpr int kNExchanges = kNChunks / kChunksPerExchange;
    static_assert(kNExchanges * kChunksPerExchange == kNChunks);

    // Shared memory.
    extern __shared__ char smem_[];
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_);

    const int batch_id = blockIdx.x;
    const int warp_id = threadIdx.x / 32;

    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride;
    output_t *out = reinterpret_cast<output_t *>(params.out_ptr) + batch_id * params.out_batch_stride;

    float x_vals[kNChunks][kNElts];
    
    load_input<kNChunks, kNElts, input_t>(x, x_vals, params.dim);
    
    //FUSE BEGIN
    if constexpr (fuse_gelu){
        fused_gelu<kNChunks, kNElts>(x_vals);
    }

    if constexpr (fuse_rms_norm){
        weights_t *weights = norm_affine ? reinterpret_cast<weights_t*>(params.weights_ptr) : nullptr;
        float weights_vals[kNChunks][kNElts];
        if constexpr (norm_affine){
            load_input<kNChunks, kNElts, weights_t>(weights, weights_vals, params.dim);
        }
        float *smem_sum = reinterpret_cast<float*>(smem_);
        fused_rms_norm<kNChunks, kNElts, kNWarps, norm_affine>(x_vals, weights_vals, smem_sum, float(params.dim));
    }

    if constexpr (fuse_rope){
        float sin_freqs_vals[kNChunks][kNElts];
        float cos_freqs_vals[kNChunks][kNElts];

        freqs_t *cos_freqs = reinterpret_cast<freqs_t*>(params.cos_freq_ptr) + batch_id * params.cos_freq_batch_stride;
        freqs_t *sin_freqs = reinterpret_cast<freqs_t*>(params.sin_freq_ptr) + batch_id * params.sin_freq_batch_stride;

        load_input<kNChunks, kNElts, freqs_t>(cos_freqs, cos_freqs_vals, params.dim);
        load_input<kNChunks, kNElts, freqs_t>(sin_freqs, sin_freqs_vals, params.dim);

        fused_rope<kNChunks, kNElts>(x_vals, sin_freqs_vals, cos_freqs_vals);
    }

    if constexpr (fuse_multiply_add){
        float y_scale_vals[kNChunks][kNElts];
        float z_shift_vals[kNChunks][kNElts];
    
        int fma_batch_id = batch_id / params.batch_fma_change;
        fma_yz_t* y_scale = reinterpret_cast<fma_yz_t*>(params.y_scale_ptr) + fma_batch_id * params.fma_batch_stride;
        fma_yz_t* z_shift = reinterpret_cast<fma_yz_t*>(params.z_shift_ptr) + fma_batch_id * params.fma_batch_stride;
    
        load_input<kNChunks, kNElts, fma_yz_t>(y_scale, y_scale_vals, params.dim);
        load_input<kNChunks, kNElts, fma_yz_t>(z_shift, z_shift_vals, params.dim);

        fused_multiply_add<kNChunks, kNElts, add_one>(x_vals, y_scale_vals, z_shift_vals);
    }
    //FUSE END

    hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
    hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);

    if constexpr (kNWarps > 1) {
        exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange);
        hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
        exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange);
    }

    if constexpr (kNChunks > 1) {
        float x_vals_transposed[kNElts][kNChunks];
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals_transposed[i][c] = x_vals[c][i]; }
        }
        if constexpr (kNChunks == 12) {
            hadamard_mult_thread_chunk_12<kNElts>(x_vals_transposed);
        } else {
            constexpr int kLogNChunks = cilog2(kNChunks);
            static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
            hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
        }
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals[c][i] = x_vals_transposed[i][c]; }
        }
    }
    //QUANTIZATION
    // constexpr bool do_round = std::is_same_v<output_t, int8_t>;
    if constexpr (std::is_same_v<output_t, int8_t>){
        float thread_max = -9999999.0;
        float *out_scales = reinterpret_cast<float *>(params.out_scales_ptr) + batch_id;

        #pragma unroll
        for(int c = 0; c < kNChunks; ++c){
            #pragma unroll
            for(int i = 0; i < kNElts; ++i){
                x_vals[c][i] = x_vals[c][i] * params.scale; 
                thread_max  = max(abs(x_vals[c][i]), thread_max);
            }
        }
        MaxOp<float> max_op;
        float warp_max = Allreduce<32>::run(thread_max, max_op);
        float *smem_scale = reinterpret_cast<float *>(smem_);
        if(threadIdx.x % 32 == 0){
            smem_scale[warp_id] = warp_max;
        }
        __syncthreads();
        #pragma unroll  
        for (size_t i = 0; i < kNWarps; i++)
        {
            thread_max = max(thread_max, smem_scale[i]);
        }
        float scale = 127.0f/thread_max;
        store_output<kNChunks, kNElts, output_t, true>(out, x_vals, params.dim, scale);
        *out_scales = 1/scale;
    } else {
        store_output<kNChunks, kNElts, output_t, false>(out, x_vals, params.dim, params.scale);
    }
}

template<int kNThreads, int kLogN, typename input_t, typename output_t, 
         bool fuse_gelu_,
         bool fuse_rms_norm_, bool norm_affine_, 
         bool fuse_rope_, 
         bool fuse_multiply_add_, bool add_one_>
void fused_fast_hadamard_transform_launch(UnifiedHadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fused_fast_hadamard_transform_kernel_traits<kNThreads, kLogN, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fused_fast_hadamard_transform_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int kNThreads, int kLogN, typename input_t, typename output_t, 
         bool fuse_gelu_,
         bool fuse_rms_norm_, bool norm_affine_, 
         bool fuse_rope_, 
         bool fuse_multiply_add_, bool add_one_>
void fused_fast_hadamard_transform_12N_launch(UnifiedHadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fused_fast_hadamard_transform_12N_kernel_traits<kNThreads, kLogN, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fused_fast_hadamard_transform_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}



template<typename input_t, typename output_t,  
         bool fuse_gelu_,
         bool fuse_rms_norm_, bool norm_affine_, 
         bool fuse_rope_, 
         bool fuse_multiply_add_, bool add_one_>
void fused_fast_hadamard_transform_cuda(UnifiedHadamardParamsBase &params, cudaStream_t stream) {
    if (params.log_N == 3) {
        fused_fast_hadamard_transform_launch<1, 3, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 4) {
        fused_fast_hadamard_transform_launch<2, 4, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 5) {
        fused_fast_hadamard_transform_launch<4, 5, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 6) {
        fused_fast_hadamard_transform_launch<8, 6, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 7) {
        fused_fast_hadamard_transform_launch<16, 7, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 8) {
        fused_fast_hadamard_transform_launch<32, 8, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 9) {
        fused_fast_hadamard_transform_launch<32, 9, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 10) {
        fused_fast_hadamard_transform_launch<128, 10, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 11) {
        fused_fast_hadamard_transform_launch<256, 11, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 12) {
        fused_fast_hadamard_transform_launch<256, 12, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 13) {
        fused_fast_hadamard_transform_launch<256, 13, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 14) {
        fused_fast_hadamard_transform_launch<256, 14, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 15) {
        fused_fast_hadamard_transform_launch<256, 15, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    }
}

template<typename input_t, typename output_t,  
         bool fuse_gelu_,
         bool fuse_rms_norm_, bool norm_affine_, 
         bool fuse_rope_, 
         bool fuse_multiply_add_, bool add_one_>
void fused_fast_hadamard_transform_12N_cuda(UnifiedHadamardParamsBase &params, cudaStream_t stream) {
    if (params.log_N == 4) {
        fused_fast_hadamard_transform_12N_launch<4, 4, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 5) {
        fused_fast_hadamard_transform_12N_launch<8, 5, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 6) {
        fused_fast_hadamard_transform_12N_launch<16, 6, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 7) {
        fused_fast_hadamard_transform_12N_launch<32, 7, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 8) {
        fused_fast_hadamard_transform_12N_launch<64, 8, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 9) {
        fused_fast_hadamard_transform_12N_launch<128, 9, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } else if (params.log_N == 10) {
        fused_fast_hadamard_transform_12N_launch<256, 10, input_t, output_t, fuse_gelu_, fuse_rms_norm_, norm_affine_, fuse_rope_, fuse_multiply_add_, add_one_>(params, stream);
    } 
}

template void fused_fast_hadamard_transform_cuda<at::BFloat16, at::Float8_e4m3fn, false, false, false, false, false, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_cuda<at::BFloat16, at::Float8_e4m3fn, true, false, false, false, false, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_cuda<at::BFloat16, at::Float8_e4m3fn, false, true, false, false, true, true>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_cuda<at::BFloat16, at::Float8_e4m3fn, false, true, false, false, true, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_cuda<at::BFloat16, at::Float8_e4m3fn, false, true, true, false, true, true>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_cuda<at::BFloat16, at::Float8_e4m3fn, false, true, true, false, true, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);

template void fused_fast_hadamard_transform_cuda<at::BFloat16, int8_t, false, false, false, false, false, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_cuda<at::BFloat16, int8_t, true, false, false, false, false, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_cuda<at::BFloat16, int8_t, false, true, false, false, true, true>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_cuda<at::BFloat16, int8_t, false, true, false, false, true, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_cuda<at::BFloat16, int8_t, false, true, true, false, true, true>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_cuda<at::BFloat16, int8_t, false, true, true, false, true, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);

/////

template void fused_fast_hadamard_transform_12N_cuda<at::BFloat16, at::Float8_e4m3fn, false, false, false, false, false, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_12N_cuda<at::BFloat16, at::Float8_e4m3fn, true, false, false, false, false, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_12N_cuda<at::BFloat16, at::Float8_e4m3fn, false, true, false, false, true, true>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_12N_cuda<at::BFloat16, at::Float8_e4m3fn, false, true, false, false, true, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_12N_cuda<at::BFloat16, at::Float8_e4m3fn, false, true, true, false, true, true>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_12N_cuda<at::BFloat16, at::Float8_e4m3fn, false, true, true, false, true, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);

template void fused_fast_hadamard_transform_12N_cuda<at::BFloat16, int8_t, false, false, false, false, false, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_12N_cuda<at::BFloat16, int8_t, true, false, false, false, false, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_12N_cuda<at::BFloat16, int8_t, false, true, false, false, true, true>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_12N_cuda<at::BFloat16, int8_t, false, true, false, false, true, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_12N_cuda<at::BFloat16, int8_t, false, true, true, false, true, true>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
template void fused_fast_hadamard_transform_12N_cuda<at::BFloat16, int8_t, false, true, true, false, true, false>(UnifiedHadamardParamsBase &params, cudaStream_t stream);
