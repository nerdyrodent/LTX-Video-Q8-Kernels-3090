#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>
#include <c10/cuda/CUDAException.h> 
#include <torch/extension.h>
#include <torch/python.h>
#include "kernel_traits.cuh"
#include "utils.cuh"
#include "reg2reg.h"
#include "static_switch.h"

using namespace cute;

template<typename KernelTraits>
__global__ void flash_attention_kernel(
    float_e4m3_t* Q_ptr,
    float_e4m3_t* K_ptr,
    float_e4m3_t* V_ptr,
    void* O_ptr_,
    int BATCH, int M, int N,
    float softmax_scale
) {
    using out_t = typename KernelTraits::out_t;
    using SmemLayoutQ = typename KernelTraits::SmemLayoutQ;
    using SmemLayoutK = typename KernelTraits::SmemLayoutK;
    using SmemLayoutV = typename KernelTraits::SmemLayoutV;

    out_t* O_ptr = reinterpret_cast<out_t*>(O_ptr_);

    const int m_block = blockIdx.x;
    const int base_id = blockIdx.y;
    const int tidx = threadIdx.x;
    const int V_N =  cute::ceil_div(N, 16)*16;
    extern __shared__  char smem[];

    constexpr int BLOCK_M = KernelTraits::BLOCK_M;
    constexpr int BLOCK_N = KernelTraits::BLOCK_N;
    constexpr int HEAD_DIM = KernelTraits::HEAD_DIM;
    constexpr int NUM_WARPS = KernelTraits::NUM_WARPS;
    constexpr int NUM_THREADS = KernelTraits::NUM_THREADS;

    float_e4m3_t* smem_q = reinterpret_cast<float_e4m3_t*>(smem);
    float_e4m3_t* smem_k = smem_q + cosize(SmemLayoutQ{});
    float_e4m3_t* smem_v = smem_k + cosize(SmemLayoutK{});

    const int bs_head_offset_q = base_id * HEAD_DIM * M;
    const int bs_head_offset_k = base_id * HEAD_DIM * N;
    const int bs_head_offset_v = base_id * HEAD_DIM * V_N;
    
    Tensor Q = make_tensor(
        make_gmem_ptr(Q_ptr + bs_head_offset_q),
        make_shape(M, Int<HEAD_DIM>{}),
        make_stride(Int<HEAD_DIM>{}, Int<1>{}));
    Tensor K = make_tensor(
        make_gmem_ptr(K_ptr + bs_head_offset_k),
        make_shape(N, Int<HEAD_DIM>{}),
        make_stride(Int<HEAD_DIM>{}, Int<1>{}));
    Tensor V = make_tensor(
        make_gmem_ptr(V_ptr + bs_head_offset_v),
        make_shape(Int<HEAD_DIM>{}, V_N),
        make_stride(V_N, Int<1>{}));
    Tensor O = make_tensor(
        make_gmem_ptr(O_ptr + bs_head_offset_q),
        make_shape(M, Int<HEAD_DIM>{}),
        make_stride(Int<HEAD_DIM>{}, Int<1>{}));
    
    Tensor gQ = local_tile(Q, make_tile(Int<BLOCK_M>{}, Int<HEAD_DIM>{}), make_coord(m_block, _0{})); 
    Tensor gK = local_tile(K, make_tile(Int<BLOCK_N>{}, Int<HEAD_DIM>{}), make_coord(_, _0{}));
    Tensor gV = local_tile(V, make_tile(Int<HEAD_DIM>{}, Int<BLOCK_N>{}), make_coord(_0{}, _));
    
    typename KernelTraits::MMATile tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tidx);

    Tensor sQ = make_tensor(make_smem_ptr(smem_q), SmemLayoutQ{}); 
    Tensor sK = make_tensor(make_smem_ptr(smem_k), SmemLayoutK{}); 
    Tensor sV = make_tensor(make_smem_ptr(smem_v), SmemLayoutV{}); 

    typename KernelTraits::G2STiledCopyQK gmem_tiled_copy_QK;
    auto gmem_thr_copy_QK = gmem_tiled_copy_QK.get_thread_slice(tidx);
    typename KernelTraits::G2STiledCopyV gmem_tiled_copy_V;
    auto gmem_thr_copy_V = gmem_tiled_copy_V.get_thread_slice(tidx);
    

    Tensor tQgQ = gmem_thr_copy_QK.partition_S(gQ(_, _));
    Tensor tQsQ = gmem_thr_copy_QK.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QK.partition_S(gK(_, _, _));
    Tensor tKsK = gmem_thr_copy_QK.partition_D(sK);

    Tensor tVgV = gmem_thr_copy_V.partition_S(gV(_, _, _));
    Tensor tVsV = gmem_thr_copy_V.partition_D(sV);

    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                          
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                         
    Tensor tOrV  = thr_mma.partition_fragment_B(sV);                           

    
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename KernelTraits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ); 
    auto smem_tiled_copy_K = make_tiled_copy_B(typename KernelTraits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename KernelTraits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsV = smem_thr_copy_V.partition_S(sV);

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));   
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));   
    Tensor cV = make_identity_tensor(make_shape(size<0>(sV), size<1>(sV)));   

    Tensor tQcQ = gmem_thr_copy_QK.partition_S(cQ);       
    Tensor tKcK = gmem_thr_copy_QK.partition_S(cK); 
    Tensor tVcV = gmem_thr_copy_V.partition_S(cV);  

    
    const int n_block_min = 0;
    int n_block_max = cute::ceil_div(N, BLOCK_N); 
    int n_block = n_block_max - 1;

    flash_attention::copy_qk<KernelTraits::IS_EVEN>(gmem_tiled_copy_QK, tQgQ, tQsQ, tQcQ, M - m_block*BLOCK_M);
    flash_attention::copy_qk<KernelTraits::IS_EVEN>(gmem_tiled_copy_QK, tKgK(_, _, _, n_block), tKsK, tKcK, N - n_block * BLOCK_N);
    cute::cp_async_fence();

    Tensor rAccOut_fp32 =  make_tensor_like<float>(partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M>, Int<HEAD_DIM>>{})); 
    Tensor scores_max = make_tensor<float>(Shape<Int<2 * size<1>(rAccOut_fp32)>>{});
    Tensor scores_sum = make_fragment_like(scores_max); 
    clear(rAccOut_fp32); 

    constexpr size_t n_masking_steps = 1;
    
    #pragma unroll
    for(size_t masking_step = 0; masking_step < n_masking_steps; masking_step++, --n_block){
        auto rAccScore_fp16 = partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M>, Int<BLOCK_N>>{}); 
        clear(rAccScore_fp16); 

        flash_attention::cp_async_wait<0>();
        __syncthreads();

        flash_attention::copy_v<KernelTraits::IS_EVEN>(gmem_tiled_copy_V, tVgV(_, _, _, n_block), tVsV, tVcV, N - n_block * BLOCK_N);
        cute::cp_async_fence();

        flash_attention::gemm_smem(rAccScore_fp16, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        auto rAccScore = flash_attention::convert_type<float>(rAccScore_fp16);

        flash_attention::apply_mask(rAccScore, n_block * BLOCK_N, N);
        Tensor scores = make_tensor(rAccScore.data(), flash_attention::convert_layout_acc_rowcol(rAccScore.layout()));
        flash_attention::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            flash_attention::copy_qk<true>(gmem_tiled_copy_QK, tKgK(_, _, _, n_block-1), tKsK, tKcK);           
            cute::cp_async_fence();
        }
        flash_attention::softmax_rescale_o<true>(scores, scores_max, scores_sum, rAccOut_fp32, softmax_scale);
        auto rP = flash_attention::convert_type<float_e4m3_t>(rAccScore);
        auto reg2reg = ReorgCFp8toAFp8();
        reg2reg(rP);
       
        auto tOrPLayout = Layout<Shape<Shape<_4, _2, _2>, Int<size<1>(rP)>, Int<size<2>(tOrV)>>>{};
        Tensor tOrP = make_tensor(rP.data(), tOrPLayout);
        Tensor rAccOut =  partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M>, Int<HEAD_DIM>>{}); 
        flash_attention::gemm_A_in_regs(rAccOut, tOrP, tOrV, tOsV, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        for(int i = 0; i < size(rAccOut); i++){
            rAccOut_fp32(i) += float(rAccOut(i));    
        }
    }
    __syncthreads();
    #pragma unroll
    for (;n_block>=n_block_min; n_block--) {
        auto rAccScore_fp16 = partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M>, Int<BLOCK_N>>{}); 
        clear(rAccScore_fp16); 

        flash_attention::cp_async_wait<0>();
        __syncthreads();
    
        flash_attention::copy_v<true>(gmem_tiled_copy_V, tVgV(_, _, _, n_block), tVsV, tVcV);
        cute::cp_async_fence();
        
        flash_attention::gemm_smem(rAccScore_fp16, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        auto rAccScore = flash_attention::convert_type<float>(rAccScore_fp16);
        Tensor scores = make_tensor(rAccScore.data(), flash_attention::convert_layout_acc_rowcol(rAccScore.layout()));
        flash_attention::cp_async_wait<0>();
        __syncthreads();

        if (n_block > n_block_min) {
            flash_attention::copy_qk<true>(gmem_tiled_copy_QK, tKgK(_, _, _, n_block - 1), tKsK, tKcK);
            cute::cp_async_fence();
        }

        flash_attention::softmax_rescale_o<false>(scores, scores_max, scores_sum, rAccOut_fp32, softmax_scale);
        
        auto rP = flash_attention::convert_type<float_e4m3_t>(rAccScore);
        auto reg2reg = ReorgCFp8toAFp8();
        reg2reg(rP);
       
        auto tOrPLayout = Layout<Shape<Shape<_4, _2, _2>, Int<size<1>(rP)>, Int<size<2>(tOrV)>>>{};
        Tensor tOrP = make_tensor(rP.data(), tOrPLayout);
        Tensor rAccOut =  partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M>, Int<HEAD_DIM>>{}); 
        flash_attention::gemm_A_in_regs(rAccOut, tOrP, tOrV, tOsV, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        for(int i = 0; i < size(rAccOut); i++){
            rAccOut_fp32(i) += float(rAccOut(i));    
        }
    }

    flash_attention::SumOp <float> sum_op;
    flash_attention::quad_allreduce_(scores_sum, scores_sum, sum_op);
    Tensor acc_o_rowcol = make_tensor(rAccOut_fp32.data(), flash_attention::convert_layout_acc_rowcol(rAccOut_fp32.layout()));
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
        float sum = scores_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
            acc_o_rowcol(mi, ni) *= inv_sum;
        }
    }
    
    Tensor rO = flash_attention::convert_type<out_t>(rAccOut_fp32);
    Tensor sO = make_tensor((out_t*)smem_q, typename KernelTraits::SmemLayoutO{});   
    
    auto smem_tiled_copy_O = make_tiled_copy_C(typename KernelTraits::R2SCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);   

    
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
     
    Tensor gO = local_tile(O, make_tile(Int<BLOCK_M>{}, Int<HEAD_DIM>{}), make_coord(m_block, _0{}));
    
    typename KernelTraits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);       
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_S(cO);   
    __syncthreads();
    
    flash_attention::copy_qk<KernelTraits::IS_EVEN>(gmem_tiled_copy_O, tOsO, tOgO, tOcO, M - m_block*BLOCK_M);
}


template <int BLOCK_M, int  BLOCK_N, int HEAD_DIM, bool IS_EVEN, typename accum_type>
void flash_attention_launch(void* q_ptr, void* k_ptr, void* v_ptr, void* o_ptr, int batch_size, int num_heads, int num_q_tokens, int num_k_tokens, float softmax_scale, cudaStream_t stream) {
    constexpr static int NUM_WARPS = BLOCK_M / 16;
    using KernelTraits = attention_kernel_traits<HEAD_DIM, BLOCK_M, BLOCK_N, NUM_WARPS, IS_EVEN, accum_type>;
    const int num_m_block = (num_q_tokens + KernelTraits::BLOCK_M- 1) / KernelTraits::BLOCK_M;
    dim3 grid(num_m_block, batch_size * num_heads, 1);
    auto kernel = &flash_attention_kernel<KernelTraits>;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, KernelTraits::SmemSize);
    kernel<<<grid, KernelTraits::NUM_THREADS, KernelTraits::SmemSize, stream>>>((float_e4m3_t*)q_ptr, (float_e4m3_t*)k_ptr, (float_e4m3_t*)v_ptr, o_ptr, batch_size, num_q_tokens, num_k_tokens, softmax_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<bool use_fast_accum> 
void flash_attention_fp8_cuda(void* q_ptr, void* k_ptr, void* v_ptr, void* o_ptr, int batch_size, int num_heads, int num_q_tokens, int num_k_tokens, int head_dim, float softmax_scale, cudaStream_t stream){
    using accum_type = std::conditional_t<use_fast_accum, half_t, float>;
    HEAD_SWITCH(head_dim, HEAD_DIM, 
        ISEVEN_SWITCH(num_q_tokens, num_k_tokens, IS_EVEN, flash_attention_launch<BLOCK_M, BLOCK_N, HEAD_DIM, IS_EVEN, accum_type>(q_ptr, k_ptr, v_ptr, o_ptr, batch_size, num_heads, num_q_tokens, num_k_tokens, softmax_scale, stream);)
    );
}

template void flash_attention_fp8_cuda<true>(void* q_ptr, void* k_ptr, void* v_ptr, void* o_ptr, int batch_size, int num_heads, int num_q_tokens, int num_k_tokens, int head_dim, float softmax_scale, cudaStream_t stream);
template void flash_attention_fp8_cuda<false>(void* q_ptr, void* k_ptr, void* v_ptr, void* o_ptr, int batch_size, int num_heads, int num_q_tokens, int num_k_tokens, int head_dim, float softmax_scale, cudaStream_t stream);


