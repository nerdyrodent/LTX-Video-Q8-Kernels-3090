#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/layout/layout.h>
#include <cutlass/numeric_types.h>

#include "mma_sm89_fp16.hpp"
#include "mma_traits_sm89_fp16.hpp"
using namespace cute;
template<int BYTES> struct BytesToType {};
template<> struct BytesToType<4> {
    using Type = uint32_t;
    static_assert(sizeof(Type) == 4);
};
template<> struct BytesToType<2> {
    using Type = uint16_t;
    static_assert(sizeof(Type) == 2);
};
template<int HEAD_DIM_, int BLOCK_M_, int BLOCK_N_, int NUM_WARPS_, bool IS_EVEN_, typename accum_t_>
struct attention_kernel_traits {
    using accum_t  = accum_t_;
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, float_e4m3_t>;
    using out_t = cutlass::bfloat16_t;
    
    static constexpr  bool IS_EVEN = IS_EVEN_;
    static constexpr int BLOCK_M = BLOCK_M_;
    static constexpr int BLOCK_N = BLOCK_N_;
    static constexpr int HEAD_DIM = HEAD_DIM_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;

    using SmemLayoutAtom = decltype(composition(
        Swizzle<2, 4, 3>{},
        make_layout(make_shape(Int<8>{}, Int<64>{}),
                    make_stride(Int<64>{}, Int<1>{}))));
    
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<BLOCK_M>, Int<HEAD_DIM>>{})
    );
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<BLOCK_N>, Int<HEAD_DIM>>{})
    );
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<HEAD_DIM>, Int<BLOCK_N>>{})
    );
    
    using MMA_Atom_SM89 = std::conditional_t<
        std::is_same_v<accum_t, cutlass::half_t>,
        MMA_Atom<SM89_16x8x32_F16E4M3E4M3F16_TN>,
        MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>
    >;
    using MMATile = decltype(
        make_tiled_mma(
            MMA_Atom_SM89{},
            Layout<Shape<Int<NUM_WARPS>, _1, _1>>{},
            Tile<Int<NUM_WARPS*16>, _16, _32>{}
        )
    );
    constexpr static  int INPUT_ELEMS_PER_COPY = sizeof(uint128_t) / sizeof(float_e4m3_t);
    constexpr static  int OUTPUT_ELEMS_PER_COPY = sizeof(uint128_t) / sizeof(out_t); 
    constexpr static  int THREADS_PER_ROW_QK = HEAD_DIM / INPUT_ELEMS_PER_COPY;
    constexpr static  int THREADS_PER_ROW_V = BLOCK_N / INPUT_ELEMS_PER_COPY;
    constexpr static  int THREADS_PER_ROW_O = HEAD_DIM / OUTPUT_ELEMS_PER_COPY;

    using G2SCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, float_e4m3_t>;
    using G2STiledCopyQK = decltype(
        make_tiled_copy(
            G2SCopyAtom{},
            Layout<Shape<Int<NUM_THREADS / THREADS_PER_ROW_QK>, Int<THREADS_PER_ROW_QK>>, Stride<Int<THREADS_PER_ROW_QK>, _1>>{},
            Layout<Shape<_1, Int<INPUT_ELEMS_PER_COPY>>>{}
        )
    );
    using G2STiledCopyV = decltype(
        make_tiled_copy(
            G2SCopyAtom{},
            Layout<Shape<Int<NUM_THREADS / THREADS_PER_ROW_V>, Int<THREADS_PER_ROW_V>>, Stride<Int<THREADS_PER_ROW_V>, _1>>{},
            Layout<Shape<_1, Int<INPUT_ELEMS_PER_COPY>>>{}
        )
    );
    using SwizzleLayoutO = std::conditional_t<
        std::is_same_v<out_t, cutlass::bfloat16_t>,
        Swizzle<3, 3, 3>,
        Swizzle<2, 4, 3>
    >;

    using R2SCopyAtomO = Copy_Atom<UniversalCopy<typename BytesToType<2*sizeof(out_t)>::Type>, out_t>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, out_t>{},
            Layout<Shape<Int<NUM_THREADS / THREADS_PER_ROW_O>, Int<THREADS_PER_ROW_O>>, Stride<Int<THREADS_PER_ROW_O>, _1>>{},
                        Layout<Shape<_1, Int<OUTPUT_ELEMS_PER_COPY>>>{}));  
    
    constexpr static int OUT_PIPE = 64 / sizeof(out_t);
    using SmemLayoutOAtom = decltype(composition(
        SwizzleLayoutO{},
        make_layout(make_shape(Int<8>{}, Int<OUT_PIPE>{}),
                    make_stride(Int<OUT_PIPE>{}, Int<1>{}))));
    using SmemLayoutO = decltype(
        tile_to_shape(SmemLayoutOAtom{}, make_shape(Int<BLOCK_M>{}, Int<HEAD_DIM>{}))
    );
    
    constexpr static int SmemSize = cosize(SmemLayoutQ{}) + cosize(SmemLayoutK{}) + cosize(SmemLayoutV{});
};