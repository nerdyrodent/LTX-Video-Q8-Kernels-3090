#pragma once

#include <cute/config.hpp>
#include <cute/arch/mma.hpp>
////////////////////////////////////////////////////////////////////////////////

#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4)
#  define CUTE_ARCH_MMA_F32_SM89_SUPPORTED
#endif

#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8)
#  define CUTE_ARCH_MMA_F16_SM89_SUPPORTED
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#  if defined(CUTE_ARCH_MMA_F32_SM89_SUPPORTED)
#    define CUTE_ARCH_MMA_F32_SM89_ENABLED
#  endif

#  if defined(CUTE_ARCH_MMA_F16_SM89_SUPPORTED)
#    define CUTE_ARCH_MMA_F16_SM89_ENABLED
#  endif
#endif

namespace cute {
struct SM89_16x8x32_F32E4M3E4M3F32_TN
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float      & d0, float      & d1, float      & d2, float      & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const& c0, float const& c1, float const& c2, float const& c3)
  {
#if defined(CUTE_ARCH_MMA_F32_SM89_ENABLED)
    asm(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :
        "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(c0), "f"(c1), "f"(c2), "f"(c3)
  );
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_F32E4M3E4M3F32_TN without CUTE_ARCH_MMA_F32_SM89_ENABLED");
#endif
  }
};

  // MMA 16x8x32 TN
struct SM89_16x8x32_F16E4M3E4M3F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_F16_SM89_ENABLED)
    asm(
      "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
      : "=r"(d0), "=r"(d1)
      :
        "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "r"(c0), "r"(c1)
  );
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_F32E4M3E4M3F32_TN without CUTE_ARCH_MMA_F16_SM89_ENABLED");
#endif
  }
};
}