#pragma once

#define HEAD_SWITCH(VAR_NAME, CONST_NAME, ...)    \
    if (VAR_NAME == 64) {                         \
      constexpr static int BLOCK_M = 64;                \
      constexpr static int BLOCK_N = 64;                \
      constexpr static int NUM_WARPS = 4;          \
      constexpr static int CONST_NAME = 64;       \
      __VA_ARGS__                                 \
    } else if (VAR_NAME == 128){                  \
        constexpr static int BLOCK_M = 128;                \
        constexpr static int BLOCK_N = 256;                \
        constexpr static int NUM_WARPS = 8;  \
        constexpr static int CONST_NAME = 128;      \
        __VA_ARGS__                                 \
    }                                           

  #define ISEVEN_SWITCH(VAR_NAME_1, VAR_NAME_2, CONST_NAME, ...)                                      \
      if((VAR_NAME_1 % BLOCK_M == 0) && (VAR_NAME_2 % BLOCK_N == 0)) {                                \
          constexpr static bool CONST_NAME = true;                                                       \
          __VA_ARGS__                                                                                 \
        } else  {                                                                                     \
          constexpr static bool CONST_NAME = false;                                                      \
          __VA_ARGS__                                                                                 \
        }                                                                                             