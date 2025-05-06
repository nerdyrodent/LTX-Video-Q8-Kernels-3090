#pragma once

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      __VA_ARGS__                               \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      __VA_ARGS__                               \
    }                                           \

#define K_SWITCH(COSNT_NAME, ...) \
    if (K <= 4096) {                             \
      constexpr static int COSNT_NAME = 2;    \
      __VA_ARGS__                               \
    } else if (K <= 8192) {                      \
      constexpr static int COSNT_NAME = 2;    \
      __VA_ARGS__                               \
    } else if (K <= 16384) {                      \
      constexpr static int COSNT_NAME = 4;    \
      __VA_ARGS__                               \
    } else {                                    \
      printf("Unsupported K value: %d\n", K);   \
    }

#define M_SWITCH(...) \
    if (M <= 1024) {                             \
        constexpr static int BM = 128;    \
        constexpr static int BN = 128;    \
        constexpr static int WARP_ROW = 2;    \
        constexpr static int WARP_COL = 2;    \
      __VA_ARGS__                               \
    } else {                                    \
        constexpr static int BM = 128;    \
        constexpr static int BN = 256;    \
        constexpr static int WARP_ROW = 2;    \
        constexpr static int WARP_COL = 4;    \
        __VA_ARGS__                           \
    }                                          