#ifndef RLIB_IMPL_GPMA_DEBUG_HPP_
#define RLIB_IMPL_GPMA_DEBUG_HPP_ 1

#include <time.h>
#include <cstdio>
#include <cstdint>

inline int64_t get_time_in_us() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return ((int64_t)(t.tv_sec) * (int64_t)1000000000 + (int64_t)(t.tv_nsec)) / 1000;
}
#define LOG_TIME(msg) printf("T+%lld - " msg "\n", get_time_in_us());

#endif

