#ifndef RLIB_IMPL_GPMA_DEBUG_HPP_
#define RLIB_IMPL_GPMA_DEBUG_HPP_ 1

#include <time.h>
#include <cstdio>
#include <cstdint>

inline int64_t get_time_in_us() {
    struct timespec t;
    static int64_t prev_time = 0;
    clock_gettime(CLOCK_MONOTONIC, &t);
    auto this_time = ((int64_t)(t.tv_sec) * (int64_t)1000000000 + (int64_t)(t.tv_nsec)) / 1000;
    auto delta_time = this_time - prev_time;
    prev_time = this_time;
    return delta_time;
}
#define LOG_TIME(msg) printf("T+%lld - " msg "\n", get_time_in_us());

#endif

