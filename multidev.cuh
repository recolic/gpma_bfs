#ifndef RLIB_GPMA_MULTIDEV_CUH_
#define RLIB_GPMA_MULTIDEV_CUH_ 1

#include <vector>
#include "gpma.cuh"

template <dev_type_t... devs>
struct GPMA_multidev {
    // TODO: support multi-GPU.
    std::tuple<GPMA<devs> * ...> ptrs;
    GPMA_multidev(size_t row_nums [sizeof...(devs)])
        : ptrs(new GPMA<devs>(row_nums) ...) {
    }

    GPMA_multidev() : ptrs{nullptr} {}

    ~GPMA_multidev() {
        for(auto *ptr : ptrs) {
            if(ptr) delete ptr;
        }
    }

    // dispatcher.
    void update_batch(NATIVE_VEC_KEY<CPU> &update_keys, NATIVE_VEC_VALUE<CPU> &update_values) {
        // keys MUST be transfer-ed from CPU to GPU.
        // values MAY NOT NECESSARILY to be transfer-ed. TODO: test if `device_vec<>(SIZE, VAL)` occupies the PCIE bandwidth.
        // currently, everything transfer-ed from cpu.

        if(update_keys.size() != update_values.size())
            throw std::invalid_argument("Inconsistant kv size.");

        std::tuple<NATIVE_VEC_KEY<devs> ...> keys_dispatched_buffers;
        std::tuple<NATIVE_VEC_VALUE<devs> ...> values_dispatched_buffers;

        // TODO: better dispatcher algorithm.
        static_assert(std::get<0>(std::make_tuple(devs...)) == GPU && std::get<1>(std::make_tuple(devs...)) == CPU);
#pragma omp parallel for schedule(static)
        for(auto i = 0; i < update_keys.size() / 2; ++i) {
            std::get<0>(keys_dispatched_buffers).push_back(update_keys[i]);
            std::get<0>(valuess_dispatched_buffers).push_back(update_values[i]);
        }
        // TODO: remove this barrier
#pragma omp parallel for schedule(static)
        for(auto i = update_keys.size() / 2; i < update_keys.size(); ++i) {
            std::get<1>(keys_dispatched_buffers).push_back(update_keys[i]);
            std::get<1>(valuess_dispatched_buffers).push_back(update_values[i]);
        }
        // TODO: remove this barrier

        anySync<GPU>();

#pragma omp parallel for schedule(dynamic)
        for(auto i = 0; i < sizeof...(devs); ++i) {
            ::update_gpma(*ptr, keys_dispatched_buffers[i], values_dispatched_buffers[i]);
        }
        // barrier.

        anySync<GPU>();
    }

}

#endif



