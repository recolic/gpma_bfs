#ifndef RLIB_GPMA_MULTIDEV_CUH_
#define RLIB_GPMA_MULTIDEV_CUH_ 1

#include <vector>
#include "gpma.cuh"

namespace gpma_impl {
    template <typename K, typename V>
    struct quick_1way_hash_table {
        static_assert(std::is_integral_v<K>, "only allows integral key");
        std::vector<V> buf;
        quick_1way_hash_table(K max_key, const V &def_value)
            : buf(max_key, def_value)
        {}
        const V &get(const K &k) const {return buf[k];}
        const V &set(const K &k, const V &v) {
#if DEBUG
            return buf.at(k) = v;
#else
            return buf[k] = v;
#endif
        }
    }

    template <size_t cpu_instances, size_t gpu_instances>
    struct dispatcher {
        using CpuArrT = std::array<GPMA<CPU> *, cpu_instances>;
        using GpuArrT = std::array<GPMA<GPU> *, gpu_instances>;
        constexpr KEY_TYPE hashSize = 1024;

        quick_1way_hash_table<KEY_TYPE, size_t> mapKeyToSlot;
        dispatcher()
            : mapKeyToSlot(hashSize, (size_t)(-1)) {}

        // void init(const CpuArrT &ptrs_cpu, const GpuArrT &ptrs_gpu) {}

        // Given KEY, returns the ID(offset from zero) of device, which is responsible to this KEY.
        [[gnu::always_inline]] size_t select_device(const KEY_TYPE &k) {
            auto hashKey = k % hashSize;
            auto dev_id = mapKeyToSlot.get(hashKey);
            if(dev_id == (size_t)(-1)) {
                // appoint a device for a new hash.
                return mapKeyToSlot.set(hashKey % (cpu_instances + gpu_instances));
            }
            else
                return dev_id;
        }
    }
}

template <size_t cpu_instances, size_t gpu_instances>
struct GPMA_multidev {
    constexpr size_t instances = cpu_instances + gpu_instances;
    // TODO: support multi-GPU.
    std::array<GPMA<CPU> *, cpu_instances> ptrs_cpu;
    std::array<GPMA<GPU> *, gpu_instances> ptrs_gpu;
    GPMA_multidev(std::array<size_t, instances> row_nums) {
        // Construct these actual instances.
        for(auto dev_id = 0; dev_id < instances; ++dev_id) {
            if(dev_id < cpu_instances)
                ptrs_cpu.at(dev_id) = new GPMA<CPU>(row_nums[dev_id]);
            else
                ptrs_gpu.at(dev_id - cpu_instances) = new GPMA<GPU>(row_nums[dev_id]);
        }
    }

    GPMA_multidev() : ptrs_cpu{nullptr}, ptrs_gpu{nullptr} {}

    ~GPMA_multidev() {
        for(auto *ptr : ptrs_cpu) delete ptr;
        for(auto *ptr : ptrs_gpu) delete ptr;
    }

    gpma_impl::dispatcher dispatcher;
    void update_batch(NATIVE_VEC_KEY<CPU> &update_keys, NATIVE_VEC_VALUE<CPU> &update_values) {
        // keys MUST be transfer-ed from CPU to GPU.
        // values MAY NOT NECESSARILY to be transfer-ed. TODO: test if `device_vec<>(SIZE, VAL)` occupies the PCIE bandwidth.
        // currently, everything transfer-ed from cpu.

        if(update_keys.size() != update_values.size())
            throw std::invalid_argument("Inconsistant kv size.");

        std::array<std::pair<NATIVE_VEC_KEY<CPU>, NATIVE_VEC_VALUE<CPU>>, cpu_instances> cpu_buffers;
        std::array<std::pair<NATIVE_VEC_KEY<GPU>, NATIVE_VEC_VALUE<GPU>>, gpu_instances> gpu_buffers;

        // TODO: better dispatcher algorithm.
        static_assert(cpu_instances == gpu_instances == 1, "not supporting multicpu/multigpu");
#pragma omp parallel for schedule(static)
        for(auto i = 0; i < update_keys.size(); ++i) {
            auto dev_id = dispatcher.select_device(update_keys[i]);
            if(dev_id < cpu_instances) {
                cpu_buffers[dev_id].first.push_back(update_keys[i]);
                cpu_buffers[dev_id].second.push_back(update_values[i]);
            }
            else {
                gpu_buffers[dev_id - cpu_instances].first.push_back(update_keys[i]);
                gpu_buffers[dev_id - cpu_instances].second.push_back(update_values[i]);
            }
        }
        // Maybe this barrier could be removed in the future.
        // Push GPU data, Run GPU update(in background).
        // Push CPU data, Run CPU update. Then check if GPU has done.

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



