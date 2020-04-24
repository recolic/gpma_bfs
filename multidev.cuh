#ifndef RLIB_GPMA_MULTIDEV_CUH_
#define RLIB_GPMA_MULTIDEV_CUH_ 1

#include <vector>
#include <type_traits>
#include "gpma.cuh"
#include <rlib/meta.hpp>

namespace gpma_impl {
    template <typename K, typename V>
    struct quick_1way_hash_table {
        static_assert(std::is_integral<K>::value, "only allows integral key");
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
    };

    template <size_t cpu_instances, size_t gpu_instances>
    struct dispatcher {
        static constexpr KEY_TYPE hashSize = 1024;

        quick_1way_hash_table<KEY_TYPE, size_t> mapKeyToSlot;
        dispatcher()
            : mapKeyToSlot(hashSize, (size_t)(-1)) {}

        // void init(const CpuArrT &ptrs_cpu, const GpuArrT &ptrs_gpu) {}
        static constexpr size_t gpu_factor = 7; // 1 GPU is equals to 7 CPU.

        // Given KEY, returns the ID(offset from zero) of device, which is responsible to this KEY.
        [[gnu::always_inline]] size_t select_device(const KEY_TYPE &k) {
            if(cpu_instances + gpu_instances == 1) return 0;
            auto hashKey = k % hashSize;
            auto dev_id = mapKeyToSlot.get(hashKey);
            if(dev_id == (size_t)(-1)) {
                // appoint a device for a new hash.
                dev_id = hashKey % (cpu_instances + gpu_instances * gpu_factor);
                dev_id = (dev_id > cpu_instances) ? ( cpu_instances + (dev_id-cpu_instances)/gpu_factor ) : dev_id;
                // Add link: hashKey => dev_id
                return mapKeyToSlot.set(hashKey, dev_id);
            }
            else
                return dev_id;
        }
    };

    template <dev_type_t DEV, size_t instances>
    struct thread_safe_kv_buf {
        std::array<NATIVE_VEC_KEY<DEV>, instances> k_buffers;
        std::array<NATIVE_VEC_VALUE<DEV>, instances> v_buffers;
        std::array<std::atomic<size_t>, instances> sizes;
        thread_safe_kv_buf(size_t max_size) {
            for(auto &buf : k_buffers) buf.resize(max_size);
            for(auto &buf : v_buffers) buf.resize(max_size);
            std::fill(sizes.begin(), sizes.end(), 0);
        }
        void push_back(const size_t &dev_id, const KEY_TYPE &k, const VALUE_TYPE &v) {
            // thread safe.
            const auto pos = sizes[dev_id]++;
            k_buffers[dev_id][pos] = k;
            v_buffers[dev_id][pos] = v;
        }
    };
}

template <size_t cpu_instances, size_t gpu_instances>
struct GPMA_multidev {
    static constexpr size_t instances = cpu_instances + gpu_instances;
    static_assert(instances > 0, "Need at lease one DEV instance.");
    // TODO: support multi-GPU.
    std::array<GPMA<CPU> *, cpu_instances> ptrs_cpu;
    std::array<GPMA<GPU> *, gpu_instances> ptrs_gpu;
    GPMA_multidev(size_t row_num) {
        // Construct these actual instances.
        for(auto dev_id = 0; dev_id < instances; ++dev_id) {
            if(dev_id < cpu_instances)
                ptrs_cpu.at(dev_id) = new GPMA<CPU>(row_num);
            else
                ptrs_gpu.at(dev_id - cpu_instances) = new GPMA<GPU>(row_num);
        }
    }

    GPMA_multidev() : ptrs_cpu{nullptr}, ptrs_gpu{nullptr} {}

    ~GPMA_multidev() {
        for(auto *ptr : ptrs_cpu) delete ptr;
        for(auto *ptr : ptrs_gpu) delete ptr;
    }

    gpma_impl::dispatcher<cpu_instances, gpu_instances> dispatcher;
    void update_batch(NATIVE_VEC_KEY<CPU> &update_keys, NATIVE_VEC_VALUE<CPU> &update_values) {
        // keys MUST be transfer-ed from CPU to GPU.
        // values MAY NOT NECESSARILY to be transfer-ed. TODO: test if `device_vec<>(SIZE, VAL)` occupies the PCIE bandwidth.
        // currently, everything transfer-ed from cpu.

        if(update_keys.size() != update_values.size())
            throw std::invalid_argument("Inconsistant kv size.");

        gpma_impl::thread_safe_kv_buf<CPU, cpu_instances + gpu_instances> cpu_buffers(update_keys.size());
        gpma_impl::thread_safe_kv_buf<GPU, gpu_instances> gpu_buffers(update_keys.size());

// #pragma omp parallel for schedule(static) // not supporting parallel push back
        for(auto i = 0; i < update_keys.size(); ++i) {
            auto dev_id = dispatcher.select_device(update_keys[i]);
            //rlib::printfln("PUSH, dev_id={}, i={}, k={}, v={}.", dev_id, i, update_keys[i], update_values[i]);
            // Fill cpu buffers.
            cpu_buffers.push_back(dev_id, update_keys[i], update_values[i]);
        }
        for(auto dev_id = 0; dev_id < cpu_instances + gpu_instances; ++dev_id) {
            cpu_buffers.k_buffers[dev_id].resize(cpu_buffers.sizes[dev_id]);
            cpu_buffers.v_buffers[dev_id].resize(cpu_buffers.sizes[dev_id]);
        }


#pragma omp parallel for schedule(dynamic)
        for(auto dev_id = 0; dev_id < instances; ++dev_id) {
            if(dev_id < cpu_instances)
                ::update_gpma(*(ptrs_cpu[dev_id]), cpu_buffers.k_buffers[dev_id], cpu_buffers.v_buffers[dev_id]);
            else {
                NATIVE_VEC_KEY<GPU> gpu_k_buf (cpu_buffers.k_buffers[dev_id]);
                NATIVE_VEC_VALUE<GPU> gpu_v_buf (cpu_buffers.v_buffers[dev_id]);
                auto gpu_id = dev_id - cpu_instances;
                ::update_gpma(*(ptrs_gpu[gpu_id]), gpu_k_buf, gpu_v_buf);
            }
        }
        // barrier.

        anySync<GPU>();
    }

};

#endif



