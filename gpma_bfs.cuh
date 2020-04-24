#pragma once

#include "cub/cub.cuh"
#include "utils.cuh"
#include "multidev.cuh"
#include <thread>
#include <mutex>
#include <atomic>
#include <lib/barrier.hpp>

#define FULL_MASK 0xffffffff

namespace impl {
__host__ __device__ inline bool gpma_bitmap_get(SIZE_TYPE *bitmap, size_t bit_offset) {
    // Host should call this function with bitmap in CPU.
    // Device should call this function with bitmap in GPU.
    SIZE_TYPE bit_loc = 1 << (bit_offset % 32);
    SIZE_TYPE bit_chunk = bitmap[bit_offset / 32];
    return (bit_chunk & bit_loc);
}
__host__ __device__ inline bool gpma_bitmap_set_return_old(SIZE_TYPE *bitmap, size_t bit_offset) {
    // Host should call this function with bitmap in CPU.
    // Device should call this function with bitmap in GPU.
    SIZE_TYPE bit_loc = 1 << (bit_offset % 32);
    SIZE_TYPE bit_chunk = bitmap[bit_offset / 32];
    bool old = (bit_chunk & bit_loc);
    // RWLock required here. TODO
    bitmap[bit_offset / 32] = bit_chunk | bit_loc;
    return old;
}
//__host__ inline void atomic_push_to_queue(SIZE_TYPE *queue, SIZE_TYPE &queue_offset) {
// TODO
//}

__host__ inline void gpma_bitmap_merge(SIZE_TYPE *cpu_bitmap, SIZE_TYPE *gpu_bitmap, size_t bitmap_size_in_byte) {
    // Merge the bitmaps in argument, and set both of them to the merged result.
    auto cpu_bitmap2 = (decltype(cpu_bitmap))malloc(bitmap_size_in_byte);
    assert(cpu_bitmap2 != NULL);
    anyMemcpy<GPU, CPU>(cpu_bitmap2, gpu_bitmap, bitmap_size_in_byte);

    assert(bitmap_size_in_byte % sizeof(SIZE_TYPE) == 0);
#pragma omp parallel for
    for(auto i = 0; i < bitmap_size_in_byte / sizeof(SIZE_TYPE); ++i) {
        cpu_bitmap[i] = cpu_bitmap[i] | cpu_bitmap2[i];
    }
    
    anyMemcpy<CPU, GPU>(gpu_bitmap, cpu_bitmap, bitmap_size_in_byte);
    free(cpu_bitmap2);
}
__host__ inline void gpma_queue_merge(SIZE_TYPE *cpu_queue, SIZE_TYPE *gpu_queue, SIZE_TYPE &cpu_queue_size, SIZE_TYPE *gpu_queue_size, const size_t max_size) {
    // Merge the queues in argument, and set both of them to the merged result.
    auto gpu_queue_size_in_cpu = new SIZE_TYPE;
    anyMemcpy<GPU, CPU>(gpu_queue_size_in_cpu, gpu_queue_size, sizeof(SIZE_TYPE));
    assert(max_size > cpu_queue_size + *gpu_queue_size_in_cpu);
    anyMemcpy<GPU, CPU>(cpu_queue + cpu_queue_size, gpu_queue, *gpu_queue_size_in_cpu * sizeof(SIZE_TYPE));
    anyMemcpy<CPU, GPU>(gpu_queue + *gpu_queue_size_in_cpu, cpu_queue, cpu_queue_size * sizeof(SIZE_TYPE));
    cpu_queue_size = *gpu_queue_size_in_cpu = *gpu_queue_size_in_cpu + cpu_queue_size;
    anyMemcpy<CPU, GPU>(gpu_queue_size, gpu_queue_size_in_cpu, sizeof(SIZE_TYPE));
    delete gpu_queue_size_in_cpu;
}

template <size_t cpus, size_t gpus>
struct multidev_bfs_data {
    static_assert(/*cpus < 2 && */gpus < 2, "Current code shares the same bitmap/nodeQ/edgeQ for all CPUs. Also share them for all GPUs. So at most 1 cpu/gpu allowed. cba97278");
    native_vector<CPU, SIZE_TYPE> cpu_results, cpu_bitmap, cpu_nodeQ, cpu_edgeQ;
    native_vector<GPU, SIZE_TYPE> gpu_results, gpu_bitmap, gpu_nodeQ, gpu_edgeQ;
    SIZE_TYPE cpu_nodeQ_size = 0, cpu_edgeQ_size = 0;
    SIZE_TYPE *gpu_nodeQ_size, *gpu_edgeQ_size;

    static constexpr size_t _cpus = cpus;

    static constexpr size_t get_bitmap_size(size_t node_size) {
        return ((node_size - 1) / sizeof(SIZE_TYPE) + 1); // in nSIZE_T s, rather than bytes/bits.
    }    
    multidev_bfs_data(size_t node_size, size_t edge_size)
        : cpu_results(node_size, 0), gpu_results(node_size, 0),
          cpu_nodeQ(3*node_size), gpu_nodeQ(3*node_size),
          cpu_edgeQ(3*edge_size), gpu_edgeQ(3*edge_size),
          cpu_bitmap(get_bitmap_size(node_size), 0), gpu_bitmap(get_bitmap_size(node_size), 0),
          online_guys(cpus + gpus), barrier(2), select_one_thread(false) {
     // currently the barrier is initialized as 2 threads (1 for cpu and 1 for gpu). If you have bfs fully parallelized for multi-cpu/gpu, 
     // you may use cpus+gpus threads here. cba97278
        anyMalloc<GPU>((void**)&gpu_nodeQ_size, sizeof(SIZE_TYPE));
        anyMalloc<GPU>((void**)&gpu_edgeQ_size, sizeof(SIZE_TYPE));
        anyMemcpy<CPU, GPU>(gpu_nodeQ_size, &cpu_nodeQ_size, sizeof(SIZE_TYPE));
        anyMemcpy<CPU, GPU>(gpu_edgeQ_size, &cpu_edgeQ_size, sizeof(SIZE_TYPE));
        rlib::println("DEBUG: constructing bfs_data, ptrs=", RAW_PTR(gpu_results), RAW_PTR(gpu_bitmap), RAW_PTR(gpu_nodeQ), RAW_PTR(gpu_edgeQ));
    }

    ~multidev_bfs_data() {
        rlib::println("DEBUG: de-constructing bfs_data, ptrs=", RAW_PTR(gpu_results), RAW_PTR(gpu_bitmap), RAW_PTR(gpu_nodeQ), RAW_PTR(gpu_edgeQ));
    }

    std::atomic<size_t> online_guys;
    std::atomic<bool> select_one_thread;
    Barrier barrier;
    void iteration_barrier_1(size_t dev_id) {
        barrier.wait();
        static bool _false = false;
        if(select_one_thread.compare_exchange_strong(_false, true)) { // ONLY ONE thread should do the following things:
            gpma_queue_merge(RAW_PTR(cpu_edgeQ), RAW_PTR(gpu_edgeQ), cpu_edgeQ_size, gpu_edgeQ_size, cpu_edgeQ.size());

            cpu_nodeQ_size = 0;
            anySetVal<GPU>(gpu_nodeQ_size, 0u);
        }
        barrier.wait();
    }
    bool iteration_barrier_2(size_t dev_id) {
        // return true IF ALL workers are completed.
        barrier.wait();
        static bool _false = false;
        if(select_one_thread.compare_exchange_strong(_false, true)) { // ONLY ONE thread should do the following things:
#pragma omp parallel for num_threads(3)
            for(auto thread_id = 0; thread_id < 3; ++thread_id) {
                switch(thread_id) {
                    case 0:
            gpma_queue_merge(RAW_PTR(cpu_nodeQ), RAW_PTR(gpu_nodeQ), cpu_nodeQ_size, gpu_nodeQ_size, cpu_nodeQ.size());
            cpu_edgeQ_size = 0;
            anySetVal<GPU>(gpu_edgeQ_size, 0u);
                    break; case 1:
            gpma_bitmap_merge(RAW_PTR(cpu_bitmap), RAW_PTR(gpu_bitmap), cpu_bitmap.size() * sizeof(SIZE_TYPE));
                    break; case 2:
            // bitmap bit-or merge also works for results array!
            gpma_bitmap_merge(RAW_PTR(cpu_results), RAW_PTR(gpu_results), cpu_results.size() * sizeof(SIZE_TYPE));
                }
            }
        }

        /* Code here are designed for multi-CPU or multi-GPU case.
        bool completed = 0 == anyGetVal<DEV>(node_queue_size);
        if(completed)
            -- online_guys;
        barrier.wait();
        if(online_guys == 0)
            return true; // ALL DONE
        else
            online_guys = cpus + gpus;
        return false;
        */

        bool all_completed = cpu_nodeQ_size + anyGetVal<GPU>(gpu_nodeQ_size) == 0;
        barrier.wait();
        return all_completed;
    }

};
}

// For every node in NodeQ, push its neighbor node to EdgeQ.
template <SIZE_TYPE THREADS_NUM>
__global__ void gpma_bfs_gather_kernel(SIZE_TYPE *node_queue, SIZE_TYPE *node_queue_size, SIZE_TYPE *edge_queue, SIZE_TYPE *edge_queue_size, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *row_offsets) {

    typedef cub::BlockScan<SIZE_TYPE, THREADS_NUM> BlockScan;
    __shared__ typename BlockScan::TempStorage block_temp_storage;

    volatile __shared__ SIZE_TYPE comm[THREADS_NUM / 32][3];
    volatile __shared__ SIZE_TYPE comm2[THREADS_NUM];
    volatile __shared__ SIZE_TYPE output_cta_offset;
    volatile __shared__ SIZE_TYPE output_warp_offset[THREADS_NUM / 32];

    typedef cub::WarpScan<SIZE_TYPE> WarpScan;
    __shared__ typename WarpScan::TempStorage temp_storage[THREADS_NUM / 32];

    SIZE_TYPE thread_id = threadIdx.x;
    SIZE_TYPE lane_id = thread_id % 32;
    SIZE_TYPE warp_id = thread_id / 32;

    SIZE_TYPE cta_offset = blockDim.x * blockIdx.x;
    while (cta_offset < node_queue_size[0]) {
        SIZE_TYPE node, row_begin, row_end;
        if (cta_offset + thread_id < node_queue_size[0]) {
            node = node_queue[cta_offset + thread_id];
            row_begin = row_offsets[node];
            row_end = row_offsets[node + 1];
        } else
            row_begin = row_end = 0;

        // CTA-based coarse-grained gather
        while (__syncthreads_or(row_end - row_begin >= THREADS_NUM)) {
            // vie for control of block
            if (row_end - row_begin >= THREADS_NUM)
                comm[0][0] = thread_id;
            __syncthreads();

            // winner describes adjlist
            if (comm[0][0] == thread_id) {
                comm[0][1] = row_begin;
                comm[0][2] = row_end;
                row_begin = row_end;
            }
            __syncthreads();

            SIZE_TYPE gather = comm[0][1] + thread_id;
            SIZE_TYPE gather_end = comm[0][2];
            SIZE_TYPE neighbour;
            SIZE_TYPE thread_data_in;
            SIZE_TYPE thread_data_out;
            SIZE_TYPE block_aggregate;
            while (__syncthreads_or(gather < gather_end)) {
                if (gather < gather_end) {
                    // RDEBUG: ACTUAL LOGIC BEGIN: PUSH the destination node (neighbor) INTO edgeQ.
                    KEY_TYPE cur_key = keys[gather];
                    VALUE_TYPE cur_value = values[gather];
                    neighbour = (SIZE_TYPE)(cur_key & COL_IDX_NONE); // get low 32b, which is Edge.TO.
                    thread_data_in = (neighbour == COL_IDX_NONE || cur_value == VALUE_NONE) ? 0 : 1; // DO NOTHING if NULL.
                } else
                    thread_data_in = 0; // NOTHING TO DO.

                __syncthreads();
                BlockScan(block_temp_storage).ExclusiveSum(thread_data_in, thread_data_out, block_aggregate);
                // block_aggregate stores the final sum of all threads: how many valid task in this turn?
                __syncthreads();
                if (0 == thread_id) {
                    output_cta_offset = atomicAdd(edge_queue_size, block_aggregate);
                }
                __syncthreads();
                if (thread_data_in)
                    edge_queue[output_cta_offset + thread_data_out] = neighbour; // THE ONLY USEFUL STATEMENT!
                // RDEBUG: ACTUAL LOGIC END: PUSH the destination node (neighbor) INTO edgeQ.
                gather += THREADS_NUM;
            }
        }

        // warp-based coarse-grained gather
        while (__any_sync(FULL_MASK, row_end - row_begin >= 32)) {
            // vie for control of warp
            if (row_end - row_begin >= 32)
                comm[warp_id][0] = lane_id;

            // winner describes adjlist
            if (comm[warp_id][0] == lane_id) {
                comm[warp_id][1] = row_begin;
                comm[warp_id][2] = row_end;
                row_begin = row_end;
            }

            SIZE_TYPE gather = comm[warp_id][1] + lane_id;
            SIZE_TYPE gather_end = comm[warp_id][2];
            SIZE_TYPE neighbour;
            SIZE_TYPE thread_data_in;
            SIZE_TYPE thread_data_out;
            SIZE_TYPE warp_aggregate;
            while (__any_sync(FULL_MASK, gather < gather_end)) {
                if (gather < gather_end) {
                    KEY_TYPE cur_key = keys[gather];
                    VALUE_TYPE cur_value = values[gather];
                    neighbour = (SIZE_TYPE)(cur_key & COL_IDX_NONE);
                    thread_data_in = (neighbour == COL_IDX_NONE || cur_value == VALUE_NONE) ? 0 : 1;
                } else
                    thread_data_in = 0;

                WarpScan(temp_storage[warp_id]).ExclusiveSum(thread_data_in, thread_data_out, warp_aggregate);

                if (0 == lane_id) {
                    output_warp_offset[warp_id] = atomicAdd(edge_queue_size, warp_aggregate);
                }

                if (thread_data_in)
                    edge_queue[output_warp_offset[warp_id] + thread_data_out] = neighbour;
                gather += 32;
            }
        }

        // scan-based fine-grained gather
        SIZE_TYPE thread_data = row_end - row_begin;
        SIZE_TYPE rsv_rank;
        SIZE_TYPE total;
        SIZE_TYPE remain;
        __syncthreads();
        BlockScan(block_temp_storage).ExclusiveSum(thread_data, rsv_rank, total); // total = how many tasks left in this block?
        __syncthreads();

        SIZE_TYPE cta_progress = 0;
        while (cta_progress < total) {
            remain = total - cta_progress;

            // share batch of gather offsets
            while ((rsv_rank < cta_progress + THREADS_NUM) && (row_begin < row_end)) {
                comm2[rsv_rank - cta_progress] = row_begin;
                rsv_rank++;
                row_begin++;
            }
            __syncthreads();
            SIZE_TYPE neighbour;
            // gather batch of adjlist
            if (thread_id < min(remain, THREADS_NUM)) {
                KEY_TYPE cur_key = keys[comm2[thread_id]];
                VALUE_TYPE cur_value = values[comm2[thread_id]];
                neighbour = (SIZE_TYPE)(cur_key & COL_IDX_NONE);
                thread_data = (neighbour == COL_IDX_NONE || cur_value == VALUE_NONE) ? 0 : 1;
            } else
                thread_data = 0;
            __syncthreads();

            SIZE_TYPE scatter;
            SIZE_TYPE block_aggregate;

            BlockScan(block_temp_storage).ExclusiveSum(thread_data, scatter, block_aggregate);
            __syncthreads();

            if (0 == thread_id) {
                output_cta_offset = atomicAdd(edge_queue_size, block_aggregate);
            }
            __syncthreads();

            if (thread_data)
                edge_queue[output_cta_offset + scatter] = neighbour;
            cta_progress += THREADS_NUM;
            __syncthreads();
        }

        cta_offset += blockDim.x * gridDim.x;
    }
}

// For every node in edgeQ, set results[node]=level for ONLY NEW nodes. Then put these NEW nodes in nodeQ.
template <SIZE_TYPE THREADS_NUM>
__global__ void gpma_bfs_contract_kernel(SIZE_TYPE *edge_queue, SIZE_TYPE *edge_queue_size, SIZE_TYPE *node_queue, SIZE_TYPE *node_queue_size, SIZE_TYPE level, SIZE_TYPE *label, SIZE_TYPE *bitmap) {

    typedef cub::BlockScan<SIZE_TYPE, THREADS_NUM> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    volatile __shared__ SIZE_TYPE output_cta_offset;

    volatile __shared__ SIZE_TYPE warp_cache[THREADS_NUM / 32][128];
    const SIZE_TYPE HASH_KEY1 = 1097;
    const SIZE_TYPE HASH_KEY2 = 1103;
    volatile __shared__ SIZE_TYPE cta1_cache[HASH_KEY1];
    volatile __shared__ SIZE_TYPE cta2_cache[HASH_KEY2];

    // init cta-level cache
    for (int i = threadIdx.x; i < HASH_KEY1; i += blockDim.x)
        cta1_cache[i] = SIZE_NONE;
    for (int i = threadIdx.x; i < HASH_KEY2; i += blockDim.x)
        cta2_cache[i] = SIZE_NONE;
    __syncthreads();

    SIZE_TYPE thread_id = threadIdx.x;
    SIZE_TYPE warp_id = thread_id / 32;
    SIZE_TYPE cta_offset = blockDim.x * blockIdx.x;

    while (cta_offset < edge_queue_size[0]) {
        SIZE_TYPE neighbour;
        SIZE_TYPE valid = 0;

        do {
            if (cta_offset + thread_id >= edge_queue_size[0])
                break;
            neighbour = edge_queue[cta_offset + thread_id];

            // warp cull
            SIZE_TYPE hash = neighbour & 127; // 0x7f
            warp_cache[warp_id][hash] = neighbour;
            SIZE_TYPE retrieved = warp_cache[warp_id][hash];
            if (retrieved == neighbour) {
                warp_cache[warp_id][hash] = thread_id;
                if (warp_cache[warp_id][hash] != thread_id)
                    break;
            }

            // history cull
            if (cta1_cache[neighbour % HASH_KEY1] == neighbour)
                break;
            if (cta2_cache[neighbour % HASH_KEY2] == neighbour)
                break;
            cta1_cache[neighbour % HASH_KEY1] = neighbour;
            cta2_cache[neighbour % HASH_KEY2] = neighbour;

            // bitmap check: if bitmap[neighbour].isSet(): break
            //               else bitmap[neighbour].set()
            bool oldbit = impl::gpma_bitmap_set_return_old(bitmap, neighbour);
            if(oldbit)
                break;

            // note: `label` is `results`
            SIZE_TYPE ret = atomicCAS(label + neighbour, 0, level); // if label[neighbour] == 0: label[neighbour] = level
            valid = ret ? 0 : 1; // valid = isSwapHappened?
        } while (false);
        __syncthreads();

        SIZE_TYPE scatter;
        SIZE_TYPE total; // how many new nodes reached in this turn?
        BlockScan(temp_storage).ExclusiveSum(valid, scatter, total);
        __syncthreads();

        if (0 == thread_id) {
            output_cta_offset = atomicAdd(node_queue_size, total);
        }
        __syncthreads();

        if (valid)
            node_queue[output_cta_offset + scatter] = neighbour;

        cta_offset += blockDim.x * gridDim.x;
    }
}

void gpma_bfs_gather_cpu(SIZE_TYPE *node_queue, SIZE_TYPE *_node_queue_len, SIZE_TYPE *edge_queue, SIZE_TYPE *_edge_queue_len, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *row_offsets) {
    auto &node_queue_len = *_node_queue_len;
    auto &edge_queue_len = *_edge_queue_len;

    std::mutex edgeQ_mutex;

#pragma omp parallel for
    for(auto i = 0; i < node_queue_len; ++i) {
        const auto &node = node_queue[i];
        const auto &row_begin = row_offsets[node];
        const auto &row_end = row_offsets[node+1];
        for(auto gather = row_begin; gather < row_end; ++gather) {
            auto neighbor = (SIZE_TYPE)(keys[gather] & COL_IDX_NONE);
            auto isValid = (neighbor != COL_IDX_NONE && values[gather] != VALUE_NONE);
            if(isValid) {
                std::lock_guard<std::mutex> guard (edgeQ_mutex);
                edge_queue[edge_queue_len] = neighbor;
                ++edge_queue_len;
            }
        }
    }
}
void gpma_bfs_contract_cpu(SIZE_TYPE *edge_queue, SIZE_TYPE *_edge_queue_len, SIZE_TYPE *node_queue, SIZE_TYPE *_node_queue_len, SIZE_TYPE level, SIZE_TYPE *label, SIZE_TYPE *bitmap) {
    auto &node_queue_len = *_node_queue_len;
    auto &edge_queue_len = *_edge_queue_len;
    // SIZE_TYPE zero = 0;

    //std::mutex nodeQ_mutex;

    for(auto i = 0; i < edge_queue_len; ++i) {
        const auto &neighbor = edge_queue[i];
        // TODO: Also add a small cache here.
        if(impl::gpma_bitmap_set_return_old(bitmap, neighbor))
            continue; // this node is not new.
        // Still waiting for parallelize, because bitmap_set_return is not parallelized yet.
        //auto exchanged = __atomic_compare_exchange_n(label+neighbor, &zero, level, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
        //if(exchanged) {
        //    std::lock_guard<std::mutex> guard (nodeQ_mutex);
        if(label[neighbor] != 0)
            continue; // this node is not new.
        label[neighbor] = level;
        {
            node_queue[node_queue_len] = neighbor;
            ++node_queue_len;
        }
    }
}


template <dev_type_t DEV, typename multidev_data_t, typename multidev_gpma_t>
__host__ void gpma_bfs_multidev_single(SIZE_TYPE node_size, SIZE_TYPE edge_size, SIZE_TYPE start_node, multidev_data_t &multidev_mgr, const multidev_gpma_t &gpma, size_t dev_id, size_t count = 1) {
    // If count is larger than 1, This function deal with [dev_id, dev_id+count) in SERIAL, not PARALLALIZED! Search for d5fb72b0 and cba97278 if you want to parallelize it.
    // If you want to parallelize it, you should just lock edge_queue. there's no need to lock bitmap, node_queue...
    SIZE_TYPE *results, *bitmap, *node_queue, *node_queue_size, *edge_queue, *edge_queue_size;
    if(DEV == CPU) {
        results = RAW_PTR(multidev_mgr.cpu_results);
        bitmap = RAW_PTR(multidev_mgr.cpu_bitmap);
        node_queue = RAW_PTR(multidev_mgr.cpu_nodeQ);
        edge_queue = RAW_PTR(multidev_mgr.cpu_edgeQ);
        node_queue_size = &multidev_mgr.cpu_nodeQ_size;
        edge_queue_size = &multidev_mgr.cpu_edgeQ_size;
    }
    else {
        results = RAW_PTR(multidev_mgr.gpu_results);
        bitmap = RAW_PTR(multidev_mgr.gpu_bitmap);
        node_queue = RAW_PTR(multidev_mgr.gpu_nodeQ);
        edge_queue = RAW_PTR(multidev_mgr.gpu_edgeQ);
        node_queue_size = multidev_mgr.gpu_nodeQ_size;
        edge_queue_size = multidev_mgr.gpu_edgeQ_size;
    }


    // init // TODO: if using multiCPU/multiGPU in parallel, the code block below is not allowed anymore. d5fb72b0
    anySetVal<DEV>(node_queue, start_node);
    anySetVal<DEV>(&bitmap[start_node / 32], 1u << (start_node % 32));
    anySetVal<DEV>(node_queue_size, 1u);
    anySetVal<DEV>(&results[start_node], 1u);

    SIZE_TYPE level = 1;
    const SIZE_TYPE THREADS_NUM = 256;
    while (true) {
        // gather
        SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, anyGetVal<DEV>(node_queue_size));
        anySetVal<DEV>(edge_queue_size, 0u);
        for(auto cter = 0; cter < count; ++cter) {
            if (DEV == GPU) {
                auto *gpma_impl = gpma.ptrs_gpu[dev_id + cter - multidev_mgr._cpus];
                gpma_bfs_gather_kernel<THREADS_NUM><<<BLOCKS_NUM, THREADS_NUM>>>(node_queue, node_queue_size, edge_queue, edge_queue_size, RAW_PTR(gpma_impl->keys), RAW_PTR(gpma_impl->values), RAW_PTR(gpma_impl->row_offset));
            } else {
                auto *gpma_impl = gpma.ptrs_cpu[dev_id + cter];
                gpma_bfs_gather_cpu(node_queue, node_queue_size, edge_queue, edge_queue_size, RAW_PTR(gpma_impl->keys), RAW_PTR(gpma_impl->values), RAW_PTR(gpma_impl->row_offset));
            }
        }
        multidev_mgr.iteration_barrier_1(dev_id);

        // contract
        level++;
        //rlib::println("DEBUG:E:", rlib::printable_iter(native_vector<DEV, SIZE_TYPE>(edge_queue, edge_queue + *tmpval_in_cpu)));
        BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, anyGetVal<DEV>(edge_queue_size));

        if (DEV == GPU) {
            gpma_bfs_contract_kernel<THREADS_NUM><<<BLOCKS_NUM, THREADS_NUM>>>(edge_queue, edge_queue_size, node_queue, node_queue_size, level, results, bitmap);
        } else {
            gpma_bfs_contract_cpu(edge_queue, edge_queue_size, node_queue, node_queue_size, level, results, bitmap);
        }
        //rlib::println("DEBUG:N:", rlib::printable_iter(native_vector<DEV, SIZE_TYPE>(node_queue, node_queue + *tmpval_in_cpu)));

        auto everyone_completed = multidev_mgr.iteration_barrier_2(dev_id);

        if (everyone_completed)
            break;
    }
}

template <dev_type_t DEV>
__host__ void gpma_bfs(const KEY_TYPE *keys, const VALUE_TYPE *values, const SIZE_TYPE *row_offsets, SIZE_TYPE node_size, SIZE_TYPE edge_size, SIZE_TYPE start_node, SIZE_TYPE *results) {
    impl::multidev_bfs_data<DEV==CPU?1:0, DEV==GPU?1:0> data(node_size, edge_size);
    gpma_bfs_multidev_single<DEV>(keys, values, row_offsets, node_size, edge_size, start_node, results, data, 0);
}

template <size_t cpu_n, size_t gpu_n>
__host__ void gpma_bfs(const GPMA_multidev<cpu_n, gpu_n> &gpma, SIZE_TYPE node_size, SIZE_TYPE edge_size, SIZE_TYPE start_node, SIZE_TYPE *results) {
    impl::multidev_bfs_data<cpu_n, gpu_n> data(node_size, edge_size);
#pragma omp parallel for num_threads(2)
    for(size_t cpuORgpu = 0; cpuORgpu < 2; ++cpuORgpu) {
        if(cpuORgpu == 0) {
            // CPU Deals [0, cpu_n)
            gpma_bfs_multidev_single<CPU>(node_size, edge_size, start_node, data, gpma, 0, cpu_n);
        }
        else {
            // GPU deals [cpu_n, cpu_n+gpu_n)
            gpma_bfs_multidev_single<GPU>(node_size, edge_size, start_node, data, gpma, cpu_n, gpu_n);
        }
    }

    std::memcpy(results, RAW_PTR(data.cpu_results), sizeof(SIZE_TYPE) * node_size);
}



