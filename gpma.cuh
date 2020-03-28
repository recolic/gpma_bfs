#pragma once
// TODO: adjust device inline functions. the inline keyword may not working for dev func.

#include "rdebug.hpp"
#include "utils.cuh"

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <type_traits>

#define RAW_PTR(x) thrust::raw_pointer_cast((x).data())
constexpr KEY_TYPE KEY_NONE = (KEY_TYPE)(-1);
constexpr KEY_TYPE KEY_MAX = KEY_NONE - 1;
constexpr SIZE_TYPE SIZE_NONE = (SIZE_TYPE)(-1);
constexpr VALUE_TYPE VALUE_NONE = 0;
constexpr KEY_TYPE COL_IDX_NONE = 0xFFFFFFFF;

constexpr SIZE_TYPE MAX_BLOCKS_NUM = 96 * 8;
#define CALC_BLOCKS_NUM(ITEMS_PER_BLOCK, CALC_SIZE) min(MAX_BLOCKS_NUM, (CALC_SIZE - 1) / ITEMS_PER_BLOCK + 1)

template <dev_type_t DEV>
class GPMA;
template <dev_type_t DEV>
void update_gpma(GPMA<DEV> &gpma, NATIVE_VEC_KEY<DEV> &update_keys, NATIVE_VEC_VALUE<DEV> &update_values);
template <dev_type_t DEV>
void recalculate_density(GPMA<DEV> &gpma);

template <dev_type_t DEV>
class GPMA {
  public:
    NATIVE_VEC_KEY<DEV> keys;
    NATIVE_VEC_VALUE<DEV> values;

    SIZE_TYPE segment_length;
    SIZE_TYPE tree_height;

    static constexpr double density_lower_thres_leaf = 0.08;
    static constexpr double density_lower_thres_root = 0.42;
    static constexpr double density_upper_thres_leaf = 0.92;
    static constexpr double density_upper_thres_root = 0.84;

    thrust::host_vector<SIZE_TYPE> lower_element;
    thrust::host_vector<SIZE_TYPE> upper_element;

    // addition for csr
    SIZE_TYPE row_num;
    NATIVE_VEC_SIZE<DEV> row_offset;

    inline int get_size() { return keys.size(); }

    void init_gpma_members(SIZE_TYPE row_num) {
        this->row_num = row_num;
        row_offset.resize(row_num + 1, 0);
        keys.resize(4, KEY_NONE);
        values.resize(4);
        segment_length = 2;
        tree_height = 1;
        anySync<DEV>();

        // the minimal tree structure has 2 levels with 4 elements' space, and the leaf segment's length is 2
        // put two MAX_KEY to keep minimal valid structure
        keys[0] = keys[2] = KEY_MAX;
        values[0] = values[2] = 1;

        recalculate_density(*this);
    }
    void init_gpma_insertions() {
        NATIVE_VEC_KEY<DEV> row_wall(row_num);
        NATIVE_VEC_VALUE<DEV> tmp_value(row_num, 1);
        anySync<DEV>();
        auto _transform_func_impl = [] __host__ __device__(KEY_TYPE x) { return MAKE_64b(x, COL_IDX_NONE); };
        thrust::tabulate(row_wall.begin(), row_wall.end(), _transform_func_impl);
        anySync<DEV>();
        update_gpma(*this, row_wall, tmp_value);
    }
    GPMA(SIZE_TYPE row_num) {
        init_gpma_members(row_num);
        init_gpma_insertions();
    }

    void print_status(std::string prefix = "DBG") const {
        DEBUG_PRINTFLN(prefix + ": GPMA_DUMP: keys={}, values={}, row_offset={}, seg_length,tree_height,row_num={},{},{}", keys.size(), values.size(), row_offset.size(), segment_length, tree_height, row_num);
        DEBUG_PRINTFLN(prefix + ": GPMA_DUMP: keys={}, values={}, row_offset={}", rlib::printable_iter(keys), rlib::printable_iter(values_for_print(values)), rlib::printable_iter(row_offset));
    }
};

/* returns the index of highest nonzero bit.
 * Example: x          RESULT
 *         0x00000001   1
 *         0xf0000001   32
 *         0x50000001   31
 *         0x00000101   9
 */
__host__ __device__ [[gnu::always_inline]] SIZE_TYPE fls(SIZE_TYPE x) {
    SIZE_TYPE r = 32;
    if (!x)
        return 0;
    if (!(x & 0xffff0000u))
        x <<= 16, r -= 16;
    if (!(x & 0xff000000u))
        x <<= 8, r -= 8;
    if (!(x & 0xf0000000u))
        x <<= 4, r -= 4;
    if (!(x & 0xc0000000u))
        x <<= 2, r -= 2;
    if (!(x & 0x80000000u))
        x <<= 1, r -= 1;
    return r;
}

template <typename T>
__global__ void memcpy_kernel(T *dest, const T *src, SIZE_TYPE size) {
    SIZE_TYPE tid_in_grid = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE threads_per_grid = gridDim.x * blockDim.x;
    // TODO: re-write the shits below.
    for (SIZE_TYPE i = tid_in_grid; i < size; i += threads_per_grid) {
        dest[i] = src[i];
    }
}

template <typename T>
__global__ void memset_kernel(T *data, T value, SIZE_TYPE size) {
    SIZE_TYPE tid_in_grid = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE threads_per_grid = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = tid_in_grid; i < size; i += threads_per_grid) {
        data[i] = value;
    }
}

template <dev_type_t DEV>
__host__ void recalculate_density(GPMA<DEV> &gpma) {
    // allow cpu+gpu
    gpma.lower_element.resize(gpma.tree_height + 1);
    gpma.upper_element.resize(gpma.tree_height + 1);

    SIZE_TYPE level_length = gpma.segment_length;
    for (SIZE_TYPE i = 0; i <= gpma.tree_height; i++) {
        double density_lower = gpma.density_lower_thres_root + (gpma.density_lower_thres_leaf - gpma.density_lower_thres_root) * (gpma.tree_height - i) / gpma.tree_height;
        double density_upper = gpma.density_upper_thres_root + (gpma.density_upper_thres_leaf - gpma.density_upper_thres_root) * (gpma.tree_height - i) / gpma.tree_height;

        gpma.lower_element[i] = (SIZE_TYPE)ceil(density_lower * level_length);
        gpma.upper_element[i] = (SIZE_TYPE)floor(density_upper * level_length);

        // special trim for wrong threshold introduced by float-integer conversion
        if (0 < i) {
            gpma.lower_element[i] = max(gpma.lower_element[i], 2 * gpma.lower_element[i - 1]);
            gpma.upper_element[i] = min(gpma.upper_element[i], 2 * gpma.upper_element[i - 1]);
        }
        level_length <<= 1;
    }
}

__device__ void cub_sort_key_value(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE size, KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values) {
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cErr(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, tmp_keys, values, tmp_values, size));
    cErr(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cErr(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, keys, tmp_keys, values, tmp_values, size));

    SIZE_TYPE THREADS_NUM = 128;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, size);
    memcpy_kernel<KEY_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(keys, tmp_keys, size);
    memcpy_kernel<VALUE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(values, tmp_values, size);

    cErr(cudaFree(d_temp_storage));
    anySync<GPU>();
}

__host__ __device__ SIZE_TYPE handle_del_mod(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE seg_length, KEY_TYPE key, VALUE_TYPE value, SIZE_TYPE leaf) {
    // allow cpu+gpu
    if (VALUE_NONE == value)
        leaf = SIZE_NONE;
    for (SIZE_TYPE i = 0; i < seg_length; i++) {
        if (keys[i] == key) {
            values[i] = value;
            leaf = SIZE_NONE;
            break;
        }
    }
    return leaf;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

__forceinline__ __host__ __device__ void locate_leaf_loop_body(SIZE_TYPE i, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE seg_length, SIZE_TYPE tree_height, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE *leaf) {
    // i should be: 0 <= i < update_size
    KEY_TYPE key = update_keys[i];
    VALUE_TYPE value = update_values[i];

    SIZE_TYPE prefix = 0;
    SIZE_TYPE current_bit = seg_length << tree_height >> 1;

    while (seg_length <= current_bit) {
        if (keys[prefix | current_bit] <= key)
            prefix |= current_bit;
        current_bit >>= 1;
    }

    prefix = handle_del_mod(keys + prefix, values + prefix, seg_length, key, value, prefix);
    leaf[i] = prefix;
}
__global__ void locate_leaf_kernel(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE seg_length, SIZE_TYPE tree_height, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE update_size, SIZE_TYPE *leaf) {
    SIZE_TYPE tid_in_grid = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE threads_per_grid = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = tid_in_grid; i < update_size; i += threads_per_grid)
        locate_leaf_loop_body(i, keys, values, seg_length, tree_height, update_keys, update_values, leaf);
}
template <dev_type_t DEV>
void locate_leaf_batch(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE seg_length, SIZE_TYPE tree_height, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE update_size, SIZE_TYPE *leaf) {
    if (DEV == GPU) {
        SIZE_TYPE THREADS_NUM = 32;
        SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_size);
        locate_leaf_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(keys, values, seg_length, tree_height, update_keys, update_values, update_size, leaf);
    } else {
#pragma omp parallel for schedule(dynamic, 64)
        for (SIZE_TYPE i = 0; i < update_size; ++i) {
            locate_leaf_loop_body(i, keys, values, seg_length, tree_height, update_keys, update_values, leaf);
        }
    }
    anySync<DEV>();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <SIZE_TYPE ITEM_PER_THREAD>
void block_compact_cpu_kernel(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE &compacted_size) {
    // COPY_PASTED from block_compact_kernel BEGIN
    // executed once for every cuda block.
    // sequencial
    KEY_TYPE *block_keys = keys;
    VALUE_TYPE *block_values = values;

    KEY_TYPE thread_keys[ITEM_PER_THREAD];
    VALUE_TYPE thread_values[ITEM_PER_THREAD];

    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        thread_keys[i] = block_keys[i];
        thread_values[i] = block_values[i];
        block_keys[i] = KEY_NONE;
    }

    SIZE_TYPE thread_data[ITEM_PER_THREAD];
    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        thread_data[i] = (thread_keys[i] == KEY_NONE || thread_values[i] == VALUE_NONE) ? 0 : 1;
    }

    anyExclusiveSum<CPU>(thread_data, thread_data, ITEM_PER_THREAD);

    SIZE_TYPE exscan[ITEM_PER_THREAD];
    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        exscan[i] = thread_data[i];
    }

    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        if (i == ITEM_PER_THREAD - 1)
            continue;
        if (exscan[i] != exscan[i + 1]) {
            SIZE_TYPE loc = exscan[i];
            block_keys[loc] = thread_keys[i];
            block_values[loc] = thread_values[i];
        }
    }

    // special logic for the last element
    SIZE_TYPE loc = exscan[ITEM_PER_THREAD - 1];
    if (thread_keys[ITEM_PER_THREAD - 1] == KEY_NONE || thread_values[ITEM_PER_THREAD - 1] == VALUE_NONE) {
        compacted_size = loc;
    } else {
        compacted_size = loc + 1;
        block_keys[loc] = thread_keys[ITEM_PER_THREAD - 1];
        block_values[loc] = thread_values[ITEM_PER_THREAD - 1];
    }
    // COPY_PASTED from block_compact_kernel END
}
template <SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__device__ void block_compact_kernel(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE &compacted_size) {
    constexpr size_t cubTHREADS = THREAD_PER_BLOCK;
    SIZE_TYPE thread_id = threadIdx.x;

    KEY_TYPE *block_keys = keys;
    VALUE_TYPE *block_values = values;

    KEY_TYPE thread_keys[ITEM_PER_THREAD];
    VALUE_TYPE thread_values[ITEM_PER_THREAD];

    SIZE_TYPE thread_offset = thread_id * ITEM_PER_THREAD;
    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        thread_keys[i] = block_keys[thread_offset + i];
        thread_values[i] = block_values[thread_offset + i];
        block_keys[thread_offset + i] = KEY_NONE;
    }

    using BlockScan = cub::BlockScan<SIZE_TYPE, cubTHREADS>;
    __shared__ typename BlockScan::TempStorage temp_storage;
    SIZE_TYPE thread_data[ITEM_PER_THREAD];
    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        thread_data[i] = (thread_keys[i] == KEY_NONE || thread_values[i] == VALUE_NONE) ? 0 : 1;
    }
    __syncthreads();

    BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
    __syncthreads();

    __shared__ SIZE_TYPE exscan[THREAD_PER_BLOCK * ITEM_PER_THREAD];
    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        exscan[i + thread_offset] = thread_data[i];
    }
    __syncthreads();

    for (SIZE_TYPE i = 0; i < ITEM_PER_THREAD; i++) {
        if (thread_id == THREAD_PER_BLOCK - 1 && i == ITEM_PER_THREAD - 1)
            continue;
        if (exscan[thread_offset + i] != exscan[thread_offset + i + 1]) {
            SIZE_TYPE loc = exscan[thread_offset + i];
            block_keys[loc] = thread_keys[i];
            block_values[loc] = thread_values[i];
        }
    }

    // special logic for the last element
    if (thread_id == THREAD_PER_BLOCK - 1) {
        SIZE_TYPE loc = exscan[THREAD_PER_BLOCK * ITEM_PER_THREAD - 1];
        if (thread_keys[ITEM_PER_THREAD - 1] == KEY_NONE || thread_values[ITEM_PER_THREAD - 1] == VALUE_NONE) {
            compacted_size = loc;
        } else {
            compacted_size = loc + 1;
            block_keys[loc] = thread_keys[ITEM_PER_THREAD - 1];
            block_values[loc] = thread_values[ITEM_PER_THREAD - 1];
        }
    }
}

template <typename FIRST_TYPE, typename SECOND_TYPE>
__device__ void block_pair_copy_kernel(FIRST_TYPE *dest_first, SECOND_TYPE *dest_second, FIRST_TYPE *src_first, SECOND_TYPE *src_second, SIZE_TYPE size) {
    for (SIZE_TYPE i = threadIdx.x; i < size; i += blockDim.x) {
        dest_first[i] = src_first[i];
        dest_second[i] = src_second[i];
    }
}

template <SIZE_TYPE ITEM_PER_THREAD>
void block_redispatch_cpu_kernel(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE rebalance_width, SIZE_TYPE seg_length, SIZE_TYPE merge_size, SIZE_TYPE *row_offset, SIZE_TYPE update_node) {
    // this function is run once per block.
    // COPY_PASTED from block_redispatch_kernel BEGIN

    // step1: load KV in shared memory
    KEY_TYPE block_keys[ITEM_PER_THREAD];
    VALUE_TYPE block_values[ITEM_PER_THREAD];
    assert(rebalance_width == ITEM_PER_THREAD);
    for (auto i = 0; i < rebalance_width; ++i) {
        block_keys[i] = keys[i];
        block_values[i] = values[i];
    }

    // step2: sort by key with value on shared memory
    thrust::sort_by_key(thrust::host, block_keys, block_keys + ITEM_PER_THREAD, block_values);

    // step3: evenly re-dispatch KVs to leaf segments
    // warning: many threads
    KEY_TYPE frac = rebalance_width / seg_length;
    KEY_TYPE deno = merge_size;
    for (SIZE_TYPE i = 0; i < merge_size; ++i) {
        keys[i] = KEY_NONE;
    }

    for (SIZE_TYPE i = 0; i < merge_size; ++i) {
        SIZE_TYPE seg_idx = (SIZE_TYPE)(frac * i / deno);
        SIZE_TYPE seg_lane = (SIZE_TYPE)(frac * i % deno / frac);
        SIZE_TYPE proj_location = seg_idx * seg_length + seg_lane;

        KEY_TYPE cur_key = block_keys[i];
        VALUE_TYPE cur_value = block_values[i];
        keys[proj_location] = cur_key;
        values[proj_location] = cur_value;

        // addition for csr
        if ((cur_key & COL_IDX_NONE) == COL_IDX_NONE) {
            SIZE_TYPE cur_row = (SIZE_TYPE)(cur_key >> 32);
            row_offset[cur_row + 1] = proj_location + update_node;
        }
    }
    // COPY_PASTED from block_redispatch_kernel END
}

template <SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__device__ void block_redispatch_kernel(KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE rebalance_width, SIZE_TYPE seg_length, SIZE_TYPE merge_size, SIZE_TYPE *row_offset, SIZE_TYPE update_node) {
    constexpr size_t cubTHREADS = THREAD_PER_BLOCK;

    // step1: load KV in shared memory
    __shared__ KEY_TYPE block_keys[THREAD_PER_BLOCK * ITEM_PER_THREAD];
    __shared__ VALUE_TYPE block_values[THREAD_PER_BLOCK * ITEM_PER_THREAD];
    block_pair_copy_kernel<KEY_TYPE, VALUE_TYPE>(block_keys, block_values, keys, values, rebalance_width);
    __syncthreads();

    // step2: sort by key with value on shared memory
    typedef cub::BlockLoad<KEY_TYPE, cubTHREADS, ITEM_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockKeyLoadT;
    typedef cub::BlockLoad<VALUE_TYPE, cubTHREADS, ITEM_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockValueLoadT;
    typedef cub::BlockStore<KEY_TYPE, cubTHREADS, ITEM_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockKeyStoreT;
    typedef cub::BlockStore<VALUE_TYPE, cubTHREADS, ITEM_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockValueStoreT;
    typedef cub::BlockRadixSort<KEY_TYPE, cubTHREADS, ITEM_PER_THREAD, VALUE_TYPE> BlockRadixSortT;

    __shared__ union {
        typename BlockKeyLoadT::TempStorage key_load;
        typename BlockValueLoadT::TempStorage value_load;
        typename BlockKeyStoreT::TempStorage key_store;
        typename BlockValueStoreT::TempStorage value_store;
        typename BlockRadixSortT::TempStorage sort;
    } temp_storage;

    KEY_TYPE thread_keys[ITEM_PER_THREAD];
    VALUE_TYPE thread_values[ITEM_PER_THREAD];
    BlockKeyLoadT(temp_storage.key_load).Load(block_keys, thread_keys);
    BlockValueLoadT(temp_storage.value_load).Load(block_values, thread_values);
    __syncthreads();

    BlockRadixSortT(temp_storage.sort).Sort(thread_keys, thread_values);
    __syncthreads();

    BlockKeyStoreT(temp_storage.key_store).Store(block_keys, thread_keys);
    BlockValueStoreT(temp_storage.value_store).Store(block_values, thread_values);
    __syncthreads();

    // step3: evenly re-dispatch KVs to leaf segments
    KEY_TYPE frac = rebalance_width / seg_length;
    KEY_TYPE deno = merge_size;
    for (SIZE_TYPE i = threadIdx.x; i < merge_size; i += blockDim.x) {
        keys[i] = KEY_NONE;
    }
    __syncthreads();

    for (SIZE_TYPE i = threadIdx.x; i < merge_size; i += blockDim.x) {
        SIZE_TYPE seg_idx = (SIZE_TYPE)(frac * i / deno);
        SIZE_TYPE seg_lane = (SIZE_TYPE)(frac * i % deno / frac);
        SIZE_TYPE proj_location = seg_idx * seg_length + seg_lane;

        KEY_TYPE cur_key = block_keys[i];
        VALUE_TYPE cur_value = block_values[i];
        keys[proj_location] = cur_key;
        values[proj_location] = cur_value;

        // addition for csr
        if ((cur_key & COL_IDX_NONE) == COL_IDX_NONE) {
            SIZE_TYPE cur_row = (SIZE_TYPE)(cur_key >> 32);
            row_offset[cur_row + 1] = proj_location + update_node;
        }
    }
}

template <SIZE_TYPE ITEM_PER_THREAD>
void block_rebalancing_cpu(SIZE_TYPE unique_update_size, SIZE_TYPE seg_length, SIZE_TYPE level, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE *unique_update_nodes, SIZE_TYPE *update_offset, SIZE_TYPE lower_bound, SIZE_TYPE upper_bound, SIZE_TYPE *row_offset) {
#pragma omp parallel for schedule(dynamic)
    for (SIZE_TYPE update_id = 0; update_id < unique_update_size; ++update_id) {
        // Logic of ONE block.
        // COPY_PASTED from block_rebalancing_kernel BEGIN
        SIZE_TYPE update_node = unique_update_nodes[update_id];
        KEY_TYPE *key = keys + update_node;
        VALUE_TYPE *value = values + update_node;
        SIZE_TYPE rebalance_width = seg_length << level;

        // compact
        SIZE_TYPE compacted_size;
        block_compact_cpu_kernel<ITEM_PER_THREAD>(key, value, compacted_size); // output compacted_size

        // judge whether fit the density threshold
        SIZE_TYPE interval_a = update_offset[update_id];
        SIZE_TYPE interval_b = update_offset[update_id + 1];
        SIZE_TYPE interval_size = interval_b - interval_a;
        SIZE_TYPE merge_size = compacted_size + interval_size;

        if (lower_bound <= merge_size && merge_size <= upper_bound) {
            // move
            for (auto i = 0; i < interval_size; ++i) {
                (key + compacted_size)[i] = (update_keys + interval_a)[i];
                (value + compacted_size)[i] = (update_values + interval_a)[i];
            }

            // set SIZE_NONE for executed update
            for (auto i = interval_a; i < interval_b; ++i) {
                update_nodes[i] = SIZE_NONE;
            }
            // re-dispatch
            block_redispatch_cpu_kernel<ITEM_PER_THREAD>(key, value, rebalance_width, seg_length, merge_size, row_offset, update_node);
        }
        // COPY_PASTED from block_rebalancing_kernel END
    }
}
template <SIZE_TYPE THREAD_PER_BLOCK, SIZE_TYPE ITEM_PER_THREAD>
__global__ void block_rebalancing_kernel(SIZE_TYPE seg_length, SIZE_TYPE level, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE *unique_update_nodes, SIZE_TYPE *update_offset, SIZE_TYPE lower_bound, SIZE_TYPE upper_bound, SIZE_TYPE *row_offset) {
    SIZE_TYPE update_id = blockIdx.x;
    // Every cuda BLOCK will process ONE unique_update.
    SIZE_TYPE update_node = unique_update_nodes[update_id];
    KEY_TYPE *key = keys + update_node;
    VALUE_TYPE *value = values + update_node;
    SIZE_TYPE rebalance_width = seg_length << level;

    // compact
    __shared__ SIZE_TYPE compacted_size;
    block_compact_kernel<THREAD_PER_BLOCK, ITEM_PER_THREAD>(key, value, compacted_size);
    __syncthreads();

    // judge whether fit the density threshold
    SIZE_TYPE interval_a = update_offset[update_id];
    SIZE_TYPE interval_b = update_offset[update_id + 1];
    SIZE_TYPE interval_size = interval_b - interval_a;
    SIZE_TYPE merge_size = compacted_size + interval_size;
    __syncthreads();

    if (lower_bound <= merge_size && merge_size <= upper_bound) {
        // move
        block_pair_copy_kernel<KEY_TYPE, VALUE_TYPE>(key + compacted_size, value + compacted_size, update_keys + interval_a, update_values + interval_a, interval_size);
        __syncthreads();

        // set SIZE_NONE for executed update
        for (SIZE_TYPE i = interval_a + threadIdx.x; i < interval_b; i += blockDim.x) {
            update_nodes[i] = SIZE_NONE;
        }

        // re-dispatch
        block_redispatch_kernel<THREAD_PER_BLOCK, ITEM_PER_THREAD>(key, value, rebalance_width, seg_length, merge_size, row_offset, update_node);
    }
}

__global__ void label_key_whether_none_kernel(SIZE_TYPE *label, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE size) {
    SIZE_TYPE tid_in_grid = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE threads_per_grid = gridDim.x * blockDim.x;
    for (SIZE_TYPE i = tid_in_grid; i < size; i += threads_per_grid) {
        label[i] = (keys[i] == KEY_NONE || values[i] == VALUE_NONE) ? 0 : 1;
    }
}

void copy_compacted_kv_cpu(SIZE_TYPE *exscan, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE size, KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, SIZE_TYPE *compacted_size) {
    // COPY_PASTED from copy_compacted_kv_kernel BEGIN
#pragma omp parallel for schedule(static)
    for (SIZE_TYPE i = 0; i < size; ++i) {
        if (i == size - 1)
            continue;
        if (exscan[i] != exscan[i + 1]) {
            SIZE_TYPE loc = exscan[i];
            tmp_keys[loc] = keys[i];
            tmp_values[loc] = values[i];
        }
    }

    { // single thread.
        SIZE_TYPE loc = exscan[size - 1];
        if (keys[size - 1] == KEY_NONE || values[size - 1] == VALUE_NONE) {
            *compacted_size = loc;
        } else {
            *compacted_size = loc + 1;
            tmp_keys[loc] = keys[size - 1];
            tmp_values[loc] = values[size - 1];
        }
    }
    // COPY_PASTED from copy_compacted_kv_kernel END
}
__global__ void copy_compacted_kv_kernel(SIZE_TYPE *exscan, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE size, KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, SIZE_TYPE *compacted_size) {
    SIZE_TYPE tid_in_grid = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE threads_per_grid = gridDim.x * blockDim.x;

    for (SIZE_TYPE i = tid_in_grid; i < size; i += threads_per_grid) {
        if (i == size - 1)
            continue;
        if (exscan[i] != exscan[i + 1]) {
            SIZE_TYPE loc = exscan[i];
            tmp_keys[loc] = keys[i];
            tmp_values[loc] = values[i];
        }
    }

    if (0 == tid_in_grid) {
        SIZE_TYPE loc = exscan[size - 1];
        if (keys[size - 1] == KEY_NONE || values[size - 1] == VALUE_NONE) {
            *compacted_size = loc;
        } else {
            *compacted_size = loc + 1;
            tmp_keys[loc] = keys[size - 1];
            tmp_values[loc] = values[size - 1];
        }
    }
}

void compact_kernel_cpu(SIZE_TYPE size, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *compacted_size, KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, SIZE_TYPE *exscan, SIZE_TYPE *label) {
    // working
#pragma omp parallel for schedule(static)
    for (SIZE_TYPE i = 0; i < size; ++i) {
        label[i] = (keys[i] == KEY_NONE || values[i] == VALUE_NONE) ? 0 : 1;
    }

    anyExclusiveSum<CPU>(label, exscan, size);
    copy_compacted_kv_cpu(exscan, keys, values, size, tmp_keys, tmp_values, compacted_size);
}
__device__ void compact_kernel_gpu(SIZE_TYPE size, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *compacted_size, KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, SIZE_TYPE *exscan, SIZE_TYPE *label) {
    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, size);
    label_key_whether_none_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(label, keys, values, size);
    anySync<GPU>();

    // exscan
    cudaExclusiveSum(label, exscan, size);

    // copy compacted kv to tmp, and set the original to none
    copy_compacted_kv_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(exscan, keys, values, size, tmp_keys, tmp_values, compacted_size);
    anySync<GPU>();
}

__forceinline__ __host__ __device__ void redispatch_loop_body(size_t i, KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE update_width, SIZE_TYPE seg_length, SIZE_TYPE merge_size, SIZE_TYPE *row_offset, SIZE_TYPE update_node) {
    KEY_TYPE frac = update_width / seg_length;
    KEY_TYPE deno = merge_size;

    SIZE_TYPE seg_idx = (SIZE_TYPE)(frac * i / deno);
    SIZE_TYPE seg_lane = (SIZE_TYPE)(frac * i % deno / frac);
    SIZE_TYPE proj_location = seg_idx * seg_length + seg_lane;
    KEY_TYPE cur_key = tmp_keys[i];
    VALUE_TYPE cur_value = tmp_values[i];
    keys[proj_location] = cur_key;
    values[proj_location] = cur_value;

    // addition for csr
    if ((cur_key & COL_IDX_NONE) == COL_IDX_NONE) {
        SIZE_TYPE cur_row = (SIZE_TYPE)(cur_key >> 32);
        row_offset[cur_row + 1] = proj_location + update_node;
    }
}

__global__ void redispatch_kernel(KEY_TYPE *tmp_keys, VALUE_TYPE *tmp_values, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE update_width, SIZE_TYPE seg_length, SIZE_TYPE merge_size, SIZE_TYPE *row_offset, SIZE_TYPE update_node) {
    SIZE_TYPE tid_in_grid = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE threads_per_grid = gridDim.x * blockDim.x;

    for (SIZE_TYPE i = tid_in_grid; i < merge_size; i += threads_per_grid) {
        redispatch_loop_body(i, tmp_keys, tmp_values, keys, values, update_width, seg_length, merge_size, row_offset, update_node);
    }
}

void rebalancing_impl_cpu(SIZE_TYPE unique_update_size, SIZE_TYPE seg_length, SIZE_TYPE level, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE *unique_update_nodes, SIZE_TYPE *update_offset, SIZE_TYPE lower_bound, SIZE_TYPE upper_bound, SIZE_TYPE *row_offset) {
    // COPY_PASTED from rebalancing_kernel BEGIN
    SIZE_TYPE update_width = seg_length << level;

#pragma omp parallel
    {
        // private variables.
        SIZE_TYPE *compacted_size;
        KEY_TYPE *tmp_keys;
        VALUE_TYPE *tmp_values;
        SIZE_TYPE *tmp_exscan;
        SIZE_TYPE *tmp_label;

        anyMalloc<CPU>((void **)&compacted_size, sizeof(SIZE_TYPE));
        anyMalloc<CPU>((void **)&tmp_keys, update_width * sizeof(KEY_TYPE));
        anyMalloc<CPU>((void **)&tmp_values, update_width * sizeof(VALUE_TYPE));
        anyMalloc<CPU>((void **)&tmp_exscan, update_width * sizeof(SIZE_TYPE));
        anyMalloc<CPU>((void **)&tmp_label, update_width * sizeof(SIZE_TYPE));

#pragma omp for schedule(dynamic, 8) // this loop is heavy...
        for (SIZE_TYPE i = 0; i < unique_update_size; ++i) {
            SIZE_TYPE update_node = unique_update_nodes[i];
            KEY_TYPE *key = keys + update_node;
            VALUE_TYPE *value = values + update_node;

            // compact
            compact_kernel_cpu(update_width, key, value, compacted_size, tmp_keys, tmp_values, tmp_exscan, tmp_label);

            // judge whether fit the density threshold
            SIZE_TYPE interval_a = update_offset[i];
            SIZE_TYPE interval_b = update_offset[i + 1];
            SIZE_TYPE interval_size = interval_b - interval_a;
            SIZE_TYPE merge_size = (*compacted_size) + interval_size;

            if (lower_bound <= merge_size && merge_size <= upper_bound) {
                // move
                memcpy(tmp_keys + (*compacted_size), update_keys + interval_a, interval_size);
                memcpy(tmp_values + (*compacted_size), update_values + interval_a, interval_size);

                // set SIZE_NONE for executed updates
                // std::fill(...); mXeXmXsXeXt(update_nodes + interval_a, SIZE_NONE, interval_size);
                std::fill(update_nodes + interval_a, update_nodes + interval_a + interval_size, SIZE_NONE);
                thrust::sort_by_key(thrust::host, key, key + merge_size, value);
                // In original cub_sort_by_key, tmp_key should be equal to key(sorted), tmp_value should equal to value(sorted).
                // TODO: conflict|| std::copy(key, key + merge_size, tmp_key);
                // TODO: conflict|| std::copy(value, value + merge_size, tmp_value);

                // re-dispatch
                // std::fill(); mXeXmXsXeXt(key, KEY_NONE, update_width);
                std::fill(key, key + update_width, KEY_NONE);

                // redispatch_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(tmp_keys, tmp_values, key, value, update_width, seg_length, merge_size, row_offset, update_node);
                for (auto i = 0; i < merge_size; ++i) {
                    redispatch_loop_body(i, tmp_keys, tmp_values, key, value, update_width, seg_length, merge_size, row_offset, update_node);
                }
            }
        }

        anyFree<CPU>(compacted_size);
        anyFree<CPU>(tmp_keys);
        anyFree<CPU>(tmp_values);
        anyFree<CPU>(tmp_exscan);
        anyFree<CPU>(tmp_label);
    } // omp parallel end

    // COPY_PASTED from rebalancing_kernel END
}
__global__ void rebalancing_kernel(SIZE_TYPE unique_update_size, SIZE_TYPE seg_length, SIZE_TYPE level, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE *unique_update_nodes, SIZE_TYPE *update_offset, SIZE_TYPE lower_bound, SIZE_TYPE upper_bound, SIZE_TYPE *row_offset) {
    SIZE_TYPE tid_in_grid = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE threads_per_grid = gridDim.x * blockDim.x;
    SIZE_TYPE update_width = seg_length << level;

    SIZE_TYPE *compacted_size;
    KEY_TYPE *tmp_keys;
    VALUE_TYPE *tmp_values;
    SIZE_TYPE *tmp_exscan;
    SIZE_TYPE *tmp_label;

    anyMalloc<GPU>((void **)&compacted_size, sizeof(SIZE_TYPE));
    anyMalloc<GPU>((void **)&tmp_keys, update_width * sizeof(KEY_TYPE));
    anyMalloc<GPU>((void **)&tmp_values, update_width * sizeof(VALUE_TYPE));
    anyMalloc<GPU>((void **)&tmp_exscan, update_width * sizeof(SIZE_TYPE));
    anyMalloc<GPU>((void **)&tmp_label, update_width * sizeof(SIZE_TYPE));

    for (SIZE_TYPE i = tid_in_grid; i < unique_update_size; i += threads_per_grid) {
        SIZE_TYPE update_node = unique_update_nodes[i];
        KEY_TYPE *key = keys + update_node;
        VALUE_TYPE *value = values + update_node;

        // compact
        compact_kernel_gpu(update_width, key, value, compacted_size, tmp_keys, tmp_values, tmp_exscan, tmp_label);
        anySync<GPU>(); // Necessary!

        // judge whether fit the density threshold
        SIZE_TYPE interval_a = update_offset[i];
        SIZE_TYPE interval_b = update_offset[i + 1];
        SIZE_TYPE interval_size = interval_b - interval_a;
        SIZE_TYPE merge_size = (*compacted_size) + interval_size;

        if (lower_bound <= merge_size && merge_size <= upper_bound) {
            SIZE_TYPE THREADS_NUM = 32;
            SIZE_TYPE BLOCKS_NUM;

            // move
            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, interval_size);
            memcpy_kernel<KEY_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(tmp_keys + (*compacted_size), update_keys + interval_a, interval_size);
            memcpy_kernel<VALUE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(tmp_values + (*compacted_size), update_values + interval_a, interval_size);

            // set SIZE_NONE for executed updates
            memset_kernel<SIZE_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(update_nodes + interval_a, SIZE_NONE, interval_size);
            anySync<GPU>(); // Necessary here, since there's multiple CUDA stream.

            cub_sort_key_value(tmp_keys, tmp_values, merge_size, key, value);

            // re-dispatch
            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_width);
            memset_kernel<KEY_TYPE><<<BLOCKS_NUM, THREADS_NUM>>>(key, KEY_NONE, update_width);
            anySync<GPU>();

            BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, merge_size);
            redispatch_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(tmp_keys, tmp_values, key, value, update_width, seg_length, merge_size, row_offset, update_node);
            anySync<GPU>();
        }
    }

    anyFree<GPU>(compacted_size);
    anyFree<GPU>(tmp_keys);
    anyFree<GPU>(tmp_values);
    anyFree<GPU>(tmp_exscan);
    anyFree<GPU>(tmp_label);
}

template <dev_type_t DEV>
void rebalance_batch(SIZE_TYPE level, SIZE_TYPE seg_length, KEY_TYPE *keys, VALUE_TYPE *values, SIZE_TYPE *update_nodes, KEY_TYPE *update_keys, VALUE_TYPE *update_values, SIZE_TYPE update_size, SIZE_TYPE *unique_update_nodes, SIZE_TYPE *update_offset, SIZE_TYPE unique_update_size, SIZE_TYPE lower_bound, SIZE_TYPE upper_bound, SIZE_TYPE *row_offset) {
    // TryInsert+ is this function.
    SIZE_TYPE update_width = seg_length << level; // real seg_length of this level
    if (false && update_width <= 1024) {
        assert(IsPowerOfTwo(update_width));
        if (DEV == GPU) {
            // func pointer for each template
            decltype(&block_rebalancing_kernel<1, 1>) func_arr[10];
            func_arr[0] = block_rebalancing_kernel<2, 1>;
            func_arr[1] = block_rebalancing_kernel<4, 1>;
            func_arr[2] = block_rebalancing_kernel<8, 1>;
            func_arr[3] = block_rebalancing_kernel<16, 1>;
            func_arr[4] = block_rebalancing_kernel<32, 1>;
            func_arr[5] = block_rebalancing_kernel<32, 2>;
            func_arr[6] = block_rebalancing_kernel<32, 4>;
            func_arr[7] = block_rebalancing_kernel<32, 8>;
            func_arr[8] = block_rebalancing_kernel<32, 16>;
            func_arr[9] = block_rebalancing_kernel<32, 32>;

            // operate each tree node by cuda-block
            SIZE_TYPE THREADS_NUM = update_width > 32 ? 32 : update_width;
            SIZE_TYPE BLOCKS_NUM = unique_update_size;

            DEBUG_PRINTFLN("ARG-=DEBUG: block_rebalancing_kernel calling func_arr[{}]<<<{}, {}>>>, update_width={}", fls(update_width) - 2, BLOCKS_NUM, THREADS_NUM, update_width);
            func_arr[fls(update_width) - 2]<<<BLOCKS_NUM, THREADS_NUM>>>(seg_length, level, keys, values, update_nodes, update_keys, update_values, unique_update_nodes, update_offset, lower_bound, upper_bound, row_offset);
        } else {
            // func pointer for each template
            decltype(&block_rebalancing_cpu<1>) func_arr[10];
            func_arr[0] = block_rebalancing_cpu<2>;
            func_arr[1] = block_rebalancing_cpu<4>;
            func_arr[2] = block_rebalancing_cpu<8>;
            func_arr[3] = block_rebalancing_cpu<16>;
            func_arr[4] = block_rebalancing_cpu<32>;
            func_arr[5] = block_rebalancing_cpu<64>;
            func_arr[6] = block_rebalancing_cpu<128>;
            func_arr[7] = block_rebalancing_cpu<256>;
            func_arr[8] = block_rebalancing_cpu<512>;
            func_arr[9] = block_rebalancing_cpu<1024>;

            DEBUG_PRINTFLN("ARG-=DEBUG: CPU block_rebalancing_cpu calling func_arr[{}], update_width={}", fls(update_width) - 2, update_width);
            func_arr[fls(update_width) - 2](unique_update_size, seg_length, level, keys, values, update_nodes, update_keys, update_values, unique_update_nodes, update_offset, lower_bound, upper_bound, row_offset);
        }
    } else {
        if (DEV == GPU) {
            // operate each tree node by cub-kernel (dynamic parallelsim)
            SIZE_TYPE BLOCKS_NUM = min(2048, unique_update_size);
            DEBUG_PRINTFLN("ARG-=DEBUG: rebalance_batch calling rebalancing_kernel<<<{}, {}>>>, update_width={}", BLOCKS_NUM, 1, update_width);
            rebalancing_kernel<<<BLOCKS_NUM, 1>>>(unique_update_size, seg_length, level, keys, values, update_nodes, update_keys, update_values, unique_update_nodes, update_offset, lower_bound, upper_bound, row_offset);
        } else {
            DEBUG_PRINTFLN("ARG-=DEBUG: CPU rebalance_batch calling rebalancing_impl_cpu, blocks={}, update_width={}", unique_update_size, update_width);
            rebalancing_impl_cpu(unique_update_size, seg_length, level, keys, values, update_nodes, update_keys, update_values, unique_update_nodes, update_offset, lower_bound, upper_bound, row_offset);
        }
    }

    anySync<DEV>(); // after previous kernel launch
}

// this function is working for cpu
template <dev_type_t DEV>
void compact_insertions(NATIVE_VEC_SIZE<DEV> &update_nodes, NATIVE_VEC_KEY<DEV> &update_keys, NATIVE_VEC_VALUE<DEV> &update_values, SIZE_TYPE &update_size) {
    auto three_tuple_first_none = [] __host__ __device__(const thrust::tuple<SIZE_TYPE, KEY_TYPE, VALUE_TYPE> &a) { return SIZE_NONE == thrust::get<0>(a); };
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(update_nodes.begin(), update_keys.begin(), update_values.begin()));
    auto zip_end = thrust::remove_if(zip_begin, zip_begin + update_size, three_tuple_first_none);
    anySync<DEV>();
    update_size = zip_end - zip_begin;
}

template <dev_type_t DEV>
SIZE_TYPE group_insertion_by_node(SIZE_TYPE *update_nodes, SIZE_TYPE update_size, SIZE_TYPE *unique_update_nodes, SIZE_TYPE *update_offset) {

    // step1: encode
    SIZE_TYPE *tmp_offset;
    anyMalloc<DEV>((void **)&tmp_offset, sizeof(SIZE_TYPE) * update_size);

    SIZE_TYPE *num_runs_out;
    anyMalloc<DEV>((void **)&num_runs_out, sizeof(SIZE_TYPE));
    anySync<DEV>();
    anyRunLengthEncoding<DEV>(update_nodes, update_size, unique_update_nodes, tmp_offset, num_runs_out);
    anySync<DEV>();

    SIZE_TYPE unique_node_size;
    anyMemcpy<DEV, CPU>(&unique_node_size, num_runs_out, sizeof(SIZE_TYPE));
    anySync<DEV>();
    anyFree<DEV>(num_runs_out);

    // step2: exclusive scan
    anyExclusiveSum<DEV>(tmp_offset, update_offset, unique_node_size);
    anySync<DEV>();

    anyMemcpy<CPU, DEV>(update_offset + unique_node_size, &update_size, sizeof(SIZE_TYPE));
    anySync<DEV>();
    anyFree<DEV>(tmp_offset);

    return unique_node_size;
}

template <dev_type_t DEV>
__host__ void compress_insertions_by_node(NATIVE_VEC_SIZE<DEV> &update_nodes, SIZE_TYPE update_size, NATIVE_VEC_SIZE<DEV> &unique_update_nodes, NATIVE_VEC_SIZE<DEV> &update_offset, SIZE_TYPE &unique_node_size) {
    unique_node_size = group_insertion_by_node<DEV>(RAW_PTR(update_nodes), update_size, RAW_PTR(unique_update_nodes), RAW_PTR(update_offset));
    anySync<DEV>();
}

__global__ void up_level_kernel(SIZE_TYPE *update_nodes, SIZE_TYPE update_size, SIZE_TYPE update_width) {
    SIZE_TYPE tid_in_grid = blockDim.x * blockIdx.x + threadIdx.x;
    SIZE_TYPE threads_per_grid = gridDim.x * blockDim.x;

    for (SIZE_TYPE i = tid_in_grid; i < update_size; i += threads_per_grid) {
        SIZE_TYPE node = update_nodes[i];
        update_nodes[i] = node & ~update_width;
    }
}

template <dev_type_t DEV>
__host__ void up_level_batch(SIZE_TYPE *update_nodes, SIZE_TYPE update_size, SIZE_TYPE update_width) {
    SIZE_TYPE THREADS_NUM = 32;
    SIZE_TYPE BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, update_size);
    if (DEV == GPU) {
        up_level_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(update_nodes, update_size, update_width);
        anySync<DEV>();
    } else {
#pragma omp parallel for schedule(static)
        for (SIZE_TYPE i = 0; i < update_size; ++i) {
            update_nodes[i] &= ~update_width;
        }
    }
}

template <dev_type_t DEV>
__host__ int resize_gpma(GPMA<DEV> &gpma, SIZE_TYPE increased_size) {
    // allow cpu + gpu
    auto kv_tuple_none = [] __host__ __device__(const thrust::tuple<KEY_TYPE, VALUE_TYPE> &a) { return KEY_NONE == thrust::get<0>(a) || VALUE_NONE == thrust::get<1>(a); };
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(gpma.keys.begin(), gpma.values.begin()));
    auto zip_end = thrust::remove_if(zip_begin, zip_begin + gpma.keys.size(), kv_tuple_none);
    anySync<DEV>();
    SIZE_TYPE compacted_size = zip_end - zip_begin;
    thrust::fill(gpma.keys.begin() + compacted_size, gpma.keys.end(), KEY_NONE);
    anySync<DEV>();

    SIZE_TYPE merge_size = compacted_size + increased_size;
    SIZE_TYPE original_tree_size = gpma.keys.size();

    SIZE_TYPE tree_size = 4;
    while (floor(gpma.density_upper_thres_root * tree_size) < merge_size)
        tree_size *= 2;
    rlib::printfln("resize gpma: FROM keys.size=(BUFSIZE={})(COMPACTED={}), tree_height={}, seg_length={}", original_tree_size, compacted_size, gpma.tree_height, gpma.segment_length);
    gpma.segment_length = 1 << (fls(fls(tree_size)) - 1);
    gpma.tree_height = fls(tree_size / gpma.segment_length) - 1;

    gpma.keys.resize(tree_size, KEY_NONE);
    gpma.values.resize(tree_size);
    anySync<DEV>();
    recalculate_density(gpma);
    rlib::printfln("               TO keys.size={}, tree_height={}, seg_length={}", tree_size, gpma.tree_height, gpma.segment_length);

    return compacted_size;
}

template <dev_type_t DEV>
__host__ void significant_insert(GPMA<DEV> &gpma, NATIVE_VEC_KEY<DEV> &update_keys, NATIVE_VEC_VALUE<DEV> &update_values, int update_size) {
    int valid_size = resize_gpma(gpma, update_size);
    thrust::copy(update_keys.begin(), update_keys.begin() + update_size, gpma.keys.begin() + valid_size);
    thrust::copy(update_values.begin(), update_values.begin() + update_size, gpma.values.begin() + valid_size);

    NATIVE_VEC_KEY<DEV> tmp_update_keys(gpma.get_size());
    NATIVE_VEC_VALUE<DEV> tmp_update_values(gpma.get_size());
    anySync<DEV>();

    int merge_size = valid_size + update_size;
    thrust::sort_by_key(gpma.keys.begin(), gpma.keys.begin() + merge_size, gpma.values.begin());
    anySync<DEV>();

    if (DEV == GPU) {
        SIZE_TYPE THREADS_NUM = 32;
        SIZE_TYPE BLOCKS_NUM;
        BLOCKS_NUM = CALC_BLOCKS_NUM(THREADS_NUM, merge_size);
        redispatch_kernel<<<BLOCKS_NUM, THREADS_NUM>>>(RAW_PTR(gpma.keys), RAW_PTR(gpma.values), RAW_PTR(tmp_update_keys), RAW_PTR(tmp_update_values), gpma.get_size(), gpma.segment_length, merge_size, RAW_PTR(gpma.row_offset), 0);
        anySync<DEV>();
    } else {
#pragma omp parallel for schedule(dynamic, 64)
        for (auto i = 0; i < merge_size; ++i) {
            redispatch_loop_body(i, RAW_PTR(gpma.keys), RAW_PTR(gpma.values), RAW_PTR(tmp_update_keys), RAW_PTR(tmp_update_values), gpma.get_size(), gpma.segment_length, merge_size, RAW_PTR(gpma.row_offset), 0);
        }
    }

    gpma.keys = tmp_update_keys;
    gpma.values = tmp_update_values;
    anySync<DEV>();
}

template <dev_type_t DEV>
__host__ void update_gpma(GPMA<DEV> &gpma, NATIVE_VEC_KEY<DEV> &update_keys, NATIVE_VEC_VALUE<DEV> &update_values) {
    DEBUG_PRINTFLN("DBG: (ENTER UPDATE)update_gpma args, update_keys={}, values={}", rlib::printable_iter(update_keys), rlib::printable_iter(values_for_print(update_values)));
    gpma.print_status("ENTER update_gpma");

    //LOG_TIME("enter_update_gpma")

    // step1: sort update keys with values
    thrust::sort_by_key(update_keys.begin(), update_keys.end(), update_values.begin());
    anySync<DEV>();
    //LOG_TIME("1-2")

    // step2: get leaf node of each update (execute del and mod)
    NATIVE_VEC_SIZE<DEV> update_nodes(update_keys.size());
    anySync<DEV>();
    locate_leaf_batch<DEV>(RAW_PTR(gpma.keys), RAW_PTR(gpma.values), gpma.segment_length, gpma.tree_height, RAW_PTR(update_keys), RAW_PTR(update_values), update_keys.size(), RAW_PTR(update_nodes));
    //LOG_TIME("2-3")
    DEBUG_PRINTFLN("STATE 2-3: update_nodes={}", rlib::printable_iter(update_nodes));

    // step3: extract insertions
    NATIVE_VEC_SIZE<DEV> unique_update_nodes(update_keys.size());
    NATIVE_VEC_SIZE<DEV> update_offset(update_keys.size() + 1);
    anySync<DEV>();
    SIZE_TYPE update_size = update_nodes.size();
    SIZE_TYPE unique_node_size = 0;
    compact_insertions(update_nodes, update_keys, update_values,
                       update_size); // update_size was modified to be compacted. update_nodes,keys,values removed if nodes is NONE.
    DEBUG_PRINTFLN("STATE 3-4: unique_update_nodes={}, update_offset={}, update_size={}", rlib::printable_iter(unique_update_nodes), rlib::printable_iter(update_offset), update_size);
    compress_insertions_by_node(update_nodes, update_size, unique_update_nodes, update_offset, unique_node_size);
    anySync<DEV>();
    //LOG_TIME("3-4")
    DEBUG_PRINTFLN("STATE 3-4: unique_update_nodes={}, update_offset={}, update_size={}", rlib::printable_iter(unique_update_nodes), rlib::printable_iter(update_offset), update_size);

    // step4: rebuild for significant update
    int threshold = 5 * 1000 * 1000;
    if (update_size >= threshold) {
        significant_insert(gpma, update_keys, update_values, update_size);
        return;
    }
    //LOG_TIME("4-5")
    gpma.print_status("STATE 4-5");

    // step5: rebalance each tree level
    for (SIZE_TYPE level = 0; level <= gpma.tree_height && update_size; level++) {
        rlib::printfln("debug: rebalance tree level {}, tree_height={} update_size={}", level, gpma.tree_height, update_size);
        SIZE_TYPE lower_bound = gpma.lower_element[level];
        SIZE_TYPE upper_bound = gpma.upper_element[level];

        // re-balance
        //rlib::println("REBAL ARGS: ", level, gpma.segment_length, rlib::printable_iter(gpma.keys), rlib::printable_iter(gpma.values), rlib::printable_iter(update_nodes), rlib::printable_iter(update_keys), rlib::printable_iter(update_values), update_size, "|||", rlib::printable_iter(unique_update_nodes), "|", rlib::printable_iter(update_offset), "|", unique_node_size, lower_bound, upper_bound, rlib::printable_iter(gpma.row_offset));
        rebalance_batch<DEV>(level, gpma.segment_length, RAW_PTR(gpma.keys), RAW_PTR(gpma.values), RAW_PTR(update_nodes), RAW_PTR(update_keys), RAW_PTR(update_values), update_size, RAW_PTR(unique_update_nodes), RAW_PTR(update_offset), unique_node_size, lower_bound, upper_bound, RAW_PTR(gpma.row_offset));
        gpma.print_status("IN 5, after REBAL");

        // compact
        compact_insertions(update_nodes, update_keys, update_values, update_size);

        // up level
        up_level_batch<DEV>(RAW_PTR(update_nodes), update_size, gpma.segment_length << level);

        // re-compress
        compress_insertions_by_node(update_nodes, update_size, unique_update_nodes, update_offset, unique_node_size);
    }
    //LOG_TIME("5-6")
    gpma.print_status("STATE 5-6");

    // step6: rebalance the root node if necessary
    if (update_size > 0) {
        resize_gpma(gpma, update_size);

        SIZE_TYPE level = gpma.tree_height;
        SIZE_TYPE lower_bound = gpma.lower_element[level];
        SIZE_TYPE upper_bound = gpma.upper_element[level];

        anySync<DEV>();
        // re-balance
        rebalance_batch<DEV>(level, gpma.segment_length, RAW_PTR(gpma.keys), RAW_PTR(gpma.values), RAW_PTR(update_nodes), RAW_PTR(update_keys), RAW_PTR(update_values), update_size, RAW_PTR(unique_update_nodes), RAW_PTR(update_offset), unique_node_size, lower_bound, upper_bound, RAW_PTR(gpma.row_offset));
    }
    //LOG_TIME("6-7 LEAVE update_gpma")

    anySync<DEV>();
    gpma.print_status("LEAVE update_gpma");
    rlib::printfln("DBG: (LEAVE UPDATE) =====================================================================================");
}
