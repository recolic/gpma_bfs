#pragma once

#include "cpu_alg.hpp"
#include "rdebug.hpp"

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <type_traits>

#define MAKE_64b(high, low) (((uint64_t)(high) << 32) + low)

template <typename T>
constexpr bool IsPowerOfTwo(T x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

using KEY_TYPE = uint64_t;
using VALUE_TYPE = char;
using SIZE_TYPE = uint32_t;

namespace runtime_info {
// some magic for template argument.
using dev_type_t = size_t;
constexpr dev_type_t GPU = 19990823;
constexpr dev_type_t CPU = 19981223;

template <dev_type_t dev_type, typename value_type>
struct native_vector;
template <typename value_type>
struct native_vector<GPU, value_type> : thrust::device_vector<value_type> {
    using par_type = thrust::device_vector<value_type>;
    using par_type::device_vector;
    native_vector(const par_type &ano)
        : par_type::device_vector(ano) {}
};
template <typename value_type>
struct native_vector<CPU, value_type> : thrust::host_vector<value_type> {
    using par_type = thrust::host_vector<value_type>;
    using par_type::host_vector;
    native_vector(const par_type &ano)
        : par_type::host_vector(ano) {}
};

template <dev_type_t dev_type>
using NATIVE_VEC_KEY = native_vector<dev_type, KEY_TYPE>;
template <dev_type_t dev_type>
using NATIVE_VEC_VALUE = native_vector<dev_type, VALUE_TYPE>;
template <dev_type_t dev_type>
using NATIVE_VEC_SIZE = native_vector<dev_type, SIZE_TYPE>;
} // namespace runtime_info
using namespace runtime_info;

inline __host__ __device__ void cErr(cudaError_t code) {
    if (code != cudaSuccess) {
        printf("GPUassert: %s\n", cudaGetErrorString(code));
    }
}

template <dev_type_t DEV>
__host__ __device__ void anySync() {
    if (DEV == GPU)
        cudaDeviceSynchronize();
}

template <dev_type_t DEV>
void anyMalloc(void **, size_t);
template <dev_type_t DEV>
void anyFree(void *);
template <>
void anyMalloc<CPU>(void **out_ptr, size_t size) {
    *out_ptr = malloc(size);
    cErr(*out_ptr == NULL ? cudaErrorMemoryAllocation : cudaSuccess);
}
template <>
void anyFree<CPU>(void *ptr) {
    free(ptr);
}
template <>
__host__ __device__ void anyMalloc<GPU>(void **out_ptr, size_t size) {
    cErr(cudaMalloc(out_ptr, size));
}
template <>
__host__ __device__ void anyFree<GPU>(void *ptr) {
    cErr(cudaFree(ptr));
}

template <dev_type_t DEV>
void anyMemset(void *dst, int value, size_t count) {
    if(DEV == GPU)
        cErr(cudaMemset(dst, value, count));
    else
        memset(dst, value, count);
}

template <dev_type_t DEV_SRC, dev_type_t DEV_DST>
void anyMemcpy(void *dst, const void *src, size_t count) {
    cudaMemcpyKind kind = DEV_SRC == GPU ? (DEV_DST == GPU ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost) : (DEV_DST == GPU ? cudaMemcpyHostToDevice : cudaMemcpyHostToHost);
    cErr(cudaMemcpy(dst, src, count, kind));
}

template <dev_type_t DEV>
void anyRunLengthEncoding(const SIZE_TYPE *inputVec, SIZE_TYPE inputLen, SIZE_TYPE *outputVec, SIZE_TYPE *outputLenVec, SIZE_TYPE *outputLen) {
    // ALL input pointers should be located on DEV.
    if (DEV == GPU) {
        void *temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cErr(cub::DeviceRunLengthEncode::Encode(temp_storage, temp_storage_bytes, inputVec, outputVec, outputLenVec, outputLen, inputLen));
        anySync<DEV>(); // TODO: test and remove them.
        anyMalloc<DEV>(&temp_storage, temp_storage_bytes);
        cErr(cub::DeviceRunLengthEncode::Encode(temp_storage, temp_storage_bytes, inputVec, outputVec, outputLenVec, outputLen, inputLen));
        anySync<DEV>();
        anyFree<DEV>(temp_storage);

        SIZE_TYPE tmp;
        anyMemcpy<GPU, CPU>(&tmp, outputLen, sizeof(SIZE_TYPE));
    printf("RLE result: outputSize=%d, inputSize=%d\n", tmp, inputLen);
    } else {
        *outputLen = rlib::cpu_rle_simple(inputVec, inputLen, outputVec, outputLenVec);
    printf("RLE result: outputSize=%d, inputSize=%d\n", *outputLen, inputLen);
//    for(auto cter = 0; cter < inputLen; ++cter)
//        printf("IN %d\n", inputVec[cter]);
//    for(auto cter = 0; cter < *outputLen; ++cter)
//        printf("OUT %d*%d\n", outputLenVec[cter], outputVec[cter]);
    }


}

// Sometimes we need to call exsum from gpu code...
__host__ __device__ void cudaExclusiveSum(const SIZE_TYPE *inputVec, SIZE_TYPE *outputVec, SIZE_TYPE len) {
    void *temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cErr(cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_bytes, inputVec, outputVec, len));
    anyMalloc<GPU>(&temp_storage, temp_storage_bytes);
    cErr(cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_bytes, inputVec, outputVec, len));
    anySync<GPU>();
    anyFree<GPU>(temp_storage);
}
template <dev_type_t DEV>
void anyExclusiveSum(const SIZE_TYPE *inputVec, SIZE_TYPE *outputVec, SIZE_TYPE len) {
    if (DEV == GPU) {
        cudaExclusiveSum(inputVec, outputVec, len);
    } else
        rlib::exclusiveSumParallel(inputVec, outputVec, len);
}
