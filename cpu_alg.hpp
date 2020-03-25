#ifndef RLIB_CPU_RLE_HPP_
#define RLIB_CPU_RLE_HPP_ 1

#include <omp.h>

namespace rlib {
template <typename T, typename SIZE_T>
size_t cpu_rle_simple(const T *inputVector, const size_t inputLength, T *outputVector, SIZE_T *outputLengthVector) {
    // returns outputLength.
    if (inputLength == 0)
        return 0;

    size_t currRunBeginIndex = 0;
    size_t outputVectorCurrIndex = 0;
    for (auto cter = 0; cter < inputLength; ++cter) {
        if (inputVector[currRunBeginIndex] != inputVector[cter]) {
            outputVector[outputVectorCurrIndex] = inputVector[currRunBeginIndex];
            outputLengthVector[outputVectorCurrIndex] = cter - currRunBeginIndex;
            ++outputVectorCurrIndex;

            currRunBeginIndex = cter;
        }
    }

    outputVector[outputVectorCurrIndex] = inputVector[currRunBeginIndex];
    outputLengthVector[outputVectorCurrIndex] = inputLength - currRunBeginIndex;
    ++outputVectorCurrIndex;

    return outputVectorCurrIndex;
}
template <typename T, typename SIZE_T>
size_t cpu_rle_parallel(const T *inputVector, size_t inputLength, T *outputVector, SIZE_T *outputLengthVector) {
    // returns outputLength.

    throw std::runtime_error("not implemented yet.");
}

template <typename T>
void prefixsumSimple(T *inputVec, size_t inputLen) {
    for (auto i = 1; i < inputLen; ++i)
        inputVec[i] += inputVec[i - 1];
}

template <typename T>
void prefixsumParallel(T *inputVec, size_t inputLen) {
    T *suma;
    const auto nthreads = omp_get_num_threads();
    if (inputLen < 4 * nthreads || nthreads < 3)
        return prefixsumSimple(inputVec, inputLen);
#pragma omp parallel
    {
        const auto ithread = omp_get_thread_num();
#pragma omp single
        {
            suma = new T[nthreads + 1];
            suma[0] = 0;
        }
        T sum(0);
#pragma omp for schedule(static)
        for (auto i = 0; i < inputLen; i++) {
            sum += inputVec[i];
            inputVec[i] = sum;
        }
        suma[ithread + 1] = sum;
#pragma omp barrier
        T offset(0);
        for (auto i = 0; i < (ithread + 1); i++) {
            offset += suma[i];
        }
#pragma omp for schedule(static)
        for (auto i = 0; i < inputLen; i++) {
            inputVec[i] += offset;
        }
    }
    delete[] suma;
}

template <typename T>
void exclusiveSumParallel(const T *inputVec, T *outputVec, const size_t len) {
    if (len == 0)
        return;
#pragma omp parallel for schedule(static)
    for (auto i = 1; i < len; ++i) {
        outputVec[i] = inputVec[i - 1];
    }
    outputVec[0] = T(0);
    return prefixsumParallel(outputVec, len);
}

} // namespace rlib

#endif
