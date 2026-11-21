#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

inline void mustCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

inline float* allocFloats(size_t count) {
    float* pointer = nullptr;
    mustCuda(cudaMalloc(&pointer, count * sizeof(float)));
    return pointer;
}

inline int* allocInts(size_t count) {
    int* pointer = nullptr;
    mustCuda(cudaMalloc(&pointer, count * sizeof(int)));
    return pointer;
}

inline void freeFloats(float*& pointer) {
    if (pointer != nullptr) {
        mustCuda(cudaFree(pointer));
        pointer = nullptr;
    }
}

inline void freeInts(int*& pointer) {
    if (pointer != nullptr) {
        mustCuda(cudaFree(pointer));
        pointer = nullptr;
    }
}
