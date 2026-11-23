#include <cuda_runtime.h>

__global__ void cudaReLUForward(
    const float* A,
    float* B,
    int* mask,
    size_t N
);

__global__ void cudaReLUBackward(
    const float* A,
    float* B,
    int* mask,
    size_t N
);