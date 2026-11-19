#include <cuda_runtime.h>

__global__ void cudaTiledGemm(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N,
    size_t K
);

__global__ void cudaTranspose(
    const float* A,
    float* B,
    size_t M,
    size_t N
);