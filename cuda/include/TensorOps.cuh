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

__global__ void cudaAddBias(
    float* A,
    const float* B,
    size_t M,
    size_t N
);

__global__ void cudaSpatialToNCHW(
    const float* spatial,
    float* nchw,
    size_t batch,
    size_t channels,
    size_t height,
    size_t width
);