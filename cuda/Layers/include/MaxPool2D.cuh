#include <cuda_runtime.h>

__global__ void cudaMaxPoolForward(
    const float* A,
    float* B,
    int* maxIdx,
    size_t pool_size,
    size_t stride,
    size_t batch_size,
    size_t channels,
    size_t height,
    size_t width
);

__global__ void cudaMaxPoolBackward(
    const float* A,
    float* B,
    int* maxId,
    size_t N
);