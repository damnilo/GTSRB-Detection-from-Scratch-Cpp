#include <cuda_runtime.h>

__global__ void cudaIm2Col(
    const float* A,
    float* B,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    size_t batch_size,
    size_t channels,
    size_t height,
    size_t width
);

__global__ void cudaCol2Im(
    const float* A,
    float* B,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    size_t batch_size,
    size_t channels,
    size_t height,
    size_t width
);
