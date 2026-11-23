#include <cuda_runtime.h>

__global__ void cudaSGDStep(
    float* params,
    const float* grads,
    float* velocity,
    float lr,
    float momentum,
    float decay,
    size_t N
);