#include <cuda_runtime.h>

__global__ void cudaDropout(
    const float* input,
    float* output,
    const float* mask,
    size_t n
);
