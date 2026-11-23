#include "../include/Dropout.cuh"

__global__ void cudaDropout(
    const float* input,
    float* output,
    const float* mask,
    size_t n
) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    output[i] = input[i] * mask[i];
}
