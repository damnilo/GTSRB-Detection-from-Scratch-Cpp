#include "../include/ReLU.cuh"

__global__ void cudaReLUForward(
    const float* A,
    float* B,
    int* mask,
    size_t N
) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    B[i] = (A[i] > 0.0f) ? A[i] : 0.0f;
    mask[i] = (A[i] > 0.0f) ? 1 : 0;
}

__global__ void cudaReLUBackward(
    const float* A,
    float* B,
    int* mask,
    size_t N
) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    B[i] = A[i] * static_cast<float>(mask[i]);
}