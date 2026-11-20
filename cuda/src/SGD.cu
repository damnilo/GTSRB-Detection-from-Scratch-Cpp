#include "../include/SGD.cuh"

__global__ void cudaSGDStep(
    float* params,
    const float* grads,
    float* velocity,
    float lr,
    float momentum,
    float decay,
    size_t N
) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    float effective = grads[i] + decay * params[i];
    velocity[i] = momentum * velocity[i] + effective;
    params[i] = params[i] - lr * velocity[i];
}