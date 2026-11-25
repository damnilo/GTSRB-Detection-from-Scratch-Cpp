#include <cuda_runtime.h>

__global__ void cudaAdamWStep(
    float* params,
    const float* grads,
    float* moments,
    float* velocities,
    float lr,
    float weightDecay,
    float beta1,
    float beta2,
    float eps,
    size_t timestep,
    size_t N
);
