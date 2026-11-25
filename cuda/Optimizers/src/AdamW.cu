#include "../include/AdamW.cuh"

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
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float biasCorr1 = 1.0f - powf(beta1, static_cast<float>(timestep));
    const float biasCorr2 = 1.0f - powf(beta2, static_cast<float>(timestep));
    const float decay = 1.0f - lr * weightDecay;

    params[i] *= decay;
    moments[i] = beta1 * moments[i] + (1.0f - beta1) * grads[i];
    velocities[i] = beta2 * velocities[i] + (1.0f - beta2) * grads[i] * grads[i];
    const float corrMoment = moments[i] / biasCorr1;
    const float corrVelocity = velocities[i] / biasCorr2;
    params[i] -= lr * corrMoment / (sqrtf(corrVelocity) + eps);
}
