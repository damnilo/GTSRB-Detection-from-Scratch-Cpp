#include <cuda_runtime.h>

__global__ void cudaBatchNormForward(
    const float* input,
    float* output,
    float* gamma,
    float* beta,
    float* runningMean,
    float* runningVar,
    float* xHat,
    float* invStd,
    float eps,
    float momentum,
    size_t batch,
    size_t channels,
    size_t height,
    size_t width,
    bool training
);

__global__ void cudaBatchNormBackward(
    const float* input,
    float* output,
    float* gamma,
    float* gradGamma,
    float* gradBeta,
    float* xHat,
    float* invStd,
    size_t batch,
    size_t channels,
    size_t height,
    size_t width,
    bool training
);