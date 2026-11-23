#include "../include/BatchNorm2D.cuh"

__device__ float blockReduce(float value) {
    
    __shared__ float shared[256];
    
    const unsigned lane = threadIdx.x;
    shared[lane] = value;
    
    __syncthreads();
    
    for (unsigned offset = blockDim.x >> 1; offset > 0; offset >>= 1) {

        if (lane < offset) {
            shared[lane] += shared[lane + offset];
        }

        __syncthreads();

    }

    return shared[0];
}

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
) {

    const int c = blockIdx.x;
    
    if (c >= channels) return;

    const int spatial = height * width;
    const int count = batch * spatial;
    const float inverseCount = 1.0f / static_cast<float>(count);

    float sum = 0.0f;
    if (training) {
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            const int n = i / spatial;
            const int hw = i - n * spatial;
            const int index = (n * channels + c) * spatial + hw;
    
            sum += input[index];
        }
    }

    sum = blockReduce(sum);

    __shared__ float meanShared;
    __shared__ float invStdShared;

    if (threadIdx.x == 0) {
        meanShared = training ? sum * inverseCount : runningMean[c];
    }

    __syncthreads();
    
    const float mean = meanShared;

    float sumSquares = 0.0f;
    
    if (training) {
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            const int n = i / spatial;
            const int hw = i - n * spatial;
            const int index = (n * channels + c) * spatial + hw;
    
            const float delta = input[index] - mean;
            sumSquares += delta * delta;
        }
    }

    sumSquares = blockReduce(sumSquares);

    if (threadIdx.x == 0) {
        if (training) {
            const float variance = sumSquares * inverseCount;
            invStdShared = rsqrtf(variance + eps);
            float unbiased = variance;
            if (count > 1) {
                unbiased *= static_cast<float>(count) / static_cast<float>(count - 1);
            }
            const float keep = 1.0f - momentum;
            runningMean[c] = keep * runningMean[c] + momentum * mean;
            runningVar[c] = keep * runningVar[c] + momentum * unbiased;
        } else {
            invStdShared = rsqrtf(runningVar[c] + eps);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        invStd[c] = invStdShared;
    }
    const float inverseStd = invStdShared;
    const float scale = gamma[c];
    const float shift = beta[c];

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const int n = i / spatial;
        const int hw = i - n * spatial;
        const int index = (n * channels + c) * spatial + hw;
        xHat[index] = (input[index] - mean) * inverseStd;
        output[index] = scale * xHat[index] + shift;
    }
}


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
) {
    
    const int c = blockIdx.x;

    if (c >= channels) return;
    
    const int spatial = height * width;
    const int count = batch * spatial;
    const float inverseCount = 1.0f / static_cast<float>(count);

    const float scale = gamma[c];

    float partialBeta = 0.0f;
    float partialGamma = 0.0f;
    float partialDxHat = 0.0f;
    float partialDxHatXHat = 0.0f;

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const int n = i / spatial;
        const int hw = i - n * spatial;
        const int index = (n * channels + c) * spatial + hw;
    
        const float dy = input[index];
        const float dxHat = dy * scale;
    
        partialBeta += dy;
        partialGamma += dy * xHat[index];
        partialDxHat += dxHat;
        partialDxHatXHat += dxHat * xHat[index];
    }

    const float sumBeta = blockReduce(partialBeta);
    const float sumGamma = blockReduce(partialGamma);
    const float sumDxHat = blockReduce(partialDxHat);
    const float sumDxHatXHat = blockReduce(partialDxHatXHat);

    if (threadIdx.x == 0) {
        gradBeta[c] = sumBeta;
        gradGamma[c] = sumGamma;
    }

    const float inverseStd = invStd[c];

    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const int n = i / spatial;
        const int hw = i - n * spatial;
        const int index = (n * channels + c) * spatial + hw;
        const float dxHat = input[index] * scale;
    
        if (training) {
            output[index] = inverseStd * inverseCount * (
                static_cast<float>(count) * dxHat - sumDxHat - xHat[index] * sumDxHatXHat
            );
        } else {
            output[index] = dxHat * inverseStd;
        }
    }
}