#include "Layers/include/BatchNorm2D.h"
#include "cuda/Layers/include/BatchNorm2D.cuh"
#include "cuda/Utils/include/DeviceMemory.cuh"

struct BatchNormDevice {
    float* output = nullptr;
    float* xHat = nullptr;
    float* invStd = nullptr;
    float* gradInput = nullptr;
    float* gamma = nullptr;
    float* beta = nullptr;
    float* runningMean = nullptr;
    float* runningVar = nullptr;
    float* gradGamma = nullptr;
    float* gradBeta = nullptr;
    size_t capacity = 0;
    size_t batch = 0;
    size_t height = 0;
    size_t width = 0;
    bool parametersOnDevice = false;
};

void BatchNorm2D::releaseDevice() {
    auto* state = static_cast<BatchNormDevice*>(device);
    if (state == nullptr) {
        return;
    }
    freeFloats(state->output);
    freeFloats(state->xHat);
    freeFloats(state->invStd);
    freeFloats(state->gradInput);
    freeFloats(state->gamma);
    freeFloats(state->beta);
    freeFloats(state->runningMean);
    freeFloats(state->runningVar);
    freeFloats(state->gradGamma);
    freeFloats(state->gradBeta);
    delete state;
    device = nullptr;
}

float* BatchNorm2D::forwardDevice(const float* input, size_t batch, size_t channelCount, size_t height, size_t width) {
    if (channelCount != channels) {
        throw std::runtime_error("BatchNorm2D device forward got a different channel count");
    }
    auto* state = static_cast<BatchNormDevice*>(device);
    if (state == nullptr) {
        state = new BatchNormDevice();
        device = state;
    }

    const size_t parameterBytes = channels * sizeof(float);
    if (!state->parametersOnDevice) {
        state->gamma = allocFloats(channels);
        state->beta = allocFloats(channels);
        state->runningMean = allocFloats(channels);
        state->runningVar = allocFloats(channels);
        state->gradGamma = allocFloats(channels);
        state->gradBeta = allocFloats(channels);
        state->invStd = allocFloats(channels);
        mustCuda(cudaMemcpy(state->gamma, gamma.raw().data(), parameterBytes, cudaMemcpyHostToDevice));
        mustCuda(cudaMemcpy(state->beta, beta.raw().data(), parameterBytes, cudaMemcpyHostToDevice));
        mustCuda(cudaMemcpy(state->runningMean, runningMean.raw().data(), parameterBytes, cudaMemcpyHostToDevice));
        mustCuda(cudaMemcpy(state->runningVar, runningVar.raw().data(), parameterBytes, cudaMemcpyHostToDevice));
        state->parametersOnDevice = true;
    }

    const size_t count = batch * channels * height * width;
    if (count > state->capacity) {
        freeFloats(state->output);
        freeFloats(state->xHat);
        freeFloats(state->gradInput);
        state->output = allocFloats(count);
        state->xHat = allocFloats(count);
        state->gradInput = allocFloats(count);
        state->capacity = count;
    }
    state->batch = batch;
    state->height = height;
    state->width = width;

    cudaBatchNormForward<<<static_cast<unsigned>(channels), 256>>>(
        input,
        state->output,
        state->gamma,
        state->beta,
        state->runningMean,
        state->runningVar,
        state->xHat,
        state->invStd,
        eps,
        momentum,
        batch,
        channels,
        height,
        width,
        training
    );
    mustCuda(cudaGetLastError());
    mustCuda(cudaMemcpyAsync(runningMean.raw().data(), state->runningMean, parameterBytes, cudaMemcpyDeviceToHost));
    mustCuda(cudaMemcpyAsync(runningVar.raw().data(), state->runningVar, parameterBytes, cudaMemcpyDeviceToHost));
    return state->output;
}

float* BatchNorm2D::backwardDevice(const float* gradOutput, size_t batch) {
    auto* state = static_cast<BatchNormDevice*>(device);
    if (state == nullptr || state->batch == 0 || batch != state->batch) {
        throw std::runtime_error("BatchNorm2D device backward requires a matching forward");
    }
    cudaBatchNormBackward<<<static_cast<unsigned>(channels), 256>>>(
        gradOutput,
        state->gradInput,
        state->gamma,
        state->gradGamma,
        state->gradBeta,
        state->xHat,
        state->invStd,
        batch,
        channels,
        state->height,
        state->width,
        training
    );
    mustCuda(cudaGetLastError());
    const size_t parameterBytes = channels * sizeof(float);
    mustCuda(cudaMemcpyAsync(gradGamma.raw().data(), state->gradGamma, parameterBytes, cudaMemcpyDeviceToHost));
    mustCuda(cudaMemcpyAsync(gradBeta.raw().data(), state->gradBeta, parameterBytes, cudaMemcpyDeviceToHost));
    return state->gradInput;
}
