#include "Layers/include/MaxPool2D.h"
#include "cuda/include/DeviceMemory.cuh"
#include "cuda/include/MaxPool2D.cuh"

struct PoolDevice {
    float* output = nullptr;
    float* gradInput = nullptr;
    int* indices = nullptr;
    size_t capacity = 0;
    size_t inputCapacity = 0;
    size_t batch = 0;
    size_t channels = 0;
    size_t height = 0;
    size_t width = 0;
    size_t outCount = 0;
};

void MaxPool2D::releaseDevice() {
    auto* state = static_cast<PoolDevice*>(device);
    if (state == nullptr) {
        return;
    }
    freeFloats(state->output);
    freeFloats(state->gradInput);
    freeInts(state->indices);
    delete state;
    device = nullptr;
}

float* MaxPool2D::forwardDevice(const float* input, size_t batch, size_t channels, size_t height, size_t width) {
    auto* state = static_cast<PoolDevice*>(device);
    if (state == nullptr) {
        state = new PoolDevice();
        device = state;
    }
    size_t outH = 0;
    size_t outW = 0;
    outputSize(height, width, outH, outW);
    const size_t count = batch * channels * outH * outW;
    const size_t inputCount = batch * channels * height * width;
    if (count > state->capacity) {
        freeFloats(state->output);
        freeInts(state->indices);
        state->output = allocFloats(count);
        state->indices = allocInts(count);
        state->capacity = count;
    }
    if (inputCount > state->inputCapacity) {
        freeFloats(state->gradInput);
        state->gradInput = allocFloats(inputCount);
        state->inputCapacity = inputCount;
    }
    state->batch = batch;
    state->channels = channels;
    state->height = height;
    state->width = width;
    state->outCount = count;
    dim3 block(32, 1);
    dim3 grid(static_cast<unsigned>((batch + block.x - 1) / block.x), static_cast<unsigned>(channels));
    cudaMaxPoolForward<<<grid, block>>>(
        input, state->output, state->indices,
        pool_size, stride,
        batch, channels, height, width
    );
    mustCuda(cudaGetLastError());
    return state->output;
}

float* MaxPool2D::backwardDevice(const float* gradOutput, size_t batch) {
    auto* state = static_cast<PoolDevice*>(device);
    if (state == nullptr || state->batch == 0 || batch != state->batch) {
        throw std::runtime_error("MaxPool2D device backward requires a matching forward");
    }
    const size_t inputCount = batch * state->channels * state->height * state->width;
    mustCuda(cudaMemset(state->gradInput, 0, inputCount * sizeof(float)));
    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((state->outCount + block - 1) / block);
    cudaMaxPoolBackward<<<grid, block>>>(gradOutput, state->gradInput, state->indices, state->outCount);
    mustCuda(cudaGetLastError());
    return state->gradInput;
}
