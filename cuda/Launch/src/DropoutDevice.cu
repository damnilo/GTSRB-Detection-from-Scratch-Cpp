#include "Layers/include/Dropout.h"
#include "cuda/Layers/include/Dropout.cuh"
#include "cuda/Utils/include/DeviceMemory.cuh"

struct DropoutState {
    float* output = nullptr;
    float* gradInput = nullptr;
    float* mask = nullptr;
    size_t capacity = 0;
    size_t count = 0;
    size_t batch = 0;
};

void Dropout::releaseDevice() {
    auto* state = static_cast<DropoutState*>(device);
    if (state == nullptr) {
        return;
    }
    freeFloats(state->output);
    freeFloats(state->gradInput);
    freeFloats(state->mask);
    delete state;
    device = nullptr;
}

float* Dropout::forwardDevice(const float* input, size_t batch, size_t channels, size_t height, size_t width) {
    if (!training || rate <= 0.0f) {
        return const_cast<float*>(input);
    }
    auto* state = static_cast<DropoutState*>(device);
    if (state == nullptr) {
        state = new DropoutState();
        device = state;
    }
    const size_t count = batch * channels * height * width;
    if (count > state->capacity) {
        freeFloats(state->output);
        freeFloats(state->gradInput);
        freeFloats(state->mask);
        state->output = allocFloats(count);
        state->gradInput = allocFloats(count);
        state->mask = allocFloats(count);
        state->capacity = count;
    }
    fillMask(count);
    mustCuda(cudaMemcpy(state->mask, maskCache.raw().data(), count * sizeof(float), cudaMemcpyHostToDevice));
    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((count + block - 1) / block);
    cudaDropout<<<grid, block>>>(input, state->output, state->mask, count);
    mustCuda(cudaGetLastError());
    state->count = count;
    state->batch = batch;
    return state->output;
}

float* Dropout::backwardDevice(const float* gradOutput, size_t batch) {
    if (!training || rate <= 0.0f) {
        return const_cast<float*>(gradOutput);
    }
    auto* state = static_cast<DropoutState*>(device);
    if (state == nullptr || state->batch == 0 || batch != state->batch) {
        throw std::runtime_error("Dropout device backward requires a matching forward");
    }
    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((state->count + block - 1) / block);
    cudaDropout<<<grid, block>>>(gradOutput, state->gradInput, state->mask, state->count);
    mustCuda(cudaGetLastError());
    return state->gradInput;
}
