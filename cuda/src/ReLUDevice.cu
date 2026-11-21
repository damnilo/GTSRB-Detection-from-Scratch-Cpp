#include "Layers/include/ReLU.h"
#include "cuda/include/DeviceMemory.cuh"
#include "cuda/include/ReLU.cuh"

struct ReLUDevice {
    float* output = nullptr;
    int* mask = nullptr;
    size_t capacity = 0;
};

void ReLU::releaseDevice() {
    auto* state = static_cast<ReLUDevice*>(device);
    if (state == nullptr) {
        return;
    }
    freeFloats(state->output);
    freeInts(state->mask);
    delete state;
    device = nullptr;
}

float* ReLU::forwardDevice(const float* input, size_t batch, size_t channels, size_t height, size_t width) {
    auto* state = static_cast<ReLUDevice*>(device);
    if (state == nullptr) {
        state = new ReLUDevice();
        device = state;
    }
    const size_t count = batch * channels * height * width;
    if (count > state->capacity) {
        freeFloats(state->output);
        freeInts(state->mask);
        state->output = allocFloats(count);
        state->mask = allocInts(count);
        state->capacity = count;
    }
    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((count + block - 1) / block);
    cudaReLUForward<<<grid, block>>>(input, state->output, state->mask, count);
    mustCuda(cudaGetLastError());
    return state->output;
}
