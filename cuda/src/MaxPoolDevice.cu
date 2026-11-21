#include "Layers/include/MaxPool2D.h"
#include "cuda/include/DeviceMemory.cuh"
#include "cuda/include/MaxPool2D.cuh"

struct PoolDevice {
    float* output = nullptr;
    int* indices = nullptr;
    size_t capacity = 0;
};

void MaxPool2D::releaseDevice() {
    auto* state = static_cast<PoolDevice*>(device);
    if (state == nullptr) {
        return;
    }
    freeFloats(state->output);
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
    if (count > state->capacity) {
        freeFloats(state->output);
        freeInts(state->indices);
        state->output = allocFloats(count);
        state->indices = allocInts(count);
        state->capacity = count;
    }
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
