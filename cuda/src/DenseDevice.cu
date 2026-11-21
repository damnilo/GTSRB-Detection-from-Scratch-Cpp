#include "Layers/include/Dense.h"
#include "cuda/include/DeviceMemory.cuh"
#include "cuda/include/TensorOps.cuh"

struct DenseDeviceState {
    float* output = nullptr;
    float* weights = nullptr;
    float* bias = nullptr;
    size_t batchCapacity = 0;
    bool weightsOnDevice = false;
};

void Dense::releaseDevice() {
    auto* state = static_cast<DenseDeviceState*>(device);
    if (state == nullptr) {
        return;
    }
    freeFloats(state->output);
    freeFloats(state->weights);
    freeFloats(state->bias);
    delete state;
    device = nullptr;
}

float* Dense::forwardDevice(const float* input, size_t batch, size_t channels, size_t height, size_t width) {
    if (channels * height * width != inputSize) {
        throw std::runtime_error("Dense device forward got a different input size");
    }
    auto* state = static_cast<DenseDeviceState*>(device);
    if (state == nullptr) {
        state = new DenseDeviceState();
        device = state;
    }
    if (!state->weightsOnDevice) {
        state->weights = allocFloats(inputSize * outputSize);
        state->bias = allocFloats(outputSize);
        mustCuda(cudaMemcpy(state->weights, weights.raw().data(), inputSize * outputSize * sizeof(float), cudaMemcpyHostToDevice));
        mustCuda(cudaMemcpy(state->bias, biases.raw().data(), outputSize * sizeof(float), cudaMemcpyHostToDevice));
        state->weightsOnDevice = true;
    }
    if (batch > state->batchCapacity) {
        freeFloats(state->output);
        state->output = allocFloats(batch * outputSize);
        state->batchCapacity = batch;
    }

    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((outputSize + 63) / 64), static_cast<unsigned>((batch + 63) / 64));
    cudaTiledGemm<<<grid, block>>>(input, state->weights, state->output, batch, outputSize, inputSize);
    mustCuda(cudaGetLastError());

    const size_t count = batch * outputSize;
    const unsigned biasBlock = 256;
    const unsigned biasGrid = static_cast<unsigned>((count + biasBlock - 1) / biasBlock);
    cudaAddBias<<<biasGrid, biasBlock>>>(state->output, state->bias, batch, outputSize);
    mustCuda(cudaGetLastError());
    return state->output;
}
