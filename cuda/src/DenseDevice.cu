#include "Layers/include/Dense.h"
#include "cuda/include/DeviceMemory.cuh"
#include "cuda/include/TensorOps.cuh"

struct DenseDeviceState {
    float* output = nullptr;
    float* gradInput = nullptr;
    float* inputT = nullptr;
    float* weights = nullptr;
    float* weightsT = nullptr;
    float* bias = nullptr;
    float* gradWeights = nullptr;
    float* gradBias = nullptr;
    const float* input = nullptr;
    size_t batchCapacity = 0;
    size_t batch = 0;
    bool weightsOnDevice = false;
};

void Dense::releaseDevice() {
    auto* state = static_cast<DenseDeviceState*>(device);
    if (state == nullptr) {
        return;
    }
    freeFloats(state->output);
    freeFloats(state->gradInput);
    freeFloats(state->inputT);
    freeFloats(state->weights);
    freeFloats(state->weightsT);
    freeFloats(state->bias);
    freeFloats(state->gradWeights);
    freeFloats(state->gradBias);
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
        state->weightsT = allocFloats(inputSize * outputSize);
        state->bias = allocFloats(outputSize);
        state->gradWeights = allocFloats(inputSize * outputSize);
        state->gradBias = allocFloats(outputSize);
        mustCuda(cudaMemcpy(state->weights, weights.raw().data(), inputSize * outputSize * sizeof(float), cudaMemcpyHostToDevice));
        mustCuda(cudaMemcpy(state->bias, biases.raw().data(), outputSize * sizeof(float), cudaMemcpyHostToDevice));
        dim3 transposeBlock(16, 16);
        dim3 transposeGrid(
            static_cast<unsigned>((outputSize + transposeBlock.x - 1) / transposeBlock.x),
            static_cast<unsigned>((inputSize + transposeBlock.y - 1) / transposeBlock.y)
        );
        cudaTranspose<<<transposeGrid, transposeBlock>>>(state->weights, state->weightsT, inputSize, outputSize);
        mustCuda(cudaGetLastError());
        state->weightsOnDevice = true;
    }
    if (batch > state->batchCapacity) {
        freeFloats(state->output);
        freeFloats(state->gradInput);
        freeFloats(state->inputT);
        state->output = allocFloats(batch * outputSize);
        state->gradInput = allocFloats(batch * inputSize);
        state->inputT = allocFloats(batch * inputSize);
        state->batchCapacity = batch;
    }
    state->input = input;
    state->batch = batch;

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

float* Dense::backwardDevice(const float* gradOutput, size_t batch) {
    auto* state = static_cast<DenseDeviceState*>(device);
    if (state == nullptr || state->input == nullptr || state->batch == 0 || batch != state->batch) {
        throw std::runtime_error("Dense device backward requires a matching forward");
    }

    dim3 transposeBlock(16, 16);
    dim3 transposeGrid(
        static_cast<unsigned>((inputSize + transposeBlock.x - 1) / transposeBlock.x),
        static_cast<unsigned>((batch + transposeBlock.y - 1) / transposeBlock.y)
    );
    cudaTranspose<<<transposeGrid, transposeBlock>>>(state->input, state->inputT, batch, inputSize);
    mustCuda(cudaGetLastError());

    dim3 block(16, 16);
    dim3 weightGrid(
        static_cast<unsigned>((outputSize + 63) / 64),
        static_cast<unsigned>((inputSize + 63) / 64)
    );
    cudaTiledGemm<<<weightGrid, block>>>(state->inputT, gradOutput, state->gradWeights, inputSize, outputSize, batch);
    mustCuda(cudaGetLastError());

    const unsigned sumBlock = 256;
    const unsigned sumGrid = static_cast<unsigned>((outputSize + sumBlock - 1) / sumBlock);
    cudaSumRows<<<sumGrid, sumBlock>>>(gradOutput, state->gradBias, batch, outputSize);
    mustCuda(cudaGetLastError());

    dim3 inputGrid(
        static_cast<unsigned>((inputSize + 63) / 64),
        static_cast<unsigned>((batch + 63) / 64)
    );
    cudaTiledGemm<<<inputGrid, block>>>(gradOutput, state->weightsT, state->gradInput, batch, inputSize, outputSize);
    mustCuda(cudaGetLastError());

    mustCuda(cudaMemcpyAsync(gradWeights.raw().data(), state->gradWeights, inputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    mustCuda(cudaMemcpyAsync(gradBiases.raw().data(), state->gradBias, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    return state->gradInput;
}
