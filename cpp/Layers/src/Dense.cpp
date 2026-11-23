#include "Layers/include/Dense.h"
#include <cmath>
#include <iostream>

// Xavier/Glorot uniform initialization: limit = sqrt(6 / (fan_in + fan_out)).
// This is not He init (that would be sqrt(6 / fan_in)). Biases start at zero.
Dense::Dense(size_t inputSize, size_t outputSize)
    : inputSize(inputSize), outputSize(outputSize),
      weights({inputSize, outputSize}),
      biases({outputSize}),
      gradWeights({inputSize, outputSize}),
      gradBiases({outputSize})
{
    float limit = std::sqrt(6.0f / (inputSize + outputSize));
    weights.randomize(-limit, limit);
    biases.fill(0.0f);
}

// output = input @ weights + biases (biases broadcast across the batch axis)
Tensor Dense::forward(const Tensor& input)
{
    inputCache = input; // Needed in backward() to compute gradWeights

    return input.matmul(weights)
               .broadcastAdd(biases);
}

// Standard linear-layer backward pass:
//   gradWeights = input^T @ gradOutput
//   gradBiases  = sum(gradOutput) over the batch axis
//   gradInput   = gradOutput @ weights^T
Tensor Dense::backward(const Tensor& gradOutput) {
    gradWeights = inputCache.transpose().matmul(gradOutput);
    gradBiases = gradOutput.sum(0);

    Tensor gradInput = gradOutput.matmul(weights.transpose());

    return gradInput;
}

std::vector<Tensor*> Dense::parameters() {
    return {&weights, &biases};
}

std::vector<Tensor*> Dense::gradients() {
    return {&gradWeights, &gradBiases};
}

Dense::~Dense() { releaseDevice(); }

#ifndef GTSRB_CUDA
void Dense::releaseDevice() {}

float* Dense::forwardDevice(const float* input, size_t batch, size_t channels, size_t height, size_t width) {
    (void)input;
    (void)batch;
    (void)channels;
    (void)height;
    (void)width;
    throw std::runtime_error("CUDA was not built");
}

float* Dense::backwardDevice(const float* gradOutput, size_t batch) {
    (void)gradOutput;
    (void)batch;
    throw std::runtime_error("CUDA was not built");
}
#endif