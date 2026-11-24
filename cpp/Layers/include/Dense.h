#pragma once
#include "Layers/include/Layer.h"

// Fully-connected (linear) layer: output = input @ weights + biases.
// Expects 2D input of shape (N, inputSize); produces (N, outputSize).
class Dense : public Layer
{
private:
    size_t inputSize;
    size_t outputSize;

    Tensor weights;     // Shape (inputSize, outputSize)
    Tensor biases;      // Shape (outputSize,)
    Tensor inputCache;  // Input from forward(), needed to compute gradWeights in backward()
    Tensor gradWeights;
    Tensor gradBiases;
    void* device = nullptr;

public:
    ~Dense() override;
    void releaseDevice();
    Dense(size_t inputSize, size_t outputSize);
    size_t outputFeatures() const { return outputSize; }

    Tensor forward(const Tensor& input) override;
    float* forwardDevice(const float* input, size_t batch, size_t channels, size_t height, size_t width) override;

    Tensor backward(const Tensor& gradOutput) override;
    float* backwardDevice(const float* gradOutput, size_t batch) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    // Copies uploaded by the first device forward. The optimizer steps these in place.
    float* deviceWeights();
    float* deviceBias();
    float* deviceGradWeights();
    float* deviceGradBias();
    // Backward reads the transpose. Rebuild it after the weights change.
    void refreshDeviceTranspose();
};