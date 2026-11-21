#pragma once
#include "Layers/include/Layer.h"

// Elementwise ReLU activation: output = max(0, input). No learnable parameters.
class ReLU : public Layer
{
private:
    Tensor maskCache; // 1.0 where input was positive, 0.0 otherwise — reused in backward()
    void* device = nullptr;

public:
    ~ReLU() override;
    void releaseDevice();

    Tensor forward(const Tensor& input) override;
    float* forwardDevice(const float* input, size_t batch, size_t channels, size_t height, size_t width) override;
    
    Tensor backward(const Tensor& gradOutput) override;
};