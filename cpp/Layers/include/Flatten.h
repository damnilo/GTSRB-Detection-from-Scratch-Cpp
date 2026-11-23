#pragma once
#include "Layers/include/Layer.h"
#include <vector>

// Reshapes a multi-dimensional tensor (e.g. (N, C, H, W) from a conv stack)
// into a flat 2D tensor (N, C*H*W) suitable for feeding into a Dense layer.
class Flatten : public Layer
{
private:
    std::vector<size_t> inputShapeCache; // Original shape, needed to reshape gradients back in backward()
public:
    Tensor forward(const Tensor& input) override;
    float* forwardDevice(const float* input, size_t batch, size_t channels, size_t height, size_t width) override;
    
    Tensor backward(const Tensor& gradOutput) override;
    float* backwardDevice(const float* gradOutput, size_t batch) override;
};