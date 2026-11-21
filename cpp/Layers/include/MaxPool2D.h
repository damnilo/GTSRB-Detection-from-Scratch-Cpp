#pragma once
#include "Layers/include/Layer.h"
#include <vector>

// 2D max pooling: downsamples each (pool_size x pool_size) window to its max value,
// independently per channel and batch item.
class MaxPool2D : public Layer
{
private:
    size_t pool_size;
    size_t stride;

    std::vector<size_t> inputShapeCache; // Original (N,C,H,W) shape, needed to size backward()'s output
    std::vector<size_t> maxIndicesCache; // Flat input-index of the max element for each output position
    void* device = nullptr;

public:
    MaxPool2D(size_t pool_size, size_t stride);
    ~MaxPool2D() override;
    void releaseDevice();
    void outputSize(size_t inH, size_t inW, size_t& outH, size_t& outW) const {
        outH = (inH - pool_size) / stride + 1;
        outW = (inW - pool_size) / stride + 1;
    }

    Tensor forward(const Tensor& input) override;
    float* forwardDevice(const float* input, size_t batch, size_t channels, size_t height, size_t width) override;
    
    Tensor backward(const Tensor& gradOutput) override;
};