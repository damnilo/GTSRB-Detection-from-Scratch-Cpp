#ifndef LAYER_H
#define LAYER_H
#include "Utils/include/Tensor.h"
#include <stdexcept>
#include <vector>

// Abstract base class for every layer in the network (Dense, Conv2D, ReLU,
// MaxPool2D, Dropout, SoftMax, Flatten). Network composes layers polymorphically
// through this interface, so any new layer type only needs to implement these four.
class Layer
{
public:
    // Computes the layer's output given its input, and caches whatever internal
    // state (input, mask, etc.) is needed to compute gradients in backward().
    virtual Tensor forward(const Tensor& input) = 0;
    virtual float* forwardDevice(const float* input, size_t batch, size_t channels, size_t height, size_t width) {
        throw std::runtime_error("CUDA was not built");
    }
    // Given dL/dOutput, returns dL/dInput, using state cached during forward().
    virtual Tensor backward(const Tensor& gradOutput) = 0;
    virtual float* backwardDevice(const float* gradOutput, size_t batch) {
        (void)gradOutput;
        (void)batch;
        throw std::runtime_error("CUDA was not built");
    }
    // Layers with learnable weights (Dense, Conv2D) override these to expose
    // pointers to their parameter/gradient tensors so Network and the optimizer
    // can collect and update them. Layers with no parameters (ReLU, Dropout,
    // MaxPool2D, Flatten, SoftMax) use the default empty-vector implementation.
    virtual std::vector<Tensor*> parameters() { return {}; };
    virtual std::vector<Tensor*> gradients() { return {}; };

    virtual ~Layer() = default;
};

#endif // LAYER_H