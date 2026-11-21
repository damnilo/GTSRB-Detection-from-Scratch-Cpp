#pragma once
#include "Layers/include/Layer.h"

// Inverted dropout regularization layer. During training, randomly zeroes a
// fraction (`rate`) of activations and scales the survivors by 1/(1-rate) so
// the expected activation magnitude is preserved; this means eval mode needs
// no special-case scaling, just a pass-through.
class Dropout : public Layer
{
private:
    float rate;
    bool training;
    Tensor maskCache; // Per-element scale (0 if dropped, 1/(1-rate) if kept) — reused in backward()

public:
    explicit Dropout(float rate);

    // Seeds the generator that draws dropout masks. Independent from Tensor::seed,
    // so weight initialization does not shift the mask stream.
    static void seed(unsigned int seed);

    Tensor forward(const Tensor& input) override;
    float* forwardDevice(const float* input, size_t batch, size_t channels, size_t height, size_t width) override;
    
    Tensor backward(const Tensor& gradOutput) override;

    void setTraining(bool isTraining); // Must be set to false before evaluation/inference
    bool isTraining() const;
};