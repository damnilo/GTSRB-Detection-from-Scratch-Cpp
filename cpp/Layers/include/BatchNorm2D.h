#pragma once
#include "Layers/include/Layer.h"

// Per-channel batch normalization over NCHW. Training uses the current batch
// mean and variance; evaluation uses the running estimates. gamma and beta are
// the only learnable parameters.
class BatchNorm2D : public Layer
{
private:
    size_t channels;
    float eps;
    float momentum;

    Tensor gamma;
    Tensor beta;
    Tensor runningMean;
    Tensor runningVar;
    Tensor gradGamma;
    Tensor gradBeta;

    Tensor xHatCache;
    Tensor invStdCache;
    bool training;
    void* device = nullptr;

public:
    explicit BatchNorm2D(size_t channels);
    ~BatchNorm2D() override;
    void releaseDevice();

    Tensor forward(const Tensor& input) override;
    float* forwardDevice(const float* input, size_t batch, size_t channels, size_t height, size_t width) override;
    Tensor backward(const Tensor& gradOutput) override;
    float* backwardDevice(const float* gradOutput, size_t batch) override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    void setTraining(bool isTraining);
    bool isTraining() const;

    // Not parameters: Serialization does not write these. Checkpoints store them separately.
    Tensor* runningMeanBuffer();
    Tensor* runningVarBuffer();
};
