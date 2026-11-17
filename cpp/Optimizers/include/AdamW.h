#pragma once
#include "Utils/include/Tensor.h"
#include <vector>

// Decoupled AdamW, matching torch.optim.AdamW with a single parameter group.
// Weight decay scales the parameter itself. It is not added to the gradient.
class AdamW {
private:
    float learningRate;
    float weightDecay;
    float beta1;
    float beta2;
    float epsilon;
    size_t timestep;
    std::vector<Tensor> moments;
    std::vector<Tensor> velocities;

public:
    explicit AdamW(float learningRate = 1e-3f, float weightDecay = 1e-2f,
                   float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    // params and grads are parallel vectors, same as SGD::update.
    void update(std::vector<Tensor*>& params, std::vector<Tensor*>& grads);
    void setLearningRate(float newLearningRate);

    void restore(size_t step, std::vector<Tensor> moments, std::vector<Tensor> velocities);

    float getLearningRate() const;
    float getWeightDecay() const;
    float getBeta1() const;
    float getBeta2() const;
    float getEpsilon() const;
    size_t getTimestep() const;
    std::vector<Tensor> getMoments() const;
    std::vector<Tensor> getVelocities() const;
};
