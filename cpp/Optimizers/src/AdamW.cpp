#include "Optimizers/include/AdamW.h"
#include <cmath>
#include <stdexcept>

AdamW::AdamW(float learningRate, float weightDecay, float beta1, float beta2, float epsilon)
    : learningRate(learningRate),
      weightDecay(weightDecay),
      beta1(beta1),
      beta2(beta2),
      epsilon(epsilon),
      timestep(0) {}

// For each parameter, with t starting at 1:
//   param = param * (1 - lr * weightDecay)
//   m = beta1 * m + (1 - beta1) * grad
//   v = beta2 * v + (1 - beta2) * grad^2
//   param -= lr * (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + eps)
void AdamW::update(std::vector<Tensor*>& params, std::vector<Tensor*>& grads) {
    if (params.size() != grads.size()) {
        throw std::invalid_argument("AdamW parameter count does not match gradient count");
    }

    if (moments.empty()) {
        for (Tensor* param : params) {
            moments.emplace_back(param->getShape());
            velocities.emplace_back(param->getShape());
        }
    }

    ++timestep;
    const float biasCorrection1 = 1.0f - std::pow(beta1, static_cast<float>(timestep));
    const float biasCorrection2 = 1.0f - std::pow(beta2, static_cast<float>(timestep));
    const float decay = 1.0f - learningRate * weightDecay;
    const float oneMinusBeta1 = 1.0f - beta1;
    const float oneMinusBeta2 = 1.0f - beta2;

    for (size_t i = 0; i < params.size(); ++i) {
        float* param = params[i]->raw().data();
        const float* grad = grads[i]->raw().data();
        float* moment = moments[i].raw().data();
        float* velocity = velocities[i].raw().data();
        const size_t count = params[i]->size();

        for (size_t j = 0; j < count; ++j) {
            param[j] *= decay;
            moment[j] = beta1 * moment[j] + oneMinusBeta1 * grad[j];
            velocity[j] = beta2 * velocity[j] + oneMinusBeta2 * grad[j] * grad[j];
            const float correctedMoment = moment[j] / biasCorrection1;
            const float correctedVelocity = velocity[j] / biasCorrection2;
            param[j] -= learningRate * correctedMoment / (std::sqrt(correctedVelocity) + epsilon);
        }
    }
}

void AdamW::setLearningRate(float newLearningRate) {
    learningRate = newLearningRate;
}

void AdamW::restore(size_t step, std::vector<Tensor> savedMoments, std::vector<Tensor> savedVelocities) {
    if (savedMoments.size() != savedVelocities.size()) {
        throw std::invalid_argument("AdamW moments and velocities differ in count");
    }
    for (size_t i = 0; i < savedMoments.size(); ++i) {
        if (savedMoments[i].getShape() != savedVelocities[i].getShape()) {
            throw std::invalid_argument("AdamW moment shape does not match velocity shape");
        }
    }
    timestep = step;
    moments = std::move(savedMoments);
    velocities = std::move(savedVelocities);
}

float AdamW::getLearningRate() const { return learningRate; }
float AdamW::getWeightDecay() const { return weightDecay; }
float AdamW::getBeta1() const { return beta1; }
float AdamW::getBeta2() const { return beta2; }
float AdamW::getEpsilon() const { return epsilon; }
size_t AdamW::getTimestep() const { return timestep; }
std::vector<Tensor> AdamW::getMoments() const { return moments; }
std::vector<Tensor> AdamW::getVelocities() const { return velocities; }
