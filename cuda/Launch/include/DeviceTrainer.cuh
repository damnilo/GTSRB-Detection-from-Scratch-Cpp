#pragma once

#include "Layers/include/Network.h"

#include <memory>
#include <string>

// One device optimizer for a whole training run. step() uploads the batch,
// runs forward through the logits, applies host SoftMax and cross-entropy,
// backpropagates, clips the global gradient norm, and updates the weights.
// Velocity or AdamW moments stay inside this object. Host weights are stale
// until copyParametersToHost().
class DeviceTrainer {
public:
    // optimizer is "sgd" or "adamw". second is momentum for SGD and weight
    // decay for AdamW. clip is the global-norm threshold from model.py.
    DeviceTrainer(const std::string& optimizer, float learningRate, float second, float clip);
    ~DeviceTrainer();
    DeviceTrainer(const DeviceTrainer&) = delete;
    DeviceTrainer& operator=(const DeviceTrainer&) = delete;

    float step(
        Network& network,
        const float* input,
        size_t batch,
        size_t channels,
        size_t height,
        size_t width,
        const float* targets,
        size_t classes
    );
    void setLearningRate(float learningRate);
    float getLearningRate() const;
    void copyParametersToHost(Network& network);

private:
    struct State;
    std::unique_ptr<State> state;
};
