#include "cuda/Launch/include/DeviceTrainer.cuh"

#include "Layers/include/BatchNorm2D.h"
#include "Layers/include/Conv2D.h"
#include "Layers/include/Dense.h"
#include "Layers/include/SoftMax.h"
#include "Loss/include/CrossEntropy.h"
#include "cuda/Optimizers/include/AdamWDevice.cuh"
#include "cuda/Optimizers/include/SGDDevice.cuh"
#include "cuda/Utils/include/DeviceMemory.cuh"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

struct DeviceTrainer::State {
    float learningRate = 0.0f;
    float clip = 0.0f;
    std::unique_ptr<SGDDevice> sgd;
    std::unique_ptr<AdamWDevice> adam;
    float* input = nullptr;
    float* grad = nullptr;
    size_t inputCapacity = 0;
    size_t gradCapacity = 0;
};

namespace {

struct TrainSlot {
    float* values = nullptr;
    float* grads = nullptr;
    Tensor* hostGrad = nullptr;
};

struct TrainSlots {
    std::vector<TrainSlot> slots;
    std::vector<Conv2D*> convs;
    std::vector<Dense*> denses;
};

void grow(float*& buffer, size_t& capacity, size_t count) {
    if (count <= capacity) {
        return;
    }
    freeFloats(buffer);
    buffer = allocFloats(count);
    capacity = count;
}

TrainSlots collect_slots(Network& network) {
    TrainSlots found;
    for (size_t index = 0; index < network.numLayers(); ++index) {
        Layer* layer = network.getLayer(index);
        if (auto* conv = dynamic_cast<Conv2D*>(layer)) {
            const std::vector<Tensor*> grads = conv->gradients();
            found.slots.push_back({conv->deviceWeights(), conv->deviceGradWeights(), grads[0]});
            found.slots.push_back({conv->deviceBias(), conv->deviceGradBias(), grads[1]});
            found.convs.push_back(conv);
        } else if (auto* norm = dynamic_cast<BatchNorm2D*>(layer)) {
            const std::vector<Tensor*> grads = norm->gradients();
            found.slots.push_back({norm->deviceGamma(), norm->deviceGradGamma(), grads[0]});
            found.slots.push_back({norm->deviceBeta(), norm->deviceGradBeta(), grads[1]});
        } else if (auto* dense = dynamic_cast<Dense*>(layer)) {
            const std::vector<Tensor*> grads = dense->gradients();
            found.slots.push_back({dense->deviceWeights(), dense->deviceGradWeights(), grads[0]});
            found.slots.push_back({dense->deviceBias(), dense->deviceGradBias(), grads[1]});
            found.denses.push_back(dense);
        }
    }
    return found;
}

void clip_slots(const TrainSlots& slots, float clip) {
    double square = 0.0;
    for (const TrainSlot& slot : slots.slots) {
        for (float value : slot.hostGrad->raw()) {
            square += static_cast<double>(value) * static_cast<double>(value);
        }
    }
    const double total = std::sqrt(square);
    if (total <= static_cast<double>(clip)) {
        return;
    }
    const float scale = static_cast<float>(static_cast<double>(clip) / (total + 1e-6));
    for (const TrainSlot& slot : slots.slots) {
        for (float& value : slot.hostGrad->raw()) {
            value *= scale;
        }
        mustCuda(cudaMemcpy(
            slot.grads,
            slot.hostGrad->raw().data(),
            slot.hostGrad->size() * sizeof(float),
            cudaMemcpyHostToDevice
        ));
    }
}

void step_slots(SGDDevice* sgd, AdamWDevice* adam, const TrainSlots& slots) {
    if (sgd != nullptr) {
        for (const TrainSlot& slot : slots.slots) {
            sgd->update(slot.values, slot.grads, slot.hostGrad->size());
        }
        return;
    }
    std::vector<AdamWBuffer> buffers;
    buffers.reserve(slots.slots.size());
    for (const TrainSlot& slot : slots.slots) {
        buffers.push_back({slot.values, slot.grads, slot.hostGrad->size()});
    }
    adam->update(buffers);
}

void refresh_transposes(const TrainSlots& slots) {
    for (Conv2D* conv : slots.convs) {
        conv->refreshDeviceTranspose();
    }
    for (Dense* dense : slots.denses) {
        dense->refreshDeviceTranspose();
    }
}

void copy_device_parameter(float* device, Tensor* host) {
    mustCuda(cudaMemcpy(host->raw().data(), device, host->size() * sizeof(float), cudaMemcpyDeviceToHost));
}

}  // namespace

DeviceTrainer::DeviceTrainer(const std::string& optimizer, float learningRate, float second, float clip)
    : state(std::make_unique<State>()) {
    if (!(clip > 0.0f)) {
        throw std::invalid_argument("DeviceTrainer clip must be positive");
    }
    state->learningRate = learningRate;
    state->clip = clip;
    if (optimizer == "sgd") {
        state->sgd = std::make_unique<SGDDevice>(learningRate, second);
    } else if (optimizer == "adamw") {
        state->adam = std::make_unique<AdamWDevice>(learningRate, second);
    } else {
        throw std::invalid_argument("DeviceTrainer optimizer must be sgd or adamw");
    }
}

DeviceTrainer::~DeviceTrainer() {
    if (state == nullptr) {
        return;
    }
    freeFloats(state->input);
    freeFloats(state->grad);
}

float DeviceTrainer::step(
    Network& network,
    const float* input,
    size_t batch,
    size_t channels,
    size_t height,
    size_t width,
    const float* targets,
    size_t classes
) {
    if (batch < 1 || channels < 1 || height < 1 || width < 1 || classes < 1) {
        throw std::invalid_argument("DeviceTrainer step got an empty batch");
    }
    const size_t inputCount = batch * channels * height * width;
    const size_t logitCount = batch * classes;
    grow(state->input, state->inputCapacity, inputCount);
    grow(state->grad, state->gradCapacity, logitCount);

    mustCuda(cudaMemcpy(state->input, input, inputCount * sizeof(float), cudaMemcpyHostToDevice));
    float* logits = network.forwardDevice(state->input, batch, channels, height, width);
    Tensor logitTensor({batch, classes});
    mustCuda(cudaMemcpy(logitTensor.raw().data(), logits, logitCount * sizeof(float), cudaMemcpyDeviceToHost));

    Tensor targetTensor({batch, classes});
    std::memcpy(targetTensor.raw().data(), targets, logitCount * sizeof(float));
    SoftMax softmax;
    CrossEntropy loss;
    const float batchLoss = loss.forward(softmax.forward(logitTensor), targetTensor);
    const Tensor gradLogits = softmax.backward(loss.backward());
    mustCuda(cudaMemcpy(state->grad, gradLogits.raw().data(), logitCount * sizeof(float), cudaMemcpyHostToDevice));
    network.backwardDevice(state->grad, batch);
    mustCuda(cudaDeviceSynchronize());

    const TrainSlots slots = collect_slots(network);
    clip_slots(slots, state->clip);
    step_slots(state->sgd.get(), state->adam.get(), slots);
    refresh_transposes(slots);
    mustCuda(cudaDeviceSynchronize());
    return batchLoss;
}

void DeviceTrainer::setLearningRate(float learningRate) {
    state->learningRate = learningRate;
    if (state->sgd) {
        state->sgd->setLearningRate(learningRate);
    }
    if (state->adam) {
        state->adam->setLearningRate(learningRate);
    }
}

float DeviceTrainer::getLearningRate() const {
    return state->learningRate;
}

void DeviceTrainer::copyParametersToHost(Network& network) {
    mustCuda(cudaDeviceSynchronize());
    for (size_t index = 0; index < network.numLayers(); ++index) {
        Layer* layer = network.getLayer(index);
        if (auto* conv = dynamic_cast<Conv2D*>(layer)) {
            const std::vector<Tensor*> params = conv->parameters();
            copy_device_parameter(conv->deviceWeights(), params[0]);
            copy_device_parameter(conv->deviceBias(), params[1]);
        } else if (auto* norm = dynamic_cast<BatchNorm2D*>(layer)) {
            const std::vector<Tensor*> params = norm->parameters();
            copy_device_parameter(norm->deviceGamma(), params[0]);
            copy_device_parameter(norm->deviceBeta(), params[1]);
        } else if (auto* dense = dynamic_cast<Dense*>(layer)) {
            const std::vector<Tensor*> params = dense->parameters();
            copy_device_parameter(dense->deviceWeights(), params[0]);
            copy_device_parameter(dense->deviceBias(), params[1]);
        }
    }
}
