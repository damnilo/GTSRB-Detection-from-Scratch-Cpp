#include "cuda/Optimizers/include/AdamWDevice.cuh"
#include "cuda/Optimizers/include/AdamW.cuh"
#include "cuda/Utils/include/DeviceMemory.cuh"

#include <stdexcept>

AdamWDevice::AdamWDevice(float learningRate, float weightDecay, float beta1, float beta2, float epsilon)
    : learningRate(learningRate),
      weightDecay(weightDecay),
      beta1(beta1),
      beta2(beta2),
      epsilon(epsilon) {}

AdamWDevice::~AdamWDevice() {
    for (auto& entry : state) {
        freeFloats(entry.second.moment);
        freeFloats(entry.second.velocity);
    }
}

void AdamWDevice::setLearningRate(float newLearningRate) {
    learningRate = newLearningRate;
}

void AdamWDevice::update(const std::vector<AdamWBuffer>& buffers) {
    ++timestep;
    const unsigned block = 256;
    for (const AdamWBuffer& buffer : buffers) {
        auto found = state.find(buffer.params);
        if (found == state.end()) {
            float* moment = allocFloats(buffer.count);
            float* velocity = allocFloats(buffer.count);
            mustCuda(cudaMemset(moment, 0, buffer.count * sizeof(float)));
            mustCuda(cudaMemset(velocity, 0, buffer.count * sizeof(float)));
            found = state.emplace(buffer.params, Slot{moment, velocity, buffer.count}).first;
        } else if (found->second.count != buffer.count) {
            throw std::runtime_error("AdamW device step got a different parameter size");
        }

        const unsigned grid = static_cast<unsigned>((buffer.count + block - 1) / block);
        cudaAdamWStep<<<grid, block>>>(
            buffer.params,
            buffer.grads,
            found->second.moment,
            found->second.velocity,
            learningRate,
            weightDecay,
            beta1,
            beta2,
            epsilon,
            timestep,
            buffer.count
        );
        mustCuda(cudaGetLastError());
    }
}
