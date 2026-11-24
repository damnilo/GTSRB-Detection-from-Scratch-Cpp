#include "cuda/Optimizers/include/SGDDevice.cuh"
#include "cuda/Optimizers/include/SGD.cuh"
#include "cuda/Utils/include/DeviceMemory.cuh"

#include <stdexcept>

SGDDevice::SGDDevice(float learningRate, float momentum)
    : learningRate(learningRate), momentum(momentum) {}

SGDDevice::~SGDDevice() {
    for (auto& entry : velocity) {
        freeFloats(entry.second.velocity);
    }
}

void SGDDevice::update(float* params, const float* grads, size_t count) {
    auto found = velocity.find(params);
    if (found == velocity.end()) {
        float* buffer = allocFloats(count);
        mustCuda(cudaMemset(buffer, 0, count * sizeof(float)));
        found = velocity.emplace(params, Slot{buffer, count}).first;
    } else if (found->second.count != count) {
        throw std::runtime_error("SGD device step got a different parameter size");
    }

    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((count + block - 1) / block);
    cudaSGDStep<<<grid, block>>>(
        params,
        grads,
        found->second.velocity,
        learningRate,
        momentum,
        1e-4f,
        count
    );
    mustCuda(cudaGetLastError());
}
