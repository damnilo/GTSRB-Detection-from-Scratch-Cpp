#pragma once
#include <cstddef>
#include <unordered_map>

// One SGD step on buffers that already live on the device. Velocity is keyed by
// the parameter pointer and starts at zero. Decay is 1e-4, the same term
// SGD::update adds on the host. There is no gradient clip.
class SGDDevice {
private:
    struct Slot {
        float* velocity = nullptr;
        size_t count = 0;
    };

    float learningRate;
    float momentum;
    std::unordered_map<float*, Slot> velocity;

public:
    SGDDevice(float learningRate, float momentum);
    ~SGDDevice();
    SGDDevice(const SGDDevice&) = delete;
    SGDDevice& operator=(const SGDDevice&) = delete;

    void update(float* params, const float* grads, size_t count);
};
