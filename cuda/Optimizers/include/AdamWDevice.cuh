#pragma once
#include <cstddef>
#include <unordered_map>
#include <vector>

// One AdamW step on buffers that already live on the device. Moments are keyed
// by the parameter pointer and start at zero. timestep advances once per
// update, then every buffer in that call uses the same bias correction.
struct AdamWBuffer {
    float* params = nullptr;
    const float* grads = nullptr;
    size_t count = 0;
};

class AdamWDevice {
private:
    struct Slot {
        float* moment = nullptr;
        float* velocity = nullptr;
        size_t count = 0;
    };

    float learningRate;
    float weightDecay;
    float beta1;
    float beta2;
    float epsilon;
    size_t timestep = 0;
    std::unordered_map<float*, Slot> state;

public:
    explicit AdamWDevice(float learningRate = 1e-3f, float weightDecay = 1e-2f,
                         float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    ~AdamWDevice();
    AdamWDevice(const AdamWDevice&) = delete;
    AdamWDevice& operator=(const AdamWDevice&) = delete;

    void update(const std::vector<AdamWBuffer>& buffers);
    void setLearningRate(float newLearningRate);
};
