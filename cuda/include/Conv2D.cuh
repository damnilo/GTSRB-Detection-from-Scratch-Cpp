#pragma once

// Workspace for Conv2D::forwardDevice. Included only from Conv2D.cu.
struct ConvDevice {
    float* col = nullptr;
    float* spatial = nullptr;
    float* nchw = nullptr;
    float* weights = nullptr;
    float* weightsT = nullptr;
    float* bias = nullptr;
    size_t batch = 0;
    size_t channels = 0;
    size_t height = 0;
    size_t width = 0;
    bool weightsOnDevice = false;
};
