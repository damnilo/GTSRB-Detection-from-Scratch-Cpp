#include "Layers/include/BatchNorm2D.h"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

size_t nchw(size_t n, size_t c, size_t h, size_t w, size_t channels, size_t height, size_t width) {
    return ((n * channels + c) * height + h) * width + w;
}

}

BatchNorm2D::BatchNorm2D(size_t channels)
    : channels(channels),
      eps(1e-5f),
      momentum(0.1f),
      gamma({channels}),
      beta({channels}),
      runningMean({channels}),
      runningVar({channels}),
      gradGamma({channels}),
      gradBeta({channels}),
      training(true)
{
    gamma.fill(1.0f);
    beta.fill(0.0f);
    runningMean.fill(0.0f);
    runningVar.fill(1.0f);
    gradGamma.fill(0.0f);
    gradBeta.fill(0.0f);
}

BatchNorm2D::~BatchNorm2D() { releaseDevice(); }

#ifndef GTSRB_CUDA
void BatchNorm2D::releaseDevice() {}

float* BatchNorm2D::forwardDevice(const float* input, size_t batch, size_t channels, size_t height, size_t width) {
    (void)input;
    (void)batch;
    (void)channels;
    (void)height;
    (void)width;
    throw std::runtime_error("CUDA was not built");
}

float* BatchNorm2D::backwardDevice(const float* gradOutput, size_t batch) {
    (void)gradOutput;
    (void)batch;
    throw std::runtime_error("CUDA was not built");
}
#endif

// y = gamma * (x - mean) / sqrt(var + eps) + beta.
// The forward variance divides by N*H*W. The running variance stores the
// unbiased estimate, divided by N*H*W - 1, which is what nn.BatchNorm2d keeps.
Tensor BatchNorm2D::forward(const Tensor& input) {
    const auto& shape = input.getShape();
    if (shape.size() != 4 || shape[1] != channels) {
        throw std::invalid_argument("BatchNorm2D expects NCHW input with C channels");
    }

    const size_t batch = shape[0];
    const size_t height = shape[2];
    const size_t width = shape[3];
    const size_t count = batch * height * width;
    const float* source = input.raw().data();

    xHatCache = Tensor(shape);
    invStdCache = Tensor({channels});
    Tensor output(shape);
    float* xHat = xHatCache.raw().data();
    float* inverseStd = invStdCache.raw().data();
    float* destination = output.raw().data();

    std::vector<float> mean(channels, 0.0f);
    std::vector<float> variance(channels, 0.0f);

    if (training) {
        for (size_t n = 0; n < batch; ++n) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        mean[c] += source[nchw(n, c, h, w, channels, height, width)];
                    }
                }
            }
        }
        for (size_t c = 0; c < channels; ++c) {
            mean[c] /= static_cast<float>(count);
        }

        for (size_t n = 0; n < batch; ++n) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t h = 0; h < height; ++h) {
                    for (size_t w = 0; w < width; ++w) {
                        float delta = source[nchw(n, c, h, w, channels, height, width)] - mean[c];
                        variance[c] += delta * delta;
                    }
                }
            }
        }
        for (size_t c = 0; c < channels; ++c) {
            variance[c] /= static_cast<float>(count);
            inverseStd[c] = 1.0f / std::sqrt(variance[c] + eps);
        }

        const float keep = 1.0f - momentum;
        for (size_t c = 0; c < channels; ++c) {
            float unbiased = variance[c];
            if (count > 1) {
                unbiased *= static_cast<float>(count) / static_cast<float>(count - 1);
            }
            runningMean[c] = keep * runningMean[c] + momentum * mean[c];
            runningVar[c] = keep * runningVar[c] + momentum * unbiased;
        }
    } else {
        for (size_t c = 0; c < channels; ++c) {
            inverseStd[c] = 1.0f / std::sqrt(runningVar[c] + eps);
            mean[c] = runningMean[c];
        }
    }

    for (size_t n = 0; n < batch; ++n) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    size_t index = nchw(n, c, h, w, channels, height, width);
                    xHat[index] = (source[index] - mean[c]) * inverseStd[c];
                    destination[index] = gamma[c] * xHat[index] + beta[c];
                }
            }
        }
    }

    return output;
}

// Training backward accounts for the batch mean and variance being functions
// of x. Evaluation backward is the fixed affine map used at inference.
Tensor BatchNorm2D::backward(const Tensor& gradOutput) {
    const auto& shape = gradOutput.getShape();
    if (xHatCache.getShape() != shape) {
        throw std::invalid_argument("BatchNorm2D backward requires a matching forward");
    }

    const size_t batch = shape[0];
    const size_t height = shape[2];
    const size_t width = shape[3];
    const size_t count = batch * height * width;
    const float* gradY = gradOutput.raw().data();
    const float* xHat = xHatCache.raw().data();
    const float* inverseStd = invStdCache.raw().data();

    Tensor gradInput(shape);
    float* gradX = gradInput.raw().data();
    gradGamma.fill(0.0f);
    gradBeta.fill(0.0f);

    std::vector<float> sumDxHat(channels, 0.0f);
    std::vector<float> sumDxHatXHat(channels, 0.0f);

    for (size_t n = 0; n < batch; ++n) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    size_t index = nchw(n, c, h, w, channels, height, width);
                    float dxHat = gradY[index] * gamma[c];
                    gradBeta[c] += gradY[index];
                    gradGamma[c] += gradY[index] * xHat[index];
                    sumDxHat[c] += dxHat;
                    sumDxHatXHat[c] += dxHat * xHat[index];
                }
            }
        }
    }

    const float inverseCount = 1.0f / static_cast<float>(count);
    for (size_t n = 0; n < batch; ++n) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    size_t index = nchw(n, c, h, w, channels, height, width);
                    float dxHat = gradY[index] * gamma[c];
                    if (training) {
                        gradX[index] = inverseStd[c] * inverseCount * (
                            static_cast<float>(count) * dxHat - sumDxHat[c] - xHat[index] * sumDxHatXHat[c]);
                    } else {
                        gradX[index] = dxHat * inverseStd[c];
                    }
                }
            }
        }
    }

    return gradInput;
}

std::vector<Tensor*> BatchNorm2D::parameters() {
    return {&gamma, &beta};
}

std::vector<Tensor*> BatchNorm2D::gradients() {
    return {&gradGamma, &gradBeta};
}

void BatchNorm2D::setTraining(bool isTraining) {
    training = isTraining;
}

bool BatchNorm2D::isTraining() const {
    return training;
}

Tensor* BatchNorm2D::runningMeanBuffer() {
    return &runningMean;
}

Tensor* BatchNorm2D::runningVarBuffer() {
    return &runningVar;
}
