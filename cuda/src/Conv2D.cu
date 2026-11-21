#include "Layers/include/Conv2D.h"
#include "cuda/include/Conv2D.cuh"
#include "cuda/include/Convolution.cuh"
#include "cuda/include/DeviceMemory.cuh"
#include "cuda/include/TensorOps.cuh"

namespace {

void launchIm2Col(
    const float* input,
    float* col,
    size_t batch,
    size_t channels,
    size_t height,
    size_t width,
    size_t kernel,
    size_t stride,
    size_t padding
) {
    dim3 block(32, 1);
    dim3 grid(static_cast<unsigned>((batch + block.x - 1) / block.x), static_cast<unsigned>(channels));
    cudaIm2Col<<<grid, block>>>(input, col, kernel, stride, padding, batch, channels, height, width);
    mustCuda(cudaGetLastError());
}

void launchGemm(const float* a, const float* b, float* c, size_t rows, size_t cols, size_t inner) {
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((cols + 63) / 64), static_cast<unsigned>((rows + 63) / 64));
    cudaTiledGemm<<<grid, block>>>(a, b, c, rows, cols, inner);
    mustCuda(cudaGetLastError());
}

void launchTranspose(const float* input, float* output, size_t rows, size_t cols) {
    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned>((cols + block.x - 1) / block.x),
        static_cast<unsigned>((rows + block.y - 1) / block.y)
    );
    cudaTranspose<<<grid, block>>>(input, output, rows, cols);
    mustCuda(cudaGetLastError());
}

void launchBias(float* activations, const float* bias, size_t rows, size_t cols) {
    const size_t count = rows * cols;
    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((count + block - 1) / block);
    cudaAddBias<<<grid, block>>>(activations, bias, rows, cols);
    mustCuda(cudaGetLastError());
}

void launchSpatial(const float* spatial, float* nchw, size_t batch, size_t channels, size_t height, size_t width) {
    const size_t count = batch * channels * height * width;
    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((count + block - 1) / block);
    cudaSpatialToNCHW<<<grid, block>>>(spatial, nchw, batch, channels, height, width);
    mustCuda(cudaGetLastError());
}

}  // namespace

void Conv2D::releaseDevice() {
    auto* state = static_cast<ConvDevice*>(device);
    if (state == nullptr) {
        return;
    }
    freeFloats(state->col);
    freeFloats(state->spatial);
    freeFloats(state->nchw);
    freeFloats(state->weights);
    freeFloats(state->weightsT);
    freeFloats(state->bias);
    delete state;
    device = nullptr;
}

float* Conv2D::forwardDevice(const float* input, size_t batch, size_t channels, size_t height, size_t width) {
    if (channels != in_channels) {
        throw std::runtime_error("Conv2D device forward got a different channel count");
    }
    auto* state = static_cast<ConvDevice*>(device);
    if (state == nullptr) {
        state = new ConvDevice();
        device = state;
    }

    const size_t row = channels * kernel_size * kernel_size;
    const size_t outH = (height + 2 * padding - kernel_size) / stride + 1;
    const size_t outW = (width + 2 * padding - kernel_size) / stride + 1;
    const size_t spatial = batch * outH * outW;

    if (!state->weightsOnDevice) {
        state->weights = allocFloats(out_channels * row);
        state->weightsT = allocFloats(out_channels * row);
        state->bias = allocFloats(out_channels);
        mustCuda(cudaMemcpy(state->weights, weights.raw().data(), out_channels * row * sizeof(float), cudaMemcpyHostToDevice));
        mustCuda(cudaMemcpy(state->bias, biases.raw().data(), out_channels * sizeof(float), cudaMemcpyHostToDevice));
        launchTranspose(state->weights, state->weightsT, out_channels, row);
        state->weightsOnDevice = true;
    }

    const bool shapeChanged = state->batch != batch || state->channels != channels
        || state->height != height || state->width != width;
    if (shapeChanged) {
        freeFloats(state->col);
        freeFloats(state->spatial);
        freeFloats(state->nchw);
        state->col = allocFloats(spatial * row);
        state->spatial = allocFloats(spatial * out_channels);
        state->nchw = allocFloats(batch * out_channels * outH * outW);
        state->batch = batch;
        state->channels = channels;
        state->height = height;
        state->width = width;
    }

    launchIm2Col(input, state->col, batch, channels, height, width, kernel_size, stride, padding);
    launchGemm(state->col, state->weightsT, state->spatial, spatial, out_channels, row);
    launchBias(state->spatial, state->bias, spatial, out_channels);
    launchSpatial(state->spatial, state->nchw, batch, out_channels, outH, outW);
    return state->nchw;
}

float* Conv2D::deviceOutput() const {
    auto* state = static_cast<ConvDevice*>(device);
    return state == nullptr ? nullptr : state->nchw;
}
