#include "Layers/include/Conv2D.h"
#include "Layers/include/Dense.h"
#include "Layers/include/Dropout.h"
#include "Layers/include/Flatten.h"
#include "Layers/include/MaxPool2D.h"
#include "Layers/include/ReLU.h"
#include "Layers/include/SoftMax.h"
#include "Loss/include/CrossEntropy.h"
#include "Optimizers/include/SGD.h"
#include "cpp_bench/bench_cuda.h"
#include "cuda/include/SGD.cuh"
#include "cuda/include/MaxPool2D.cuh"
#include "cuda/include/ReLU.cuh"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

using cuda_bench::must;
using cuda_bench::report;
using cuda_bench::launch_im2col;
using cuda_bench::launch_col2im;
using cuda_bench::launch_transpose;
using cuda_bench::launch_gemm;
using cuda_bench::median_ms;
using cuda_bench::images_per_second;
using cuda_bench::DeviceBuffer;

constexpr size_t kBatch = 64;
constexpr float kLearningRate = 0.015f;
constexpr float kMomentum = 0.95f;
constexpr float kDecay = 1e-4f;
constexpr size_t kClasses = 43;
constexpr size_t kHidden = 128;
constexpr size_t kConv1Out = 16;
constexpr size_t kConv2Out = 32;
constexpr size_t kKernel = 3;
constexpr size_t kStride = 1;
constexpr size_t kPadding = 1;

constexpr size_t kConv1Spatial = kBatch * 32 * 32;
constexpr size_t kConv1K = 3 * kKernel * kKernel;
constexpr size_t kConv1Count = kConv1Spatial * kConv1Out;
constexpr size_t kPool1Count = kBatch * kConv1Out * 16 * 16;
constexpr size_t kConv2Spatial = kBatch * 16 * 16;
constexpr size_t kConv2K = kConv1Out * kKernel * kKernel;
constexpr size_t kConv2Count = kConv2Spatial * kConv2Out;
constexpr size_t kFlatCount = kBatch * kConv2Out * 8 * 8;
constexpr size_t kHiddenCount = kBatch * kHidden;
constexpr size_t kLogitCount = kBatch * kClasses;

struct GpuEval {
    float* x;
    float* col1;
    float* y1s;
    float* y1;
    float* y1a;
    int* mask1;
    float* pool1;
    int* poolIdx1;
    float* col2;
    float* y2s;
    float* y2;
    float* y2a;
    int* mask2;
    float* pool2;
    int* poolIdx2;
    float* h;
    float* ha;
    int* mask3;
    float* logits;
    float* w1t;
    float* b1;
    float* w2t;
    float* b2;
    float* w3;
    float* b3;
    float* w4;
    float* b4;
};

void launch_bias(float* activations, const float* bias, size_t rows, size_t cols) {
    const unsigned block = 256;
    const size_t count = rows * cols;
    const unsigned grid = static_cast<unsigned>((count + block - 1) / block);
    cudaAddBias<<<grid, block>>>(activations, bias, rows, cols);
    must(cudaGetLastError(), "cudaAddBias");
}

void launch_spatial(const float* spatial, float* nchw, size_t batch, size_t channels, size_t height, size_t width) {
    const unsigned block = 256;
    const size_t count = batch * channels * height * width;
    const unsigned grid = static_cast<unsigned>((count + block - 1) / block);
    cudaSpatialToNCHW<<<grid, block>>>(spatial, nchw, batch, channels, height, width);
    must(cudaGetLastError(), "cudaSpatialToNCHW");
}

void launch_relu(const float* input, float* output, int* mask, size_t count) {
    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((count + block - 1) / block);
    cudaReLUForward<<<grid, block>>>(input, output, mask, count);
    must(cudaGetLastError(), "cudaReLUForward");
}

void launch_pool(
    const float* input,
    float* output,
    int* maxIdx,
    size_t batch,
    size_t channels,
    size_t height,
    size_t width
) {
    dim3 block(32, 1);
    dim3 grid(static_cast<unsigned>((batch + block.x - 1) / block.x), static_cast<unsigned>(channels));
    cudaMaxPoolForward<<<grid, block>>>(
        input, output, maxIdx,
        2, 2,
        batch, channels, height, width
    );
    must(cudaGetLastError(), "cudaMaxPoolForward");
}

// Weights are already transposed. Dropout is eval, so it is skipped. Flatten is
// the pool2 buffer read as (batch, 2048).
void forward_network(const GpuEval& g) {
    launch_im2col(g.x, g.col1, kBatch, 3, 32, 32, kKernel, kStride, kPadding);
    launch_gemm(g.col1, g.w1t, g.y1s, kConv1Spatial, kConv1Out, kConv1K);
    launch_bias(g.y1s, g.b1, kConv1Spatial, kConv1Out);
    launch_spatial(g.y1s, g.y1, kBatch, kConv1Out, 32, 32);
    launch_relu(g.y1, g.y1a, g.mask1, kConv1Count);
    launch_pool(g.y1a, g.pool1, g.poolIdx1, kBatch, kConv1Out, 32, 32);

    launch_im2col(g.pool1, g.col2, kBatch, kConv1Out, 16, 16, kKernel, kStride, kPadding);
    launch_gemm(g.col2, g.w2t, g.y2s, kConv2Spatial, kConv2Out, kConv2K);
    launch_bias(g.y2s, g.b2, kConv2Spatial, kConv2Out);
    launch_spatial(g.y2s, g.y2, kBatch, kConv2Out, 16, 16);
    launch_relu(g.y2, g.y2a, g.mask2, kConv2Count);
    launch_pool(g.y2a, g.pool2, g.poolIdx2, kBatch, kConv2Out, 16, 16);

    launch_gemm(g.pool2, g.w3, g.h, kBatch, kHidden, kFlatCount / kBatch);
    launch_bias(g.h, g.b3, kBatch, kHidden);
    launch_relu(g.h, g.ha, g.mask3, kHiddenCount);

    launch_gemm(g.ha, g.w4, g.logits, kBatch, kClasses, kHidden);
    launch_bias(g.logits, g.b4, kBatch, kClasses);
}

void launch_sum_rows(const float* matrix, float* sum, size_t rows, size_t cols) {
    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((cols + block - 1) / block);
    cudaSumRows<<<grid, block>>>(matrix, sum, rows, cols);
    must(cudaGetLastError(), "cudaSumRows");
}

void launch_to_spatial(const float* nchw, float* spatial, size_t batch, size_t channels, size_t height, size_t width) {
    const unsigned block = 256;
    const size_t count = batch * channels * height * width;
    const unsigned grid = static_cast<unsigned>((count + block - 1) / block);
    cudaNCHWSpatial<<<grid, block>>>(nchw, spatial, batch, channels, height, width);
    must(cudaGetLastError(), "cudaNCHWSpatial");
}

void launch_relu_backward(const float* grad, float* input_grad, int* mask, size_t count) {
    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((count + block - 1) / block);
    cudaReLUBackward<<<grid, block>>>(grad, input_grad, mask, count);
    must(cudaGetLastError(), "cudaReLUBackward");
}

void launch_sgd(float* weight, const float* grad, float* velocity, size_t count) {
    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((count + block - 1) / block);
    cudaSGDStep<<<grid, block>>>(weight, grad, velocity, kLearningRate, kMomentum, kDecay, count);
    must(cudaGetLastError(), "cudaSGDStep");
}

std::vector<float> copy_floats(const float* device, size_t count) {
    std::vector<float> host(count);
    must(cudaMemcpy(host.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy DtoH");
    return host;
}

void launch_pool_backward(const float* grad, float* input_grad, int* max_idx, size_t count) {
    const unsigned block = 256;
    const unsigned grid = static_cast<unsigned>((count + block - 1) / block);
    cudaMaxPoolBackward<<<grid, block>>>(grad, input_grad, max_idx, count);
    must(cudaGetLastError(), "cudaMaxPoolBackward");
}

// Reads the forward caches still on the device: columns, masks, pool indices,
// and the inputs of the two dense layers. Writes parameter gradients and dX.
bool backward_matches(
    const GpuEval& g,
    float* w1,
    float* w2,
    float* w3,
    float* w4,
    const Tensor& d_logits,
    const Tensor& grad_flat,
    const Tensor& grad_pool1,
    const Tensor& grad_x,
    const std::vector<Tensor*>& conv1_g,
    const std::vector<Tensor*>& conv2_g,
    const std::vector<Tensor*>& dense1_g,
    const std::vector<Tensor*>& dense2_g,
    const std::vector<Tensor*>& conv1_p,
    const std::vector<Tensor*>& conv2_p,
    const std::vector<Tensor*>& dense1_p,
    const std::vector<Tensor*>& dense2_p
) {
    DeviceBuffer<float> grad_logits(kLogitCount);
    DeviceBuffer<float> ha_t(kHiddenCount);
    DeviceBuffer<float> w4_t(kHidden * kClasses);
    DeviceBuffer<float> d_w4(kHidden * kClasses);
    DeviceBuffer<float> d_b4(kClasses);
    DeviceBuffer<float> d_ha(kHiddenCount);
    DeviceBuffer<float> d_h(kHiddenCount);

    DeviceBuffer<float> pool_t(kFlatCount);
    DeviceBuffer<float> w3_t(kFlatCount / kBatch * kHidden);
    DeviceBuffer<float> d_w3(kFlatCount / kBatch * kHidden);
    DeviceBuffer<float> d_b3(kHidden);
    DeviceBuffer<float> d_pool2(kFlatCount);

    DeviceBuffer<float> d_y2a(kConv2Count);
    DeviceBuffer<float> d_y2(kConv2Count);
    DeviceBuffer<float> d_y2s(kConv2Count);
    DeviceBuffer<float> d_y2s_t(kConv2Count);
    DeviceBuffer<float> d_w2(kConv2Out * kConv2K);
    DeviceBuffer<float> d_b2(kConv2Out);
    DeviceBuffer<float> d_col2(kConv2Spatial * kConv2K);
    DeviceBuffer<float> d_pool1(kPool1Count);

    DeviceBuffer<float> d_y1a(kConv1Count);
    DeviceBuffer<float> d_y1(kConv1Count);
    DeviceBuffer<float> d_y1s(kConv1Count);
    DeviceBuffer<float> d_y1s_t(kConv1Count);
    DeviceBuffer<float> d_w1(kConv1Out * kConv1K);
    DeviceBuffer<float> d_b1(kConv1Out);
    DeviceBuffer<float> d_col1(kConv1Spatial * kConv1K);
    DeviceBuffer<float> d_x(kBatch * 3 * 32 * 32);

    grad_logits.copy_from(d_logits.raw());

    launch_transpose(g.ha, ha_t.get(), kBatch, kHidden);
    launch_gemm(ha_t.get(), grad_logits.get(), d_w4.get(), kHidden, kClasses, kBatch);
    launch_sum_rows(grad_logits.get(), d_b4.get(), kBatch, kClasses);
    launch_transpose(w4, w4_t.get(), kHidden, kClasses);
    launch_gemm(grad_logits.get(), w4_t.get(), d_ha.get(), kBatch, kHidden, kClasses);
    launch_relu_backward(d_ha.get(), d_h.get(), g.mask3, kHiddenCount);

    launch_transpose(g.pool2, pool_t.get(), kBatch, kFlatCount / kBatch);
    launch_gemm(pool_t.get(), d_h.get(), d_w3.get(), kFlatCount / kBatch, kHidden, kBatch);
    launch_sum_rows(d_h.get(), d_b3.get(), kBatch, kHidden);
    launch_transpose(w3, w3_t.get(), kFlatCount / kBatch, kHidden);
    launch_gemm(d_h.get(), w3_t.get(), d_pool2.get(), kBatch, kFlatCount / kBatch, kHidden);

    d_y2a.zero();
    launch_pool_backward(d_pool2.get(), d_y2a.get(), g.poolIdx2, kFlatCount);
    launch_relu_backward(d_y2a.get(), d_y2.get(), g.mask2, kConv2Count);
    launch_to_spatial(d_y2.get(), d_y2s.get(), kBatch, kConv2Out, 16, 16);
    launch_transpose(d_y2s.get(), d_y2s_t.get(), kConv2Spatial, kConv2Out);
    launch_gemm(d_y2s_t.get(), g.col2, d_w2.get(), kConv2Out, kConv2K, kConv2Spatial);
    launch_sum_rows(d_y2s.get(), d_b2.get(), kConv2Spatial, kConv2Out);
    launch_gemm(d_y2s.get(), w2, d_col2.get(), kConv2Spatial, kConv2K, kConv2Out);
    d_pool1.zero();
    launch_col2im(d_col2.get(), d_pool1.get(), kBatch, kConv1Out, 16, 16, kKernel, kStride, kPadding);

    d_y1a.zero();
    launch_pool_backward(d_pool1.get(), d_y1a.get(), g.poolIdx1, kPool1Count);
    launch_relu_backward(d_y1a.get(), d_y1.get(), g.mask1, kConv1Count);
    launch_to_spatial(d_y1.get(), d_y1s.get(), kBatch, kConv1Out, 32, 32);
    launch_transpose(d_y1s.get(), d_y1s_t.get(), kConv1Spatial, kConv1Out);
    launch_gemm(d_y1s_t.get(), g.col1, d_w1.get(), kConv1Out, kConv1K, kConv1Spatial);
    launch_sum_rows(d_y1s.get(), d_b1.get(), kConv1Spatial, kConv1Out);
    launch_gemm(d_y1s.get(), w1, d_col1.get(), kConv1Spatial, kConv1K, kConv1Out);
    d_x.zero();
    launch_col2im(d_col1.get(), d_x.get(), kBatch, 3, 32, 32, kKernel, kStride, kPadding);
    must(cudaDeviceSynchronize(), "eval backward");

    bool ok = true;
    ok = report("eval", "dW4", d_w4.copy_to_host(), dense2_g[0]->raw()) && ok;
    ok = report("eval", "db4", d_b4.copy_to_host(), dense2_g[1]->raw()) && ok;
    ok = report("eval", "dW3", d_w3.copy_to_host(), dense1_g[0]->raw()) && ok;
    ok = report("eval", "db3", d_b3.copy_to_host(), dense1_g[1]->raw()) && ok;
    ok = report("eval", "dFlat", d_pool2.copy_to_host(), grad_flat.raw()) && ok;
    ok = report("eval", "dW2", d_w2.copy_to_host(), conv2_g[0]->raw()) && ok;
    ok = report("eval", "db2", d_b2.copy_to_host(), conv2_g[1]->raw()) && ok;
    ok = report("eval", "dPool1", d_pool1.copy_to_host(), grad_pool1.raw()) && ok;
    ok = report("eval", "dW1", d_w1.copy_to_host(), conv1_g[0]->raw()) && ok;
    ok = report("eval", "db1", d_b1.copy_to_host(), conv1_g[1]->raw()) && ok;
    ok = report("eval", "dX", d_x.copy_to_host(), grad_x.raw()) && ok;
    if (!ok) {
        return false;
    }

    // Velocity starts at zero, matching a freshly constructed SGD. The device
    // weights are still the originals copied before this update.
    const size_t w1_n = kConv1Out * kConv1K;
    const size_t w2_n = kConv2Out * kConv2K;
    const size_t w3_n = kFlatCount / kBatch * kHidden;
    const size_t w4_n = kHidden * kClasses;
    DeviceBuffer<float> v_w1(w1_n);
    DeviceBuffer<float> v_b1(kConv1Out);
    DeviceBuffer<float> v_w2(w2_n);
    DeviceBuffer<float> v_b2(kConv2Out);
    DeviceBuffer<float> v_w3(w3_n);
    DeviceBuffer<float> v_b3(kHidden);
    DeviceBuffer<float> v_w4(w4_n);
    DeviceBuffer<float> v_b4(kClasses);
    v_w1.zero(); v_b1.zero(); v_w2.zero(); v_b2.zero();
    v_w3.zero(); v_b3.zero(); v_w4.zero(); v_b4.zero();

    launch_sgd(w1, d_w1.get(), v_w1.get(), w1_n);
    launch_sgd(g.b1, d_b1.get(), v_b1.get(), kConv1Out);
    launch_sgd(w2, d_w2.get(), v_w2.get(), w2_n);
    launch_sgd(g.b2, d_b2.get(), v_b2.get(), kConv2Out);
    launch_sgd(w3, d_w3.get(), v_w3.get(), w3_n);
    launch_sgd(g.b3, d_b3.get(), v_b3.get(), kHidden);
    launch_sgd(w4, d_w4.get(), v_w4.get(), w4_n);
    launch_sgd(g.b4, d_b4.get(), v_b4.get(), kClasses);
    must(cudaDeviceSynchronize(), "sgd step");

    std::vector<Tensor*> params = {
        conv1_p[0], conv1_p[1], conv2_p[0], conv2_p[1],
        dense1_p[0], dense1_p[1], dense2_p[0], dense2_p[1]
    };
    std::vector<Tensor*> grads = {
        conv1_g[0], conv1_g[1], conv2_g[0], conv2_g[1],
        dense1_g[0], dense1_g[1], dense2_g[0], dense2_g[1]
    };
    SGD optimizer(kLearningRate, kMomentum);
    optimizer.update(params, grads);

    ok = report("eval", "W1", copy_floats(w1, w1_n), conv1_p[0]->raw()) && ok;
    ok = report("eval", "b1", copy_floats(g.b1, kConv1Out), conv1_p[1]->raw()) && ok;
    ok = report("eval", "W2", copy_floats(w2, w2_n), conv2_p[0]->raw()) && ok;
    ok = report("eval", "b2", copy_floats(g.b2, kConv2Out), conv2_p[1]->raw()) && ok;
    ok = report("eval", "W3", copy_floats(w3, w3_n), dense1_p[0]->raw()) && ok;
    ok = report("eval", "b3", copy_floats(g.b3, kHidden), dense1_p[1]->raw()) && ok;
    ok = report("eval", "W4", copy_floats(w4, w4_n), dense2_p[0]->raw()) && ok;
    ok = report("eval", "b4", copy_floats(g.b4, kClasses), dense2_p[1]->raw()) && ok;
    return ok;
}

void write_section(const std::string& path, const std::string& gpu, float resident_ms, float transfer_ms) {
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::string existing;
    {
        std::ifstream in(path);
        if (in) {
            existing.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
    }
    const std::string marker = "## Model A eval";
    const size_t marker_at = existing.find(marker);
    if (marker_at != std::string::npos) {
        existing.resize(marker_at);
    }
    while (!existing.empty() && (existing.back() == '\n' || existing.back() == '\r')) {
        existing.pop_back();
    }

    std::ofstream out(path);
    if (!out) {
        std::cerr << "Could not write " << path << "\n";
        std::exit(1);
    }
    if (!existing.empty()) {
        out << existing << "\n\n";
    }
    out << marker << "\n\n";
    out << "GPU: " << gpu << ".\n\n";
    out << "Model A, batch 64, no BatchNorm, dropout off. ";
    out << "Logits are the last dense output, before SoftMax. ";
    out << "They matched the CPU stack within 1e-4 of the result size. ";
    out << "Median of 10 runs after one warmup. ";
    out << "Resident keeps the batch and the weights on the device. ";
    out << "Transfer sends the batch once, runs the same kernels, and reads the logits back once.\n\n";
    out << "| path | ms | img/s |\n";
    out << "| --- | ---: | ---: |\n";
    out << std::fixed << std::setprecision(3);
    out << "| kernels on device | " << resident_ms << " | " << images_per_second(kBatch, resident_ms) << " |\n";
    out << "| one batch copy | " << transfer_ms << " | " << images_per_second(kBatch, transfer_ms) << " |\n";
    std::cout << "Wrote " << path << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    Tensor::seed(42);
    Dropout::seed(42);

    Conv2D conv1(3, kConv1Out, kKernel, kStride, kPadding);
    ReLU relu1;
    MaxPool2D pool1(2, 2);
    Conv2D conv2(kConv1Out, kConv2Out, kKernel, kStride, kPadding);
    ReLU relu2;
    MaxPool2D pool2(2, 2);
    Flatten flatten;
    Dense dense1(kConv2Out * 8 * 8, kHidden);
    ReLU relu3;
    Dropout dropout(0.3f);
    dropout.setTraining(false);
    Dense dense2(kHidden, kClasses);

    Tensor input({kBatch, 3, 32, 32});
    input.randomize(-1.0f, 1.0f);

    const Tensor cpu_conv1 = conv1.forward(input);
    const Tensor cpu_relu1 = relu1.forward(cpu_conv1);
    const Tensor cpu_pool1 = pool1.forward(cpu_relu1);
    const Tensor cpu_conv2 = conv2.forward(cpu_pool1);
    const Tensor cpu_relu2 = relu2.forward(cpu_conv2);
    const Tensor cpu_pool2 = pool2.forward(cpu_relu2);
    const Tensor cpu_flat = flatten.forward(cpu_pool2);
    const Tensor cpu_dense1 = dense1.forward(cpu_flat);
    const Tensor cpu_relu3 = relu3.forward(cpu_dense1);
    const Tensor cpu_drop = dropout.forward(cpu_relu3);
    const Tensor cpu_logits = dense2.forward(cpu_drop);

    Tensor targets({kBatch, kClasses});
    targets.fill(0.0f);
    for (size_t i = 0; i < kBatch; ++i) {
        targets.at({i, i % kClasses}) = 1.0f;
    }
    SoftMax softmax;
    CrossEntropy loss;
    const Tensor probabilities = softmax.forward(cpu_logits);
    loss.forward(probabilities, targets);
    const Tensor grad_logits = softmax.backward(loss.backward());

    const Tensor grad_ha = dense2.backward(grad_logits);
    const Tensor grad_h = relu3.backward(dropout.backward(grad_ha));
    const Tensor grad_flat = dense1.backward(grad_h);
    const Tensor grad_pool2 = flatten.backward(grad_flat);
    const Tensor grad_y2 = relu2.backward(pool2.backward(grad_pool2));
    const Tensor grad_pool1 = conv2.backward(grad_y2);
    const Tensor grad_y1 = relu1.backward(pool1.backward(grad_pool1));
    const Tensor grad_x = conv1.backward(grad_y1);
    const std::vector<Tensor*> conv1_g = conv1.gradients();
    const std::vector<Tensor*> conv2_g = conv2.gradients();
    const std::vector<Tensor*> dense1_g = dense1.gradients();
    const std::vector<Tensor*> dense2_g = dense2.gradients();

    const std::vector<Tensor*> conv1_params = conv1.parameters();
    const std::vector<Tensor*> conv2_params = conv2.parameters();
    const std::vector<Tensor*> dense1_params = dense1.parameters();
    const std::vector<Tensor*> dense2_params = dense2.parameters();

    DeviceBuffer<float> d_x(input.size());
    DeviceBuffer<float> d_col1(kConv1Spatial * kConv1K);
    DeviceBuffer<float> d_y1s(kConv1Count);
    DeviceBuffer<float> d_y1(kConv1Count);
    DeviceBuffer<float> d_y1a(kConv1Count);
    DeviceBuffer<int> d_mask1(kConv1Count);
    DeviceBuffer<float> d_pool1(kPool1Count);
    DeviceBuffer<int> d_poolIdx1(kPool1Count);
    DeviceBuffer<float> d_col2(kConv2Spatial * kConv2K);
    DeviceBuffer<float> d_y2s(kConv2Count);
    DeviceBuffer<float> d_y2(kConv2Count);
    DeviceBuffer<float> d_y2a(kConv2Count);
    DeviceBuffer<int> d_mask2(kConv2Count);
    DeviceBuffer<float> d_pool2(kFlatCount);
    DeviceBuffer<int> d_poolIdx2(kFlatCount);
    DeviceBuffer<float> d_h(kHiddenCount);
    DeviceBuffer<float> d_ha(kHiddenCount);
    DeviceBuffer<int> d_mask3(kHiddenCount);
    DeviceBuffer<float> d_logits(kLogitCount);

    DeviceBuffer<float> d_w1(conv1_params[0]->size());
    DeviceBuffer<float> d_b1(conv1_params[1]->size());
    DeviceBuffer<float> d_w1t(conv1_params[0]->size());
    DeviceBuffer<float> d_w2(conv2_params[0]->size());
    DeviceBuffer<float> d_b2(conv2_params[1]->size());
    DeviceBuffer<float> d_w2t(conv2_params[0]->size());
    DeviceBuffer<float> d_w3(dense1_params[0]->size());
    DeviceBuffer<float> d_b3(dense1_params[1]->size());
    DeviceBuffer<float> d_w4(dense2_params[0]->size());
    DeviceBuffer<float> d_b4(dense2_params[1]->size());

    d_w1.copy_from(conv1_params[0]->raw());
    d_b1.copy_from(conv1_params[1]->raw());
    d_w2.copy_from(conv2_params[0]->raw());
    d_b2.copy_from(conv2_params[1]->raw());
    d_w3.copy_from(dense1_params[0]->raw());
    d_b3.copy_from(dense1_params[1]->raw());
    d_w4.copy_from(dense2_params[0]->raw());
    d_b4.copy_from(dense2_params[1]->raw());
    launch_transpose(d_w1.get(), d_w1t.get(), kConv1Out, kConv1K);
    launch_transpose(d_w2.get(), d_w2t.get(), kConv2Out, kConv2K);
    must(cudaDeviceSynchronize(), "weight transpose");

    d_x.copy_from(input.raw());

    const GpuEval gpu{
        d_x.get(),
        d_col1.get(), d_y1s.get(), d_y1.get(), d_y1a.get(), d_mask1.get(),
        d_pool1.get(), d_poolIdx1.get(),
        d_col2.get(), d_y2s.get(), d_y2.get(), d_y2a.get(), d_mask2.get(),
        d_pool2.get(), d_poolIdx2.get(),
        d_h.get(), d_ha.get(), d_mask3.get(), d_logits.get(),
        d_w1t.get(), d_b1.get(), d_w2t.get(), d_b2.get(),
        d_w3.get(), d_b3.get(), d_w4.get(), d_b4.get()
    };
    forward_network(gpu);
    must(cudaDeviceSynchronize(), "eval forward");

    bool ok = true;
    ok = report("eval", "conv1", d_y1.copy_to_host(), cpu_conv1.raw()) && ok;
    ok = report("eval", "relu1", d_y1a.copy_to_host(), cpu_relu1.raw()) && ok;
    ok = report("eval", "pool1", d_pool1.copy_to_host(), cpu_pool1.raw()) && ok;
    ok = report("eval", "conv2", d_y2.copy_to_host(), cpu_conv2.raw()) && ok;
    ok = report("eval", "relu2", d_y2a.copy_to_host(), cpu_relu2.raw()) && ok;
    ok = report("eval", "pool2", d_pool2.copy_to_host(), cpu_pool2.raw()) && ok;
    ok = report("eval", "flatten", d_pool2.copy_to_host(), cpu_flat.raw()) && ok;
    ok = report("eval", "dense1", d_h.copy_to_host(), cpu_dense1.raw()) && ok;
    ok = report("eval", "relu3", d_ha.copy_to_host(), cpu_relu3.raw()) && ok;
    ok = report("eval", "logits", d_logits.copy_to_host(), cpu_logits.raw()) && ok;

    if (!ok) {
        std::cerr << "CUDA eval disagrees with the CPU stack by more than 1e-4 of the result size\n";
        return 1;
    }

    const float resident_ms = median_ms([&] { forward_network(gpu); });
    std::vector<float> logits_host(kLogitCount);
    const float transfer_ms = median_ms([&] {
        d_x.copy_from(input.raw());
        forward_network(gpu);
        must(cudaMemcpy(logits_host.data(), d_logits.get(), kLogitCount * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy DtoH");
    });

    std::cout << "resident median " << resident_ms << " ms, " << images_per_second(kBatch, resident_ms) << " img/s\n";
    std::cout << "transfer median " << transfer_ms << " ms, " << images_per_second(kBatch, transfer_ms) << " img/s\n";

    if (!backward_matches(
            gpu, d_w1.get(), d_w2.get(), d_w3.get(), d_w4.get(),
            grad_logits, grad_flat, grad_pool1, grad_x,
            conv1_g, conv2_g, dense1_g, dense2_g,
            conv1_params, conv2_params, dense1_params, dense2_params)) {
        std::cerr << "CUDA backward or the SGD step disagrees with the CPU stack by more than 1e-4 of the result size\n";
        return 1;
    }

    cudaDeviceProp prop{};
    must(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    const std::string path = argc > 1 ? argv[1] : "results/bench_cuda.md";
    write_section(path, prop.name, resident_ms, transfer_ms);
    std::cout << "CUDA eval matches the CPU logits, gradients, and one SGD step within 1e-4\n";
    return 0;
}
