#include "Layers/include/Network.h"
#include "Layers/include/BatchNorm2D.h"
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
#include "cuda/Optimizers/include/SGDDevice.cuh"
#include "cuda/Optimizers/include/SGD.cuh"
#include "cuda/Layers/include/MaxPool2D.cuh"
#include "cuda/Layers/include/ReLU.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
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
using cuda_bench::kWarmup;
using cuda_bench::kRepeats;
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

void must_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << what << ": cuBLAS status " << static_cast<int>(status) << "\n";
        std::exit(1);
    }
}

// Row-major (M,K) x (K,N) is column-major (N,K) x (K,M), so B is the first operand.
void launch_cublas(
    cublasHandle_t handle,
    const float* a,
    const float* b,
    float* c,
    size_t M,
    size_t N,
    size_t K
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int n = static_cast<int>(N);
    const int m = static_cast<int>(M);
    const int k = static_cast<int>(K);
    must_cublas(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        b, n,
        a, k,
        &beta,
        c, n
    ), "cublasSgemm");
}

void forward_conv(
    const float* input,
    float* col,
    float* weights,
    float* weights_t,
    float* out,
    size_t batch,
    size_t in_channels,
    size_t height,
    size_t width,
    size_t out_channels,
    size_t kernel,
    size_t stride,
    size_t padding,
    size_t spatial,
    size_t row_size
) {
    launch_im2col(input, col, batch, in_channels, height, width, kernel, stride, padding);
    launch_transpose(weights, weights_t, out_channels, row_size);
    launch_gemm(col, weights_t, out, spatial, out_channels, row_size);
    must(cudaDeviceSynchronize(), "forward sync");
}

struct ConvCase {
    const char* name;
    size_t batch;
    size_t in_channels;
    size_t height;
    size_t width;
    size_t out_channels;
};

struct BenchRow {
    std::string shape;
    std::string backend;
    float ms;
    float images_per_s;
};

bool check_conv(const ConvCase& spec, cublasHandle_t handle, std::vector<BenchRow>& rows) {
    const size_t kernel = 3;
    const size_t stride = 1;
    const size_t padding = 1;
    const size_t out_h = (spec.height + 2 * padding - kernel) / stride + 1;
    const size_t out_w = (spec.width + 2 * padding - kernel) / stride + 1;
    const size_t spatial = spec.batch * out_h * out_w;
    const size_t row_size = spec.in_channels * kernel * kernel;

    Tensor input({spec.batch, spec.in_channels, spec.height, spec.width});
    Tensor weights({spec.out_channels, row_size});
    input.randomize(-1.0f, 1.0f);
    weights.randomize(-0.1f, 0.1f);

    Tensor cpu_col = Conv2D::im2col(input, kernel, stride, padding);
    Tensor cpu_out = cpu_col.matmul(weights.transpose());

    Tensor d_out_col({spatial, spec.out_channels});
    d_out_col.randomize(-1.0f, 1.0f);
    Tensor cpu_grad_w = d_out_col.matmulTransposedLeft(cpu_col);
    Tensor cpu_d_col = d_out_col.matmul(weights);
    Tensor cpu_grad_x = Conv2D::col2im(cpu_d_col, input.getShape(), kernel, stride, padding);

    DeviceBuffer<float> d_input(input.size());
    DeviceBuffer<float> d_weights(weights.size());
    DeviceBuffer<float> d_col(cpu_col.size());
    DeviceBuffer<float> d_weights_t(weights.size());
    DeviceBuffer<float> d_out(cpu_out.size());
    DeviceBuffer<float> d_cublas(cpu_out.size());
    DeviceBuffer<float> d_dout(d_out_col.size());
    DeviceBuffer<float> d_dout_t(d_out_col.size());
    DeviceBuffer<float> d_grad_w(cpu_grad_w.size());
    DeviceBuffer<float> d_cublas_grad(cpu_grad_w.size());
    DeviceBuffer<float> d_d_col(cpu_d_col.size());
    DeviceBuffer<float> d_grad_x(cpu_grad_x.size());

    d_input.copy_from(input.raw());
    d_weights.copy_from(weights.raw());
    d_dout.copy_from(d_out_col.raw());

    forward_conv(
        d_input.get(),
        d_col.get(),
        d_weights.get(),
        d_weights_t.get(),
        d_out.get(),
        spec.batch,
        spec.in_channels,
        spec.height,
        spec.width,
        spec.out_channels,
        kernel,
        stride,
        padding,
        spatial,
        row_size
    );

    bool ok = true;
    ok = report(spec.name, "im2col", d_col.copy_to_host(), cpu_col.raw()) && ok;
    ok = report(spec.name, "transpose", d_weights_t.copy_to_host(), weights.transpose().raw()) && ok;
    const std::vector<float> tiled_host = d_out.copy_to_host();
    ok = report(spec.name, "gemm", tiled_host, cpu_out.raw()) && ok;

    launch_cublas(handle, d_col.get(), d_weights_t.get(), d_cublas.get(), spatial, spec.out_channels, row_size);
    must(cudaDeviceSynchronize(), "cublas sync");
    const std::vector<float> cublas_host = d_cublas.copy_to_host();
    ok = report(spec.name, "cublas", cublas_host, cpu_out.raw()) && ok;
    ok = report(spec.name, "gemm vs cublas", tiled_host, cublas_host) && ok;

    launch_transpose(d_dout.get(), d_dout_t.get(), spatial, spec.out_channels);

    launch_gemm(
        d_dout_t.get(),
        d_col.get(),
        d_grad_w.get(),
        spec.out_channels,
        row_size,
        spatial
    );

    must(cudaDeviceSynchronize(), "gradW sync");
    const std::vector<float> grad_w_host = d_grad_w.copy_to_host();
    ok = report(spec.name, "gradW", grad_w_host, cpu_grad_w.raw()) && ok;

    launch_cublas(
        handle,
        d_dout_t.get(),
        d_col.get(),
        d_cublas_grad.get(),
        spec.out_channels,
        row_size,
        spatial
    );
    must(cudaDeviceSynchronize(), "cublas gradW sync");
    ok = report(spec.name, "gradW vs cublas", grad_w_host, d_cublas_grad.copy_to_host()) && ok;

    launch_gemm(
        d_dout.get(),
        d_weights.get(),
        d_d_col.get(),
        spatial,
        row_size,
        spec.out_channels
    );

    d_grad_x.zero();

    launch_col2im(
        d_d_col.get(),
        d_grad_x.get(),
        spec.batch,
        spec.in_channels,
        spec.height,
        spec.width,
        kernel,
        stride,
        padding
    );

    must(cudaDeviceSynchronize(), "col2im sync");

    ok = report(spec.name, "col2im", d_grad_x.copy_to_host(), cpu_grad_x.raw()) && ok;

    forward_conv(
        d_input.get(),
        d_col.get(),
        d_weights.get(),
        d_weights_t.get(),
        d_out.get(),
        spec.batch,
        spec.in_channels,
        spec.height,
        spec.width,
        spec.out_channels,
        kernel,
        stride,
        padding,
        spatial,
        row_size
    );

    std::vector<float> samples;
    samples.reserve(kRepeats);

    cudaEvent_t start, stop;

    must(cudaEventCreate(&start), "cudaEventCreate");
    must(cudaEventCreate(&stop), "cudaEventCreate");

    for (int i = 0; i < kWarmup + kRepeats; ++i) {
        must(cudaEventRecord(start), "cudaEventRecord");

        forward_conv(
            d_input.get(),
            d_col.get(),
            d_weights.get(),
            d_weights_t.get(),
            d_out.get(),
            spec.batch,
            spec.in_channels,
            spec.height,
            spec.width,
            spec.out_channels,
            kernel,
            stride,
            padding,
            spatial,
            row_size
        );

        must(cudaEventRecord(stop), "cudaEventRecord");
        must(cudaEventSynchronize(stop), "cudaEventSynchronize");

        if (i >= kWarmup) {
            float ms = 0.0f;
            must(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
            samples.push_back(ms);
        }
    }

    std::sort(samples.begin(), samples.end());

    const float forward_ms = samples[samples.size() / 2];
    std::cout << spec.name << " forward median " << forward_ms << " ms, "
              << images_per_second(spec.batch, forward_ms) << " img/s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const std::string shape = "(" + std::to_string(spatial) + ", " + std::to_string(row_size)
        + ") x (" + std::to_string(row_size) + ", " + std::to_string(spec.out_channels) + ")";
    const std::vector<float> weights_t = weights.transpose().raw();
    std::vector<float> copied(cpu_out.size());

    const float tiled_ms = median_ms([&] {
        launch_gemm(d_col.get(), d_weights_t.get(), d_out.get(), spatial, spec.out_channels, row_size);
    });
    const float cublas_ms = median_ms([&] {
        launch_cublas(handle, d_col.get(), d_weights_t.get(), d_cublas.get(), spatial, spec.out_channels, row_size);
    });
    std::vector<float> cpu_samples;
    cpu_samples.reserve(kRepeats);
    for (int i = 0; i < kWarmup + kRepeats; ++i) {
        const auto start_cpu = std::chrono::steady_clock::now();
        Tensor ignored = cpu_col.matmul(weights.transpose());
        const auto stop_cpu = std::chrono::steady_clock::now();
        if (i >= kWarmup) {
            cpu_samples.push_back(std::chrono::duration<float, std::milli>(stop_cpu - start_cpu).count());
        }
        (void)ignored;
    }
    std::sort(cpu_samples.begin(), cpu_samples.end());
    const float cpu_ms = cpu_samples[cpu_samples.size() / 2];
    const float copy_ms = median_ms([&] {
        d_col.copy_from(cpu_col.raw());
        d_weights_t.copy_from(weights_t);
        launch_gemm(d_col.get(), d_weights_t.get(), d_out.get(), spatial, spec.out_channels, row_size);
        must(cudaMemcpy(copied.data(), d_out.get(), copied.size() * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy DtoH");
    });

    rows.push_back({shape, "tiled GEMM", tiled_ms, images_per_second(spec.batch, tiled_ms)});
    rows.push_back({shape, "cuBLAS", cublas_ms, images_per_second(spec.batch, cublas_ms)});
    rows.push_back({shape, "CPU matmul", cpu_ms, images_per_second(spec.batch, cpu_ms)});
    rows.push_back({shape, "tiled GEMM + copies", copy_ms, images_per_second(spec.batch, copy_ms)});

    return ok;
}

void write_gemm_table(const std::string& path, const std::string& gpu, const std::vector<BenchRow>& rows) {
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Could not write " << path << "\n";
        std::exit(1);
    }

    out << "# CUDA conv GEMM\n\n";
    out << "GPU: " << gpu << ".\n\n";
    out << "Batch 64, kernel 3, stride 1, padding 1. ";
    out << "Each row is the forward product of the unrolled image and the transposed weights. ";
    out << "Median of 10 runs after one warmup. ";
    out << "Each shape has one row that also sends both inputs to the device and the product back. ";
    out << "Tiled GEMM and cuBLAS matched the CPU product within 1e-4.\n\n";
    out << "| shape | backend | ms | img/s |\n";
    out << "| --- | --- | ---: | ---: |\n";
    out << std::fixed << std::setprecision(3);
    for (const BenchRow& row : rows) {
        out << "| " << row.shape << " | " << row.backend << " | " << row.ms << " | " << row.images_per_s << " |\n";
    }
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
    out << "| one batch copy | " << transfer_ms << " | " << images_per_second(kBatch, transfer_ms) << " |\n\n";
    out << "BatchNorm2D train and eval, batch 64, matched the CPU layer within 1e-4 ";
    out << "for 16 channels at 32x32 and 32 channels at 16x16.\n";
    out << "Network::backwardDevice, from the logit gradient and stopping before SoftMax, matched the CPU stack within 1e-4.\n";
    out << "One SGD step on BatchNorm gamma and beta, velocity zero and no clip, matched the CPU update within 1e-4.\n";
    std::cout << "Wrote " << path << "\n";
}

bool batch_norm_matches(size_t channels, size_t height, size_t width) {
    BatchNorm2D layer(channels);
    const std::vector<Tensor*> params = layer.parameters();
    for (size_t c = 0; c < channels; ++c) {
        params[0]->raw()[c] = 0.5f + 0.01f * static_cast<float>(c);
        params[1]->raw()[c] = -0.25f + 0.02f * static_cast<float>(c);
    }

    Tensor input({kBatch, channels, height, width});
    for (size_t i = 0; i < input.size(); ++i) {
        input.raw()[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.05f;
    }
    const std::vector<float> initial_mean = layer.runningMeanBuffer()->raw();
    const std::vector<float> initial_var = layer.runningVarBuffer()->raw();

    DeviceBuffer<float> device_input(input.size());
    device_input.copy_from(input.raw());
    float* device_y = layer.forwardDevice(device_input.get(), kBatch, channels, height, width);
    must(cudaDeviceSynchronize(), "batch norm forward");

    const std::vector<float> got_y = copy_floats(device_y, input.size());
    const std::vector<float> got_mean = layer.runningMeanBuffer()->raw();
    const std::vector<float> got_var = layer.runningVarBuffer()->raw();
    layer.runningMeanBuffer()->raw() = initial_mean;
    layer.runningVarBuffer()->raw() = initial_var;
    const Tensor cpu_y = layer.forward(input);

    const std::string name = "bn" + std::to_string(channels);
    bool ok = true;
    ok = report(name, "forward", got_y, cpu_y.raw()) && ok;
    ok = report(name, "running mean", got_mean, layer.runningMeanBuffer()->raw()) && ok;
    ok = report(name, "running var", got_var, layer.runningVarBuffer()->raw()) && ok;

    Tensor grad_y(input.getShape());
    for (size_t i = 0; i < grad_y.size(); ++i) {
        grad_y.raw()[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.03f;
    }
    const Tensor cpu_grad = layer.backward(grad_y);
    const std::vector<Tensor*> grads = layer.gradients();
    const std::vector<float> cpu_gamma = grads[0]->raw();
    const std::vector<float> cpu_beta = grads[1]->raw();
    DeviceBuffer<float> device_grad_y(grad_y.size());
    device_grad_y.copy_from(grad_y.raw());
    float* device_grad = layer.backwardDevice(device_grad_y.get(), kBatch);
    must(cudaDeviceSynchronize(), "batch norm backward");
    ok = report(name, "grad input", copy_floats(device_grad, input.size()), cpu_grad.raw()) && ok;
    ok = report(name, "grad gamma", grads[0]->raw(), cpu_gamma) && ok;
    ok = report(name, "grad beta", grads[1]->raw(), cpu_beta) && ok;

    const std::vector<float> trained_mean = layer.runningMeanBuffer()->raw();
    const std::vector<float> trained_var = layer.runningVarBuffer()->raw();
    layer.setTraining(false);
    const Tensor cpu_eval = layer.forward(input);
    float* device_eval = layer.forwardDevice(device_input.get(), kBatch, channels, height, width);
    must(cudaDeviceSynchronize(), "batch norm eval");
    ok = report(name, "eval", copy_floats(device_eval, input.size()), cpu_eval.raw()) && ok;
    ok = report(name, "eval running mean", layer.runningMeanBuffer()->raw(), trained_mean) && ok;
    ok = report(name, "eval running var", layer.runningVarBuffer()->raw(), trained_var) && ok;

    grads[0]->raw() = cpu_gamma;
    grads[1]->raw() = cpu_beta;
    SGDDevice device_step(kLearningRate, kMomentum);
    device_step.update(layer.deviceGamma(), layer.deviceGradGamma(), channels);
    device_step.update(layer.deviceBeta(), layer.deviceGradBeta(), channels);
    must(cudaDeviceSynchronize(), "batch norm sgd");
    const std::vector<float> got_gamma = copy_floats(layer.deviceGamma(), channels);
    const std::vector<float> got_beta = copy_floats(layer.deviceBeta(), channels);
    SGD optimizer(kLearningRate, kMomentum);
    std::vector<Tensor*> step_params = layer.parameters();
    std::vector<Tensor*> step_grads = layer.gradients();
    optimizer.update(step_params, step_grads);
    ok = report(name, "sgd gamma", got_gamma, step_params[0]->raw()) && ok;
    ok = report(name, "sgd beta", got_beta, step_params[1]->raw()) && ok;
    return ok;
}

// The network does not own these layers. Their device caches already come from forwardDevice.
bool network_backward_matches(
    Conv2D& conv1,
    ReLU& relu1,
    MaxPool2D& pool1,
    Conv2D& conv2,
    ReLU& relu2,
    MaxPool2D& pool2,
    Flatten& flatten,
    Dense& dense1,
    ReLU& relu3,
    Dropout& dropout,
    Dense& dense2,
    SoftMax& softmax,
    const Tensor& grad_logits,
    const Tensor& grad_x,
    const std::vector<float>& cpu_w1,
    const std::vector<float>& cpu_b1,
    const std::vector<float>& cpu_w2,
    const std::vector<float>& cpu_b2,
    const std::vector<float>& cpu_w3,
    const std::vector<float>& cpu_b3,
    const std::vector<float>& cpu_w4,
    const std::vector<float>& cpu_b4
) {
    auto alias = [](Layer* layer) {
        return std::shared_ptr<Layer>(layer, [](Layer*) {});
    };
    Network network;
    network.addLayer(alias(&conv1));
    network.addLayer(alias(&relu1));
    network.addLayer(alias(&pool1));
    network.addLayer(alias(&conv2));
    network.addLayer(alias(&relu2));
    network.addLayer(alias(&pool2));
    network.addLayer(alias(&flatten));
    network.addLayer(alias(&dense1));
    network.addLayer(alias(&relu3));
    network.addLayer(alias(&dropout));
    network.addLayer(alias(&dense2));
    network.addLayer(alias(&softmax));

    DeviceBuffer<float> device_grad_logits(grad_logits.size());
    device_grad_logits.copy_from(grad_logits.raw());
    float* d_x = network.backwardDevice(device_grad_logits.get(), kBatch);
    must(cudaDeviceSynchronize(), "network backward");

    bool ok = true;
    ok = report("network", "dX", copy_floats(d_x, kBatch * 3 * 32 * 32), grad_x.raw()) && ok;
    ok = report("network", "dW1", conv1.gradients()[0]->raw(), cpu_w1) && ok;
    ok = report("network", "db1", conv1.gradients()[1]->raw(), cpu_b1) && ok;
    ok = report("network", "dW2", conv2.gradients()[0]->raw(), cpu_w2) && ok;
    ok = report("network", "db2", conv2.gradients()[1]->raw(), cpu_b2) && ok;
    ok = report("network", "dW3", dense1.gradients()[0]->raw(), cpu_w3) && ok;
    ok = report("network", "db3", dense1.gradients()[1]->raw(), cpu_b3) && ok;
    ok = report("network", "dW4", dense2.gradients()[0]->raw(), cpu_w4) && ok;
    ok = report("network", "db4", dense2.gradients()[1]->raw(), cpu_b4) && ok;

    conv1.gradients()[0]->raw() = cpu_w1;
    conv1.gradients()[1]->raw() = cpu_b1;
    conv2.gradients()[0]->raw() = cpu_w2;
    conv2.gradients()[1]->raw() = cpu_b2;
    dense1.gradients()[0]->raw() = cpu_w3;
    dense1.gradients()[1]->raw() = cpu_b3;
    dense2.gradients()[0]->raw() = cpu_w4;
    dense2.gradients()[1]->raw() = cpu_b4;
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    Tensor::seed(42);

    cublasHandle_t handle = nullptr;
    must_cublas(cublasCreate(&handle), "cublasCreate");

    const ConvCase cases[] = {
        {"conv1", 64, 3, 32, 32, 16},
        {"conv2", 64, 16, 16, 16, 32},
    };

    std::vector<BenchRow> rows;
    bool gemm_ok = true;
    for (const ConvCase& spec : cases) {
        gemm_ok = check_conv(spec, handle, rows) && gemm_ok;
    }
    cublasDestroy(handle);

    if (!gemm_ok) {
        std::cerr << "CUDA conv disagrees with the CPU path or cuBLAS by more than 1e-4 of the result size\n";
        return 1;
    }

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

    const Tensor grad_flat = dense1.backward(relu3.backward(dropout.backward(dense2.backward(grad_logits))));
    const Tensor grad_pool1 = conv2.backward(relu2.backward(pool2.backward(flatten.backward(grad_flat))));
    const Tensor grad_x = conv1.backward(relu1.backward(pool1.backward(grad_pool1)));
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
    struct LayerOut {
        float* conv1;
        float* relu1;
        float* pool1;
        float* conv2;
        float* relu2;
        float* pool2;
        float* flat;
        float* dense1;
        float* relu3;
        float* logits;
    };
    auto run_layers = [&] {
        LayerOut out{};
        out.conv1 = conv1.forwardDevice(d_x.get(), kBatch, 3, 32, 32);
        out.relu1 = relu1.forwardDevice(out.conv1, kBatch, kConv1Out, 32, 32);
        out.pool1 = pool1.forwardDevice(out.relu1, kBatch, kConv1Out, 32, 32);
        out.conv2 = conv2.forwardDevice(out.pool1, kBatch, kConv1Out, 16, 16);
        out.relu2 = relu2.forwardDevice(out.conv2, kBatch, kConv2Out, 16, 16);
        out.pool2 = pool2.forwardDevice(out.relu2, kBatch, kConv2Out, 16, 16);
        out.flat = flatten.forwardDevice(out.pool2, kBatch, kConv2Out, 8, 8);
        out.dense1 = dense1.forwardDevice(out.flat, kBatch, kFlatCount / kBatch, 1, 1);
        out.relu3 = relu3.forwardDevice(out.dense1, kBatch, kHidden, 1, 1);
        float* dropped = dropout.forwardDevice(out.relu3, kBatch, kHidden, 1, 1);
        out.logits = dense2.forwardDevice(dropped, kBatch, kHidden, 1, 1);
        return out;
    };
    const LayerOut layers = run_layers();
    must(cudaDeviceSynchronize(), "layer forward");

    bool ok = true;
    ok = report("eval", "conv1", copy_floats(layers.conv1, kConv1Count), cpu_conv1.raw()) && ok;
    ok = report("eval", "relu1", copy_floats(layers.relu1, kConv1Count), cpu_relu1.raw()) && ok;
    ok = report("eval", "pool1", copy_floats(layers.pool1, kPool1Count), cpu_pool1.raw()) && ok;
    ok = report("eval", "conv2", copy_floats(layers.conv2, kConv2Count), cpu_conv2.raw()) && ok;
    ok = report("eval", "relu2", copy_floats(layers.relu2, kConv2Count), cpu_relu2.raw()) && ok;
    ok = report("eval", "pool2", copy_floats(layers.pool2, kFlatCount), cpu_pool2.raw()) && ok;
    ok = report("eval", "flatten", copy_floats(layers.flat, kFlatCount), cpu_flat.raw()) && ok;
    ok = report("eval", "dense1", copy_floats(layers.dense1, kHiddenCount), cpu_dense1.raw()) && ok;
    ok = report("eval", "relu3", copy_floats(layers.relu3, kHiddenCount), cpu_relu3.raw()) && ok;
    ok = report("eval", "logits", copy_floats(layers.logits, kLogitCount), cpu_logits.raw()) && ok;

    if (!ok) {
        std::cerr << "CUDA eval disagrees with the CPU stack by more than 1e-4 of the result size\n";
        return 1;
    }

    std::vector<float> cpu_w1 = conv1_g[0]->raw();
    std::vector<float> cpu_b1 = conv1_g[1]->raw();
    std::vector<float> cpu_w2 = conv2_g[0]->raw();
    std::vector<float> cpu_b2 = conv2_g[1]->raw();
    std::vector<float> cpu_w3 = dense1_g[0]->raw();
    std::vector<float> cpu_b3 = dense1_g[1]->raw();
    std::vector<float> cpu_w4 = dense2_g[0]->raw();
    std::vector<float> cpu_b4 = dense2_g[1]->raw();
    if (!network_backward_matches(
            conv1, relu1, pool1, conv2, relu2, pool2, flatten, dense1, relu3, dropout, dense2, softmax,
            grad_logits, grad_x,
            cpu_w1, cpu_b1, cpu_w2, cpu_b2, cpu_w3, cpu_b3, cpu_w4, cpu_b4)) {
        std::cerr << "CUDA network backward disagrees with the CPU stack by more than 1e-4 of the result size\n";
        return 1;
    }

    const float resident_ms = median_ms([&] { (void)run_layers(); });
    std::vector<float> logits_host(kLogitCount);
    const float transfer_ms = median_ms([&] {
        d_x.copy_from(input.raw());
        const float* logits = run_layers().logits;
        must(cudaMemcpy(logits_host.data(), logits, kLogitCount * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy DtoH");
    });

    // Backward still reads the bench's own buffers. Refresh them after the layer timing.
    forward_network(gpu);
    must(cudaDeviceSynchronize(), "eval forward");

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
    if (!batch_norm_matches(kConv1Out, 32, 32) || !batch_norm_matches(kConv2Out, 16, 16)) {
        std::cerr << "CUDA BatchNorm disagrees with the CPU layer by more than 1e-4 of the result size\n";
        return 1;
    }

    cudaDeviceProp prop{};
    must(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    const std::string path = argc > 1 ? argv[1] : "results/bench_cuda.md";
    write_gemm_table(path, prop.name, rows);
    write_section(path, prop.name, resident_ms, transfer_ms);
    std::cout << "CUDA eval matches the CPU logits, network backward, gradients, one SGD step, BatchNorm, and the gamma/beta step within 1e-4\n";
    return 0;
}
