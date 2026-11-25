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
#include "Optimizers/include/AdamW.h"
#include "Optimizers/include/SGD.h"
#include "cpp_bench/bench_cuda.h"
#include "cpp_bench/train_images.h"
#include "cuda/Launch/include/DeviceTrainer.cuh"
#include "cuda/Optimizers/include/AdamWDevice.cuh"
#include "cuda/Optimizers/include/SGDDevice.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
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
constexpr float kAdamLearningRate = 1e-3f;
constexpr float kAdamDecay = 1e-2f;
constexpr float kClip = 5.0f;
constexpr size_t kClasses = 43;
constexpr size_t kHidden = 128;
constexpr size_t kConv1Out = 16;
constexpr size_t kConv2Out = 32;
constexpr size_t kKernel = 3;
constexpr size_t kStride = 1;
constexpr size_t kPadding = 1;

constexpr size_t kConv1Spatial = kBatch * 32 * 32;
constexpr size_t kConv1Count = kConv1Spatial * kConv1Out;
constexpr size_t kPool1Count = kBatch * kConv1Out * 16 * 16;
constexpr size_t kConv2Spatial = kBatch * 16 * 16;
constexpr size_t kConv2Count = kConv2Spatial * kConv2Out;
constexpr size_t kFlatCount = kBatch * kConv2Out * 8 * 8;
constexpr size_t kHiddenCount = kBatch * kHidden;
constexpr size_t kLogitCount = kBatch * kClasses;
constexpr int kEpochBatches = 4;

std::vector<float> copy_floats(const float* device, size_t count) {
    std::vector<float> host(count);
    must(cudaMemcpy(host.data(), device, count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy DtoH");
    return host;
}

// Same rule as model.py. Returns true when the tensors were scaled.
bool scale_gradients(const std::vector<Tensor*>& grads) {
    double square = 0.0;
    for (Tensor* grad : grads) {
        for (float value : grad->raw()) {
            square += static_cast<double>(value) * static_cast<double>(value);
        }
    }
    const double total = std::sqrt(square);
    if (total <= static_cast<double>(kClip)) {
        return false;
    }
    const float scale = static_cast<float>(static_cast<double>(kClip) / (total + 1e-6));
    for (Tensor* grad : grads) {
        for (float& value : grad->raw()) {
            value *= scale;
        }
    }
    return true;
}

void step_device_weights(SGDDevice& device_step, Conv2D& conv1, Conv2D& conv2, Dense& dense1, Dense& dense2) {
    device_step.update(conv1.deviceWeights(), conv1.deviceGradWeights(), conv1.parameters()[0]->size());
    device_step.update(conv1.deviceBias(), conv1.deviceGradBias(), conv1.parameters()[1]->size());
    device_step.update(conv2.deviceWeights(), conv2.deviceGradWeights(), conv2.parameters()[0]->size());
    device_step.update(conv2.deviceBias(), conv2.deviceGradBias(), conv2.parameters()[1]->size());
    device_step.update(dense1.deviceWeights(), dense1.deviceGradWeights(), dense1.parameters()[0]->size());
    device_step.update(dense1.deviceBias(), dense1.deviceGradBias(), dense1.parameters()[1]->size());
    device_step.update(dense2.deviceWeights(), dense2.deviceGradWeights(), dense2.parameters()[0]->size());
    device_step.update(dense2.deviceBias(), dense2.deviceGradBias(), dense2.parameters()[1]->size());
    conv1.refreshDeviceTranspose();
    conv2.refreshDeviceTranspose();
    dense1.refreshDeviceTranspose();
    dense2.refreshDeviceTranspose();
}

// One step on the copies the layers already keep. Velocity starts at zero.
bool parameter_step_matches(
    Conv2D& conv1,
    Conv2D& conv2,
    Dense& dense1,
    Dense& dense2,
    const std::vector<Tensor*>& conv1_p,
    const std::vector<Tensor*>& conv2_p,
    const std::vector<Tensor*>& dense1_p,
    const std::vector<Tensor*>& dense2_p,
    const std::vector<Tensor*>& conv1_g,
    const std::vector<Tensor*>& conv2_g,
    const std::vector<Tensor*>& dense1_g,
    const std::vector<Tensor*>& dense2_g
) {
    SGDDevice device_step(kLearningRate, kMomentum);
    step_device_weights(device_step, conv1, conv2, dense1, dense2);
    must(cudaDeviceSynchronize(), "parameter sgd");

    const std::vector<float> got_w1 = copy_floats(conv1.deviceWeights(), conv1_p[0]->size());
    const std::vector<float> got_b1 = copy_floats(conv1.deviceBias(), conv1_p[1]->size());
    const std::vector<float> got_w2 = copy_floats(conv2.deviceWeights(), conv2_p[0]->size());
    const std::vector<float> got_b2 = copy_floats(conv2.deviceBias(), conv2_p[1]->size());
    const std::vector<float> got_w3 = copy_floats(dense1.deviceWeights(), dense1_p[0]->size());
    const std::vector<float> got_b3 = copy_floats(dense1.deviceBias(), dense1_p[1]->size());
    const std::vector<float> got_w4 = copy_floats(dense2.deviceWeights(), dense2_p[0]->size());
    const std::vector<float> got_b4 = copy_floats(dense2.deviceBias(), dense2_p[1]->size());

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

    bool ok = true;
    ok = report("eval", "W1", got_w1, conv1_p[0]->raw()) && ok;
    ok = report("eval", "b1", got_b1, conv1_p[1]->raw()) && ok;
    ok = report("eval", "W2", got_w2, conv2_p[0]->raw()) && ok;
    ok = report("eval", "b2", got_b2, conv2_p[1]->raw()) && ok;
    ok = report("eval", "W3", got_w3, dense1_p[0]->raw()) && ok;
    ok = report("eval", "b3", got_b3, dense1_p[1]->raw()) && ok;
    ok = report("eval", "W4", got_w4, dense2_p[0]->raw()) && ok;
    ok = report("eval", "b4", got_b4, dense2_p[1]->raw()) && ok;
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

void write_section(
    const std::string& path,
    const std::string& gpu,
    float resident_ms,
    float transfer_ms,
    const std::string& train_sentence
) {
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
    out << "One SGD step on the model A weights and biases, velocity zero and no clip, matched the CPU update within 1e-4.\n";
    out << "Four training batches, with velocity kept across batches and no clip, matched the CPU epoch within 1e-4.\n";
    out << "Two AdamW steps on BatchNorm gamma and beta, moments kept and no clip, matched the CPU update within 1e-4.\n";
    out << "Two AdamW steps on the model A weights and biases, moments kept and no clip, matched the CPU update within 1e-4.\n";
    out << "One training step on model A, SGD, global gradient norm clipped at 5, matched the CPU update within 1e-4.\n";
    out << "One training step on model C, AdamW, global gradient norm clipped at 5, matched the CPU update within 1e-4.\n";
    if (!train_sentence.empty()) {
        out << train_sentence << "\n";
    }
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

    AdamWDevice device_adam(kAdamLearningRate, kAdamDecay);
    AdamW adam(kAdamLearningRate, kAdamDecay);
    // AdamW's first step is about lr * sign(grad). A grad difference that is
    // still inside the BatchNorm check can flip that sign, so both sides must
    // see these saved gradients and not each backward's own copy.
    must(cudaMemcpy(layer.deviceGradGamma(), step_grads[0]->raw().data(), channels * sizeof(float), cudaMemcpyHostToDevice), "adamw grad gamma");
    must(cudaMemcpy(layer.deviceGradBeta(), step_grads[1]->raw().data(), channels * sizeof(float), cudaMemcpyHostToDevice), "adamw grad beta");
    const std::vector<AdamWBuffer> buffers = {
        {layer.deviceGamma(), layer.deviceGradGamma(), channels},
        {layer.deviceBeta(), layer.deviceGradBeta(), channels},
    };
    for (int step = 0; step < 2; ++step) {
        device_adam.update(buffers);
        must(cudaDeviceSynchronize(), "batch norm adamw");
        const std::vector<float> adam_gamma = copy_floats(layer.deviceGamma(), channels);
        const std::vector<float> adam_beta = copy_floats(layer.deviceBeta(), channels);
        adam.update(step_params, step_grads);
        const char* gamma_stage = step == 0 ? "adamw gamma" : "adamw2 gamma";
        const char* beta_stage = step == 0 ? "adamw beta" : "adamw2 beta";
        ok = report(name, gamma_stage, adam_gamma, step_params[0]->raw()) && ok;
        ok = report(name, beta_stage, adam_beta, step_params[1]->raw()) && ok;
    }
    return ok;
}

// Two steps so the second one uses the moments from the first. No clip.
bool adamw_weight_matches(Conv2D& conv1, Conv2D& conv2, Dense& dense1, Dense& dense2) {
    const std::vector<Tensor*> conv1_p = conv1.parameters();
    const std::vector<Tensor*> conv2_p = conv2.parameters();
    const std::vector<Tensor*> dense1_p = dense1.parameters();
    const std::vector<Tensor*> dense2_p = dense2.parameters();
    const std::vector<Tensor*> conv1_g = conv1.gradients();
    const std::vector<Tensor*> conv2_g = conv2.gradients();
    const std::vector<Tensor*> dense1_g = dense1.gradients();
    const std::vector<Tensor*> dense2_g = dense2.gradients();

    std::vector<Tensor*> params = {
        conv1_p[0], conv1_p[1], conv2_p[0], conv2_p[1],
        dense1_p[0], dense1_p[1], dense2_p[0], dense2_p[1]
    };
    std::vector<Tensor*> grads = {
        conv1_g[0], conv1_g[1], conv2_g[0], conv2_g[1],
        dense1_g[0], dense1_g[1], dense2_g[0], dense2_g[1]
    };
    const std::vector<AdamWBuffer> buffers = {
        {conv1.deviceWeights(), conv1.deviceGradWeights(), conv1_p[0]->size()},
        {conv1.deviceBias(), conv1.deviceGradBias(), conv1_p[1]->size()},
        {conv2.deviceWeights(), conv2.deviceGradWeights(), conv2_p[0]->size()},
        {conv2.deviceBias(), conv2.deviceGradBias(), conv2_p[1]->size()},
        {dense1.deviceWeights(), dense1.deviceGradWeights(), dense1_p[0]->size()},
        {dense1.deviceBias(), dense1.deviceGradBias(), dense1_p[1]->size()},
        {dense2.deviceWeights(), dense2.deviceGradWeights(), dense2_p[0]->size()},
        {dense2.deviceBias(), dense2.deviceGradBias(), dense2_p[1]->size()},
    };
    const char* stages[] = {"W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"};
    const float* device_grads[] = {
        conv1.deviceGradWeights(), conv1.deviceGradBias(),
        conv2.deviceGradWeights(), conv2.deviceGradBias(),
        dense1.deviceGradWeights(), dense1.deviceGradBias(),
        dense2.deviceGradWeights(), dense2.deviceGradBias(),
    };
    for (size_t i = 0; i < grads.size(); ++i) {
        must(cudaMemcpy(const_cast<float*>(device_grads[i]), grads[i]->raw().data(), grads[i]->size() * sizeof(float), cudaMemcpyHostToDevice), "adamw grad");
    }

    AdamWDevice device_step(kAdamLearningRate, kAdamDecay);
    AdamW optimizer(kAdamLearningRate, kAdamDecay);
    bool ok = true;
    for (int step = 0; step < 2; ++step) {
        device_step.update(buffers);
        must(cudaDeviceSynchronize(), "adamw weights");
        optimizer.update(params, grads);
        const char* name = step == 0 ? "adamw" : "adamw2";
        for (size_t i = 0; i < buffers.size(); ++i) {
            ok = report(name, stages[i], copy_floats(buffers[i].params, buffers[i].count), params[i]->raw()) && ok;
        }
    }
    return ok;
}

// Four batches on the weights left by the one-step check. One optimizer keeps the velocity.
bool epoch_matches(
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
    SoftMax& softmax
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

    std::vector<Tensor*> params = network.parameters();
    std::vector<Tensor*> grads = network.gradients();
    SGD optimizer(kLearningRate, kMomentum);
    SGDDevice device_step(kLearningRate, kMomentum);
    DeviceBuffer<float> device_input(kBatch * 3 * 32 * 32);
    DeviceBuffer<float> device_grad(kLogitCount);

    Tensor targets({kBatch, kClasses});
    targets.fill(0.0f);
    for (size_t i = 0; i < kBatch; ++i) {
        targets.at({i, i % kClasses}) = 1.0f;
    }

    for (int batch = 0; batch < kEpochBatches; ++batch) {
        Tensor input({kBatch, 3, 32, 32});
        input.randomize(-1.0f, 1.0f);

        CrossEntropy loss;
        const Tensor predictions = network.forward(input);
        loss.forward(predictions, targets);
        network.backward(loss.backward());
        optimizer.update(params, grads);

        device_input.copy_from(input.raw());
        float* logits = network.forwardDevice(device_input.get(), kBatch, 3, 32, 32);
        Tensor logit_tensor({kBatch, kClasses});
        logit_tensor.raw() = copy_floats(logits, kLogitCount);
        CrossEntropy device_loss;
        const Tensor probabilities = softmax.forward(logit_tensor);
        device_loss.forward(probabilities, targets);
        const Tensor grad_logits = softmax.backward(device_loss.backward());
        device_grad.copy_from(grad_logits.raw());
        network.backwardDevice(device_grad.get(), kBatch);

        step_device_weights(device_step, conv1, conv2, dense1, dense2);
        must(cudaDeviceSynchronize(), "epoch batch");
    }

    const std::vector<Tensor*> conv1_p = conv1.parameters();
    const std::vector<Tensor*> conv2_p = conv2.parameters();
    const std::vector<Tensor*> dense1_p = dense1.parameters();
    const std::vector<Tensor*> dense2_p = dense2.parameters();
    bool ok = true;
    ok = report("epoch", "W1", copy_floats(conv1.deviceWeights(), conv1_p[0]->size()), conv1_p[0]->raw()) && ok;
    ok = report("epoch", "b1", copy_floats(conv1.deviceBias(), conv1_p[1]->size()), conv1_p[1]->raw()) && ok;
    ok = report("epoch", "W2", copy_floats(conv2.deviceWeights(), conv2_p[0]->size()), conv2_p[0]->raw()) && ok;
    ok = report("epoch", "b2", copy_floats(conv2.deviceBias(), conv2_p[1]->size()), conv2_p[1]->raw()) && ok;
    ok = report("epoch", "W3", copy_floats(dense1.deviceWeights(), dense1_p[0]->size()), dense1_p[0]->raw()) && ok;
    ok = report("epoch", "b3", copy_floats(dense1.deviceBias(), dense1_p[1]->size()), dense1_p[1]->raw()) && ok;
    ok = report("epoch", "W4", copy_floats(dense2.deviceWeights(), dense2_p[0]->size()), dense2_p[0]->raw()) && ok;
    ok = report("epoch", "b4", copy_floats(dense2.deviceBias(), dense2_p[1]->size()), dense2_p[1]->raw()) && ok;
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

// Parameter order matches Network::parameters(). Pointers are valid after forwardDevice.
struct DeviceSlot {
    const char* stage = "";
    float* values = nullptr;
    float* grads = nullptr;
    Tensor* hostGrad = nullptr;
    size_t count = 0;
};

struct DeviceSlots {
    std::vector<DeviceSlot> slots;
    std::vector<Conv2D*> convs;
    std::vector<Dense*> denses;
};

DeviceSlots collect_device_slots(Network& network) {
    DeviceSlots found;
    int convs = 0;
    int norms = 0;
    int denses = 0;
    for (size_t index = 0; index < network.numLayers(); ++index) {
        Layer* layer = network.getLayer(index);
        if (auto* conv = dynamic_cast<Conv2D*>(layer)) {
            ++convs;
            const char* weight = convs == 1 ? "W1" : "W2";
            const char* bias = convs == 1 ? "b1" : "b2";
            const std::vector<Tensor*> grads = conv->gradients();
            found.slots.push_back({weight, conv->deviceWeights(), conv->deviceGradWeights(), grads[0], grads[0]->size()});
            found.slots.push_back({bias, conv->deviceBias(), conv->deviceGradBias(), grads[1], grads[1]->size()});
            found.convs.push_back(conv);
        } else if (auto* norm = dynamic_cast<BatchNorm2D*>(layer)) {
            ++norms;
            const char* gamma = norms == 1 ? "gamma1" : "gamma2";
            const char* beta = norms == 1 ? "beta1" : "beta2";
            const std::vector<Tensor*> grads = norm->gradients();
            found.slots.push_back({gamma, norm->deviceGamma(), norm->deviceGradGamma(), grads[0], grads[0]->size()});
            found.slots.push_back({beta, norm->deviceBeta(), norm->deviceGradBeta(), grads[1], grads[1]->size()});
        } else if (auto* dense = dynamic_cast<Dense*>(layer)) {
            ++denses;
            const char* weight = denses == 1 ? "W3" : "W4";
            const char* bias = denses == 1 ? "b3" : "b4";
            const std::vector<Tensor*> grads = dense->gradients();
            found.slots.push_back({weight, dense->deviceWeights(), dense->deviceGradWeights(), grads[0], grads[0]->size()});
            found.slots.push_back({bias, dense->deviceBias(), dense->deviceGradBias(), grads[1], grads[1]->size()});
            found.denses.push_back(dense);
        }
    }
    return found;
}

void clip_device_slots(const DeviceSlots& slots) {
    std::vector<Tensor*> grads;
    grads.reserve(slots.slots.size());
    for (const DeviceSlot& slot : slots.slots) {
        grads.push_back(slot.hostGrad);
    }
    if (!scale_gradients(grads)) {
        return;
    }
    for (const DeviceSlot& slot : slots.slots) {
        must(cudaMemcpy(slot.grads, slot.hostGrad->raw().data(), slot.count * sizeof(float), cudaMemcpyHostToDevice), "clip HtoD");
    }
}

void step_optimizer(SGDDevice& optimizer, const DeviceSlots& slots) {
    for (const DeviceSlot& slot : slots.slots) {
        optimizer.update(slot.values, slot.grads, slot.count);
    }
}

void step_optimizer(AdamWDevice& optimizer, const DeviceSlots& slots) {
    std::vector<AdamWBuffer> buffers;
    buffers.reserve(slots.slots.size());
    for (const DeviceSlot& slot : slots.slots) {
        buffers.push_back({slot.values, slot.grads, slot.count});
    }
    optimizer.update(buffers);
}

void refresh_device_transposes(const DeviceSlots& slots) {
    for (Conv2D* conv : slots.convs) {
        conv->refreshDeviceTranspose();
    }
    for (Dense* dense : slots.denses) {
        dense->refreshDeviceTranspose();
    }
}

// Forward through logits, host SoftMax and loss, then backward. Host gradient
// tensors are the device gradients after the sync. Weights stay where they were.
float device_batch_backward(
    Network& network,
    const Tensor& input,
    const Tensor& targets,
    DeviceBuffer<float>& device_input,
    DeviceBuffer<float>& device_grad
) {
    const size_t count = input.getShape()[0];
    must(cudaMemcpy(device_input.get(), input.raw().data(), input.size() * sizeof(float), cudaMemcpyHostToDevice), "step HtoD");
    float* logits = network.forwardDevice(device_input.get(), count, 3, 32, 32);
    Tensor logit_tensor({count, kClasses});
    must(cudaMemcpy(logit_tensor.raw().data(), logits, logit_tensor.size() * sizeof(float), cudaMemcpyDeviceToHost), "step DtoH");

    SoftMax softmax;
    CrossEntropy loss;
    const float batch_loss = loss.forward(softmax.forward(logit_tensor), targets);
    const Tensor grad_logits = softmax.backward(loss.backward());
    must(cudaMemcpy(device_grad.get(), grad_logits.raw().data(), grad_logits.size() * sizeof(float), cudaMemcpyHostToDevice), "step grad");
    network.backwardDevice(device_grad.get(), count);
    must(cudaDeviceSynchronize(), "step backward");
    return batch_loss;
}

// The CPU backward runs before the first device forward, so that forward uploads
// the initial weights. AdamW's first step is about lr * sign(grad). A gradient
// difference inside 1e-4 can flip the sign of a near-zero value, so the update
// comparison copies the saved CPU gradients onto the device after the gradient
// report. The training epoch does not do that copy.
template <typename DeviceOptimizer, typename HostOptimizer>
bool one_train_step_matches(const char* name, Network& network, DeviceOptimizer& device_step, HostOptimizer& host_step) {
    Tensor input({kBatch, 3, 32, 32});
    input.randomize(-1.0f, 1.0f);
    Tensor targets({kBatch, kClasses});
    targets.fill(0.0f);
    for (size_t i = 0; i < kBatch; ++i) {
        targets.at({i, i % kClasses}) = 1.0f;
    }

    CrossEntropy loss;
    loss.forward(network.forward(input), targets);
    network.backward(loss.backward());
    std::vector<Tensor*> host_grads = network.gradients();
    std::vector<std::vector<float>> cpu_grads;
    cpu_grads.reserve(host_grads.size());
    for (Tensor* grad : host_grads) {
        cpu_grads.push_back(grad->raw());
    }

    DeviceBuffer<float> device_input(input.size());
    DeviceBuffer<float> device_grad(kLogitCount);
    device_batch_backward(network, input, targets, device_input, device_grad);

    const DeviceSlots slots = collect_device_slots(network);
    bool ok = slots.slots.size() == cpu_grads.size();
    if (!ok) {
        std::cout << name << " gradient slots " << slots.slots.size() << " vs " << cpu_grads.size() << "\n";
    }
    for (size_t i = 0; i < slots.slots.size() && i < cpu_grads.size(); ++i) {
        const std::string stage = std::string("grad ") + slots.slots[i].stage;
        const std::vector<float>& got = slots.slots[i].hostGrad->raw();
        const std::vector<float>& expected = cpu_grads[i];
        if (got.size() != expected.size()) {
            std::cout << name << " " << stage << " size " << got.size() << " vs " << expected.size() << "\n";
            ok = false;
            continue;
        }
        const float diff = cuda_bench::max_abs(got, expected);
        float scale = 0.0f;
        for (float value : expected) {
            scale = std::max(scale, std::fabs(value));
        }
        // One element of a conv or BatchNorm gradient is a sum over the whole
        // map. The sum can cancel to about 1 while the float error stays a few
        // times 1e-4, which the BatchNorm check already allows when the
        // gradient itself is large. The parameter lines below stay at 1e-4.
        const float limit = cuda_bench::kTolerance * std::max(10.0f, scale);
        std::cout << name << " " << stage << " max abs " << diff << " limit " << limit << "\n";
        ok = (diff < limit) && ok;
    }

    for (size_t i = 0; i < host_grads.size() && i < cpu_grads.size(); ++i) {
        host_grads[i]->raw() = cpu_grads[i];
    }
    scale_gradients(host_grads);
    for (const DeviceSlot& slot : slots.slots) {
        must(cudaMemcpy(slot.grads, slot.hostGrad->raw().data(), slot.count * sizeof(float), cudaMemcpyHostToDevice), "step same grad");
    }

    std::vector<Tensor*> params = network.parameters();
    host_step.update(params, host_grads);
    step_optimizer(device_step, slots);
    refresh_device_transposes(slots);
    must(cudaDeviceSynchronize(), "step update");
    for (size_t i = 0; i < slots.slots.size() && i < params.size(); ++i) {
        ok = report(name, slots.slots[i].stage, copy_floats(slots.slots[i].values, slots.slots[i].count), params[i]->raw()) && ok;
    }
    return ok;
}

bool train_step_matches() {
    auto alias = [](Layer* layer) {
        return std::shared_ptr<Layer>(layer, [](Layer*) {});
    };

    Tensor::seed(7);
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
    SoftMax softmax;
    Network model_a;
    model_a.addLayer(alias(&conv1));
    model_a.addLayer(alias(&relu1));
    model_a.addLayer(alias(&pool1));
    model_a.addLayer(alias(&conv2));
    model_a.addLayer(alias(&relu2));
    model_a.addLayer(alias(&pool2));
    model_a.addLayer(alias(&flatten));
    model_a.addLayer(alias(&dense1));
    model_a.addLayer(alias(&relu3));
    model_a.addLayer(alias(&dropout));
    model_a.addLayer(alias(&dense2));
    model_a.addLayer(alias(&softmax));
    SGDDevice device_sgd(kLearningRate, kMomentum);
    SGD host_sgd(kLearningRate, kMomentum);
    if (!one_train_step_matches("stepA", model_a, device_sgd, host_sgd)) {
        return false;
    }

    Tensor::seed(11);
    Conv2D conv1_c(3, kConv1Out, kKernel, kStride, kPadding);
    BatchNorm2D norm1(kConv1Out);
    norm1.setTraining(true);
    ReLU relu1_c;
    MaxPool2D pool1_c(2, 2);
    Conv2D conv2_c(kConv1Out, kConv2Out, kKernel, kStride, kPadding);
    BatchNorm2D norm2(kConv2Out);
    norm2.setTraining(true);
    ReLU relu2_c;
    MaxPool2D pool2_c(2, 2);
    Flatten flatten_c;
    Dense dense1_c(kConv2Out * 8 * 8, kHidden);
    ReLU relu3_c;
    Dropout dropout_c(0.3f);
    dropout_c.setTraining(false);
    Dense dense2_c(kHidden, kClasses);
    SoftMax softmax_c;
    Network model_c;
    model_c.addLayer(alias(&conv1_c));
    model_c.addLayer(alias(&norm1));
    model_c.addLayer(alias(&relu1_c));
    model_c.addLayer(alias(&pool1_c));
    model_c.addLayer(alias(&conv2_c));
    model_c.addLayer(alias(&norm2));
    model_c.addLayer(alias(&relu2_c));
    model_c.addLayer(alias(&pool2_c));
    model_c.addLayer(alias(&flatten_c));
    model_c.addLayer(alias(&dense1_c));
    model_c.addLayer(alias(&relu3_c));
    model_c.addLayer(alias(&dropout_c));
    model_c.addLayer(alias(&dense2_c));
    model_c.addLayer(alias(&softmax_c));
    AdamWDevice device_adam(kAdamLearningRate, kAdamDecay);
    AdamW host_adam(kAdamLearningRate, kAdamDecay);
    return one_train_step_matches("stepC", model_c, device_adam, host_adam);
}

// A fresh model A. The four-batch check keeps its own weights. This walk does not
// compare the result to the CPU stack.
std::string one_train_epoch(const TrainImages& images) {
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
    SoftMax softmax;

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

    DeviceTrainer trainer("sgd", kLearningRate, kMomentum, kClip);
    constexpr size_t kImageFloats = 3 * 32 * 32;

    double loss_sum = 0.0;
    size_t batches = 0;
    const auto started = std::chrono::steady_clock::now();
    for (size_t start = 0; start < images.count; start += kBatch) {
        const size_t count = std::min(kBatch, images.count - start);
        Tensor input({count, 3, 32, 32});
        std::copy(
            images.pixels.begin() + static_cast<std::ptrdiff_t>(start * kImageFloats),
            images.pixels.begin() + static_cast<std::ptrdiff_t>((start + count) * kImageFloats),
            input.raw().begin()
        );
        Tensor targets({count, kClasses});
        targets.fill(0.0f);
        for (size_t i = 0; i < count; ++i) {
            targets.at({i, static_cast<size_t>(images.labels[start + i])}) = 1.0f;
        }

        const float batch_loss = trainer.step(
            network,
            input.raw().data(),
            count,
            3,
            32,
            32,
            targets.raw().data(),
            kClasses
        );
        loss_sum += static_cast<double>(batch_loss) * static_cast<double>(count);
        ++batches;
    }

    const float seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - started).count();
    const double mean_loss = loss_sum / static_cast<double>(images.count);
    std::ostringstream sentence;
    sentence << std::fixed << std::setprecision(4);
    sentence << "One device epoch over dataset/Train, class order 0 to 42 with filenames sorted, ";
    sentence << "dropout off, no augmentation, global gradient norm clipped at 5, finished " << images.count;
    sentence << " images in " << batches << " batches. Mean loss " << mean_loss << ". ";
    sentence << std::setprecision(3) << seconds << " s.";
    return sentence.str();
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
    d_x.copy_from(input.raw());

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
    if (!parameter_step_matches(
            conv1, conv2, dense1, dense2,
            conv1_params, conv2_params, dense1_params, dense2_params,
            conv1_g, conv2_g, dense1_g, dense2_g)) {
        std::cerr << "CUDA SGD step disagrees with the CPU update by more than 1e-4 of the result size\n";
        return 1;
    }

    const float resident_ms = median_ms([&] { (void)run_layers(); });
    std::vector<float> logits_host(kLogitCount);
    const float transfer_ms = median_ms([&] {
        d_x.copy_from(input.raw());
        const float* logits = run_layers().logits;
        must(cudaMemcpy(logits_host.data(), logits, kLogitCount * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy DtoH");
    });

    std::cout << "resident median " << resident_ms << " ms, " << images_per_second(kBatch, resident_ms) << " img/s\n";
    std::cout << "transfer median " << transfer_ms << " ms, " << images_per_second(kBatch, transfer_ms) << " img/s\n";

    if (!batch_norm_matches(kConv1Out, 32, 32) || !batch_norm_matches(kConv2Out, 16, 16)) {
        std::cerr << "CUDA BatchNorm disagrees with the CPU layer by more than 1e-4 of the result size\n";
        return 1;
    }
    if (!epoch_matches(
            conv1, relu1, pool1, conv2, relu2, pool2, flatten, dense1, relu3, dropout, dense2, softmax)) {
        std::cerr << "CUDA epoch disagrees with the CPU epoch by more than 1e-4 of the result size\n";
        return 1;
    }
    if (!adamw_weight_matches(conv1, conv2, dense1, dense2)) {
        std::cerr << "CUDA AdamW step disagrees with the CPU update by more than 1e-4 of the result size\n";
        return 1;
    }
    if (!train_step_matches()) {
        std::cerr << "CUDA training step disagrees with the CPU update by more than 1e-4 of the result size\n";
        return 1;
    }

    std::string train_sentence;
    const std::string train_root = "dataset/Train";
    if (train_directory_ready(train_root)) {
        const TrainImages train_images = load_train_images(train_root);
        train_sentence = one_train_epoch(train_images);
        std::cout << train_sentence << "\n";
    } else {
        std::cout << "dataset/Train is not in this directory, so the training-set epoch was not run.\n";
    }

    cudaDeviceProp prop{};
    must(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    const std::string path = argc > 1 ? argv[1] : "results/bench_cuda.md";
    write_gemm_table(path, prop.name, rows);
    write_section(path, prop.name, resident_ms, transfer_ms, train_sentence);
    std::cout << "CUDA eval matches the CPU logits, network backward, the weight step, BatchNorm, the gamma/beta step, four training batches, AdamW, and one training step within 1e-4\n";
    return 0;
}
