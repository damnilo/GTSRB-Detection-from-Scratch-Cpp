#include "Layers/include/Conv2D.h"
#include "Layers/include/Dense.h"
#include "Layers/include/Dropout.h"
#include "Layers/include/Flatten.h"
#include "Layers/include/MaxPool2D.h"
#include "Layers/include/ReLU.h"
#include "cuda/include/Convolution.cuh"
#include "cuda/include/MaxPool2D.cuh"
#include "cuda/include/ReLU.cuh"
#include "cuda/include/TensorOps.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

constexpr float kTolerance = 1e-4f;
constexpr int kWarmup = 1;
constexpr int kRepeats = 10;

constexpr size_t kBatch = 64;
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

void must(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(error) << "\n";
        std::exit(1);
    }
}

template <typename T>
class DeviceBuffer {
private:
    T* ptr = nullptr;
    size_t count = 0;

public:
    explicit DeviceBuffer(size_t count) : count(count) {
        must(cudaMalloc(&ptr, count * sizeof(T)), "cudaMalloc");
    }

    ~DeviceBuffer() { cudaFree(ptr); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr; }
    size_t size() const { return count; }

    void copy_from(const std::vector<T>& host) const {
        must(cudaMemcpy(ptr, host.data(), count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy HtoD");
    }

    std::vector<T> copy_to_host() const {
        std::vector<T> host(count);
        must(cudaMemcpy(host.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy DtoH");
        return host;
    }
};

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

void launch_im2col(
    const float* input,
    float* col,
    size_t batch,
    size_t channels,
    size_t height,
    size_t width
) {
    dim3 block(32, 1);
    dim3 grid(static_cast<unsigned>((batch + block.x - 1) / block.x), static_cast<unsigned>(channels));
    cudaIm2Col<<<grid, block>>>(
        input, col,
        kKernel, kStride, kPadding,
        batch, channels, height, width
    );
    must(cudaGetLastError(), "cudaIm2Col");
}

void launch_transpose(const float* input, float* output, size_t rows, size_t cols) {
    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned>((cols + block.x - 1) / block.x),
        static_cast<unsigned>((rows + block.y - 1) / block.y)
    );
    cudaTranspose<<<grid, block>>>(input, output, rows, cols);
    must(cudaGetLastError(), "cudaTranspose");
}

void launch_gemm(const float* a, const float* b, float* c, size_t M, size_t N, size_t K) {
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((N + 63) / 64), static_cast<unsigned>((M + 63) / 64));
    cudaTiledGemm<<<grid, block>>>(a, b, c, M, N, K);
    must(cudaGetLastError(), "cudaTiledGemm");
}

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
    launch_im2col(g.x, g.col1, kBatch, 3, 32, 32);
    launch_gemm(g.col1, g.w1t, g.y1s, kConv1Spatial, kConv1Out, kConv1K);
    launch_bias(g.y1s, g.b1, kConv1Spatial, kConv1Out);
    launch_spatial(g.y1s, g.y1, kBatch, kConv1Out, 32, 32);
    launch_relu(g.y1, g.y1a, g.mask1, kConv1Count);
    launch_pool(g.y1a, g.pool1, g.poolIdx1, kBatch, kConv1Out, 32, 32);

    launch_im2col(g.pool1, g.col2, kBatch, kConv1Out, 16, 16);
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

float max_abs(const std::vector<float>& got, const std::vector<float>& expected) {
    float diff = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        diff = std::max(diff, std::fabs(got[i] - expected[i]));
    }
    return diff;
}

bool report(const char* stage, const std::vector<float>& got, const std::vector<float>& expected) {
    if (got.size() != expected.size()) {
        std::cout << stage << " size " << got.size() << " vs " << expected.size() << "\n";
        return false;
    }
    const float diff = max_abs(got, expected);
    float scale = 0.0f;
    for (float value : expected) {
        scale = std::max(scale, std::fabs(value));
    }
    const float limit = kTolerance * std::max(1.0f, scale);
    std::cout << stage << " max abs " << diff << " limit " << limit << "\n";
    return diff < limit;
}

template <typename Launch>
float median_ms(Launch&& launch) {
    cudaEvent_t start, stop;
    must(cudaEventCreate(&start), "cudaEventCreate");
    must(cudaEventCreate(&stop), "cudaEventCreate");

    std::vector<float> samples;
    samples.reserve(kRepeats);
    for (int i = 0; i < kWarmup + kRepeats; ++i) {
        must(cudaEventRecord(start), "cudaEventRecord");
        launch();
        must(cudaEventRecord(stop), "cudaEventRecord");
        must(cudaEventSynchronize(stop), "cudaEventSynchronize");
        if (i >= kWarmup) {
            float ms = 0.0f;
            must(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
            samples.push_back(ms);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

float images_per_second(float ms) {
    return static_cast<float>(kBatch) * 1000.0f / ms;
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
    out << "| kernels on device | " << resident_ms << " | " << images_per_second(resident_ms) << " |\n";
    out << "| one batch copy | " << transfer_ms << " | " << images_per_second(transfer_ms) << " |\n";
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
    ok = report("conv1", d_y1.copy_to_host(), cpu_conv1.raw()) && ok;
    ok = report("relu1", d_y1a.copy_to_host(), cpu_relu1.raw()) && ok;
    ok = report("pool1", d_pool1.copy_to_host(), cpu_pool1.raw()) && ok;
    ok = report("conv2", d_y2.copy_to_host(), cpu_conv2.raw()) && ok;
    ok = report("relu2", d_y2a.copy_to_host(), cpu_relu2.raw()) && ok;
    ok = report("pool2", d_pool2.copy_to_host(), cpu_pool2.raw()) && ok;
    ok = report("flatten", d_pool2.copy_to_host(), cpu_flat.raw()) && ok;
    ok = report("dense1", d_h.copy_to_host(), cpu_dense1.raw()) && ok;
    ok = report("relu3", d_ha.copy_to_host(), cpu_relu3.raw()) && ok;
    ok = report("logits", d_logits.copy_to_host(), cpu_logits.raw()) && ok;

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

    std::cout << "resident median " << resident_ms << " ms, " << images_per_second(resident_ms) << " img/s\n";
    std::cout << "transfer median " << transfer_ms << " ms, " << images_per_second(transfer_ms) << " img/s\n";

    cudaDeviceProp prop{};
    must(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    const std::string path = argc > 1 ? argv[1] : "results/bench_cuda.md";
    write_section(path, prop.name, resident_ms, transfer_ms);
    std::cout << "CUDA eval matches the CPU logits within 1e-4\n";
    return 0;
}
