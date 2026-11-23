#pragma once

#include "cuda/Layers/include/Convolution.cuh"
#include "cuda/Utils/include/TensorOps.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Header-only helpers for bench_eval. The definitions stay inline because
// this header is included from one translation unit.
namespace cuda_bench {

constexpr float kTolerance = 1e-4f;
constexpr int kWarmup = 1;
constexpr int kRepeats = 10;

inline void must(cudaError_t error, const char* what) {
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

    void zero() const {
        must(cudaMemset(ptr, 0, count * sizeof(T)), "cudaMemset");
    }
};

inline float max_abs(const std::vector<float>& got, const std::vector<float>& expected) {
    float diff = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        diff = std::max(diff, std::fabs(got[i] - expected[i]));
    }
    return diff;
}

// The allowance grows with the result. A reduction of length 65536 is tens of
// units wide, and 1e-4 absolute is tighter than float32 rounding along that sum.
inline bool report(
    const std::string& name,
    const char* stage,
    const std::vector<float>& got,
    const std::vector<float>& expected
) {
    if (got.size() != expected.size()) {
        std::cout << name << " " << stage << " size " << got.size() << " vs " << expected.size() << "\n";
        return false;
    }
    const float diff = max_abs(got, expected);
    float scale = 0.0f;
    for (float value : expected) {
        scale = std::max(scale, std::fabs(value));
    }
    const float limit = kTolerance * std::max(1.0f, scale);
    std::cout << name << " " << stage << " max abs " << diff << " limit " << limit << "\n";
    return diff < limit;
}

inline void launch_im2col(
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
    cudaIm2Col<<<grid, block>>>(
        input, col,
        kernel, stride, padding,
        batch, channels, height, width
    );
    must(cudaGetLastError(), "cudaIm2Col");
}

inline void launch_col2im(
    const float* col,
    float* grad,
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
    cudaCol2Im<<<grid, block>>>(
        col, grad,
        kernel, stride, padding,
        batch, channels, height, width
    );
    must(cudaGetLastError(), "cudaCol2Im");
}

inline void launch_transpose(const float* input, float* output, size_t rows, size_t cols) {
    dim3 block(16, 16);
    dim3 grid(
        static_cast<unsigned>((cols + block.x - 1) / block.x),
        static_cast<unsigned>((rows + block.y - 1) / block.y)
    );
    cudaTranspose<<<grid, block>>>(input, output, rows, cols);
    must(cudaGetLastError(), "cudaTranspose");
}

inline void launch_gemm(const float* a, const float* b, float* c, size_t M, size_t N, size_t K) {
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((N + 63) / 64), static_cast<unsigned>((M + 63) / 64));
    cudaTiledGemm<<<grid, block>>>(a, b, c, M, N, K);
    must(cudaGetLastError(), "cudaTiledGemm");
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

inline float images_per_second(size_t batch, float ms) {
    return static_cast<float>(batch) * 1000.0f / ms;
}

}  // namespace cuda_bench
