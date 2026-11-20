#include "Layers/include/Conv2D.h"
#include "cuda/include/Convolution.cuh"
#include "cuda/include/TensorOps.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kTolerance = 1e-4f;
constexpr int kWarmup = 1;
constexpr int kRepeats = 10;

void must(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(error) << "\n";
        std::exit(1);
    }
}

void must_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << what << ": cuBLAS status " << static_cast<int>(status) << "\n";
        std::exit(1);
    }
}

class DeviceBuffer {
    
private:

    float* ptr = nullptr;
    size_t count = 0;

public:

    explicit DeviceBuffer(size_t count) : count(count) {
        must(cudaMalloc(&ptr, count * sizeof(float)), "cudaMalloc");
    }

    ~DeviceBuffer() { cudaFree(ptr); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* get() const { return ptr; }
    size_t size() const { return count; }

    void copy_from(const std::vector<float>& host) const {
        must(cudaMemcpy(ptr, host.data(), count * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy HtoD");
    }

    std::vector<float> copy_to_host() const {
        std::vector<float> host(count);
        must(cudaMemcpy(host.data(), ptr, count * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy DtoH");
        return host;
    }

    void zero() const {
        must(cudaMemset(ptr, 0, count * sizeof(float)), "cudaMemset");
    }
};

float max_abs(
    const std::vector<float>& got, 
    const std::vector<float>& expected
) {
    float diff = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        diff = std::max(diff, std::fabs(got[i] - expected[i]));
    }
    return diff;
}

bool report(
    const std::string& name,
    const char* stage,
    const std::vector<float>& got,
    const std::vector<float>& expected
) {
    const float diff = max_abs(got, expected);
    float scale = 0.0f;
    for (float value : expected) {
        scale = std::max(scale, std::fabs(value));
    }
    // A product of length 65536 is tens of units wide. 1e-4 absolute is tighter
    // than float32 rounding along that sum, so the allowance scales with it.
    const float limit = kTolerance * std::max(1.0f, scale);
    std::cout << name << " " << stage << " max abs " << diff << " limit " << limit << "\n";
    return diff < limit;
}

void launch_im2col(
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
        kernel, stride, 
        padding, batch, 
        channels, height, 
        width
    );
    must(cudaGetLastError(), "cudaIm2Col");
}

void launch_col2im(
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
        kernel, stride, 
        padding, batch, 
        channels, height, 
        width
    );
    must(cudaGetLastError(), "cudaCol2Im");
}

void launch_transpose(
    const float* input, 
    float* output, 
    size_t rows, 
    size_t cols
) {
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((cols + block.x - 1) / block.x),
              static_cast<unsigned>((rows + block.y - 1) / block.y));
    cudaTranspose<<<grid, block>>>(
        input, output, 
        rows, cols
    );
    must(cudaGetLastError(), "cudaTranspose");
}

void launch_gemm(
    const float* a, 
    const float* b, 
    float* c, 
    size_t M, 
    size_t N, 
    size_t K
) {
    dim3 block(16, 16);
    dim3 grid(static_cast<unsigned>((N + 63) / 64), static_cast<unsigned>((M + 63) / 64));
    cudaTiledGemm<<<grid, block>>>(
        a, b, c, 
        M, N, K
    );
    must(cudaGetLastError(), "cudaTiledGemm");
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

float images_per_second(size_t batch, float ms) {
    return static_cast<float>(batch) * 1000.0f / ms;
}

void forward_kernels(
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

bool check_case(const ConvCase& spec, cublasHandle_t handle, std::vector<BenchRow>& rows) {
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

    DeviceBuffer d_input(input.size());
    DeviceBuffer d_weights(weights.size());
    DeviceBuffer d_col(cpu_col.size());
    DeviceBuffer d_weights_t(weights.size());
    DeviceBuffer d_out(cpu_out.size());
    DeviceBuffer d_cublas(cpu_out.size());
    DeviceBuffer d_dout(d_out_col.size());
    DeviceBuffer d_dout_t(d_out_col.size());
    DeviceBuffer d_grad_w(cpu_grad_w.size());
    DeviceBuffer d_cublas_grad(cpu_grad_w.size());
    DeviceBuffer d_d_col(cpu_d_col.size());
    DeviceBuffer d_grad_x(cpu_grad_x.size());

    d_input.copy_from(input.raw());
    d_weights.copy_from(weights.raw());
    d_dout.copy_from(d_out_col.raw());

    forward_kernels(
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
    
    launch_col2im(d_d_col.get(), 
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

    forward_kernels(
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
        
        forward_kernels(
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

void write_table(const std::string& path, const std::string& gpu, const std::vector<BenchRow>& rows) {
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
    std::cout << "Wrote " << path << "\n";
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
    bool ok = true;
    for (const ConvCase& spec : cases) {
        ok = check_case(spec, handle, rows) && ok;
    }
    cublasDestroy(handle);

    if (!ok) {
        std::cerr << "CUDA conv disagrees with the CPU path by more than 1e-4 of the result size\n";
        return 1;
    }

    cudaDeviceProp prop{};
    must(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    const std::string path = argc > 1 ? argv[1] : "results/bench_cuda.md";
    write_table(path, prop.name, rows);
    std::cout << "CUDA conv matches the CPU path and cuBLAS within 1e-4\n";
    return 0;
}
