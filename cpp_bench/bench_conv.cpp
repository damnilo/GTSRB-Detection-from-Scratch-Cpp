#include "Layers/include/Conv2D.h"
#include "cpp_bench/bench_cuda.h"

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
using DeviceBuffer = cuda_bench::DeviceBuffer<float>;

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
