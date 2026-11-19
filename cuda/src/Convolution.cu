#include "../include/Convolution.cuh"

__global__ void cudaIm2Col(
    const float* A,
    float* B,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    size_t batch_size,
    size_t channels,
    size_t height,
    size_t width
) {

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= batch_size || c >= channels) return;

    const int k = static_cast<int>(kernel_size);
    const int s = static_cast<int>(stride);
    const int p = static_cast<int>(padding);
    const int H = static_cast<int>(height);
    const int W = static_cast<int>(width);
    const int C = static_cast<int>(channels);
    int outH = (H + 2 * p - k) / s + 1;
    int outW = (W + 2 * p - k) / s + 1;
    int rowSize = C * k * k;
    int channelOffset = c * k * k;

    for (int i = 0; i < outH; ++i) {
        int hStart = i * s - p;
        for (int j = 0; j < outW; ++j) {
            int wStart = j * s - p;
            int row = (b * outH + i) * outW + j;

            int col_idx = 0;
            for (int k1 = 0; k1 < k; ++k1) {
                int h = hStart + k1;
                for (int k2 = 0; k2 < k; ++k2) {
                    int w = wStart + k2;
                    float val = 0.0f;
                    if (h >= 0 && w >= 0 && h < H && w < W) {
                        int inputIdx = ((b * C + c) * H + h) * W + w;
                        val = A[inputIdx];
                    }
                    B[row * rowSize + channelOffset + col_idx] = val;
                    ++col_idx;
                }
            }
        }
    }
}

__global__ void cudaCol2Im(
    const float* A,
    float* B,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    size_t batch_size,
    size_t channels,
    size_t height,
    size_t width
) {
    
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= batch_size || c >= channels) return;

    const int k = static_cast<int>(kernel_size);
    const int s = static_cast<int>(stride);
    const int p = static_cast<int>(padding);
    const int H = static_cast<int>(height);
    const int W = static_cast<int>(width);
    const int C = static_cast<int>(channels);
    int outH = (H + 2 * p - k) / s + 1;
    int outW = (W + 2 * p - k) / s + 1;
    int rowSize = C * k * k;
    int channelOffset = c * k * k;

    for (int i = 0; i < outH; ++i) {
        int hStart = i * s - p;
        for (int j = 0; j < outW; ++j) {
            int wStart = j * s - p;
            int row = (b * outH + i) * outW + j;

            int col_idx = 0;
            for (int k1 = 0; k1 < k; ++k1) {
                int h = hStart + k1;

                for (int k2 = 0; k2 < k; ++k2) {
                    int w = wStart + k2;

                    if (h >= 0 && w >= 0 && h < H && w < W) {
                        int inputIdx = ((b * C + c) * H + h) * W + w;
                        B[inputIdx] += A[row * rowSize + channelOffset + col_idx];
                    }

                    ++col_idx;
                }
            }
        }
    }
}
