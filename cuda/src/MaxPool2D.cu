#include "../include/MaxPool2D.cuh"

__global__ void cudaMaxPoolForward(
    const float* A,
    float* B,
    int* maxIdx,
    size_t pool_size,
    size_t stride,
    size_t batch_size,
    size_t channels,
    size_t height,
    size_t width
) {
    
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;

    if (b >= batch_size || c >= channels) return;

    const int p = static_cast<int>(pool_size);
    const int s = static_cast<int>(stride);
    const int H = static_cast<int>(height);
    const int W = static_cast<int>(width);
    const int C = static_cast<int>(channels);
    int outH = (H - p) / s + 1;
    int outW = (W - p) / s + 1;

    for (int i = 0; i < outH; ++i) {
        int hStart = i * s;
        
        for (int j = 0; j < outW; ++j) {
            int wStart = j * s;
            float maxVal = -INFINITY;
            int maxId = 0;

            for (int k1 = 0; k1 < p; ++k1) {
                int h = hStart + k1;

                for (int k2 = 0; k2 < p; ++k2) {
                    int w = wStart + k2;

                    int flatIdx = ((b * C + c) * H + h) * W + w;
                    float val = A[flatIdx];
                    if (val > maxVal) {
                        maxVal = val;
                        maxId = flatIdx;
                    }
                }
            }

            int outIdx = (b * C + c) * outH * outW + i * outW + j;
            B[outIdx] = maxVal;
            maxIdx[outIdx] = maxId;
        }
    }
}

__global__ void cudaMaxPoolBackward(
    const float* A,
    float* B,
    int* maxIdx,
    size_t N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    atomicAdd(&B[maxIdx[i]], A[i]);
}