#include "../include/TensorOps.cuh"

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4

__global__ void cudaTiledGemm(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N,
    size_t K
) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int tid = 16 * ty + tx;

    int rowBase = blockIdx.y * BM;
    int colBase = blockIdx.x * BN;
    int row0 = rowBase + ty * TM;
    int col0 = colBase + tx * TN;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float acc[TM][TN] = {};
    float a_reg[TM];
    float b_reg[TN];

    for (int i = 0; i < static_cast<int>(K); i += BK) {
        int r = tid / 4;
        int c = (tid % 4) * 4;

        for (int j = 0; j < 4; ++j) {
            if ((rowBase + r) < static_cast<int>(M) && (i + c + j) < static_cast<int>(K)) {
                As[r][c + j] = A[(rowBase + r) * K + (i + c + j)];
            } else {
                As[r][c + j] = 0.0f;
            }
        }

        r = tid / 16;
        c = (tid % 16) * 4;

        for (int j = 0; j < 4; ++j) {
            if ((i + r) < static_cast<int>(K) && (colBase + c + j) < static_cast<int>(N)) {
                Bs[r][c + j] = B[(i + r) * N + c + j + colBase];
            } else {
                Bs[r][c + j] = 0.0f;
            }
        }

        __syncthreads();

        for (int j = 0; j < BK; ++j) {
            for (int k = 0; k < TM; ++k) {
                a_reg[k] = As[ty * TM + k][j];
            }
            for (int k = 0; k < TN; ++k) {
                b_reg[k] = Bs[j][tx * TN + k];
            }

            for (int k = 0; k < TM; ++k) {
                for (int l = 0; l < TN; ++l) {
                    acc[k][l] += a_reg[k] * b_reg[l];
                }
            }
        }

        __syncthreads();

    }

    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            if ((row0 + i) < static_cast<int>(M) && (col0 + j) < static_cast<int>(N)) {
                C[(row0 + i) * N + col0 + j] = acc[i][j];
            }
        }
    } 
}

__global__ void cudaTranspose(
    const float* A,
    float* B,
    size_t M,
    size_t N
) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= M || j >= N) return;

    B[j * M + i] = A[i * N + j];

}