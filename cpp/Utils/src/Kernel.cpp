#include "Utils/include/Kernel.h"

#include <algorithm>

namespace {
    constexpr int kTileM = 16;
    constexpr int kTileN = 16;
    constexpr int kTileK = 4;
}

void tiledGemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K) {
    const int tileRows = static_cast<int>((M + kTileM - 1) / kTileM);
    const int tileCols = static_cast<int>((N + kTileN - 1) / kTileN);

    #pragma omp parallel for
    for (int tile = 0; tile < tileRows * tileCols; ++tile) {
        const int i0 = (tile / tileCols) * kTileM;
        const int j0 = (tile % tileCols) * kTileN;
        const int iLimit = std::min(kTileM, static_cast<int>(M) - i0);
        const int jLimit = std::min(kTileN, static_cast<int>(N) - j0);

        float acc[kTileM * kTileN] = {};

        for (int k0 = 0; k0 < static_cast<int>(K); k0 += kTileK) {
            const int kLimit = std::min(kTileK, static_cast<int>(K) - k0);
            float aPanel[kTileM * kTileK];
            float bPanel[kTileK * kTileN];

            for (int i = 0; i < iLimit; ++i) {
                for (int k = 0; k < kLimit; ++k) {
                    aPanel[i * kTileK + k] = A[(static_cast<size_t>(i0 + i) * K) + static_cast<size_t>(k0 + k)];
                }
            }
            for (int k = 0; k < kLimit; ++k) {
                for (int j = 0; j < jLimit; ++j) {
                    bPanel[k * kTileN + j] = B[(static_cast<size_t>(k0 + k) * N) + static_cast<size_t>(j0 + j)];
                }
            }

            for (int k = 0; k < kLimit; ++k) {
                for (int i = 0; i < iLimit; ++i) {
                    const float a = aPanel[i * kTileK + k];
                    for (int j = 0; j < jLimit; ++j) {
                        acc[i * kTileN + j] += a * bPanel[k * kTileN + j];
                    }
                }
            }
        }

        for (int i = 0; i < iLimit; ++i) {
            for (int j = 0; j < jLimit; ++j) {
                C[(static_cast<size_t>(i0 + i) * N) + static_cast<size_t>(j0 + j)] = acc[i * kTileN + j];
            }
        }
    }
}
