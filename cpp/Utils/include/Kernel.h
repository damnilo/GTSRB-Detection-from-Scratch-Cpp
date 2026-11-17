#pragma once

#include <cstddef>

// Row-major C = A (M,K) * B (K,N). Tiles are 16 x 16 x 4.
// Partial edge tiles are computed, so the dimensions need not be multiples of the tile.
void tiledGemm(const float* A, const float* B, float* C, size_t M, size_t N, size_t K);
