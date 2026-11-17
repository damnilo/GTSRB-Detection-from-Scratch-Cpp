#include "Utils/include/Tensor.h"
#include "Utils/include/Kernel.h"

#include <cblas.h>

#include <limits>
#include <random>
#include <numeric>
#include <algorithm>

namespace {
    // One generator for every Tensor::randomize() call, so layer constructors
    // consume a single deterministic stream after Tensor::seed().
    std::mt19937& tensorRng() {
        static std::mt19937 generator{std::random_device{}()};
        return generator;
    }
}

Tensor::Tensor(){
    // Default-constructed tensor has empty shape and empty data — a valid but
    // "unset" state, used as a placeholder member before a real shape is known.
}

Tensor::Tensor(const std::vector<size_t>& shape) : shape(shape) {
    size_t total = 1;
    for(auto s: shape){
        total *= s;
    }
    data.resize(total); // Note: resize() default-initializes floats to 0.0f
}

Tensor::Tensor(std::initializer_list<size_t> shape) : Tensor(std::vector<size_t>(shape)) {
    // Delegates to the vector constructor
}

const std::vector<size_t>& Tensor::getShape() const {
    return shape;
}

size_t Tensor::size() const {
    return data.size();
}

size_t Tensor::dimensions() const {
    return shape.size();
}

// Converts multi-dim indices to a flat offset using row-major strides, computed
// right-to-left so the last axis has stride 1 (standard C-array layout).
size_t Tensor::flattenIndex(const std::vector<size_t>& indices) const {
    if(indices.size() != shape.size()){
        throw std::invalid_argument("Number of indices does not match number of dimensions");
    }

    size_t index = 0;
    size_t multiplier = 1;

    for(int i = shape.size() - 1; i >= 0; --i){
        if(indices[i] >= shape[i]){
            throw std::out_of_range("Index out of bounds");
        }
        index += indices[i] * multiplier;
        multiplier *= shape[i];
    }

    return index;
}

float& Tensor::at(const std::vector<size_t>& indices) {
    return data[flattenIndex(indices)];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    return data[flattenIndex(indices)];
}

float& Tensor::operator[](size_t index) {
    return data[index];
}

const float& Tensor::operator[](size_t index) const {
    return data[index];
}

// Shape-only reshape — total element count must match, since no data is moved or copied.
void Tensor::reshape(const std::vector<size_t>& newShape) {
    size_t newSize = 1;
    for(auto s: newShape){
        newSize *= s;
    }

    if(newSize != data.size()){
        throw std::invalid_argument("New shape must have the same number of elements as the old shape");
    }

    shape = newShape;
}

void Tensor::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}

void Tensor::seed(unsigned int seed) {
    tensorRng().seed(seed);
}

// Used for weight initialization. Each layer constructor controls the actual
// [minValue, maxValue] range (Xavier uniform bounds), this just fills uniformly within it.
void Tensor::randomize(float minValue, float maxValue) {
    std::uniform_real_distribution<float> dis(minValue, maxValue);

    for(auto& d: data){
        d = dis(tensorRng());
    }
}

std::vector<float>& Tensor::raw() {
    return data;
}

const std::vector<float>& Tensor::raw() const {
    return data;
}

Tensor Tensor::operator+(const Tensor& other) const {
    if(shape != other.shape){
        throw std::invalid_argument("Shapes must be the same for addition");
    }
    Tensor result(shape);
    for(size_t i = 0; i < data.size(); ++i){
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if(shape != other.shape){
        throw std::invalid_argument("Shapes must be the same for subtraction");
    }
    Tensor result(shape);
    for(size_t i = 0; i < data.size(); ++i){
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

// Elementwise multiply — used for masking (e.g. ReLU/Dropout gradients), not matrix multiplication.
Tensor Tensor::operator*(const Tensor& other) const {
    if(shape != other.shape){
        throw std::invalid_argument("Shapes must be the same for multiplication");
    }
    Tensor result(shape);
    for(size_t i = 0; i < data.size(); ++i){
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if(shape != other.shape){
        throw std::invalid_argument("Shapes must be the same for addition");
    }
    for(size_t i = 0; i < data.size(); ++i){
        data[i] += other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if(shape != other.shape){
        throw std::invalid_argument("Shapes must be the same for subtraction");
    }
    for(size_t i = 0; i < data.size(); ++i){
        data[i] -= other.data[i];
    }
    return *this;
}

Tensor Tensor::operator*(float scalar) const {
    Tensor result(shape);
    for(size_t i = 0; i < data.size(); ++i){
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Tensor Tensor::operator+(float scalar) const {
    Tensor result(shape);
    for(size_t i = 0; i < data.size(); ++i){
        result.data[i] = data[i] + scalar;
    }
    return result;
}

// (M,K) x (K,N) -> (M,N), row-major. Large narrow products use the tiled
// kernel. Short products and wide ones use OpenBLAS. beta = 0 overwrites C.
Tensor Tensor::matmul(const Tensor& other) const {
    if(shape.size() != 2 || other.shape.size() != 2){
        throw std::invalid_argument("Both tensors must be 2D for matrix multiplication");
    }
    if(shape[1] != other.shape[0]){
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication");
    }

    size_t M = shape[0];
    size_t K = shape[1];
    size_t N = other.shape[1];
    const size_t intMax = static_cast<size_t>(std::numeric_limits<blasint>::max());
    if(M > intMax || N > intMax || K > intMax){
        throw std::invalid_argument("Matrix dimension does not fit in a BLAS integer");
    }

    Tensor result({M, N});

    // The tiled kernel only pays off when it has enough 16x16 tiles to occupy
    // the cores. A short-and-wide product, such as a conv weight gradient
    // (M = 16, K = 65536, N = 27), makes two tiles, so it stays on OpenBLAS.
    constexpr size_t tile = 16;
    const size_t tileRows = (M + tile - 1) / tile;
    const size_t tileCols = (N + tile - 1) / tile;
    if (N < 64 && tileRows * tileCols >= 32) {
        tiledGemm(data.data(), other.data.data(), result.data.data(), M, N, K);
    } else {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<blasint>(M), static_cast<blasint>(N), static_cast<blasint>(K),
            1.0f, data.data(), static_cast<blasint>(K),
            other.data.data(), static_cast<blasint>(N),
            0.0f, result.data.data(), static_cast<blasint>(N));
    }

    return result;
}

Tensor Tensor::matmulTransposedLeft(const Tensor& other) const {
    if(shape.size() != 2 || other.shape.size() != 2){
        throw std::invalid_argument("Both tensors must be 2D for matrix multiplication");
    }
    if(shape[0] != other.shape[0]){
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication");
    }

    size_t K = shape[0];
    size_t M = shape[1];
    size_t N = other.shape[1];
    const size_t intMax = static_cast<size_t>(std::numeric_limits<blasint>::max());
    if(M > intMax || N > intMax || K > intMax){
        throw std::invalid_argument("Matrix dimension does not fit in a BLAS integer");
    }

    Tensor result({M, N});
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        static_cast<blasint>(M), static_cast<blasint>(N), static_cast<blasint>(K),
        1.0f, data.data(), static_cast<blasint>(M),
        other.data.data(), static_cast<blasint>(N),
        0.0f, result.data.data(), static_cast<blasint>(N));
    return result;
}

// 2D transpose: (M,N) -> (N,M). Straightforward element swap, parallelized over rows.
Tensor Tensor::transpose() const {
    if(shape.size() != 2){
        throw std::invalid_argument("Tensor must be 2D for transpose");
    }

    Tensor result({shape[1], shape[0]});

    const float* src = data.data();
    float* dst = result.data.data();

    #pragma omp parallel for
    for(size_t i = 0; i < shape[0]; ++i){
        for(size_t j = 0; j < shape[1]; ++j){
            dst[j * shape[0] + i] = src[i * shape[1] + j];
        }
    }

    return result;
}

// Sums the tensor along one axis, collapsing it out of the result shape.
// Used primarily for bias gradients: summing dL/dOutput over the batch axis (axis 0).
Tensor Tensor::sum(size_t axis) const {
    if(axis >= shape.size()){
        throw std::invalid_argument("Axis out of bounds");
    }

    // Fast path: summing over axis 0 of a 2D tensor (the common case for bias gradients)
    // avoids the general N-dimensional index bookkeeping below.
    if(axis == 0){
        Tensor result({shape[1]});
        for(size_t j = 0; j < shape[0]; ++j){
            for(size_t i = 0; i < shape[1]; ++i){
                result[i] += data[j * shape[1] + i];
            }
        }
        return result;
    }

    // General N-dimensional case: build the result shape by dropping `axis`,
    // then walk every element of the source tensor, map its multi-index to the
    // corresponding (axis-removed) index in the result, and accumulate.
    std::vector<size_t> newShape = shape;
    newShape.erase(newShape.begin() + axis);

    Tensor result(newShape);
    std::fill(result.raw().begin(), result.raw().end(), 0.0f);

    std::vector<size_t> indices(shape.size(), 0);
    for(size_t i = 0; i < data.size(); ++i){
        // NOTE: this inner loop computes `flatIndex`/`dimPos` but the result is never
        // used — the actual flat-index calculation happens below via `reduced`/`fi`/`m`.
        // Harmless dead code, safe to remove if cleaning up.
        size_t flatIndex = 0, multiplier = 1;
        for(int j = shape.size() - 1; j >= 0; --j){
            if(j == axis) continue;
            size_t dimPos = (j > axis) ? j - 1 : j;
        }

        // Build the index tuple with `axis` removed, then flatten it against newShape.
        std::vector<size_t> reduced;
        for(size_t j = 0; j < shape.size(); ++j){
            if(j != axis){
                reduced.push_back(indices[j]);
            }
        }
        size_t fi = 0, m = 1;
        for(int j = reduced.size() - 1; j >= 0; --j){
            fi += reduced[j] * m;
            m *= newShape[j];
        }

        result.raw()[fi] += data[i];

        // Odometer-style increment of the multi-index, skipping the reduced axis.
        for(int j = shape.size() - 1; j >= 0; --j){
            if(j == axis) continue;
            if(++indices[j] < shape[j]) break;
            indices[j] = 0;
        }
    }

    return result;
}

// Adds a 1D row vector to every row of a 2D tensor — the only broadcasting op in the
// library. Used to add per-output-channel biases after a Dense or Conv2D matmul.
Tensor Tensor::broadcastAdd(const Tensor& rowVector) const {
    if(rowVector.dimensions() != 1 || rowVector.size() != shape[1]){
        throw std::invalid_argument("Row vector must be 1D and match the number of columns");
    }

    Tensor result(shape);

    const float* a = data.data();
    const float* b = rowVector.data.data();
    float* c = result.data.data();
    for(size_t i = 0; i < shape[0]; ++i){
        for(size_t j = 0; j < shape[1]; ++j){
            c[i * shape[1] + j] = a[i * shape[1] + j] + b[j];
        }
    }

    return result;
}

Tensor Tensor::zeros(const std::vector<size_t>& shape) {
    Tensor result(shape);
    result.fill(0.0f);
    return result;
}