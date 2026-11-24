#include "cpp_bench/train_images.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

constexpr int kClasses = 43;
constexpr int kSize = 32;

void skip_space_and_comments(std::istream& in) {
    while (in) {
        const int next = in.peek();
        if (next == EOF) {
            return;
        }
        if (std::isspace(next)) {
            in.get();
            continue;
        }
        if (next == '#') {
            std::string ignored;
            std::getline(in, ignored);
            continue;
        }
        return;
    }
}

int read_number(std::istream& in) {
    skip_space_and_comments(in);
    int value = 0;
    if (!(in >> value)) {
        throw std::runtime_error("PPM header is missing a number");
    }
    return value;
}

std::vector<unsigned char> read_ppm(const std::filesystem::path& path, int& width, int& height) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Could not open " + path.string());
    }
    std::string magic;
    in >> magic;
    if (magic != "P6" && magic != "P3") {
        throw std::runtime_error(path.string() + " is not a PPM image");
    }
    width = read_number(in);
    height = read_number(in);
    const int max_value = read_number(in);
    if (width <= 0 || height <= 0 || max_value != 255) {
        throw std::runtime_error(path.string() + " is not an 8-bit PPM image");
    }
    const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    std::vector<unsigned char> rgb(count);
    if (magic == "P6") {
        if (in.peek() == '\n' || in.peek() == ' ' || in.peek() == '\r' || in.peek() == '\t') {
            in.get();
        }
        in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(count));
        if (in.gcount() != static_cast<std::streamsize>(count)) {
            throw std::runtime_error(path.string() + " ended before the pixel data");
        }
        return rgb;
    }
    for (size_t i = 0; i < count; ++i) {
        rgb[i] = static_cast<unsigned char>(read_number(in));
    }
    return rgb;
}

void store_resized(const std::vector<unsigned char>& rgb, int width, int height, std::vector<float>& pixels) {
    const size_t base = pixels.size();
    pixels.resize(base + static_cast<size_t>(3 * kSize * kSize));
    if (width == kSize && height == kSize) {
        for (int y = 0; y < kSize; ++y) {
            for (int x = 0; x < kSize; ++x) {
                for (int channel = 0; channel < 3; ++channel) {
                    const size_t source = (static_cast<size_t>(y * width + x) * 3) + static_cast<size_t>(channel);
                    const size_t dest = base + ((static_cast<size_t>(channel) * kSize + static_cast<size_t>(y)) * kSize) + static_cast<size_t>(x);
                    pixels[dest] = static_cast<float>(rgb[source]) / 255.0f;
                }
            }
        }
        return;
    }

    for (int y = 0; y < kSize; ++y) {
        const float source_y = (static_cast<float>(y) + 0.5f) * static_cast<float>(height) / kSize - 0.5f;
        const float clamped_y = std::clamp(source_y, 0.0f, static_cast<float>(height - 1));
        const int y0 = static_cast<int>(std::floor(clamped_y));
        const int y1 = std::min(y0 + 1, height - 1);
        const float weight_y = clamped_y - static_cast<float>(y0);
        for (int x = 0; x < kSize; ++x) {
            const float source_x = (static_cast<float>(x) + 0.5f) * static_cast<float>(width) / kSize - 0.5f;
            const float clamped_x = std::clamp(source_x, 0.0f, static_cast<float>(width - 1));
            const int x0 = static_cast<int>(std::floor(clamped_x));
            const int x1 = std::min(x0 + 1, width - 1);
            const float weight_x = clamped_x - static_cast<float>(x0);
            for (int channel = 0; channel < 3; ++channel) {
                auto at = [&](int sy, int sx) {
                    const size_t index = (static_cast<size_t>(sy * width + sx) * 3) + static_cast<size_t>(channel);
                    return static_cast<float>(rgb[index]) / 255.0f;
                };
                const float top = at(y0, x0) * (1.0f - weight_x) + at(y0, x1) * weight_x;
                const float bottom = at(y1, x0) * (1.0f - weight_x) + at(y1, x1) * weight_x;
                const size_t dest = base + ((static_cast<size_t>(channel) * kSize + static_cast<size_t>(y)) * kSize) + static_cast<size_t>(x);
                pixels[dest] = top * (1.0f - weight_y) + bottom * weight_y;
            }
        }
    }
}

std::string lower_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension;
}

void standardize(TrainImages& images) {
    const size_t pixels_per_channel = images.count * static_cast<size_t>(kSize * kSize);
    double sum[3] = {};
    for (size_t image = 0; image < images.count; ++image) {
        const size_t base = image * 3 * kSize * kSize;
        for (int channel = 0; channel < 3; ++channel) {
            const size_t channel_base = base + static_cast<size_t>(channel * kSize * kSize);
            for (int pixel = 0; pixel < kSize * kSize; ++pixel) {
                sum[channel] += images.pixels[channel_base + static_cast<size_t>(pixel)];
            }
        }
    }
    double second[3] = {};
    for (int channel = 0; channel < 3; ++channel) {
        images.mean[channel] = static_cast<float>(sum[channel] / static_cast<double>(pixels_per_channel));
    }
    for (size_t image = 0; image < images.count; ++image) {
        const size_t base = image * 3 * kSize * kSize;
        for (int channel = 0; channel < 3; ++channel) {
            const size_t channel_base = base + static_cast<size_t>(channel * kSize * kSize);
            const double mean = images.mean[channel];
            for (int pixel = 0; pixel < kSize * kSize; ++pixel) {
                const double delta = images.pixels[channel_base + static_cast<size_t>(pixel)] - mean;
                second[channel] += delta * delta;
            }
        }
    }
    for (int channel = 0; channel < 3; ++channel) {
        const double variance = second[channel] / static_cast<double>(pixels_per_channel);
        images.deviation[channel] = static_cast<float>(std::sqrt(variance));
        if (images.deviation[channel] == 0.0f) {
            images.deviation[channel] = 1.0f;
        }
    }
    for (size_t image = 0; image < images.count; ++image) {
        const size_t base = image * 3 * kSize * kSize;
        for (int channel = 0; channel < 3; ++channel) {
            const size_t channel_base = base + static_cast<size_t>(channel * kSize * kSize);
            const float mean = images.mean[channel];
            const float deviation = images.deviation[channel];
            for (int pixel = 0; pixel < kSize * kSize; ++pixel) {
                float& value = images.pixels[channel_base + static_cast<size_t>(pixel)];
                value = (value - mean) / deviation;
            }
        }
    }
}

}  // namespace

bool train_directory_ready(const std::string& train_root) {
    namespace fs = std::filesystem;
    const fs::path root(train_root);
    if (!fs::is_directory(root)) {
        return false;
    }
    for (int label = 0; label < kClasses; ++label) {
        if (!fs::is_directory(root / std::to_string(label))) {
            return false;
        }
    }
    return true;
}

TrainImages load_train_images(const std::string& train_root) {
    namespace fs = std::filesystem;
    TrainImages images;
    const fs::path root(train_root);
    for (int label = 0; label < kClasses; ++label) {
        const fs::path directory = root / std::to_string(label);
        std::vector<fs::path> files;
        for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file() && lower_extension(entry.path()) == ".ppm") {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) {
            throw std::runtime_error(directory.string() + " has no PPM images");
        }
        for (const fs::path& file : files) {
            int width = 0;
            int height = 0;
            const std::vector<unsigned char> rgb = read_ppm(file, width, height);
            store_resized(rgb, width, height, images.pixels);
            images.labels.push_back(label);
        }
    }
    images.count = images.labels.size();
    standardize(images);
    return images;
}
