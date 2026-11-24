#pragma once

#include <string>
#include <vector>

// Images from dataset/Train, class directories 0..42, filenames sorted inside each class.
// PNG, JPEG, and PPM are accepted. Each image is RGB, resized to 32x32, divided by 255,
// then standardized by the per-channel mean and standard deviation of this same set.
struct TrainImages {
    std::vector<float> pixels;
    std::vector<int> labels;
    size_t count = 0;
    float mean[3] = {};
    float deviation[3] = {};
};

bool train_directory_ready(const std::string& train_root);
TrainImages load_train_images(const std::string& train_root);
