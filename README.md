# Road Sign Classification with a CNN from Scratch (C++ & Python)

## Overview

This project implements a **Convolutional Neural Network (CNN) from scratch** for traffic sign classification using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

Dataset can be found on this link:

 https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign 

Unlike frameworks such as PyTorch or TensorFlow, all neural network components—including convolution, pooling, activation functions, dropout, optimization, loss computation, and backpropagation—are implemented manually in **C++**. The C++ library is exposed to Python through **pybind11**, while Python is used for dataset handling, training, evaluation, and visualization.

---

## Features

* CNN implementation from scratch in C++
* Forward and backward propagation
* Convolutional, pooling, and fully connected layers
* ReLU activation
* Dropout regularization
* Softmax + Cross Entropy loss
* SGD optimizer
* Model serialization (save/load trained networks)
* Python bindings using pybind11
* Training and validation pipeline
* Test dataset evaluation
* Group-aware train/validation split to prevent data leakage

---

## Project Structure

```
.
├── cpp/
│   ├── include/          # Header files
│   ├── src/              # C++ implementation
│   └── bindings.cpp      # Python bindings
│
├── model.py              # Network architecture and training
├── load_dataset.py       # Dataset loading utilities
├── test.py               # Test set evaluation
├── CMakeLists.txt
└── README.md
```

---

## Network Architecture

The implemented CNN consists of:

```
Input (3×32×32)
        │
Conv2D (3 → 16, 3×3, stride 1, padding 1)
        │
ReLU
        │
MaxPool2D (2×2)
        │
Conv2D (16 → 32, 3×3, stride 1, padding 1)
        │
ReLU
        │
MaxPool2D (2×2)
        │
Flatten (32×8×8 = 2048)
        │
Dense (2048 → 128)
        │
ReLU
        │
Dropout (p = 0.3)
        │
Dense (128 → 43)
        │
Softmax
```

---

## Dataset

The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** consisting of **43 traffic sign classes**.

Images are:

* resized to **32×32**
* converted to RGB
* normalized
* stored in channel-first format `(N, C, H, W)`

Training and validation sets are generated using a **group-aware stratified split**, ensuring images from the same recording sequence are never split between both sets.

---

## Building

### Requirements

* C++17
* CMake
* Python 3
* pybind11
* NumPy
* OpenCV
* pandas

Build the C++ extension:

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

---

## Training

Run:

```bash
python model.py
```

The best-performing model is automatically saved as `best_model.bin`. Training also writes `checkpoints/norm.npz` (per-channel mean and standard deviation of the training split) and `checkpoints/config.json`.

---

## Testing

Evaluate the trained model on the official test dataset:

```bash
python test.py
```

`test.py` loads `best_model.bin`, standardizes images with `checkpoints/norm.npz`, disables dropout, and prints test accuracy.

```bash
python eval.py
```

`eval.py` runs the same protocol and writes `results/test_metrics.json`, `results/per_class.csv`, `results/confusion.csv`, `results/confusion.png`, and `results/top_confusions.csv`. It also reports accuracy under the old test-set normalization so the two protocols can be compared. The figures below were produced by the old protocol.

---

## Evaluation

Checkpoint: `best_model.bin`, loaded into the architecture in `create_network()`.

Current `test.py` protocol:

* 12,630 images from `dataset/Test.csv` (official GTSRB test split)
* resized to 32×32, converted from BGR to RGB, scaled to `[0, 1]`, stored as `(N, C, H, W)`
* standardized per channel with the training-set mean and standard deviation (`checkpoints/norm.npz`)
* dropout in evaluation mode (pass-through)
* predicted class is the argmax of the softmax output

The table below was measured when the test set was standardized with its own mean and standard deviation. Replace it with the training-set-normalization numbers from `results/test_metrics.json` after running `python eval.py`.

### Overall results

| Metric | Value |
| --- | --- |
| Test images | 12,630 |
| Correct predictions | 12,082 |
| Accuracy | 95.6611% (12,082 / 12,630) |
| Macro precision | 94.5825% |
| Macro recall | 94.0629% |
| Macro F1 | 94.0867% |
| Weighted precision | 95.7216% |
| Weighted recall | 95.6611% |
| Weighted F1 | 95.6020% |

Macro scores are the unweighted mean of the 43 per-class scores. Weighted scores use the number of test images in each class as the weight. Weighted recall equals accuracy.

Precision, recall, and F1 for class `c` use the confusion matrix `M` (rows = true class, columns = predicted class):

* precision = `M[c, c] / sum(M[:, c])`
* recall = `M[c, c] / sum(M[c, :])`
* F1 = `2 * precision * recall / (precision + recall)`

### Per-class results

| ID | Class | Support | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Speed limit (20km/h) | 60 | 95.00% | 95.00% | 95.00% |
| 1 | Speed limit (30km/h) | 720 | 97.24% | 97.78% | 97.51% |
| 2 | Speed limit (50km/h) | 750 | 94.67% | 97.07% | 95.85% |
| 3 | Speed limit (60km/h) | 450 | 90.45% | 94.67% | 92.51% |
| 4 | Speed limit (70km/h) | 660 | 96.11% | 97.27% | 96.69% |
| 5 | Speed limit (80km/h) | 630 | 89.97% | 91.11% | 90.54% |
| 6 | End of speed limit (80km/h) | 150 | 100.00% | 86.00% | 92.47% |
| 7 | Speed limit (100km/h) | 450 | 97.94% | 84.44% | 90.69% |
| 8 | Speed limit (120km/h) | 450 | 88.84% | 95.56% | 92.08% |
| 9 | No passing | 480 | 98.10% | 96.88% | 97.48% |
| 10 | No passing veh over 3.5 tons | 660 | 98.32% | 97.73% | 98.02% |
| 11 | Right-of-way at intersection | 420 | 93.44% | 98.33% | 95.82% |
| 12 | Priority road | 690 | 99.41% | 98.26% | 98.83% |
| 13 | Yield | 720 | 98.49% | 99.58% | 99.03% |
| 14 | Stop | 270 | 99.63% | 99.63% | 99.63% |
| 15 | No vehicles | 210 | 96.76% | 99.52% | 98.12% |
| 16 | Veh > 3.5 tons prohibited | 150 | 99.33% | 99.33% | 99.33% |
| 17 | No entry | 360 | 99.13% | 95.00% | 97.02% |
| 18 | General caution | 390 | 96.32% | 93.85% | 95.06% |
| 19 | Dangerous curve left | 60 | 98.36% | 100.00% | 99.17% |
| 20 | Dangerous curve right | 90 | 88.24% | 100.00% | 93.75% |
| 21 | Double curve | 90 | 94.94% | 83.33% | 88.76% |
| 22 | Bumpy road | 120 | 87.79% | 95.83% | 91.63% |
| 23 | Slippery road | 150 | 91.25% | 97.33% | 94.19% |
| 24 | Road narrows on the right | 90 | 90.00% | 90.00% | 90.00% |
| 25 | Road work | 480 | 97.26% | 96.25% | 96.75% |
| 26 | Traffic signals | 180 | 90.42% | 83.89% | 87.03% |
| 27 | Pedestrians | 60 | 84.38% | 45.00% | 58.70% |
| 28 | Children crossing | 150 | 95.95% | 94.67% | 95.30% |
| 29 | Bicycles crossing | 90 | 86.14% | 96.67% | 91.10% |
| 30 | Beware of ice/snow | 150 | 92.80% | 77.33% | 84.36% |
| 31 | Wild animals crossing | 270 | 97.37% | 95.93% | 96.64% |
| 32 | End speed + passing limits | 60 | 98.28% | 95.00% | 96.61% |
| 33 | Turn right ahead | 210 | 99.05% | 99.05% | 99.05% |
| 34 | Turn left ahead | 120 | 95.97% | 99.17% | 97.54% |
| 35 | Ahead only | 390 | 98.47% | 98.97% | 98.72% |
| 36 | Go straight or right | 120 | 98.31% | 96.67% | 97.48% |
| 37 | Go straight or left | 60 | 93.65% | 98.33% | 95.93% |
| 38 | Keep right | 690 | 97.30% | 99.28% | 98.28% |
| 39 | Keep left | 90 | 95.60% | 96.67% | 96.13% |
| 40 | Roundabout mandatory | 90 | 92.55% | 96.67% | 94.57% |
| 41 | End of no passing | 60 | 85.07% | 95.00% | 89.76% |
| 42 | End no passing veh > 3.5 tons | 90 | 88.78% | 96.67% | 92.55% |

The lowest F1 is class 27 (Pedestrians): 27 of 60 test images are correct, and 10 of the errors are predicted as class 29 (Bicycles crossing). The highest F1 is class 14 (Stop), 99.63%, with 269 of 270 correct.

### Largest off-diagonal counts

These are the most frequent mistakes. Each count is one cell of the confusion matrix.

| Count | True class | Predicted class |
| ---: | --- | --- |
| 47 | 7 Speed limit (100km/h) | 8 Speed limit (120km/h) |
| 27 | 5 Speed limit (80km/h) | 2 Speed limit (50km/h) |
| 22 | 3 Speed limit (60km/h) | 5 Speed limit (80km/h) |
| 19 | 7 Speed limit (100km/h) | 5 Speed limit (80km/h) |
| 13 | 5 Speed limit (80km/h) | 3 Speed limit (60km/h) |
| 10 | 27 Pedestrians | 29 Bicycles crossing |
| 10 | 9 No passing | 3 Speed limit (60km/h) |
| 10 | 6 End of speed limit (80km/h) | 38 Keep right |
| 9 | 30 Beware of ice/snow | 11 Right-of-way at intersection |
| 9 | 8 Speed limit (120km/h) | 3 Speed limit (60km/h) |

### Confusion matrix

Rows are the true class, columns are the predicted class. The first column of each row is the true class id. The remaining 43 numbers are counts for predicted classes 0 through 42, in that order. The diagonal sums to 12,082 and the whole matrix sums to 12,630.

```
        0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42
   0   57   0   0   0   0   0   0   0   3   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   1    1 704   5   0   7   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   1   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   2    0   9 728   2   2   7   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0
   3    0   0   0 426   0  22   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0
   4    0   3   2   5 642   2   0   3   1   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0
   5    0   3  27  13   4 574   0   2   0   0   2   0   0   2   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   1   1   0   0   0   0   0   0   0   0   0   0   0
   6    0   0   0   0   0   0 129   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0  10   0   0   1   9
   7    0   0   0   3   0  19   0 380  47   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
   8    0   0   0   9   1   4   0   2 430   3   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0
   9    0   0   0  10   0   0   0   0   0 465   5   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  10    0   0   0   0   1   8   0   1   1   2 645   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0
  11    0   0   0   0   0   0   0   0   0   0   0 413   0   0   0   0   0   0   3   0   1   1   0   0   1   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  12    0   0   0   0   0   0   0   0   0   0   0   0 678   1   0   7   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   1   0   0   0   2   0   0
  13    0   0   0   0   0   0   0   0   0   0   0   0   2 717   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  14    0   0   0   0   0   0   0   0   0   0   0   0   0   0 269   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  15    0   0   0   0   0   0   0   0   0   0   0   0   0   1   0 209   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  16    0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0 149   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  17    0   0   0   0   0   0   0   0   0   0   1   0   1   1   0   0   0 342   2   0   0   0   4   0   0   0   0   0   0   0   0   0   0   1   0   0   0   4   1   3   0   0   0
  18    2   0   0   2   2   0   0   0   1   0   0   0   0   0   0   0   0   0 366   0   1   0   0   1   2   1   6   0   1   0   3   0   0   0   0   0   0   0   0   0   1   1   0
  19    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  60   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  20    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  90   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  21    0   0   5   0   0   0   0   0   0   0   0   5   0   0   0   0   0   0   0   0   0  75   0   0   0   0   0   0   1   0   0   1   0   0   0   0   0   0   0   0   0   3   0
  22    0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 115   0   0   0   1   1   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  23    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0 146   0   0   0   0   0   0   2   1   0   0   0   0   0   0   0   0   0   0   0
  24    0   0   0   0   0   0   0   0   0   0   0   2   0   1   0   0   0   0   0   0   0   0   0   0  81   0   3   2   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0
  25    0   0   1   0   5   0   0   0   0   0   0   8   0   0   0   0   0   0   0   0   0   0   0   0   1 462   1   2   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
  26    0   1   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   2   2   0   1   0   8   0   0   8 151   0   0   0   0   3   0   0   0   0   0   0   0   0   0   0   2
  27    0   3   0   0   0   0   0   0   0   1   0   5   0   2   0   0   0   0   5   0   0   0   0   6   0   0   1  27   0  10   0   0   0   0   0   0   0   0   0   0   0   0   0
  28    0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   3   0   0   0 142   1   2   0   0   0   0   0   0   0   0   0   0   0   0
  29    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   3   0   0   0   0   0   0  87   0   0   0   0   0   0   0   0   0   0   0   0   0
  30    0   1   0   0   0   0   0   0   1   0   2   9   0   1   0   0   0   0   1   0   4   0   0   2   0   3   3   0   3   3 116   0   0   0   0   0   0   0   1   0   0   0   0
  31    0   0   0   0   0   1   0   0   0   0   1   0   0   0   0   0   0   0   0   0   2   3   0   3   0   0   0   0   0   0   1 259   0   0   0   0   0   0   0   0   0   0   0
  32    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  57   0   0   0   0   0   0   0   1   2   0
  33    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 208   0   1   1   0   0   0   0   0   0
  34    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 119   0   0   0   0   0   1   0   0
  35    0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2 386   0   0   0   0   0   0   0
  36    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 116   0   4   0   0   0   0
  37    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0  59   0   0   0   0   0
  38    0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0 685   1   0   0   0
  39    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   1   0   0   0  87   1   0   0
  40    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   2   0   0   0   0  87   0   0
  41    0   0   0   0   0   0   0   0   0   3   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  57   0
  42    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   3  87
```

---

## Implemented Components

### Layers

* Conv2D
* Dense
* MaxPool2D
* Flatten
* Dropout
* ReLU

### Loss

* Cross Entropy

### Optimizer

* Stochastic Gradient Descent (SGD)

### Utilities

* Tensor class
* Serialization
* Dataset loading
* Evaluation

---

## Technologies

* C++
* Python
* pybind11
* OpenCV
* NumPy
* pandas
* CMake

---

## Future Improvements

* Adam optimizer
* Data augmentation
* GPU acceleration
* Additional activation functions
* Residual architectures

---

## License

This project was developed for educational purposes as part of a university course in Artificial Intelligence.
