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
* SGD with momentum and decoupled AdamW
* Batch normalization and train-time augmentation
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
│   ├── Layers/           # Conv, Dense, BatchNorm, Dropout, and the rest
│   ├── Loss/
│   ├── Optimizers/       # SGD and AdamW
│   ├── Utils/            # Tensor, GEMM, serialization
│   └── bindings.cpp
├── model.py              # Training loop
├── ablate.py             # Augmentation, BatchNorm, and AdamW ablations
├── eval.py               # Official-test metrics
├── reference/pytorch_cnn.py
├── bench/bench_cpu.py
└── CMakeLists.txt
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

Ablation variants C, D, and E insert `BatchNorm2D` after each convolution and before ReLU. Variants B, D, and E augment training batches before normalization. Validation and test images are never augmented.

---

## Dataset

The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** consisting of **43 traffic sign classes**.

Images are:

* resized to **32×32**
* converted to RGB
* scaled to `[0, 1]` and stored as `(N, C, H, W)`
* standardized with the training-split mean and standard deviation, including at test time

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

The best-performing model is saved as `best_model.bin`, with `checkpoints/norm.npz` and `checkpoints/config.json`. `python ablate.py` trains variants A–E on seeds 42–46 and writes each run under `checkpoints/<variant>/seed<N>/`. A finished seed is skipped when its `metrics.json` already exists. Variant E is AdamW on the augmented BatchNorm model; A–D use SGD.

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

`eval.py` runs the same protocol and writes `results/test_metrics.json`, `results/per_class.csv`, `results/confusion.csv`, `results/confusion.png`, and `results/top_confusions.csv`.

---

## Evaluation

Official test images are standardized with the training-split mean and standard deviation. An earlier README table reported 95.66% (12,082 / 12,630). That number used the test set's own statistics and is not the current protocol.

One SGD run without augmentation or BatchNorm, evaluated with the corrected protocol, scores 94.96% (11,994 / 12,630) and a macro F1 of 92.57%. The same weights under test-set normalization score 94.88% (11,983 / 12,630).

The ablation below is the result to quote. Each cell is the mean ± sample standard deviation over seeds 42–46. A–D use SGD (learning rate 0.015, momentum 0.95, weight decay 1e-4). E uses AdamW (learning rate 1e-3, weight decay 1e-2). Every run except A seed 42 lasted 10 epochs; that one seed lasted 20. Its accuracy, 95.27%, sits inside the range of the other A seeds, so the A mean barely moves if it is left out. Validation and test images are not augmented. The headline is the mean, not the best seed.

| Variant | Augmentation | BatchNorm | Optimizer | Accuracy | Macro F1 | Class 27 recall | 7→8 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| A | no | no | SGD | 95.06 ± 0.27 | 92.83 ± 0.47 | 48.7 ± 4.2 | 37.2 ± 9.6 |
| B | yes | no | SGD | 94.92 ± 0.60 | 93.17 ± 0.94 | 61.3 ± 16.1 | 55.0 ± 13.8 |
| C | no | yes | SGD | 94.97 ± 0.31 | 92.93 ± 0.21 | 70.7 ± 15.9 | 39.2 ± 14.8 |
| D | yes | yes | SGD | 93.81 ± 0.75 | 92.48 ± 0.59 | 59.3 ± 11.6 | 82.4 ± 21.9 |
| E | yes | yes | AdamW | 90.64 ± 1.75 | 88.83 ± 2.01 | 59.0 ± 17.1 | 81.2 ± 17.8 |

Per-seed metrics live in `checkpoints/<variant>/seed<N>/metrics.json`. The aggregate is `results/ablation.json`.

Forward activations and parameter gradients match a PyTorch twin of the same network to within `1e-4` (`python reference/pytorch_cnn.py`). Matrix products use a tiled kernel when there are enough 16×16 tiles, and OpenBLAS otherwise. Training sets `OMP_NUM_THREADS` to 4 unless that variable is already set, so the convolution weight gradient stays on the performance cores.

---

## Implemented Components

### Layers

* Conv2D
* BatchNorm2D
* Dense
* MaxPool2D
* Flatten
* Dropout
* ReLU
* SoftMax

### Loss

* Cross entropy on probabilities (the network already applies softmax)

### Optimizers

* SGD with momentum and L2 weight decay
* AdamW with decoupled weight decay

### Utilities

* Tensor class, tiled GEMM, and OpenBLAS
* Serialization
* Dataset loading and group-aware split
* Official-test evaluation and the ablation runner

---

## Technologies

* C++
* Python
* pybind11
* OpenCV
* NumPy
* pandas
* PyTorch (numeric reference only)
* OpenBLAS
* CMake

---

## Future Improvements

* AVX2 microkernel and fewer activation copies
* Grad-CAM and a gallery of the worst test errors
* GPU acceleration
* More than 10 epochs for the AdamW run, at the learning rate used above
* Residual architectures

---

## License

This project was developed for educational purposes as part of a university course in Artificial Intelligence.

