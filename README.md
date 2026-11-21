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
├── cuda/                 # Device kernels
├── cpp_bench/            # CUDA checks against the CPU path and cuBLAS
├── model.py              # Training loop
├── ablate.py             # Augmentation, BatchNorm, and AdamW ablations
├── eval.py               # Official-test metrics
├── gradcam.py            # Last-convolution Grad-CAM and the mistake gallery
├── tests/test_reference.py
├── bench/bench_cpu.py
├── results/
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

The best-performing model is saved as `best_model.bin`, with `checkpoints/norm.npz` and `checkpoints/config.json`. `python ablate.py` trains variants A–E on seeds 42–46 and writes each run under `checkpoints/<variant>/seed<N>/`. A finished seed is skipped when its `metrics.json` already exists. Variant E is AdamW on the augmented BatchNorm model; A–D use SGD. `.gitignore` ignores every `best_model.bin` except `checkpoints/A/seed42/best_model.bin`. That run has no BatchNorm, so it has no `bn.npz`. `norm.npz`, `config.json`, and `metrics.json` are committed for the ablation.

---

## Testing

Evaluate the trained model on the official test dataset:

```bash
python test.py
```

`test.py` loads `best_model.bin`, standardizes images with `checkpoints/norm.npz`, disables dropout, and prints test accuracy.

```bash
python -m pytest tests/test_reference.py
```

That file is the PyTorch twin. It checks forward activations and parameter gradients to `1e-4`, a finite-difference of `Conv2D` and `Dense`, a softmax row sum of 1, dropout as the identity in eval, a save/load roundtrip, and that one recording track is not in both train and validation.

```bash
python eval.py
```

`eval.py` runs the same protocol and writes `results/test_metrics.json`, `results/per_class.csv`, `results/confusion.csv`, `results/confusion.png`, and `results/top_confusions.csv`. A checkpoint run does not overwrite that file:

```bash
python eval.py --checkpoint checkpoints/A/seed42 --device cpu
python eval.py --checkpoint checkpoints/A/seed42 --device cuda
```

The CUDA command needs a module built with `nvcc`. It refuses a BatchNorm checkpoint. The first batch must match the CPU logits to `1e-4` before the official test runs. Results go to `results/test_metrics_cuda.json`. `python gradcam.py` writes `results/failures.md` and the panels in `results/gradcam/`. The gallery checked in with this repository is checkpoint `checkpoints/C/seed42`: 679 of 12,630 test images are wrong, 38 of them read the 100 km/h sign as 120 km/h, and 14 read pedestrians as bicycles crossing.

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

Forward activations and parameter gradients match a PyTorch twin of the same network to within `1e-4` (`python -m pytest tests/test_reference.py`). Matrix products use a tiled kernel when there are enough 16×16 tiles, and OpenBLAS otherwise. Training sets `OMP_NUM_THREADS` to 4 unless that variable is already set, so the convolution weight gradient stays on the performance cores.

---

## CPU benchmark

`python bench/bench_cpu.py` writes `results/bench_cpu.md`. The run below used 4 threads on Intel64 Family 6 Model 197 Stepping 2. Each cell is the median of 10 timings after 5 warmups. Dropout is off, and the inputs are built outside the timer.

At batch 64 the full C++ forward takes 30.933 ms (2,069 images/s). The PyTorch twin takes 4.672 ms (13,699 images/s). The C++ network is slower on this CPU. The same file has the batch 1 and 16 rows, and separate rows for each convolution and dense layer.

---

## CUDA

The device path is checked by `cpp_bench/bench_eval`. It is not what `python eval.py` calls. Numbers below are from a Tesla T4, in `results/bench_cuda.md`. Tiled GEMM and cuBLAS matched the CPU product to `1e-4`. The custom tiled GEMM is not faster than cuBLAS on either convolution shape.

| shape | backend | ms | img/s |
| --- | --- | ---: | ---: |
| (65536, 27) x (27, 16) | tiled GEMM | 0.151 | 424088 |
| (65536, 27) x (27, 16) | cuBLAS | 0.092 | 699301 |
| (65536, 27) x (27, 16) | CPU matmul | 115.442 | 554 |
| (65536, 27) x (27, 16) | tiled GEMM + copies | 2.925 | 21878 |
| (16384, 144) x (144, 32) | tiled GEMM | 0.292 | 219154 |
| (16384, 144) x (144, 32) | cuBLAS | 0.126 | 508001 |
| (16384, 144) x (144, 32) | CPU matmul | 298.500 | 214 |
| (16384, 144) x (144, 32) | tiled GEMM + copies | 2.879 | 22231 |

Model A, batch 64, no BatchNorm, dropout off. Logits are the last dense output, before SoftMax. They matched the CPU stack, as did the backward pass and one SGD step, to `1e-4` of the result size. Median of 10 runs after one warmup.

| path | ms | img/s |
| --- | ---: | ---: |
| kernels on device | 3.104 | 20616.643 |
| one batch copy | 3.336 | 19185.020 |

The CPU matmul row includes allocating the result. On the multiply alone, copies dominate. On the full forward they do not: one upload of the batch and one download of the logits adds about 0.2 ms.

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
* CUDA (optional; the bench and the Python module skip it when `nvcc` is absent)
* CMake

---

## Future Improvements

* AVX2 microkernel and fewer activation copies
* Official-test accuracy of `eval.py --device cuda` on a Tesla T4 (the command exists; the number is not recorded yet)
* One GPU training epoch
* More than 10 epochs for the AdamW run, at the learning rate used above
* Residual architectures

---

## License

This project was developed for educational purposes as part of a university course in Artificial Intelligence.

