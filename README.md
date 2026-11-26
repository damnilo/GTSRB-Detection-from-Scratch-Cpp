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
├── .github/workflows/cpu.yml # CPU build and the PyTorch reference tests
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

The CUDA command needs a module built with `nvcc`. A checkpoint without BatchNorm writes `results/test_metrics_cuda.json`. A BatchNorm checkpoint writes `results/test_metrics_cuda_bn.json`. The first batch must match the CPU logits to `1e-4` before the official test runs. On a Tesla T4, `results/test_metrics_cuda.json` records 95.273% (12,033 / 12,630) and 27,096.176 images/s for `checkpoints/A/seed42`. `python gradcam.py` writes `results/failures.md` and the panels in `results/gradcam/`. The gallery checked in with this repository is checkpoint `checkpoints/C/seed42`: 679 of 12,630 test images are wrong, 38 of them read the 100 km/h sign as 120 km/h, and 14 read pedestrians as bicycles crossing.

---

## Evaluation

Official test images are standardized with the training-split mean and standard deviation. An earlier README table reported 95.66% (12,082 / 12,630). That number used the test set's own statistics and is not the current protocol.

One SGD run without augmentation or BatchNorm, evaluated with the corrected protocol, scores 94.96% (11,994 / 12,630) and a macro F1 of 92.57%. The same weights under test-set normalization score 94.88% (11,983 / 12,630).

The 10-epoch ablation below is the result to quote. Each cell is the mean ± sample standard deviation over seeds 42–46. A–D use SGD (learning rate 0.015, momentum 0.95, weight decay 1e-4). E uses AdamW (learning rate 1e-3, weight decay 1e-2). Every run except A seed 42 lasted 10 epochs; that one seed lasted 20. Its accuracy, 95.27%, sits inside the range of the other A seeds, so the A mean barely moves if it is left out. Validation and test images are not augmented. The headline is that A mean, 95.06 ± 0.27, not the best seed.

| Variant | Augmentation | BatchNorm | Optimizer | Accuracy | Macro F1 | Class 27 recall | 7→8 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| A | no | no | SGD | 95.06 ± 0.27 | 92.83 ± 0.47 | 48.7 ± 4.2 | 37.2 ± 9.6 |
| B | yes | no | SGD | 94.92 ± 0.60 | 93.17 ± 0.94 | 61.3 ± 16.1 | 55.0 ± 13.8 |
| C | no | yes | SGD | 94.97 ± 0.31 | 92.93 ± 0.21 | 70.7 ± 15.9 | 39.2 ± 14.8 |
| D | yes | yes | SGD | 93.81 ± 0.75 | 92.48 ± 0.59 | 59.3 ± 11.6 | 82.4 ± 21.9 |
| E | yes | yes | AdamW | 90.64 ± 1.75 | 88.83 ± 2.01 | 59.0 ± 17.1 | 81.2 ± 17.8 |

The same five variants and the same five seeds were trained again for 20 epochs each. `python ablate.py 20` wrote `results/ablation_e20.json`.

| Variant | Augmentation | BatchNorm | Optimizer | Accuracy | Macro F1 | Class 27 recall | 7→8 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| A | no | no | SGD | 95.65 ± 0.30 | 93.69 ± 0.73 | 61.0 ± 17.5 | 43.4 ± 4.9 |
| B | yes | no | SGD | 95.46 ± 0.53 | 93.70 ± 0.80 | 49.0 ± 1.5 | 48.2 ± 6.9 |
| C | no | yes | SGD | 95.59 ± 0.15 | 93.30 ± 0.42 | 62.3 ± 16.0 | 42.0 ± 6.6 |
| D | yes | yes | SGD | 95.92 ± 0.64 | 94.34 ± 1.01 | 57.7 ± 12.3 | 53.2 ± 4.1 |
| E | yes | yes | AdamW | 92.60 ± 1.17 | 90.37 ± 1.30 | 46.3 ± 3.2 | 68.0 ± 16.3 |

D has the highest accuracy mean at 20 epochs, 0.26 points above A, with a standard deviation of 0.64. The 100 km/h sign is still read as 120 km/h more often for D (53.2) than for A (43.4). E remains last. The headline stays the 10-epoch A mean.

Per-seed metrics for the 10-epoch runs live in `checkpoints/<variant>/seed<N>/metrics.json`. That aggregate is `results/ablation.json`. The 20-epoch aggregate is `results/ablation_e20.json`.

`python robustness.py` repeats the official test of `checkpoints/A/seed42` after a fixed change to every test image. Pixels stay in [0, 1], the change is clipped back into that range, and the checkpoint's training-set mean and standard deviation are applied afterwards. Brightness adds a constant. Contrast is `(pixel - 0.5) * alpha + 0.5`. Blur is a Gaussian kernel; sigma 0 lets the 3×3 kernel set the width. ECE is the 15-bin expected calibration error of the softmax probability on the predicted class. This is one seed, not the five-seed mean above. The clean row is the same 95.27% (12,033 / 12,630) as that checkpoint's `metrics.json`.

| Condition | Accuracy | ECE | 7→8 |
| --- | ---: | ---: | ---: |
| clean | 95.27 | 2.65 | 40 |
| brightness −0.2 | 76.74 | 13.00 | 37 |
| brightness −0.1 | 91.31 | 5.10 | 54 |
| brightness +0.1 | 94.35 | 2.61 | 33 |
| brightness +0.2 | 91.50 | 3.20 | 20 |
| contrast 0.6 | 91.35 | 0.78 | 20 |
| contrast 0.8 | 94.90 | 1.88 | 35 |
| contrast 1.2 | 93.41 | 3.99 | 56 |
| contrast 1.4 | 86.10 | 8.50 | 43 |
| blur 3×3 | 93.06 | 3.38 | 40 |
| blur 5×5, sigma 1 | 90.15 | 4.51 | 37 |

Darkening every test image by 0.2 drops this seed to 76.74%. The same shift brighter costs 0.92 points. The full table is `results/robustness_a_seed42.json`.

Forward activations and parameter gradients match a PyTorch twin of the same network to within `1e-4` (`python -m pytest tests/test_reference.py`). The same test runs on GitHub Actions from `.github/workflows/cpu.yml`, without CUDA and without the dataset. Matrix products use a tiled kernel when there are enough 16×16 tiles, and OpenBLAS otherwise. Training sets `OMP_NUM_THREADS` to 4 unless that variable is already set, so the convolution weight gradient stays on the performance cores.

---

## CPU benchmark

`python bench/bench_cpu.py` writes `results/bench_cpu.md`. The run below used 4 threads on Intel64 Family 6 Model 197 Stepping 2. Each cell is the median of 10 timings after 5 warmups. Dropout is off, and the inputs are built outside the timer.

At batch 64 the full C++ forward takes 30.933 ms (2,069 images/s). The PyTorch twin takes 4.672 ms (13,699 images/s). The C++ network is slower on this CPU. The same file has the batch 1 and 16 rows, and separate rows for each convolution and dense layer.

---

## CUDA

The device path is checked by `cpp_bench/bench_eval` and by `python eval.py --checkpoint checkpoints/A/seed42 --device cuda`. Numbers below are from a Tesla T4. Tiled GEMM and cuBLAS matched the CPU product to `1e-4`. The custom tiled GEMM is not faster than cuBLAS on either convolution shape. A later session rewrote the timing table in `results/bench_cuda.md` with slower rows for every backend. That shift is not copied here.

| shape | backend | ms | img/s |
| --- | --- | ---: | ---: |
| (65536, 27) x (27, 16) | tiled GEMM | 0.154 | 415368.625 |
| (65536, 27) x (27, 16) | cuBLAS | 0.092 | 699056.312 |
| (65536, 27) x (27, 16) | CPU matmul | 10.271 | 6230.892 |
| (65536, 27) x (27, 16) | tiled GEMM + copies | 2.562 | 24982.824 |
| (16384, 144) x (144, 32) | tiled GEMM | 0.183 | 348979.219 |
| (16384, 144) x (144, 32) | cuBLAS | 0.096 | 669120.125 |
| (16384, 144) x (144, 32) | CPU matmul | 27.081 | 2363.273 |
| (16384, 144) x (144, 32) | tiled GEMM + copies | 2.775 | 23064.592 |

Model A, batch 64, no BatchNorm, dropout off. Logits are the last dense output, before SoftMax. They matched the CPU stack, as did the backward pass and one SGD step, to `1e-4` of the result size. Median of 10 runs after one warmup.

| path | ms | img/s |
| --- | ---: | ---: |
| kernels on device | 3.020 | 21192.053 |
| one batch copy | 3.259 | 19638.650 |

The CPU matmul row includes allocating the result. On the multiply alone, copies dominate. On the full forward they do not: one upload of the batch and one download of the logits adds about 0.2 ms.

One device epoch over `dataset/Train`, classes in order 0 to 42 with filenames sorted, dropout off, no augmentation, and the global gradient norm clipped at 5, finished 39,209 images in 613 batches. Mean loss was 24.8015. The epoch took 12.967 s. Softmax and the loss stayed on the host. That sentence is in `results/bench_cuda.md`.

The official test of `checkpoints/A/seed42` is in `results/test_metrics_cuda.json`. Accuracy is 95.273% (12,033 / 12,630) and macro F1 is 92.629%, the same values as that checkpoint's CPU `metrics.json`. The first batch matched the CPU logits to `1e-4`. The timed pass took 466.117 ms (27,096.176 images/s): every test batch is uploaded, the kernels run, and the logits are downloaded, with the weights already on the device. The table above is the median of ten batch-64 runs.

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
* Residual architectures

---

## License

This project was developed for educational purposes as part of a university course in Artificial Intelligence.

