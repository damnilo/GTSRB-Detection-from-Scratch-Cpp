# GTSRB CNN from scratch

The network is a 43-class classifier for the German Traffic Sign Recognition Benchmark. Convolution, pooling, batch normalization, dropout, SGD, AdamW, and cross-entropy are implemented in C++17 and exposed to Python with pybind11. The number quoted for the original training recipe is variant A at 10 epochs: **95.06 ± 0.27%** official-test accuracy, the mean ± sample standard deviation over seeds 42–46. That is not the best seed, and it is not the later follow-up variant F.

## Data protocol

Images are resized to 32×32, converted to RGB, scaled to [0, 1], and stored as NCHW. Training and validation come from a group-aware split, so frames from one recording track do not sit in both sets. Official-test images use the training-split per-channel mean and standard deviation. An earlier figure of 95.66% (12,082 / 12,630) used the test set's own statistics and is not this protocol.

One SGD run without augmentation or batch normalization, under the corrected protocol, scores 94.96% (11,994 / 12,630) and a macro F1 of 92.57%. The same weights with test-set normalization score 94.88% (11,983 / 12,630).

Validation and test images are never augmented. Training augmentation, when it is on, jitters a batch in [0, 1] and only then applies the training-set mean and standard deviation.

## Model and training

The stack is Conv 3→16, ReLU, MaxPool 2, Conv 16→32, ReLU, MaxPool 2, Flatten to 2048, Dense 2048→128, ReLU, Dropout 0.3, Dense 128→43, SoftMax. Variants with batch normalization insert it after each convolution and before ReLU. The loss is mean cross-entropy on probabilities, because the network already applies softmax.

A–D use SGD with learning rate 0.015, momentum 0.95, and weight decay 1e-4. E uses AdamW with learning rate 1e-3 and weight decay 1e-2. Gradients are clipped by a global norm of 5 before the optimizer step. Class imbalance uses weighted sampling with replacement. The learning rate is halved when validation accuracy does not improve. Dropout is an identity at evaluation time.

The full augmentation rotates by up to 15 degrees, scales, shifts, sometimes changes brightness, and sometimes applies a 3×3 blur. The mild schedule used by variant F rotates by at most 5 degrees and does not blur.

## Ablations

Each cell below is the mean ± sample standard deviation over seeds 42–46, except the single G seed. Sources are `results/ablation.json`, `results/ablation_e20.json`, and `results/ablation_followup.json`.

Ten epochs. A seed 42 in this file lasted 20 epochs and scored 95.27%, inside the range of the other A seeds.

| Variant | Augmentation | BatchNorm | Optimizer | Accuracy | Macro F1 | Class 27 recall | 7→8 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| A | no | no | SGD | 95.06 ± 0.27 | 92.83 ± 0.47 | 48.7 ± 4.2 | 37.2 ± 9.6 |
| B | yes | no | SGD | 94.92 ± 0.60 | 93.17 ± 0.94 | 61.3 ± 16.1 | 55.0 ± 13.8 |
| C | no | yes | SGD | 94.97 ± 0.31 | 92.93 ± 0.21 | 70.7 ± 15.9 | 39.2 ± 14.8 |
| D | yes | yes | SGD | 93.81 ± 0.75 | 92.48 ± 0.59 | 59.3 ± 11.6 | 82.4 ± 21.9 |
| E | yes | yes | AdamW | 90.64 ± 1.75 | 88.83 ± 2.01 | 59.0 ± 17.1 | 81.2 ± 17.8 |

Twenty epochs, same five variants and the same full augmentation.

| Variant | Augmentation | BatchNorm | Optimizer | Accuracy | Macro F1 | Class 27 recall | 7→8 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| A | no | no | SGD | 95.65 ± 0.30 | 93.69 ± 0.73 | 61.0 ± 17.5 | 43.4 ± 4.9 |
| B | yes | no | SGD | 95.46 ± 0.53 | 93.70 ± 0.80 | 49.0 ± 1.5 | 48.2 ± 6.9 |
| C | no | yes | SGD | 95.59 ± 0.15 | 93.30 ± 0.42 | 62.3 ± 16.0 | 42.0 ± 6.6 |
| D | yes | yes | SGD | 95.92 ± 0.64 | 94.34 ± 1.01 | 57.7 ± 12.3 | 53.2 ± 4.1 |
| E | yes | yes | AdamW | 92.60 ± 1.17 | 90.37 ± 1.30 | 46.3 ± 3.2 | 68.0 ± 16.3 |

Among A–E at 20 epochs, D has the highest accuracy mean, 0.26 points above A, with a standard deviation of 0.64. The 100 km/h sign is still read as 120 km/h more often for D (53.2) than for A (43.4). E remains last in that table.

F repeats D for 20 epochs with the mild augmentation. G repeats E for seed 42 only, for 20 epochs, with the AdamW learning rate set to 5e-4. G is not a five-seed mean.

| Variant | Augmentation | BatchNorm | Optimizer | Accuracy | Macro F1 | Class 27 recall | 7→8 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| F | mild | yes | SGD | 96.49 ± 0.41 | 95.04 ± 1.11 | 61.7 ± 12.0 | 18.4 ± 12.0 |
| G | yes | yes | AdamW, 5e-4 | 92.66 | 90.84 | 56.7 | 47 |

F is 0.84 points above the 20-epoch A mean and 0.57 points above D. The 100 km/h sign is read as 120 km/h 18.4 times, compared with 43.4 for A and 53.2 for D. G seed 42 scores 92.66%, next to the E mean of 92.60 ± 1.17. The quoted headline stays the 10-epoch A mean.

## Reference checks

`python -m pytest tests/test_reference.py` compares the C++ network with a PyTorch twin. Forward activations and parameter gradients match to 1e-4. Conv2D and Dense also pass a central finite-difference check. A softmax row sums to 1, dropout in eval mode is the identity, and a save/load roundtrip restores the parameters. The same test runs on GitHub Actions from `.github/workflows/cpu.yml`, without CUDA and without the dataset. BatchNorm does not have a finite-difference test.

## CPU speed

`results/bench_cpu.md` records 4 threads on Intel64 Family 6 Model 197 Stepping 2. At batch 64 the C++ forward takes 30.933 ms (2,069 images/s). The PyTorch twin takes 4.672 ms (13,699 images/s). The C++ network is slower on this CPU.

## CUDA

The device numbers below are from a Tesla T4. Tiled GEMM and cuBLAS matched the CPU product to 1e-4. The custom tiled GEMM is not faster than cuBLAS. A later session rewrote `results/bench_cuda.md` with slower rows for every backend. That shift is not copied here. The table is the earlier median.

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

Model A, batch 64, no batch normalization, dropout off: kernels resident **3.020 ms / 21,192.053 images/s**, and one batch copy **3.259 ms / 19,638.650 images/s**.

The official test of `checkpoints/A/seed42` is in `results/test_metrics_cuda.json`: 95.273% (12,033 / 12,630), macro F1 92.629%, 466.117 ms, 27,096.176 images/s. Those accuracy figures match that checkpoint's CPU `metrics.json`. The first batch matched the CPU logits to 1e-4.

The official test of `checkpoints/C/seed42`, which uses batch normalization, is in `results/test_metrics_cuda_bn.json`: 94.624% (11,951 / 12,630), macro F1 92.610%, 606.930 ms, 20,809.659 images/s. Those accuracy figures match that checkpoint's CPU `metrics.json`. The first batch matched to 1e-4.

One device epoch over `dataset/Train` finished 39,209 images in 613 batches. Mean loss was 24.8015. The epoch took 12.967 s. Classes were in order 0 to 42 with filenames sorted, dropout was off, there was no augmentation and no batch normalization, and the global gradient norm was clipped at 5. Softmax and the loss stayed on the host. The loss is not the loss of a shuffled `model.py` epoch.

When the Python module is built with nvcc, `model.py` trains on the device. Augmentation, the permutation, and the learning-rate schedule stay on the host.

## Robustness of one checkpoint

`results/robustness_a_seed42.json` repeats the official test of `checkpoints/A/seed42` after a fixed change in [0, 1], clipped, then standardized with that checkpoint's training statistics. ECE is the 15-bin expected calibration error of the softmax probability on the predicted class. This is one seed.

The clean row is 95.27% (12,033 / 12,630), ECE 2.65%, and 40 confusions of class 7 into class 8. Darkening every test image by 0.2 drops the same seed to 76.74% with ECE 13.00%. The same shift brighter scores 91.50%. A 3×3 blur scores 93.06%. The full condition list is in that JSON file and in the README.

## What these measurements do not say

The custom GEMM is not faster than cuBLAS. The C++ CPU forward is not faster than the PyTorch twin. Variant E and the single lower-learning-rate seed G do not catch A. F is a separate 20-epoch schedule, and the quoted headline remains the 10-epoch A mean until that choice is changed. One device epoch with sorted class order is not a training run of `model.py`.

## Reproduce

Build the extension with CMake, then run `python -m pytest tests/test_reference.py`. `python ablate.py` trains A–E. `python ablate.py 20` repeats them for 20 epochs. `python ablate.py F 20` and `python ablate.py G 42 20` write `results/ablation_followup.json`.

The CPU image has no CUDA and no dataset:

```bash
docker build -t gtsrb-cnn .
docker run --rm gtsrb-cnn
```

The second command runs the reference tests. Mount a GTSRB tree at `/src/dataset` before training or official-test evaluation.
