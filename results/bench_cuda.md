# CUDA conv GEMM

GPU: Tesla T4.

Batch 64, kernel 3, stride 1, padding 1. Each row is the forward product of the unrolled image and the transposed weights. Median of 10 runs after one warmup. Each shape has one row that also sends both inputs to the device and the product back. Tiled GEMM and cuBLAS matched the CPU product within 1e-4.

| shape | backend | ms | img/s |
| --- | --- | ---: | ---: |
| (65536, 27) x (27, 16) | tiled GEMM | 0.139 | 459347.719 |
| (65536, 27) x (27, 16) | cuBLAS | 0.086 | 745156.500 |
| (65536, 27) x (27, 16) | CPU matmul | 12.002 | 5332.309 |
| (65536, 27) x (27, 16) | tiled GEMM + copies | 2.667 | 23995.490 |
| (16384, 144) x (144, 32) | tiled GEMM | 0.166 | 386025.844 |
| (16384, 144) x (144, 32) | cuBLAS | 0.092 | 694444.438 |
| (16384, 144) x (144, 32) | CPU matmul | 27.908 | 2293.248 |
| (16384, 144) x (144, 32) | tiled GEMM + copies | 2.724 | 23498.449 |

## Model A eval

GPU: Tesla T4.

Model A, batch 64, no BatchNorm, dropout off. Logits are the last dense output, before SoftMax. They matched the CPU stack within 1e-4 of the result size. Median of 10 runs after one warmup. Resident keeps the batch and the weights on the device. Transfer sends the batch once, runs the same kernels, and reads the logits back once.

| path | ms | img/s |
| --- | ---: | ---: |
| kernels on device | 2.740 | 23357.119 |
| one batch copy | 2.978 | 21492.203 |

BatchNorm2D train and eval, batch 64, matched the CPU layer within 1e-4 for 16 channels at 32x32 and 32 channels at 16x16.
Device backward of Conv2D, ReLU, MaxPool2D, and Dense matched the CPU stack within 1e-4.
