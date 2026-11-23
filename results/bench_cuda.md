# CUDA conv GEMM

GPU: Tesla T4.

Batch 64, kernel 3, stride 1, padding 1. Each row is the forward product of the unrolled image and the transposed weights. Median of 10 runs after one warmup. Each shape has one row that also sends both inputs to the device and the product back. Tiled GEMM and cuBLAS matched the CPU product within 1e-4.

| shape | backend | ms | img/s |
| --- | --- | ---: | ---: |
| (65536, 27) x (27, 16) | tiled GEMM | 0.208 | 307172.469 |
| (65536, 27) x (27, 16) | cuBLAS | 0.117 | 548395.938 |
| (65536, 27) x (27, 16) | CPU matmul | 10.536 | 6074.292 |
| (65536, 27) x (27, 16) | tiled GEMM + copies | 2.679 | 23888.582 |
| (16384, 144) x (144, 32) | tiled GEMM | 0.250 | 256180.344 |
| (16384, 144) x (144, 32) | cuBLAS | 0.113 | 568181.812 |
| (16384, 144) x (144, 32) | CPU matmul | 27.010 | 2369.455 |
| (16384, 144) x (144, 32) | tiled GEMM + copies | 2.798 | 22877.537 |

## Model A eval

GPU: Tesla T4.

Model A, batch 64, no BatchNorm, dropout off. Logits are the last dense output, before SoftMax. They matched the CPU stack within 1e-4 of the result size. Median of 10 runs after one warmup. Resident keeps the batch and the weights on the device. Transfer sends the batch once, runs the same kernels, and reads the logits back once.

| path | ms | img/s |
| --- | ---: | ---: |
| kernels on device | 3.983 | 16066.838 |
| one batch copy | 4.235 | 15112.589 |

BatchNorm2D train and eval, batch 64, matched the CPU layer within 1e-4 for 16 channels at 32x32 and 32 channels at 16x16.
