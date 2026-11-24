# CUDA conv GEMM

GPU: Tesla T4.

Batch 64, kernel 3, stride 1, padding 1. Each row is the forward product of the unrolled image and the transposed weights. Median of 10 runs after one warmup. Each shape has one row that also sends both inputs to the device and the product back. Tiled GEMM and cuBLAS matched the CPU product within 1e-4.

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

## Model A eval

GPU: Tesla T4.

Model A, batch 64, no BatchNorm, dropout off. Logits are the last dense output, before SoftMax. They matched the CPU stack within 1e-4 of the result size. Median of 10 runs after one warmup. Resident keeps the batch and the weights on the device. Transfer sends the batch once, runs the same kernels, and reads the logits back once.

| path | ms | img/s |
| --- | ---: | ---: |
| kernels on device | 3.020 | 21192.053 |
| one batch copy | 3.259 | 19638.650 |

BatchNorm2D train and eval, batch 64, matched the CPU layer within 1e-4 for 16 channels at 32x32 and 32 channels at 16x16.
Network::backwardDevice, from the logit gradient and stopping before SoftMax, matched the CPU stack within 1e-4.
