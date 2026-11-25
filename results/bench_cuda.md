# CUDA conv GEMM

GPU: Tesla T4.

Batch 64, kernel 3, stride 1, padding 1. Each row is the forward product of the unrolled image and the transposed weights. Median of 10 runs after one warmup. Each shape has one row that also sends both inputs to the device and the product back. Tiled GEMM and cuBLAS matched the CPU product within 1e-4.

| shape | backend | ms | img/s |
| --- | --- | ---: | ---: |
| (65536, 27) x (27, 16) | tiled GEMM | 0.192 | 332502.094 |
| (65536, 27) x (27, 16) | cuBLAS | 0.109 | 589622.625 |
| (65536, 27) x (27, 16) | CPU matmul | 10.019 | 6387.765 |
| (65536, 27) x (27, 16) | tiled GEMM + copies | 2.670 | 23974.492 |
| (16384, 144) x (144, 32) | tiled GEMM | 0.266 | 240731.828 |
| (16384, 144) x (144, 32) | cuBLAS | 0.118 | 541565.125 |
| (16384, 144) x (144, 32) | CPU matmul | 26.673 | 2399.464 |
| (16384, 144) x (144, 32) | tiled GEMM + copies | 2.740 | 23356.572 |

## Model A eval

GPU: Tesla T4.

Model A, batch 64, no BatchNorm, dropout off. Logits are the last dense output, before SoftMax. They matched the CPU stack within 1e-4 of the result size. Median of 10 runs after one warmup. Resident keeps the batch and the weights on the device. Transfer sends the batch once, runs the same kernels, and reads the logits back once.

| path | ms | img/s |
| --- | ---: | ---: |
| kernels on device | 4.622 | 13847.250 |
| one batch copy | 4.856 | 13178.964 |

BatchNorm2D train and eval, batch 64, matched the CPU layer within 1e-4 for 16 channels at 32x32 and 32 channels at 16x16.
Network::backwardDevice, from the logit gradient and stopping before SoftMax, matched the CPU stack within 1e-4.
One SGD step on BatchNorm gamma and beta, velocity zero and no clip, matched the CPU update within 1e-4.
One SGD step on the model A weights and biases, velocity zero and no clip, matched the CPU update within 1e-4.
Four training batches, with velocity kept across batches and no clip, matched the CPU epoch within 1e-4.
Two AdamW steps on BatchNorm gamma and beta, moments kept and no clip, matched the CPU update within 1e-4.
Two AdamW steps on the model A weights and biases, moments kept and no clip, matched the CPU update within 1e-4.
One training step on model A, SGD, global gradient norm clipped at 5, matched the CPU update within 1e-4.
One training step on model C, AdamW, global gradient norm clipped at 5, matched the CPU update within 1e-4.
One device epoch over dataset/Train, class order 0 to 42 with filenames sorted, dropout off, no augmentation, global gradient norm clipped at 5, finished 39209 images in 613 batches. Mean loss 24.8015. 12.967 s.
