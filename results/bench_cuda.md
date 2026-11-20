# CUDA conv GEMM

GPU: Tesla T4.

Batch 64, kernel 3, stride 1, padding 1. Each row is the forward product of the unrolled image and the transposed weights. Median of 10 runs after one warmup. Each shape has one row that also sends both inputs to the device and the product back. Tiled GEMM and cuBLAS matched the CPU product within 1e-4.

| shape | backend | ms | img/s |
| --- | --- | ---: | ---: |
| (65536, 27) x (27, 16) | tiled GEMM | 0.151 | 424088.219 |
| (65536, 27) x (27, 16) | cuBLAS | 0.092 | 699300.750 |
| (65536, 27) x (27, 16) | CPU matmul | 115.442 | 554.393 |
| (65536, 27) x (27, 16) | tiled GEMM + copies | 2.925 | 21877.529 |
| (16384, 144) x (144, 32) | tiled GEMM | 0.292 | 219154.062 |
| (16384, 144) x (144, 32) | cuBLAS | 0.126 | 508001.031 |
| (16384, 144) x (144, 32) | CPU matmul | 298.500 | 214.405 |
| (16384, 144) x (144, 32) | tiled GEMM + copies | 2.879 | 22231.113 |
