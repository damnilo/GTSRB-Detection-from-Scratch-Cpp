# CPU network

CPU: Intel64 Family 6 Model 197 Stepping 2, GenuineIntel.

Threads: 4. Warmup 5, then the median of 10. 
Dropout is off. Inputs are built once, outside the timer. 
The full row counts the four matmuls inside one forward. 
A layer row is that convolution or dense layer alone.

| case | batch | cpp ms | torch ms | cpp img/s | torch img/s | cpp GFLOP/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 1 | 1.149 | 0.136 | 870.4 | 7363.8 | 3.29 |
| conv1 | 1 | 0.355 | 0.021 | 2817.7 | 48192.8 | 2.49 |
| conv2 | 1 | 0.359 | 0.026 | 2789.0 | 38461.5 | 6.58 |
| fc1 | 1 | 0.045 | 0.008 | 22346.4 | 131579.0 | 11.72 |
| fc2 | 1 | 0.001 | 0.004 | 714287.7 | 256410.2 | 7.86 |
| full | 16 | 8.097 | 0.992 | 1976.1 | 16125.8 | 7.47 |
| conv1 | 16 | 2.581 | 0.080 | 6200.1 | 200000.0 | 5.49 |
| conv2 | 16 | 2.269 | 0.125 | 7052.0 | 128307.9 | 16.64 |
| fc1 | 16 | 0.109 | 0.042 | 146721.7 | 380499.4 | 76.92 |
| fc2 | 16 | 0.003 | 0.007 | 4705886.8 | 2237763.4 | 51.80 |
| full | 64 | 30.933 | 4.672 | 2069.0 | 13698.9 | 7.82 |
| conv1 | 64 | 9.694 | 0.327 | 6602.2 | 195988.4 | 5.84 |
| conv2 | 64 | 8.368 | 0.359 | 7648.2 | 178521.6 | 18.04 |
| fc1 | 64 | 0.257 | 0.132 | 249221.2 | 483931.9 | 130.66 |
| fc2 | 64 | 0.010 | 0.010 | 6597937.4 | 6095238.1 | 72.63 |
