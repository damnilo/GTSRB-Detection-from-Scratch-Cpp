# Test mistakes

Checkpoint `checkpoints/C/seed42`. Dropout and BatchNorm are in eval mode.
Normalization is the training-set mean and standard deviation saved beside the checkpoint.
Grad-CAM is the ReLU after the last convolution, from the gradient of the predicted logit.

Mistakes: 679 / 12630.
Speed limit 100 read as 120 (7→8): 38.
Pedestrians read as bicycles crossing (27→29): 14.

Classes 7 and 8 are the same round speed-limit sign. At 32×32 the digits collapse, so the heatmap sits on the sign body rather than on the number.
Classes 27 and 29 are both triangular warnings with a dark figure. The resolution removes the difference between a person and a bicycle.

| file | true | predicted | confidence |
| --- | --- | --- | ---: |
| `00_true07_pred08.png` | 7 Speed limit (100km/h) | 8 Speed limit (120km/h) | 0.450 |
| `01_true07_pred08.png` | 7 Speed limit (100km/h) | 8 Speed limit (120km/h) | 0.472 |
| `02_true07_pred08.png` | 7 Speed limit (100km/h) | 8 Speed limit (120km/h) | 0.477 |
| `03_true07_pred08.png` | 7 Speed limit (100km/h) | 8 Speed limit (120km/h) | 0.481 |
| `04_true07_pred08.png` | 7 Speed limit (100km/h) | 8 Speed limit (120km/h) | 0.483 |
| `05_true07_pred08.png` | 7 Speed limit (100km/h) | 8 Speed limit (120km/h) | 0.495 |
| `06_true27_pred29.png` | 27 Pedestrians | 29 Bicycles crossing | 0.320 |
| `07_true27_pred29.png` | 27 Pedestrians | 29 Bicycles crossing | 0.483 |
| `08_true27_pred29.png` | 27 Pedestrians | 29 Bicycles crossing | 0.485 |
| `09_true27_pred29.png` | 27 Pedestrians | 29 Bicycles crossing | 0.493 |
| `10_true27_pred29.png` | 27 Pedestrians | 29 Bicycles crossing | 0.667 |
| `11_true27_pred29.png` | 27 Pedestrians | 29 Bicycles crossing | 0.724 |
| `12_true31_pred20.png` | 31 Wild animals crossing | 20 Dangerous curve right | 0.142 |
| `13_true30_pred13.png` | 30 Beware of ice/snow | 13 Yield | 0.168 |
| `14_true25_pred36.png` | 25 Road work | 36 Go straight or right | 0.202 |
| `15_true30_pred08.png` | 30 Beware of ice/snow | 8 Speed limit (120km/h) | 0.216 |
| `16_true30_pred11.png` | 30 Beware of ice/snow | 11 Right-of-way at intersection | 0.232 |
| `17_true25_pred24.png` | 25 Road work | 24 Road narrows on the right | 0.240 |
| `18_true25_pred31.png` | 25 Road work | 31 Wild animals crossing | 0.241 |
| `19_true08_pred03.png` | 8 Speed limit (120km/h) | 3 Speed limit (60km/h) | 0.246 |
| `20_true39_pred40.png` | 39 Keep left | 40 Roundabout mandatory | 0.249 |
| `21_true30_pred09.png` | 30 Beware of ice/snow | 9 No passing | 0.250 |
| `22_true18_pred26.png` | 18 General caution | 26 Traffic signals | 0.256 |
| `23_true25_pred26.png` | 25 Road work | 26 Traffic signals | 0.258 |
