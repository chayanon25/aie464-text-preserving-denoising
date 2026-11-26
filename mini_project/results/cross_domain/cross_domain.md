# Cross-domain benchmark (zero-shot)
Rows: model trained on Source. Columns: test set. Bold = in-domain. "-" = not evaluated yet.

## PSNR (dB)

| Model | Source | Kaggle test | Augraphy test | DIBCO test |
|---|---|---|---|---|
| SmallUNet (edge+aug) | Kaggle | **31.19** | 17.48 | 12.80 |
| SmallUNet (edge+aug) | Augraphy | 23.43 | **17.67** | 12.98 |
| SmallUNet (edge+aug) | DIBCO | 19.49 | 15.54 | **16.26** |
| DnCNN-lite | Kaggle | **31.94** | 16.72 | 11.80 |
| DnCNN-lite | Augraphy | 19.41 | **17.85** | 12.64 |
| DnCNN-lite | DIBCO | 18.27 | 15.39 | **15.45** |
| ShallowCNN | Kaggle | **28.79** | 16.47 | 12.01 |
| ShallowCNN | Augraphy | 17.99 | **17.47** | 12.77 |
| ShallowCNN | DIBCO | 11.07 | 11.07 | **10.32** |
| KD-CNN | Kaggle | **28.84** | 16.55 | 12.14 |
| KD-CNN | Augraphy | 17.80 | **17.47** | 12.79 |
| KD-CNN | DIBCO | 11.07 | 11.07 | **10.32** |
| NAFNet | Kaggle | **31.19** | 17.16 | 13.02 |
| NAFNet | Augraphy | 24.10 | **17.83** | 12.63 |
| NAFNet | DIBCO | 22.51 | 16.00 | **16.45** |

## FM (%)

| Model | Source | Kaggle test | Augraphy test | DIBCO test |
|---|---|---|---|---|
| SmallUNet (edge+aug) | Kaggle | **88.28** | 78.92 | 61.91 |
| SmallUNet (edge+aug) | Augraphy | 85.10 | **80.02** | 71.37 |
| SmallUNet (edge+aug) | DIBCO | 89.40 | 75.19 | **85.12** |
| DnCNN-lite | Kaggle | **88.13** | 75.96 | 35.97 |
| DnCNN-lite | Augraphy | 79.23 | **79.93** | 68.97 |
| DnCNN-lite | DIBCO | 83.71 | 69.22 | **80.93** |
| ShallowCNN | Kaggle | **88.14** | 73.17 | 41.61 |
| ShallowCNN | Augraphy | 79.23 | **79.98** | 67.69 |
| ShallowCNN | DIBCO | 0.00 | 0.00 | **0.00** |
| KD-CNN | Kaggle | **87.98** | 73.72 | 43.46 |
| KD-CNN | Augraphy | 79.05 | **79.78** | 68.97 |
| KD-CNN | DIBCO | 0.00 | 0.00 | **0.00** |
| NAFNet | Kaggle | **88.88** | 77.63 | 66.53 |
| NAFNet | Augraphy | 84.72 | **80.30** | 68.35 |
| NAFNet | DIBCO | 88.92 | 73.61 | **86.63** |

## Transfer gap, PSNR (dB) (cross-domain minus in-domain on the same test set)

| Model | Source | Kaggle test | Augraphy test | DIBCO test |
|---|---|---|---|---|
| SmallUNet (edge+aug) | Kaggle |  | -0.18 | -3.46 |
| SmallUNet (edge+aug) | Augraphy | -7.77 |  | -3.28 |
| SmallUNet (edge+aug) | DIBCO | -11.70 | -2.13 |  |
| DnCNN-lite | Kaggle |  | -1.13 | -3.65 |
| DnCNN-lite | Augraphy | -12.54 |  | -2.81 |
| DnCNN-lite | DIBCO | -13.67 | -2.45 |  |
| ShallowCNN | Kaggle |  | -1.00 | +1.69 |
| ShallowCNN | Augraphy | -10.80 |  | +2.45 |
| ShallowCNN | DIBCO | -17.72 | -6.40 |  |
| KD-CNN | Kaggle |  | -0.92 | +1.83 |
| KD-CNN | Augraphy | -11.05 |  | +2.47 |
| KD-CNN | DIBCO | -17.77 | -6.40 |  |
| NAFNet | Kaggle |  | -0.67 | -3.43 |
| NAFNet | Augraphy | -7.09 |  | -3.83 |
| NAFNet | DIBCO | -8.68 | -1.83 |  |

## Transfer gap, FM (%) (cross-domain minus in-domain on the same test set)

| Model | Source | Kaggle test | Augraphy test | DIBCO test |
|---|---|---|---|---|
| SmallUNet (edge+aug) | Kaggle |  | -1.10 | -23.22 |
| SmallUNet (edge+aug) | Augraphy | -3.18 |  | -13.75 |
| SmallUNet (edge+aug) | DIBCO | +1.12 | -4.83 |  |
| DnCNN-lite | Kaggle |  | -3.97 | -44.96 |
| DnCNN-lite | Augraphy | -8.89 |  | -11.96 |
| DnCNN-lite | DIBCO | -4.41 | -10.71 |  |
| ShallowCNN | Kaggle |  | -6.81 | +41.61 |
| ShallowCNN | Augraphy | -8.91 |  | +67.69 |
| ShallowCNN | DIBCO | -88.14 | -79.98 |  |
| KD-CNN | Kaggle |  | -6.06 | +43.46 |
| KD-CNN | Augraphy | -8.93 |  | +68.97 |
| KD-CNN | DIBCO | -87.98 | -79.78 |  |
| NAFNet | Kaggle |  | -2.67 | -20.10 |
| NAFNet | Augraphy | -4.16 |  | -18.28 |
| NAFNet | DIBCO | +0.03 | -6.69 |  |
