# Text detection

Text detection models localize the words of a page. In docTR, a *word* is an uninterrupted sequence of characters.
All models predict a segmentation map that is turned into boxes, and support both straight and rotated boxes
(`assume_straight_pages=False`).

## Architectures

| Family | Paper | Variants |
| --- | --- | --- |
| **DBNet** | [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/pdf/1911.08947.pdf) | {func}`~doctr.models.detection.db_resnet34`, {func}`~doctr.models.detection.db_resnet50`, {func}`~doctr.models.detection.db_mobilenet_v3_large` |
| **LinkNet** | [LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation](https://arxiv.org/pdf/1707.03718.pdf) | {func}`~doctr.models.detection.linknet_resnet18`, {func}`~doctr.models.detection.linknet_resnet34`, {func}`~doctr.models.detection.linknet_resnet50` |
| **FAST** | [FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation](https://arxiv.org/pdf/2111.02394.pdf) | {func}`~doctr.models.detection.fast_tiny`, {func}`~doctr.models.detection.fast_small`, {func}`~doctr.models.detection.fast_base` {bdg-secondary}`default` |

## Benchmark

Recall and precision on FUNSD and CORD, and latency with a batch size of 1:

| Architecture | Input shape | # params | FUNSD recall | FUNSD precision | CORD recall | CORD precision | sec/it |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `db_resnet34` | (1024, 1024, 3) | 22.4 M | 82.76 | 76.75 | 89.20 | 71.74 | 0.8 |
| `db_resnet50` | (1024, 1024, 3) | 25.4 M | 83.56 | 86.68 | 92.61 | 86.39 | 1.1 |
| `db_mobilenet_v3_large` | (1024, 1024, 3) | 4.2 M | 82.69 | 84.63 | **94.51** | 70.28 | 0.5 |
| `linknet_resnet18` | (1024, 1024, 3) | 11.5 M | 81.64 | 85.52 | 88.92 | 82.74 | 0.6 |
| `linknet_resnet34` | (1024, 1024, 3) | 21.6 M | 81.62 | 82.95 | 86.26 | 81.06 | 0.7 |
| `linknet_resnet50` | (1024, 1024, 3) | 28.8 M | 81.78 | 82.47 | 87.29 | 85.54 | 1.0 |
| `fast_tiny` | (1024, 1024, 3) | 13.5 M (8.5 M) | 84.90 | 85.04 | 93.73 | 76.26 | 0.7 (0.4) |
| `fast_small` | (1024, 1024, 3) | 14.7 M (9.7 M) | **85.36** | 86.68 | 94.09 | 78.53 | 0.7 (0.5) |
| `fast_base` | (1024, 1024, 3) | 16.3 M (10.6 M) | 84.95 | **86.73** | 94.39 | 85.36 | 0.8 (0.5) |

:::{note}
FAST models are **reparameterized** by `detection_predictor` and `ocr_predictor`: their branches are fused into
single convolutions for inference. Values in parentheses are measured after reparameterization.
:::

## Usage

```python
from doctr.models import ocr_predictor, detection_predictor

# In the OCR pipeline
predictor = ocr_predictor(det_arch="db_resnet50", pretrained=True)

# On its own
det_predictor = detection_predictor("db_resnet50", pretrained=True)
```

Tune the binarization and box thresholds of the post-processor to detect more (or fewer) words, see
[](../user_guide/ocr_predictor.md#tune-the-text-detection).
