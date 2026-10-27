# Layout analysis

Layout models localize and classify the regions of a page (titles, paragraphs, tables, figures, headers...). Like
the text detection models, they work on rotated and skewed documents.

## Architectures

| Architecture | Paper | Pretrained weights |
| --- | --- | --- |
| {func}`~doctr.models.layout.lw_detr_s` {bdg-secondary}`default` | [LW-DETR: A Transformer Replacement to YOLO for Real-Time Detection](https://arxiv.org/pdf/2406.03459v1) | {bdg-success}`available` |
| {func}`~doctr.models.layout.lw_detr_m` | Same, larger backbone | {bdg-warning}`coming soon` |

The pretrained model predicts 11 classes: `Caption`, `Footnote`, `Formula`, `List-item`, `Page-footer`,
`Page-header`, `Picture`, `Section-header`, `Table`, `Text` and `Title`.

## Benchmark

| Architecture | Input shape | # params | mAP@[.5:.95] | AP@.5 | AP@.75 | sec/it (batch 1) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lw_detr_s` | (1024, 1024, 3) | 15.1 M | 66.89 | 82.83 | 73.75 | 0.5 |
| `lw_detr_m` | (1024, 1024, 3) | 29.5 M | soon | soon | soon | 0.7 |

## Usage

```python
from doctr.models import layout_predictor, ocr_predictor

# In the OCR pipeline
predictor = ocr_predictor(pretrained=True, detect_layout=True, layout_arch="lw_detr_s")

# On its own
layout = layout_predictor("lw_detr_s", pretrained=True)
```

See [](../user_guide/layout_and_tables.md) for how the regions are used, and
[](../training/custom_models.md#layout-models) to plug a model trained on your own classes.
