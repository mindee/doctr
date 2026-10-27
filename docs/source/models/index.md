# Model zoo

docTR implements a range of published architectures for each task of the pipeline, all with pretrained weights.
This section lists them with their benchmarks, so you can pick the right trade-off between speed and accuracy.

## At a glance

| Task | Architectures | Default | Use it with |
| --- | --- | --- | --- |
| [Text detection](detection.md) | DBNet, LinkNet, FAST | `fast_base` | `ocr_predictor(det_arch=...)` |
| [Text recognition](recognition.md) | CRNN, SAR, MASTER, ViTSTR, PARSeq, VIPTR | `crnn_vgg16_bn` | `ocr_predictor(reco_arch=...)` |
| [Layout analysis](layout.md) | LW-DETR | `lw_detr_s` | `ocr_predictor(detect_layout=True, layout_arch=...)` |
| [Table structure](table_structure.md) | TableCenterNet | `tablecenternet` | `ocr_predictor(detect_tables=True)` |
| [Orientation](#orientation-classifiers) | MobileNetV3 | `mobilenet_v3_small_*_orientation` | `ocr_predictor(assume_straight_pages=False)` |

[End-to-end benchmarks](end_to_end.md) compare detection + recognition pairs, including commercial OCR services.

## How to choose

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {octicon}`check-circle;1.1em` Balanced
The defaults, `fast_base` + `crnn_vgg16_bn`, are accurate on most documents and run fine on a CPU.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Fast
`fast_tiny` or `db_mobilenet_v3_large` with `crnn_mobilenet_v3_small` or `viptr_tiny` for CPUs, edge devices and
high-volume pipelines.
:::

:::{grid-item-card} {octicon}`goal;1.1em` Accurate
`db_resnet50` or `fast_base` with `parseq`, `vitstr_base` or `master`, ideally on a GPU.
:::
::::

Benchmarks on public datasets are a starting point: the best way to choose is to compare a few combinations on a
sample of **your** documents. If none is good enough, [fine-tune a model](../training/index.md) on your data.

## Orientation classifiers

Two lightweight MobileNetV3 classifiers handle [rotated documents](../user_guide/rotated_documents.md). They are
loaded automatically when needed.

| Architecture                             | Predicts                                         | Factory                                                    |
| ---------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------- |
| `mobilenet_v3_small_page_orientation`    | The orientation of a page: 0°, 90°, 180°, 270°   | {func}`~doctr.models.classification.page_orientation_predictor`             |
| `mobilenet_v3_small_crop_orientation`    | The orientation of a word crop: 0°, 90°, 180°, 270° | {func}`~doctr.models.classification.crop_orientation_predictor`          |

## About the benchmarks

- **Datasets**: unless stated otherwise, models are evaluated on both the training and test sets of
  [FUNSD](https://guillaumejaume.github.io/FUNSD/) and [CORD](https://github.com/clovaai/cord) (see
  [](../training/datasets.md)). FUNSD has 199 pages (30,595 word crops), which might not be representative of every
  use case.
- **Metrics** are described in [](../api/utils.md#task-evaluation).
- **Speed** (`sec/it`) is measured on an *11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz* CPU, after a warm-up of
  100 tensors, as the average over 1,000 samples. The batch size is given in each table.
- Reproduce or extend them with the `evaluate.py` and `latency.py` scripts of the
  [references](https://github.com/mindee/doctr/tree/main/references).

```{toctree}
:hidden:

detection
recognition
layout
table_structure
end_to_end
vocabs
```
