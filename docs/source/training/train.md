# Training a model

The [`references/`](https://github.com/mindee/doctr/tree/main/references) folder of the repository contains
ready-to-use training, evaluation and latency scripts for every task. This page summarizes how to use them; each
task folder has a README with all the details.

## Setup

The scripts live in the repository, not in the package. Clone it and install the extra requirements:

```bash
git clone https://github.com/mindee/doctr.git
cd doctr
pip install -e .
pip install -r references/requirements.txt
```

All scripts share the same interface: the architecture as first argument, `--train_path` and `--val_path`
pointing to your data, and options such as `--epochs`, `--batch_size`, `--lr`, `--pretrained` (start from the
pretrained weights, recommended for fine-tuning) or `--name` (experiment name). Run any script with `--help` to
list them all.

:::{tip}
Annotating data is the hardest part. The [docTR Labeler](../community/tools.md#doctr-labeler) lets you draw and
correct polygons with automatic suggestions, and exports the formats below.
:::

## Data formats

Every task uses the same layout on disk, one folder per split:

```text
train_set/
├── images/
│   ├── sample_img_01.png
│   ├── sample_img_02.png
│   └── ...
└── labels.json
```

`labels.json` maps each image file name to its annotation, which depends on the task. Coordinates are **absolute
pixel coordinates**.

::::{tab-set}

:::{tab-item} Detection
```{code-block} javascript
:caption: labels.json

{
  "sample_img_01.png": {
    "img_dimensions": [900, 600],
    "img_hash": "<sha256 of the image file>",
    "polygons": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
  }
}
```

For a multi-class model (e.g. for [KIE](../user_guide/kie.md)), `polygons` is a dictionary with one list per class:

```javascript
"polygons": {
  "date": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...],
  "total": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
}
```

Every class must appear in both splits; an image without a box of a class has an empty list for it.
:::

:::{tab-item} Recognition
`images/` contains **word crops** and `labels.json` their transcription, in UTF-8:

```{code-block} javascript
:caption: labels.json

{
  "img_1.jpg": "I",
  "img_2.jpg": "am",
  "img_3.jpg": "a",
  "img_4.jpg": "Jedi"
}
```

Labels must not contain spaces, and their characters must be part of the vocabulary you train with.
:::

:::{tab-item} Layout
```{code-block} javascript
:caption: labels.json

{
  "sample_img_01.png": {
    "img_dimensions": [900, 600],
    "img_hash": "<sha256 of the image file>",
    "polygons": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...],
    "classes": ["Title", "Text", ...]
  }
}
```
:::

:::{tab-item} Table structure
`images/` contains images of **tables**:

```{code-block} javascript
:caption: labels.json

{
  "sample_img_01.png": {
    "cells": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...],
    "logic": [[start_col, end_col, start_row, end_row], ...]
  }
}
```

Cell polygons are ordered top-left, top-right, bottom-right, bottom-left. Logical coordinates are 0-indexed and
inclusive: a cell spanning one row and one column has equal start and end indices.
:::

:::{tab-item} Orientation
No labels: a folder of **upright** images. The script rotates them to generate the training targets.

```text
train_set/
├── sample_img_01.png
├── sample_img_02.png
└── ...
```
:::
::::

## Train

### Text detection

```bash
python references/detection/train.py db_resnet50 \
  --train_path path/to/train_set --val_path path/to/val_set \
  --epochs 5 --pretrained
```

Add `--rotation` to train on rotated documents. Built-in datasets can replace local folders:
`--train_datasets FUNSD CORD --val_datasets FUNSD`.
[Full guide](https://github.com/mindee/doctr/tree/main/references/detection#readme).

### Text recognition

```bash
python references/recognition/train.py crnn_vgg16_bn \
  --train_path path/to/train_set --val_path path/to/val_set \
  --vocab german --epochs 5 --pretrained
```

`--vocab` selects one of the [vocabularies](../models/vocabs.md) (`french` by default). Without any dataset, the
script trains on synthetic words generated from the vocabulary, optionally with your own fonts (`--font`).
[Full guide](https://github.com/mindee/doctr/tree/main/references/recognition#readme).

### Layout analysis

```bash
python references/layout/train.py lw_detr_s \
  --train_path path/to/train_set --val_path path/to/val_set --epochs 5
```

[Full guide](https://github.com/mindee/doctr/tree/main/references/layout#readme).

### Table structure

```bash
python references/table/train.py tablecenternet \
  --train_path path/to/train_set --val_path path/to/val_set --epochs 5
```

[Full guide](https://github.com/mindee/doctr/tree/main/references/table#readme), including a toy dataset to try
the pipeline end to end.

### Orientation classifiers

```bash
# --type page for whole pages, --type crop for word crops
python references/classification/train_orientation.py mobilenet_v3_small_page_orientation \
  --type page --train_path path/to/train_set --val_path path/to/val_set --epochs 5
```

[Full guide](https://github.com/mindee/doctr/tree/main/references/classification#readme).

## Hardware

::::{tab-set}

:::{tab-item} Single GPU
Every script accepts `--device`: a CUDA index (`0`), `cuda:N`, `mps` or `cpu`. By default, the script picks CUDA,
then MPS, then the CPU.

`--amp` enables automatic mixed precision on CUDA; add `--amp-dtype bfloat16` on Ampere GPUs or newer to avoid the
overflows of float16:

```bash
python references/detection/train.py db_resnet50 --train_path train --val_path val \
  --device 0 --amp --amp-dtype bfloat16
```
:::

:::{tab-item} Multi-GPU
Launch the same scripts with [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) to train with
DistributedDataParallel:

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 references/detection/train.py db_resnet50 \
  --train_path train --val_path val --epochs 5 --backend nccl
```
:::

:::{tab-item} Apple Silicon
```bash
# Let the few operations MPS lacks fall back to the CPU
PYTORCH_ENABLE_MPS_FALLBACK=1 python references/detection/train.py db_resnet50 \
  --train_path train --val_path val --device mps
```
:::
::::

## Outputs, logging and evaluation

- **Checkpoints** are saved in `--output_dir` as `<experiment name>.pt`, next to a `<experiment name>.json` file
  with the run metadata: architecture and task settings (vocabulary, class names...), dataset hashes, library
  versions and all the arguments.
- **Tracking**: `--wb` logs to Weights & Biases and `--clearml` to ClearML.
- **Evaluation**: each task folder has an `evaluate.py` script, and a `latency.py` script to benchmark speed.
- **Sharing**: `--push-to-hub` publishes the final model on the [Hugging Face Hub](sharing_models.md).

Then [load your checkpoint](custom_models.md) into a predictor.
