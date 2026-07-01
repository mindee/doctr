# Table structure recognition

The sample scripts in this folder let you train, evaluate and benchmark table structure recognition models with docTR.
A table structure model localizes every cell of a table (its spatial structure) and recovers the rows and columns each cell spans (its logical structure).

## Setup

First, you need to install `doctr` (with pip, for instance)

```shell
pip install -e . --upgrade
pip install -r references/requirements.txt
```

## Usage

You can start your training in PyTorch:

```shell
python references/table/train.py tablecenternet --train_path path/to/your/train_set --val_path path/to/your/val_set --epochs 5
```

To try the pipeline end-to-end on a small toy dataset:

```shell
wget https://github.com/mindee/doctr/releases/download/v1.0.1/toy_table_set-ea091e15.zip
unzip toy_table_set-ea091e15.zip -d table_set
python references/table/train.py tablecenternet --train_path ./table_set --val_path ./table_set -b 2 --epochs 1
```

### Multi-GPU support

We now use the built-in [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html) launcher to spawn your DDP workers. `torchrun` will set all the necessary environment variables (`LOCAL_RANK`, `RANK`, etc.) for you. Arguments are the same than the ones from single GPU, except:

- `--backend`: you can specify another `backend` for `DistributedDataParallel` if the default one is not available on
your operating system. Fastest one is `nccl` according to [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

#### Key `torchrun` parameters

- `--nproc_per_node=<N>`
  Spawn `<N>` processes on the local machine (typically equal to the number of GPUs you want to use).
- `--nnodes=<M>`
  (Optional) Total number of nodes in your job. Default is 1.
- `--rdzv_backend`, `--rdzv_endpoint`, `--rdzv_id`
  (Optional) Rendezvous settings for multi-node jobs. See the [torchrun docs](https://pytorch.org/docs/stable/elastic/run.html) for details.

#### GPU selection

By default all visible GPUs will be used. To limit which GPUs participate, set the `CUDA_VISIBLE_DEVICES` environment variable **before** running `torchrun`. For example, to use only CUDA devices 0 and 2:

```shell
CUDA_VISIBLE_DEVICES=0,2 \
torchrun --nproc_per_node=2 references/table/train.py \
  tablecenternet \
  --train_path path/to/train \
  --val_path   path/to/val \
  --epochs 5 \
  --backend nccl
  ```

## Evaluation

You can evaluate a model (the pretrained one by default, or your own checkpoint with `--resume`) on a dataset:

```shell
python references/table/evaluate.py tablecenternet path/to/your/dataset
python references/table/evaluate.py tablecenternet path/to/your/dataset --resume path/to/your/checkpoint.pt
```

The script reports the cell-detection recall, precision and F1, along with the structure accuracy (the share of cells whose logical coordinates are correctly predicted). Cells are matched to the ground truth above the IoU threshold set with `--iou_thresh` (default `0.5`).

## Latency benchmark

You can measure the inference latency of an architecture:

```shell
python references/table/latency.py tablecenternet --it 100 --size 1024 --gpu
```

## Data format

You need to provide both `train_path` and `val_path` arguments to start training (`evaluate.py` takes a single `dataset_path`).
Each path must lead to a folder with 1 subfolder and 1 file:

```shell
├── images
│   ├── sample_img_01.png
│   ├── sample_img_02.png
│   ├── sample_img_03.png
│   └── ...
└── labels.json
```

`labels.json` is a dictionary mapping each image file name to its annotation. Each annotation has 2 entries:

- `cells`: the list of cell polygons. Each polygon is a quadrilateral given as 4 `(x, y)` **absolute** coordinates ordered top-left, top-right, bottom-right, bottom-left.
- `logic`: the list of logical coordinates, one per cell, given as `[start_col, end_col, start_row, end_row]`. Indices are **0-indexed** and ends are inclusive, so a cell that spans a single row and a single column has equal start and end indices.

Both lists must have the same length (one entry per cell): `cells` has shape `(N, 4, 2)` and `logic` has shape `(N, 4)`.

labels.json

```shell
{
    "sample_img_01.png": {
        "cells": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...],
        "logic": [[start_col, end_col, start_row, end_row], ...]
    },
    "sample_img_02.png": {
        "cells": [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...],
        "logic": [[start_col, end_col, start_row, end_row], ...]
    }
    ...
}
```

## Slack Logging with tqdm

To enable Slack logging using `tqdm`, you need to set the following environment variables:

- `TQDM_SLACK_TOKEN`: the Slack Bot Token
- `TQDM_SLACK_CHANNEL`: you can retrieve it using `Right Click on Channel > Copy > Copy link`. You should get something like `https://xxxxxx.slack.com/archives/yyyyyyyy`. Keep only the `yyyyyyyy` part.

You can follow this page on [how to create a Slack App](https://api.slack.com/quickstart).

## Advanced options

Feel free to inspect the multiple script option to customize your training to your own needs!

```python
python references/table/train.py --help
```
