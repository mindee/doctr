# Layout detection

The sample training script was made to train layout detection model with docTR.

## Setup

First, you need to install `doctr` (with pip, for instance)

```shell
pip install -e . --upgrade
pip install -r references/requirements.txt
```

## Usage

You can start your training in PyTorch:

```shell
python references/layout/train.py lw_detr_s --train_path path/to/your/train_set --val_path path/to/your/val_set --epochs 5
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
torchrun --nproc_per_node=2 references/layout/train.py \
  lw_detr_s \
  --train_path path/to/train \
  --val_path   path/to/val \
  --epochs 5 \
  --backend nccl
  ```

## Device and mixed precision

Every training and evaluation script accepts `--device`: a CUDA index (`0`), `cuda:N`, `mps` (Apple Silicon GPU) or `cpu`. Without it the script picks CUDA if available, then MPS, then CPU. In distributed mode (`torchrun`) the argument is ignored and each process uses its own GPU.

`--amp` enables automatic mixed precision and is only supported on CUDA. `--amp-dtype bfloat16` (Ampere or newer GPUs) uses bfloat16 instead of float16: it has the range of float32, so it needs no loss scaling and avoids the overflows float16 can produce in some losses.

```shell
# Apple Silicon: set the fallback so the few ops MPS lacks run on CPU
PYTORCH_ENABLE_MPS_FALLBACK=1 python references/layout/train.py lw_detr_s --train_path path/to/train --val_path path/to/val --epochs 5 --device mps
# NVIDIA GPU with bfloat16 mixed precision
python references/layout/train.py lw_detr_s --train_path path/to/train --val_path path/to/val --epochs 5 --device 0 --amp --amp-dtype bfloat16
```

## Checkpoints

Each run writes its metadata once, as `<experiment name>.json` next to the checkpoints it saves. It holds what is needed to rebuild the model for inference and to reproduce the run: the architecture and its task settings (`class_names`, `assume_straight_pages`), the dataset hashes when local data is used, the docTR / PyTorch versions, the git revision and the full list of arguments.

## Data format

You need to provide both `train_path` and `val_path` arguments to start training.
Each path must lead to folder with 1 subfolder and 1 file:

```shell
├── images
│   ├── sample_img_01.png
│   ├── sample_img_02.png
│   ├── sample_img_03.png
│   └── ...
└── labels.json
```

Each JSON file must be a dictionary, where the keys are the image file names and the value is a dictionary with 4 entries: `img_dimensions` (spatial shape of the image), `img_hash` (SHA256 of the image file), `polygons` (the set of 2D points forming the localization polygon), `classes` (list of class names for each polygon).
The order of the points does not matter inside a polygon. Points are (x, y) absolutes coordinates.

labels.json

```shell
{
    "sample_img_01.png" = {
        'img_dimensions': (900, 600),
        'img_hash': "theimagedumpmyhash",
        'polygons': [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...],
        'classes': ["class_name_1", "class_name_2", ...]
     },
     "sample_img_02.png" = {
        'img_dimensions': (900, 600),
        'img_hash': "thisisahash",
        'polygons': [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...],
        'classes': ["class_name_1", "class_name_2", ...]
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
python references/layout/train.py --help
```
