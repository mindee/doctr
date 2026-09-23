# Character classification

The sample training scripts was made to train a character classification model or a orientation classifier with docTR.

## Setup

First, you need to install `doctr` (with pip, for instance)

```shell
pip install -e . --upgrade
pip install -r references/requirements.txt
```

## Usage character classification

You can start your training in PyTorch:

```shell
python references/classification/train_character.py mobilenet_v3_large --epochs 5 --device 0
```

## Usage orientation classification

You can start your training in PyTorch:

```shell
python references/classification/train_orientation.py resnet18 --type page --train_path path/to/your/train_set --val_path path/to/your/val_set --epochs 5
```

The type can be either `page` for document images or `crop` for word crops.

## Device and mixed precision

Every training and evaluation script accepts `--device`: a CUDA index (`0`), `cuda:N`, `mps` (Apple Silicon GPU) or `cpu`. Without it the script picks CUDA if available, then MPS, then CPU. In distributed mode (`torchrun`) the argument is ignored and each process uses its own GPU.

`--amp` enables automatic mixed precision and is only supported on CUDA. `--amp-dtype bfloat16` (Ampere or newer GPUs) uses bfloat16 instead of float16: it has the range of float32, so it needs no loss scaling and avoids the overflows float16 can produce in some losses.

```shell
# Apple Silicon: set the fallback so the few ops MPS lacks run on CPU
PYTORCH_ENABLE_MPS_FALLBACK=1 python references/classification/train_character.py mobilenet_v3_small --epochs 5 --device mps
# NVIDIA GPU with bfloat16 mixed precision
python references/classification/train_character.py mobilenet_v3_small --epochs 5 --device 0 --amp --amp-dtype bfloat16
```

## Checkpoints

Each run writes its metadata once, as `<experiment name>.json` next to the checkpoints it saves. It holds what is needed to rebuild the model for inference and to reproduce the run: the architecture and its task settings (`classes` / `vocab_name`), the dataset hashes when local data is used, the docTR / PyTorch versions, the git revision and the full list of arguments.

## Data format

You need to provide both `train_path` and `val_path` arguments to start training.
Each path must lead to a folder where the images are stored. For example:

```shell
 images
    ├── sample_img_01.png
    ├── sample_img_02.png
    ├── sample_img_03.png
    └── ...
```

## Slack Logging with tqdm

To enable Slack logging using `tqdm`, you need to set the following environment variables:

- `TQDM_SLACK_TOKEN`: the Slack Bot Token
- `TQDM_SLACK_CHANNEL`: you can retrieve it using `Right Click on Channel > Copy > Copy link`. You should get something like `https://xxxxxx.slack.com/archives/yyyyyyyy`. Keep only the `yyyyyyyy` part.

You can follow this page on [how to create a Slack App](https://api.slack.com/quickstart).

## Advanced options

Feel free to inspect the multiple script option to customize your training to your own needs!

Character classification:

```shell
python references/classification/train_character.py --help
```

Orientation classification:

```shell
python references/classification/train_orientation.py --help
```
