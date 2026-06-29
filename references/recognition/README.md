# Text recognition

The sample training script was made to train text recognition model with docTR.

## Setup

First, you need to install `doctr` (with pip, for instance)

```shell
pip install -e . --upgrade
pip install -r references/requirements.txt
```

## Usage

You can start your training in PyTorch:

```shell
python references/recognition/train.py crnn_vgg16_bn --train_path path/to/your/train_set --val_path path/to/your/val_set --epochs 5
```

Alternatively, instead of providing local folders you can train directly on one or several built-in datasets, which are downloaded automatically. When several are passed, the first one is loaded and extended with the others:

```shell
python references/recognition/train.py crnn_vgg16_bn --train_datasets FUNSD SVHN --val_datasets FUNSD --epochs 5
```

The available built-in datasets are `CORD`, `FUNSD`, `IC03`, `IIIT5K`, `SVHN`, `SVT` and `SynthText`. For each split use either the local path or the built-in datasets (not both): `--train_path` or `--train_datasets`, and `--val_path` or `--val_datasets`. If neither is provided, the script falls back to synthetic data generated on the fly with the `WordGenerator` (see [Using custom fonts](#using-custom-fonts)).

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
torchrun --nproc_per_node=2 references/recognition/train.py \
  crnn_vgg16_bn \
  --train_path path/to/train \
  --val_path   path/to/val \
  --epochs 5 \
  --backend nccl
```

## Data format

To train on your own data you need to provide both `train_path` and `val_path` arguments (alternatively, use the built-in datasets shown above, or the synthetic `WordGenerator`).
Each of these paths must lead to a 2-elements folder:

```shell
├── images
    ├── img_1.jpg
    ├── img_2.jpg
    ├── img_3.jpg
    └── ...
├── labels.json
```

The JSON files must contain word-labels for each picture as a string.
The order of entries in the json does not matter.

```shell
# labels.json
{
    "img_1.jpg": "I",
    "img_2.jpg": "am",
    "img_3.jpg": "a",
    "img_4.jpg": "Jedi",
    "img_5.jpg": "!",
    ...
}
```

When typing your labels, be aware that the VOCAB doesn't handle spaces. Also make sure your `labels.json` file is using UTF-8 encoding.

## Slack Logging with tqdm

To enable Slack logging using `tqdm`, you need to set the following environment variables:

- `TQDM_SLACK_TOKEN`: the Slack Bot Token
- `TQDM_SLACK_CHANNEL`: you can retrieve it using `Right Click on Channel > Copy > Copy link`. You should get something like `https://xxxxxx.slack.com/archives/yyyyyyyy`. Keep only the `yyyyyyyy` part.

You can follow this page on [how to create a Slack App](https://api.slack.com/quickstart).

## Advanced options

Feel free to inspect the multiple script option to customize your training to your own needs!

```shell
python references/recognition/train.py --help
```

## Using custom fonts

If you want to use your own custom fonts for training, make sure the font is installed on your OS.
Do so on linux by copying the .ttf file to the desired directory with: ```sudo cp custom-font.ttf /usr/local/share/fonts/``` and then running ```fc-cache -f -v``` to build the font cache.

Keep in mind that passing fonts to the training script will only work with the WordGenerator which will not augment or change images from the dataset if it is passed as argument. If no path to a dataset is passed like in this command ```python3 doctr/references/recognition/train.py crnn_mobilenet_v3_small --vocab french --font "custom-font.ttf"```  only then is the WordGenerator "triggered" to create random images from the given vocab and font.

Running the training script should look like this for multiple custom fonts:

```shell
python references/recognition/train.py crnn_vgg16_bn --epochs 5 --font "custom-font-1.ttf,custom-font-2.ttf"
```
