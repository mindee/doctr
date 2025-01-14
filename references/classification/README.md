# Character classification

The sample training scripts was made to train a character classification model or a orientation classifier with docTR.

## Setup

First, you need to install `doctr` (with pip, for instance)

```shell
pip install -e . --upgrade
pip install -r references/requirements.txt
```

## Usage character classification

You can start your training in TensorFlow:

```shell
python references/classification/train_tensorflow_character.py mobilenet_v3_large --epochs 5
```

or PyTorch:

```shell
python references/classification/train_pytorch_character.py mobilenet_v3_large --epochs 5 --device 0
```

## Usage orientation classification

You can start your training in TensorFlow:

```shell
python references/classification/train_tensorflow_orientation.py resnet18 --type page --train_path path/to/your/train_set --val_path path/to/your/val_set --epochs 5
```

or PyTorch:

```shell
python references/classification/train_pytorch_orientation.py resnet18 --type page --train_path path/to/your/train_set --val_path path/to/your/val_set --epochs 5
```

The type can be either `page` for document images or `crop` for word crops.

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
python references/classification/train_tensorflow_character.py --help
```

Orientation classification:

```shell
python references/classification/train_tensorflow_orientation.py --help
```
