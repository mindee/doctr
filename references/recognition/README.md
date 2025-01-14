# Text recognition

The sample training script was made to train text recognition model with docTR.

## Setup

First, you need to install `doctr` (with pip, for instance)

```shell
pip install -e . --upgrade
pip install -r references/requirements.txt
```

## Usage

You can start your training in TensorFlow:

```shell
python references/recognition/train_tensorflow.py crnn_vgg16_bn --train_path path/to/your/train_set --val_path path/to/your/val_set  --epochs 5
```

or PyTorch:

```shell
python references/recognition/train_pytorch.py crnn_vgg16_bn --train_path path/to/your/train_set --val_path path/to/your/val_set --epochs 5
```

### Multi-GPU support (PyTorch only - Experimental)

Multi-GPU support on recognition task with PyTorch has been added. It'll be probably added for other tasks.
Arguments are the same than the ones from single GPU, except:

- `--devices`: **by default, if you do not pass `--devices`, it will use all GPUs on your computer**.
You can use specific GPUs by passing a list of ids (ex: `0 1 2`). To find them, you can use the following snippet:

```python
import torch
devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
device_names = [torch.cuda.get_device_name(d) for d in devices]
```

- `--backend`: you can specify another `backend` for `DistribuedDataParallel` if the default one is not available on
your operating system. Fastest one is `nccl` according to [PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

```shell
python references/recognition/train_pytorch_ddp.py crnn_vgg16_bn --train_path path/to/your/train_set --val_path path/to/your/val_set --epochs 5 --devices 0 1 --backend nccl
```

## Data format

You need to provide both `train_path` and `val_path` arguments to start training.
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
python references/recognition/train_pytorch.py --help
```

## Using custom fonts

If you want to use your own custom fonts for training, make sure the font is installed on your OS.
Do so on linux by copying the .ttf file to the desired directory with: ```sudo cp custom-font.ttf /usr/local/share/fonts/``` and then running ```fc-cache -f -v``` to build the font cache.

Keep in mind that passing fonts to the training script will only work with the WordGenerator which will not augment or change images from the dataset if it is passed as argument. If no path to a dataset is passed like in this command ```python3 doctr/references/recognition/train_pytorch.py crnn_mobilenet_v3_small --vocab french --font "custom-font.ttf"```  only then is the WordGenerator "triggered" to create random images from the given vocab and font.

Running the training script should look like this for multiple custom fonts:

```shell
python references/recognition/train_pytorch.py crnn_vgg16_bn --epochs 5 --font "custom-font-1.ttf,custom-font-2.ttf"
```
