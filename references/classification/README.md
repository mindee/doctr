# Character classification

The sample training scripts were made to train a character classification model, an orientation classifier or a font style classifier with docTR.

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

## Usage font style classification

The font style classifier predicts, for each word crop, its emphasis (bold, italic, underline, strikethrough), its text and highlight colors, whether it is handwritten, its font category and its closest font family (cf. `doctr.datasets.FONT_STYLE_ATTRIBUTES`). The model outputs the logits of all these attributes concatenated, and is trained with the sum of their cross entropies.

```shell
python references/classification/train_font_style.py mobilenet_v3_small_font_style --train_path path/to/train --val_path path/to/val --epochs 100 --batch_size 256 --class-weights underline strikethrough color highlight category family --label-smoothing 0.1 --amp
```

The checkpoint with the best mean macro F1 score of the attributes is kept, and the accuracy and macro F1 score of each attribute is reported after each epoch. `--class-weights` weights the classes of each attribute (or of the listed ones) by their inverse square root frequency (rare styles such as strikethrough or highlight): leave bold, italic and handwritten out, or the classifier calls ambiguous words bold or handwritten, whereas most words of a page are regular and printed, and `--label-smoothing` helps with the many near-identical font families. The augmentations leave the colors unchanged, as they are labels.

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

### Orientation classification

Each path must lead to a folder where the images are stored. For example:

```shell
 images
    ├── sample_img_01.png
    ├── sample_img_02.png
    ├── sample_img_03.png
    └── ...
```

### Font style classification

Each path must lead to a folder with 1 subfolder and 1 file, or to a folder of such folders (e.g. dataset shards, which are concatenated):

```shell
├── images
│   ├── sample_img_01.png
│   ├── sample_img_02.png
│   ├── sample_img_03.png
│   └── ...
└── labels.json
```

The JSON file must be a dictionary, where the keys are the image file names (word crops) and the value is a dictionary with the style attributes of the word:
`bold` (`true` / `false`), `italic` (`true` / `false`), `underline` (`"none"`, `"single"` or `"double"`), `strikethrough` (`true` / `false`), `color` (the text color: `"black"`, `"gray"`, `"white"`, `"red"`, `"orange"`, `"brown"`, `"yellow"`, `"green"`, `"blue"` or `"purple"`), `highlight` (`null` or the marker color: `"yellow"`, `"green"`, `"pink"`, `"blue"` or `"orange"`), `category` (`"SANS_SERIF"`, `"SERIF"`, `"DISPLAY"`, `"HANDWRITING"`, `"MONOSPACE"` or `"UNKNOWN"`) and `family` (one of `doctr.datasets.FONT_FAMILIES`).
`handwritten` (`true` / `false`) is optional: without it, a word is handwritten when its category is `"HANDWRITING"`. Other entries (e.g. the text of the word) are ignored.

labels.json

```shell
{
    "sample_img_01.png": {
        "bold": true,
        "italic": false,
        "underline": "double",
        "strikethrough": false,
        "color": "red",
        "highlight": null,
        "category": "SERIF",
        "family": "Lora"
    },
    "sample_img_02.png": {
        "bold": false,
        "italic": true,
        "underline": "none",
        "strikethrough": false,
        "color": "black",
        "highlight": "yellow",
        "category": "SANS_SERIF",
        "family": "Roboto",
        "handwritten": false
    },
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

Character classification:

```shell
python references/classification/train_character.py --help
```

Orientation classification:

```shell
python references/classification/train_orientation.py --help
```

Font style classification:

```shell
python references/classification/train_font_style.py --help
```
