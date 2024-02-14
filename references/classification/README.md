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
python references/classification/train_tensorflow_orientation.py path/to/your/train_set path/to/your/val_set resnet18 page --epochs 5
```

or PyTorch:

```shell
python references/classification/train_pytorch_orientation.py path/to/your/train_set path/to/your/val_set resnet18 page --epochs 5
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

## Advanced options

Feel free to inspect the multiple script option to customize your training to your own needs!

Character classification:

```python
python references/classification/train_tensorflow_character.py --help
```

Orientation classification:

```python
python references/classification/train_tensorflow_orientation.py --help
```
