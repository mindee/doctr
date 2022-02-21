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
python references/recognition/train_pytorch.py crnn_vgg16_bn --train_path path/to/your/train_set --val_path path/to/your/val_set --epochs 5 --device 0
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
labels = {
    'img_1.jpg': 'I',
    'img_2.jpg': 'am',
    'img_3.jpg': 'a',
    'img_4.jpg': 'Jedi',
    'img_5.jpg': '!',
    ...
}
```

## Advanced options

Feel free to inspect the multiple script option to customize your training to your own needs!

```python
python references/recognition/train_pytorch.py --help
```
