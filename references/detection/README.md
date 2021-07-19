# Text detection

The sample training script was made to train text detection model with doctr

## Setup

First, you need to install doctr (with pip, for instance)

```shell
pip install -e . --upgrade
pip install -r references/requirements.txt
```

if you are using PyTorch back-end, there is an extra dependency (to optimize data loading):
```shell
pip install contiguous-params>=1.0.0
```

## Usage

You can start your training in TensorFlow:

```shell
python references/detection/train.py path/to/your/dataset db_resnet50 --epochs 5
```
or PyTorch:

```shell
python references/detection/train_pytorch.py path/to/your/dataset db_resnet50 --epochs 5 --device 0
```

## Data format

You need to provide a --data_path argument to start training. 
The data_path must lead to folder with 4 subfolder in it:

```shell
├── train
    ├── img_1.jpg
    ├── img_2.jpg
    ├── img_3.jpg
    └── ...
├── train_labels
    ├── img_1.jpg.json
    ├── img_2.jpg.json
    ├── img_3.jpg.json
    └── ...
├── val                    
    ├── img_a.jpg
    ├── img_b.jpg
    ├── img_c.jpg
    └── ...
├── val_labels
    ├── img_a.jpg.json
    ├── img_b.jpg.json
    ├── img_c.jpg.json
    └── ...
```

Each JSON file must contains 3 lists of boxes: boxes_1 (very confident), boxes_2 (confident) and boxes_3 (not very confident).
The order of the points does not matter inside a box. Points are (x, y) absolutes coordinates.

```shell
image.json = {
    'boxes_1': [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...],
    'boxes_2': [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...],
    'boxes_3': [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], ...]
}
```

## Advanced options

Feel free to inspect the multiple script option to customize your training to your own needs!

```python
python references/detection/train.py --help
```
