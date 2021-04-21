# Text detection

The sample training script was made to train text detection model with doctr

## Getting started

First, you need to install doctr (with pip, for instance)

```shell
pip install -e . --upgrade
pip install -r references/requirements.txt
```

Then, to run the script execute the following command

```shell
python references/detection/train.py db_resnet50 --epochs 5 --data_path path/to/your/dataset
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

## Tune arguments

You can pass the following arguments:

```shell
model (str): text-detection model to train
--epochs (int): default=10, number of epochs to train the model on
--batch_size (int): default=2, batch size for training
--input_size Tuple[int, int]: default=(1024, 1024), input size (H, W) for the model
--learning_rate, (float): default=0.001, learning rate for the optimizer (Adam)
--data_path (str), path to data folder
```
