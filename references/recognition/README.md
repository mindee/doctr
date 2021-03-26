# Text recognition

The sample training script was made to train text recognition model with doctr

## Getting started

First, you need to install doctr (with pip, for instance)

```shell
pip install python-doctr
```

Then, to run the script execute the following command

```shell
python references/recognition/train.py crnn_vgg16_bn --epochs 5 --data_path path/to/your/dataset
```

## Data format

You need to provide a --data_path argument to start training. 
The data_path must lead to a 3 elements folder:

```shell
├── labels.json
├── train
    ├── img_1.jpg
    ├── img_2.jpg
    ├── img_3.jpg
    └── ...
├── val                    
    ├── img_a.jpg
    ├── img_b.jpg
    ├── img_c.jpg
    └── ...
```

The JSON file must contains word-labels for each picture as a string. 
The order of entries in the json does not matter (train labels and val labels can be mixed)

```shell
labels = {
    'img_1.jpg': 'I',
    'img_2.jpg': 'am',
    'img_3.jpg': 'a',
    'img_4.jpg': 'Jedi',
    'img_5.jpg': '!',
    ...
    'img_a.jpg': 'I',
    'img_b.jpg': 'am',
    'img_c.jpg': 'a',
    'img_d.jpg': 'Sith',
    'img_e.jpg': '!',
    ...
}
```

## Tune arguments

You can pass the following arguments:

```shell
model (str): text-recognition model to train
--epochs (int): default=10, number of epochs to train the model on
--batch_size (int): default=64, batch size for training
--input_size Tuple[int, int]: default=(32, 128), input size (H, W) for the model
--learning_rate, (float): default=0.001, learning rate for the optimizer (Adam)
--postprocessor, (str): default='crnn', postprocessor, either crnn or sar
--teacher_forcing (bool), default=False, if True, teacher forcing during training
--data_path (str), path to data folder
```
