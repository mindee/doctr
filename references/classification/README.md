# Character classification

The sample training script was made to train a character classification model with docTR.

## Setup

First, you need to install `doctr` (with pip, for instance)

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
python references/classification/train_tensorflow.py mobilenet_v3_large --epochs 5
```
or PyTorch:

```shell
python references/classification/train_pytorch.py mobilenet_v3_large --epochs 5 --device 0
```


## Advanced options

Feel free to inspect the multiple script option to customize your training to your own needs!

```python
python references/classification/train_tensorflow.py --help
```
