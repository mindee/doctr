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

When typing your labels, be aware that the VOCAB doesn't handle spaces.

## Advanced options

Feel free to inspect the multiple script option to customize your training to your own needs!

```python
python references/recognition/train_pytorch.py --help
```
## Using custom fonts
If you want to use your own custom fonts for training, make sure the font is installed on your OS. 
You can do so in linux by either:
* clicking on the .ttf file and pressing the install button.  
* If the above doesn't work, copy the .ttf file to the desired directory with: ```sudo cp BORING.ttf /usr/local/share/fonts/``` and then run ```fc-cache -f -v``` to build the font cache.

Keep in mind that you shouldn't pass your dataset otherwise it will be used for training instead of the font.

Running the training script should look like this for multiple custom fonts:
```shell
python references/recognition/train_pytorch.py crnn_vgg16_bn --epochs 5 --font "cusomt-font-1.ttf,custom-font-2.ttf"
```
