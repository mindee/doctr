import pytest
import numpy as np
import tensorflow as tf

from doctr import datasets


def test_visiondataset():
    url = 'https://data.deepai.org/mnist.zip'
    with pytest.raises(ValueError):
        datasets.core.VisionDataset(url, download=False)

    dataset = datasets.core.VisionDataset(url, download=True, extract_archive=True)
    assert len(dataset) == 0
    assert repr(dataset) == 'VisionDataset()'


@pytest.mark.parametrize(
    "dataset_name, train, input_size, size",
    [
        ['FUNSD', True, [512, 512], 149],
        ['FUNSD', False, [512, 512], 50],
        ['SROIE', True, [512, 512], 626],
        ['SROIE', False, [512, 512], 360],
        ['CORD', True, [512, 512], 800],
        ['CORD', False, [512, 512], 100],
    ],
)
def test_dataset(dataset_name, train, input_size, size):

    ds = datasets.__dict__[dataset_name](train=train, download=True, input_size=input_size)

    assert len(ds) == size
    assert repr(ds) == f"{dataset_name}(train={train}, input_size={input_size})"
    img, target = ds[0]
    assert isinstance(img, tf.Tensor) and img.shape == (*input_size, 3)
    assert isinstance(target, dict)
