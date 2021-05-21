import pytest
import tensorflow as tf

from doctr import datasets
from doctr.transforms import Resize


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

    ds = datasets.__dict__[dataset_name](train=train, download=True, sample_transforms=Resize(input_size))

    assert len(ds) == size
    assert repr(ds) == f"{dataset_name}(train={train})"
    img, target = ds[0]
    assert isinstance(img, tf.Tensor) and img.shape == (*input_size, 3)
    assert isinstance(target, dict)

    loader = datasets.DataLoader(ds, batch_size=2)
    images, targets = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)
