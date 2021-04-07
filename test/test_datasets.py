import pytest
import numpy as np
from doctr import datasets


def test_visiondataset():
    url = 'https://data.deepai.org/mnist.zip'
    with pytest.raises(ValueError):
        datasets.core.VisionDataset(url, download=False)

    dataset = datasets.core.VisionDataset(url, download=True, extract_archive=True)
    assert len(dataset) == 0
    assert repr(dataset) == 'VisionDataset()'


@pytest.mark.parametrize(
    "dataset_name, train, size",
    [
        ['FUNSD', True, 149],
        ['FUNSD', False, 50],
        ['SROIE', True, 626],
        ['SROIE', False, 360],
        ['CORD', True, 800],
        ['CORD', False, 100],
    ],
)
def test_dataset(dataset_name, train, size):

    ds = datasets.__dict__[dataset_name](train=train, download=True)

    assert len(ds) == size
    assert repr(ds) == f"{dataset_name}()"
    img, target = ds[0]
    assert isinstance(img, np.ndarray) and img.ndim == 3
    assert isinstance(target, dict)
