from pathlib import Path

import numpy as np
import pytest

from doctr import datasets


def test_visiondataset():
    url = "https://github.com/mindee/doctr/releases/download/v0.6.0/mnist.zip"
    with pytest.raises(ValueError):
        datasets.datasets.VisionDataset(url, download=False)

    dataset = datasets.datasets.VisionDataset(url, download=True, extract_archive=True)
    assert len(dataset) == 0
    assert repr(dataset) == "VisionDataset()"


def test_abstractdataset(mock_image_path):
    with pytest.raises(ValueError):
        datasets.datasets.AbstractDataset("my/fantasy/folder")

    # Check transforms
    path = Path(mock_image_path)
    ds = datasets.datasets.AbstractDataset(path.parent)
    # Check target format
    with pytest.raises(AssertionError):
        ds.data = [(path.name, 0)]
        _ = ds[0]
    with pytest.raises(AssertionError):
        ds.data = [(path.name, dict(boxes=np.array([[0, 0, 1, 1]])))]
        _ = ds[0]
    with pytest.raises(AssertionError):
        ds.data = [(path.name, {"label": "A"})]
        _ = ds[0]

    # Patch some data
    ds.data = [(path.name, np.array([0]))]

    # Fetch the img
    sample = ds[0]
    img, target = sample.image, sample.target
    assert isinstance(target, np.ndarray)
    assert np.array_equal(target, np.array([0]))

    # Check img_transforms
    def img_transform(sample):
        sample.image = 1 - sample.image
        return sample

    ds.img_transforms = img_transform

    sample2 = ds[0]
    img2, target2 = sample2.image, sample2.target

    assert np.all(img2.numpy() == 1 - img.numpy())
    assert np.array_equal(target, target2)

    # Check sample_transforms
    ds.img_transforms = None

    def sample_transform(sample):
        sample.target = sample.target + 1
        return sample

    ds.sample_transforms = sample_transform

    sample3 = ds[0]
    img3, target3 = sample3.image, sample3.target

    assert np.all(img3.numpy() == img.numpy())
    assert np.array_equal(target3, target + 1)

    # Check inplace modifications
    ds.data = [(ds.data[0][0], "A")]

    def inplace_transfo(sample):
        sample.target += "B"
        return sample

    ds.sample_transforms = inplace_transfo

    t = ds[0].target
    t = ds[0].target

    assert t == "AB"
