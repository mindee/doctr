import pytest
import json
import numpy as np
import tensorflow as tf
from io import BytesIO

from doctr import datasets
from doctr.datasets import DataLoader
from doctr.transforms import Resize


@pytest.fixture(scope="function")
def mock_ocrdataset(tmpdir_factory, mock_image_stream):
    root = tmpdir_factory.mktemp("dataset")
    label_file = root.join("labels.json")
    label = [
        {
            "raw-archive-filepath": "mock_image_file_0.jpg",
            "coordinates": [[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]],
            "string": ['I']
        },
        {
            "raw-archive-filepath": "mock_image_file_1.jpg",
            "coordinates": [[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]],
            "string": ['am']
        },
        {
            "raw-archive-filepath": "mock_image_file_2.jpg",
            "coordinates": [[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]],
            "string": ['a']
        },
        {
            "raw-archive-filepath": "mock_image_file_3.jpg",
            "coordinates": [[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]],
            "string": ['jedi']
        },
        {
            "raw-archive-filepath": "mock_image_file_4.jpg",
            "coordinates": [[[0.1, 0.2], [0.2, 0.2], [0.2, 0.3], [0.1, 0.3]]],
            "string": ['!']
        }
    ]
    with open(label_file, 'w') as f:
        json.dump(label, f)

    file = BytesIO(mock_image_stream)
    image_folder = tmpdir_factory.mktemp("images")
    for i in range(5):
        fn = image_folder.join(f"mock_image_file_{i}.jpg")
        with open(fn, 'wb') as f:
            f.write(file.getbuffer())

    return str(image_folder), str(label_file)


def test_ocrdataset(mock_ocrdataset):

    input_size = (512, 512)

    ds = datasets.OCRDataset(
        *mock_ocrdataset,
        sample_transforms=Resize(input_size),
    )
    assert len(ds) == 5
    img, target = ds[0]
    assert isinstance(img, tf.Tensor)
    assert img.shape[:2] == input_size
    # Bounding boxes
    assert isinstance(target['boxes'], np.ndarray) and target['boxes'].dtype == np.float32
    assert np.all(np.logical_and(target['boxes'] >= 0, target['boxes'] <= 1))
    assert target['boxes'].shape[1] == 4
    # Flags
    assert isinstance(target['labels'], list) and all(isinstance(s, str) for s in target['labels'])
    # Cardinality consistency
    assert target['boxes'].shape[0] == len(target['labels'])

    loader = DataLoader(ds, batch_size=2)
    images, targets = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)
