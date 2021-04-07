import tensorflow as tf
import pytest
import json
import numpy as np

from doctr.datasets import RecognitionDataset


@pytest.fixture(scope="function")
def mock_recognition_label(tmpdir_factory):
    label_file = tmpdir_factory.mktemp("labels").join("labels.json")
    label = {
        "mock_image_file_0.jpeg": "I",
        "mock_image_file_1.jpeg": "am",
        "mock_image_file_2.jpeg": "a",
        "mock_image_file_3.jpeg": "jedi",
        "mock_image_file_4.jpeg": "!",
    }
    with open(label_file, 'w') as f:
        json.dump(label, f)
    return str(label_file)


def test_recognition_dataset(mock_image_folder, mock_recognition_label):
    ds = RecognitionDataset(
        input_size=(32, 128),
        img_folder=mock_image_folder,
        labels_path=mock_recognition_label
    )
    assert ds.__len__() == 5
    image, label = ds[0]
    assert isinstance(image, tf.Tensor)
    assert image.shape[:2] == (32, 128)
    assert isinstance(label, str)
