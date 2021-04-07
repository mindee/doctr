import tensorflow as tf
import pytest
import json
import numpy as np

from doctr.datasets import RecognitionDataGenerator
# from doctr.datasets import DataLoader, DetectionDataset


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


def test_recognition_core_generator(mock_image_folder, mock_recognition_label):
    core_loader = RecognitionDataGenerator(
        input_size=(32, 128),
        img_folder=mock_image_folder,
        labels_path=mock_recognition_label,
        batch_size=2,
    )
    assert core_loader.__len__() == 3
    for _, (images, labels) in enumerate(core_loader):
        assert isinstance(images, tf.Tensor)
        assert images.shape[1] == 32 and images.shape[2] == 128
        assert isinstance(labels, list) and all(isinstance(elt, str) for elt in labels)
        assert len(labels) == images.shape[0]
