import tensorflow as tf
import pytest
import json

from doctr.datasets import RecognitionDataset
from doctr.datasets import DataLoader
from doctr.transforms import Resize


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
    input_size = (32, 128)
    ds = RecognitionDataset(
        img_folder=mock_image_folder,
        labels_path=mock_recognition_label,
        sample_transforms=Resize(input_size),
    )
    assert ds.__len__() == 5
    image, label = ds[0]
    assert isinstance(image, tf.Tensor)
    assert image.shape[:2] == input_size
    assert isinstance(label, str)

    loader = DataLoader(ds, batch_size=2)
    images, labels = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(labels, list) and all(isinstance(elt, str) for elt in labels)
