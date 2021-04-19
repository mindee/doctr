import tensorflow as tf
import pytest
import json
import numpy as np

from doctr.datasets.detection import DetectionDataset
from doctr.datasets import DataLoader
from doctr.transforms import Resize


@pytest.fixture(scope="function")
def mock_detection_label(tmpdir_factory):
    folder = tmpdir_factory.mktemp("labels")
    label = {
        "boxes_1": [[[1, 2], [1, 3], [2, 1], [2, 3]], [[10, 20], [10, 30], [20, 10], [20, 30]]],
        "boxes_2": [[[3, 2], [3, 3], [4, 1], [4, 3]], [[30, 20], [30, 30], [40, 10], [40, 30]]],
        "boxes_3": [[[1, 5], [1, 8], [2, 5], [2, 8]]],
    }
    for i in range(5):
        fn = folder.join("mock_image_file_" + str(i) + ".jpeg.json")
        with open(fn, 'w') as f:
            json.dump(label, f)
    return str(folder)


def test_detection_dataset(mock_image_folder, mock_detection_label):

    input_size = (1024, 1024)

    ds = DetectionDataset(
        img_folder=mock_image_folder,
        label_folder=mock_detection_label,
        sample_transforms=Resize(input_size),
    )

    assert ds.__len__() == 5
    img, target = ds[0]
    assert isinstance(img, tf.Tensor)
    assert img.shape[:2] == input_size
    # Bounding boxes
    assert isinstance(target['boxes'], np.ndarray) and target['boxes'].dtype == np.float32
    assert np.all(np.logical_and(target['boxes'] >= 0, target['boxes'] <= 1))
    assert target['boxes'].shape[1] == 4
    # Flags
    assert isinstance(target['flags'], np.ndarray) and target['flags'].dtype == np.bool
    # Cardinality consistency
    assert target['boxes'].shape[0] == target['flags'].shape[0]

    loader = DataLoader(ds, batch_size=2)
    images, targets = next(iter(loader))
    assert isinstance(images, tf.Tensor) and images.shape == (2, *input_size, 3)
    assert isinstance(targets, list) and all(isinstance(elt, dict) for elt in targets)
