import tensorflow as tf
import pytest
import json
import numpy as np

from doctr.datasets.detection import DetectionDataset


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

    ds = DetectionDataset(
        input_size=(1024, 1024),
        img_folder=mock_image_folder,
        labels_path=mock_detection_label
    )

    assert ds.__len__() == 5
    img, boxes, flags = ds[0]
    assert isinstance(img, tf.Tensor)
    assert img.shape[:2] == (1024, 1024)
    # Bounding boxes
    assert isinstance(boxes, np.ndarray) and boxes.dtype == np.float32
    assert np.all(np.logical_and(boxes >= 0, boxes <= 1))
    assert boxes.shape[1] == 4
    # Flags
    assert isinstance(flags, np.ndarray) and flags.dtype == np.bool
    # Cardinality consistency
    assert boxes.shape[0] == flags.shape[0]
