import tensorflow as tf
import pytest
import json
import numpy as np

from doctr.datasets import DetectionDataGenerator, RecognitionDataGenerator


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


def test_detection_core_generator(mock_image_folder, mock_detection_label):
    core_loader = DetectionDataGenerator(
        input_size=(1024, 1024),
        images_path=mock_image_folder,
        labels_path=mock_detection_label,
        batch_size=2,
    )
    assert core_loader.__len__() == 3
    for _, (batch_images, batch_boxes, batch_flags) in enumerate(core_loader):
        assert isinstance(batch_images, tf.Tensor)
        assert batch_images.shape[1] == batch_images.shape[2] == 1024
        assert isinstance(batch_boxes, list) and isinstance(batch_flags, list)
        assert all(isinstance(boxes, np.ndarray) and boxes.dtype == np.float32 for boxes in batch_boxes)
        assert all(np.all(np.logical_and(boxes >= 0, boxes <= 1)) for boxes in batch_boxes)
        assert all(boxes.shape[1] == 4 for boxes in batch_boxes)
        assert all(isinstance(flags, np.ndarray) and flags.dtype == np.bool for flags in batch_flags)
        assert all(boxes.shape[0] == flags.shape[0] for boxes, flags in zip(batch_boxes, batch_flags))


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
        images_path=mock_image_folder,
        labels_path=mock_recognition_label,
        batch_size=2,
    )
    assert core_loader.__len__() == 3
    for _, batch in enumerate(core_loader):
        images, gts = batch
        assert isinstance(images, tf.Tensor)
        assert images.shape[1] == 32
        assert images.shape[2] == 128
        assert isinstance(gts, list)
        assert len(gts) == images.shape[0]
        for gt in gts:
            assert isinstance(gt, str)
