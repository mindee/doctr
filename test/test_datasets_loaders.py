from doctr.datasets import loaders
from typing import Tuple
import tensorflow as tf


def test_detection_core_generator(mock_image_folder, mock_detection_label):
    core_loader = loaders.detection.core.DataGenerator(
        input_size=(1024, 1024),
        images_path=mock_image_folder,
        labels_path=mock_detection_label,
        batch_size=1,
    )
    assert core_loader.__len__() == 5
    for _, batch in enumerate(core_loader):
        image, gt, mask = batch
        assert isinstance(image, tf.Tensor)
        assert isinstance(gt, tf.Tensor)
        assert isinstance(mask, tf.Tensor)
