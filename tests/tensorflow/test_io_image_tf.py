import numpy as np
import pytest
import tensorflow as tf

from doctr.io import decode_img_as_tensor, read_img_as_tensor, tensor_from_numpy


def test_read_img_as_tensor(mock_image_path):
    img = read_img_as_tensor(mock_image_path)

    assert isinstance(img, tf.Tensor)
    assert img.dtype == tf.float32
    assert img.shape == (900, 1200, 3)

    img = read_img_as_tensor(mock_image_path, dtype=tf.float16)
    assert img.dtype == tf.float16
    img = read_img_as_tensor(mock_image_path, dtype=tf.uint8)
    assert img.dtype == tf.uint8


def test_decode_img_as_tensor(mock_image_stream):
    img = decode_img_as_tensor(mock_image_stream)

    assert isinstance(img, tf.Tensor)
    assert img.dtype == tf.float32
    assert img.shape == (900, 1200, 3)

    img = decode_img_as_tensor(mock_image_stream, dtype=tf.float16)
    assert img.dtype == tf.float16
    img = decode_img_as_tensor(mock_image_stream, dtype=tf.uint8)
    assert img.dtype == tf.uint8


def test_tensor_from_numpy(mock_image_stream):
    with pytest.raises(ValueError):
        tensor_from_numpy(np.zeros((256, 256, 3)), tf.int64)

    out = tensor_from_numpy(np.zeros((256, 256, 3), dtype=np.uint8))

    assert isinstance(out, tf.Tensor)
    assert out.dtype == tf.float32
    assert out.shape == (256, 256, 3)

    out = tensor_from_numpy(np.zeros((256, 256, 3), dtype=np.uint8), dtype=tf.float16)
    assert out.dtype == tf.float16
    out = tensor_from_numpy(np.zeros((256, 256, 3), dtype=np.uint8), dtype=tf.uint8)
    assert out.dtype == tf.uint8
