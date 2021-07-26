import pytest
import tensorflow as tf
import numpy as np

from doctr.io import read_img_as_tensor, decode_img_as_tensor


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
