import pytest
import torch
import numpy as np

from doctr.io import read_img_as_tensor, decode_img_as_tensor


def test_read_img_as_tensor(mock_image_path):

    img = read_img_as_tensor(mock_image_path)

    assert isinstance(img, torch.Tensor)
    assert img.dtype == torch.float32
    assert img.shape == (3, 900, 1200)

    img = read_img_as_tensor(mock_image_path, dtype=torch.float16)
    assert img.dtype == torch.float16
    img = read_img_as_tensor(mock_image_path, dtype=torch.uint8)
    assert img.dtype == torch.uint8


def test_decode_img_as_tensor(mock_image_stream):

    img = decode_img_as_tensor(mock_image_stream)

    assert isinstance(img, torch.Tensor)
    assert img.dtype == torch.float32
    assert img.shape == (3, 900, 1200)

    img = decode_img_as_tensor(mock_image_stream, dtype=torch.float16)
    assert img.dtype == torch.float16
    img = decode_img_as_tensor(mock_image_stream, dtype=torch.uint8)
    assert img.dtype == torch.uint8
