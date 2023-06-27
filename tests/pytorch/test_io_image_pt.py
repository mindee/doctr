import numpy as np
import pytest
import torch

from doctr.io import decode_img_as_tensor, read_img_as_tensor, tensor_from_numpy


def test_read_img_as_tensor(mock_image_path):
    img = read_img_as_tensor(mock_image_path)

    assert isinstance(img, torch.Tensor)
    assert img.dtype == torch.float32
    assert img.shape == (3, 900, 1200)

    img = read_img_as_tensor(mock_image_path, dtype=torch.float16)
    assert img.dtype == torch.float16
    img = read_img_as_tensor(mock_image_path, dtype=torch.uint8)
    assert img.dtype == torch.uint8

    with pytest.raises(ValueError):
        read_img_as_tensor(mock_image_path, dtype=torch.float64)


def test_decode_img_as_tensor(mock_image_stream):
    img = decode_img_as_tensor(mock_image_stream)

    assert isinstance(img, torch.Tensor)
    assert img.dtype == torch.float32
    assert img.shape == (3, 900, 1200)

    img = decode_img_as_tensor(mock_image_stream, dtype=torch.float16)
    assert img.dtype == torch.float16
    img = decode_img_as_tensor(mock_image_stream, dtype=torch.uint8)
    assert img.dtype == torch.uint8

    with pytest.raises(ValueError):
        decode_img_as_tensor(mock_image_stream, dtype=torch.float64)


def test_tensor_from_numpy(mock_image_stream):
    with pytest.raises(ValueError):
        tensor_from_numpy(np.zeros((256, 256, 3)), torch.int64)

    out = tensor_from_numpy(np.zeros((256, 256, 3), dtype=np.uint8))

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (3, 256, 256)

    out = tensor_from_numpy(np.zeros((256, 256, 3), dtype=np.uint8), dtype=torch.float16)
    assert out.dtype == torch.float16
    out = tensor_from_numpy(np.zeros((256, 256, 3), dtype=np.uint8), dtype=torch.uint8)
    assert out.dtype == torch.uint8
