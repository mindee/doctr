import numpy as np
import pytest
import torch
from PIL import Image

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


def test_read_img_as_tensor_exif_orientation(tmpdir_factory):
    folder = tmpdir_factory.mktemp("images")
    path = str(folder.join("exif_o6.jpg"))
    pil_img = Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8))
    exif = pil_img.getexif()
    exif[0x0112] = 6  # orientation tag
    pil_img.save(path, format="JPEG", exif=exif)

    img = read_img_as_tensor(path)
    assert img.shape == (3, 60, 40)

    with open(path, "rb") as f:
        img_stream = decode_img_as_tensor(f.read())
    assert img_stream.shape == (3, 60, 40)


def test_img_as_tensor_transparency(tmpdir_factory):
    # Opaque red square on a transparent black background, placed on white like `read_img_as_numpy` does
    rgba = np.zeros((40, 60, 4), dtype=np.uint8)
    rgba[10:30, 20:40] = (255, 0, 0, 255)
    path = str(tmpdir_factory.mktemp("images").join("transparent.png"))
    Image.fromarray(rgba, "RGBA").save(path, format="PNG")
    with open(path, "rb") as f:
        img_stream = f.read()

    for img in (read_img_as_tensor(path, dtype=torch.uint8), decode_img_as_tensor(img_stream, dtype=torch.uint8)):
        assert img.shape == (3, 40, 60)
        assert torch.all(img[:, 0, 0] == 255)
        assert img[:, 20, 30].tolist() == [255, 0, 0]
    assert torch.all(read_img_as_tensor(path)[:, 0, 0] == 1)
    assert torch.all(decode_img_as_tensor(img_stream)[:, 0, 0] == 1)
