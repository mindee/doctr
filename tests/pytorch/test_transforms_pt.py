import math

import numpy as np
import pytest
import torch

from doctr.transforms import ChannelShuffle, ColorInversion, RandomCrop, RandomRotate, Resize
from doctr.transforms.functional import crop_detection, rotate


def test_resize():
    output_size = (32, 32)
    transfo = Resize(output_size)
    input_t = torch.ones((3, 64, 64), dtype=torch.float32)
    out = transfo(input_t)

    assert torch.all(out == 1)
    assert out.shape[-2:] == output_size
    assert repr(transfo) == f"Resize(output_size={output_size}, interpolation='bilinear')"

    transfo = Resize(output_size, preserve_aspect_ratio=True)
    input_t = torch.ones((3, 32, 64), dtype=torch.float32)
    out = transfo(input_t)

    assert out.shape[-2:] == output_size
    assert not torch.all(out == 1)
    # Asymetric padding
    assert torch.all(out[:, -1] == 0) and torch.all(out[:, 0] == 1)

    # Symetric padding
    transfo = Resize(output_size, preserve_aspect_ratio=True, symmetric_pad=True)
    assert repr(transfo) == (f"Resize(output_size={output_size}, interpolation='bilinear', "
                             f"preserve_aspect_ratio=True, symmetric_pad=True)")
    out = transfo(input_t)
    assert out.shape[-2:] == output_size
    # symetric padding
    assert torch.all(out[:, -1] == 0) and torch.all(out[:, 0] == 0)

    # Inverse aspect ratio
    input_t = torch.ones((3, 64, 32), dtype=torch.float32)
    out = transfo(input_t)

    assert not torch.all(out == 1)
    assert out.shape[-2:] == output_size

    # Same aspect ratio
    output_size = (32, 128)
    transfo = Resize(output_size, preserve_aspect_ratio=True)
    out = transfo(torch.ones((3, 16, 64), dtype=torch.float32))
    assert out.shape[-2:] == output_size

    # FP16
    input_t = torch.ones((3, 64, 64), dtype=torch.float16)
    out = transfo(input_t)
    assert out.dtype == torch.float16


@pytest.mark.parametrize(
    "rgb_min",
    [
        0.2,
        0.4,
        0.6,
    ],
)
def test_invert_colorize(rgb_min):

    transfo = ColorInversion(min_val=rgb_min)
    input_t = torch.ones((8, 3, 32, 32), dtype=torch.float32)
    out = transfo(input_t)
    assert torch.all(out <= 1 - rgb_min + 1e-4)
    assert torch.all(out >= 0)

    input_t = torch.full((8, 3, 32, 32), 255, dtype=torch.uint8)
    out = transfo(input_t)
    assert torch.all(out <= int(math.ceil(255 * (1 - rgb_min + 1e-4))))
    assert torch.all(out >= 0)

    # FP16
    input_t = torch.ones((8, 3, 32, 32), dtype=torch.float16)
    out = transfo(input_t)
    assert out.dtype == torch.float16


def test_rotate():
    input_t = torch.ones((3, 50, 50), dtype=torch.float32)
    boxes = np.array([
        [15, 20, 35, 30]
    ])
    r_img, r_boxes = rotate(input_t, boxes, angle=12., expand=False)
    assert r_img.shape == (3, 50, 50)
    assert r_img[0, 0, 0] == 0.

    # Expand
    r_img, r_boxes = rotate(input_t, boxes, angle=12., expand=True)
    assert r_img.shape == (3, 60, 60)
    # With the expansion, there should be a maximum of 1 pixel of the initial image on the first row
    assert r_img[0, 0, :].sum() <= 1

    # Relative coords
    rel_boxes = np.array([[.3, .4, .7, .6]])
    r_img, r_boxes = rotate(input_t, rel_boxes, angle=90)
    assert r_boxes.shape == (1, 4, 2)
    assert np.isclose(r_boxes, np.asarray([[[0.4, 0.7], [0.4, 0.3], [0.6, 0.3], [0.6, 0.7]]])).all()

    # FP16 (only on GPU)
    if torch.cuda.is_available():
        input_t = torch.ones((3, 50, 50), dtype=torch.float16).cuda()
        r_img, _ = rotate(input_t, boxes, angle=12.)
        assert r_img.dtype == torch.float16


def test_random_rotate():
    rotator = RandomRotate(max_angle=10., expand=False)
    input_t = torch.ones((3, 50, 50), dtype=torch.float32)
    boxes = np.array([
        [15, 20, 35, 30]
    ])
    r_img, r_boxes = rotator(input_t, boxes)
    assert r_img.shape == input_t.shape

    rotator = RandomRotate(max_angle=10., expand=True)
    r_img, r_boxes = rotator(input_t, boxes)
    assert r_img.shape != input_t.shape

    # FP16 (only on GPU)
    if torch.cuda.is_available():
        input_t = torch.ones((3, 50, 50), dtype=torch.float16).cuda()
        r_img, _ = rotator(input_t, boxes)
        assert r_img.dtype == torch.float16


def test_crop_detection():
    img = torch.ones((3, 50, 50), dtype=torch.float32)
    abs_boxes = np.array([
        [15, 20, 35, 30],
        [5, 10, 10, 20],
    ])
    crop_box = (12 / 50, 23 / 50, 50 / 50, 50 / 50)
    c_img, c_boxes = crop_detection(img, abs_boxes, crop_box)
    assert c_img.shape == (3, 26, 37)
    assert c_boxes.shape == (1, 4)
    rel_boxes = np.array([
        [.3, .4, .7, .6],
        [.1, .2, .2, .4],
    ])
    c_img, c_boxes = crop_detection(img, rel_boxes, crop_box)
    assert c_img.shape == (3, 26, 37)
    assert c_boxes.shape == (1, 4)

    # FP16
    img = torch.ones((3, 50, 50), dtype=torch.float16)
    c_img, _ = crop_detection(img, abs_boxes, crop_box)
    assert c_img.dtype == torch.float16


def test_random_crop():
    cropper = RandomCrop(scale=(0.5, 1.), ratio=(0.75, 1.33))
    input_t = torch.ones((3, 50, 50), dtype=torch.float32)
    boxes = np.array([
        [15, 20, 35, 30]
    ])
    img, target = cropper(input_t, dict(boxes=boxes))
    # Check the scale
    assert img.shape[-1] * img.shape[-2] >= 0.5 * input_t.shape[-1] * input_t.shape[-2]
    # Check aspect ratio
    assert 0.75 <= img.shape[-2] / img.shape[-1] <= 1.33
    # Check the target
    assert np.all(boxes >= 0)
    assert np.all(boxes[:, [0, 2]] <= img.shape[-1]) and np.all(boxes[:, [1, 3]] <= img.shape[-2])


@pytest.mark.parametrize(
    "input_dtype, input_size",
    [
        [torch.float32, (3, 32, 32)],
        [torch.uint8, (3, 32, 32)],
    ],
)
def test_channel_shuffle(input_dtype, input_size):
    transfo = ChannelShuffle()
    input_t = torch.rand(input_size, dtype=torch.float32)
    if input_dtype == torch.uint8:
        input_t = (255 * input_t).round()
    input_t = input_t.to(dtype=input_dtype)
    out = transfo(input_t)
    assert isinstance(out, torch.Tensor)
    assert out.shape == input_size
    assert out.dtype == input_dtype
    # Ensure that nothing has changed apart from channel order
    if input_dtype == torch.uint8:
        assert torch.all(input_t.sum(0) == out.sum(0))
    else:
        # Float approximation
        assert (input_t.sum(0) - out.sum(0)).abs().mean() < 1e-7
