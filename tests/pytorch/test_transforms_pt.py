import math

import numpy as np
import pytest
import torch

from doctr.transforms import (
    ChannelShuffle,
    ColorInversion,
    GaussianBlur,
    GaussianNoise,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResize,
    RandomRotate,
    RandomShadow,
    Resize,
)
from doctr.transforms.functional import crop_detection, rotate_sample


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
    assert repr(transfo) == (
        f"Resize(output_size={output_size}, interpolation='bilinear', preserve_aspect_ratio=True, symmetric_pad=True)"
    )
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

    # --- Test with target (bounding boxes) ---

    target_boxes = np.array([[0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8]])
    output_size = (64, 64)

    transfo = Resize(output_size, preserve_aspect_ratio=True)
    input_t = torch.ones((3, 32, 64), dtype=torch.float32)
    out, new_target = transfo(input_t, target_boxes)

    assert out.shape[-2:] == output_size
    assert new_target.shape == target_boxes.shape
    assert np.all(new_target >= 0) and np.all(new_target <= 1)

    out = transfo(input_t)
    assert out.shape[-2:] == output_size


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


def test_rotate_sample():
    img = torch.ones((3, 200, 100), dtype=torch.float32)
    boxes = np.array([0, 0, 100, 200])[None, ...]
    polys = np.stack((boxes[..., [0, 1]], boxes[..., [2, 1]], boxes[..., [2, 3]], boxes[..., [0, 3]]), axis=1)
    rel_boxes = np.array([0, 0, 1, 1], dtype=np.float32)[None, ...]
    rel_polys = np.stack(
        (rel_boxes[..., [0, 1]], rel_boxes[..., [2, 1]], rel_boxes[..., [2, 3]], rel_boxes[..., [0, 3]]), axis=1
    )

    # No angle
    rotated_img, rotated_geoms = rotate_sample(img, boxes, 0, False)
    assert torch.all(rotated_img == img) and np.all(rotated_geoms == rel_polys)
    rotated_img, rotated_geoms = rotate_sample(img, boxes, 0, True)
    assert torch.all(rotated_img == img) and np.all(rotated_geoms == rel_polys)
    rotated_img, rotated_geoms = rotate_sample(img, polys, 0, False)
    assert torch.all(rotated_img == img) and np.all(rotated_geoms == rel_polys)
    rotated_img, rotated_geoms = rotate_sample(img, polys, 0, True)
    assert torch.all(rotated_img == img) and np.all(rotated_geoms == rel_polys)

    # No expansion
    expected_img = torch.zeros((3, 200, 100), dtype=torch.float32)
    expected_img[:, 50:150] = 1
    expected_polys = np.array([[0, 0.75], [0, 0.25], [1, 0.25], [1, 0.75]])[None, ...]
    rotated_img, rotated_geoms = rotate_sample(img, boxes, 90, False)
    assert torch.all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)
    rotated_img, rotated_geoms = rotate_sample(img, polys, 90, False)
    assert torch.all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)
    rotated_img, rotated_geoms = rotate_sample(img, rel_boxes, 90, False)
    assert torch.all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)
    rotated_img, rotated_geoms = rotate_sample(img, rel_polys, 90, False)
    assert torch.all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)

    # Expansion
    expected_img = torch.ones((3, 100, 200), dtype=torch.float32)
    expected_polys = np.array([[0, 1], [0, 0], [1, 0], [1, 1]], dtype=np.float32)[None, ...]
    rotated_img, rotated_geoms = rotate_sample(img, boxes, 90, True)
    assert torch.all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)
    rotated_img, rotated_geoms = rotate_sample(img, polys, 90, True)
    assert torch.all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)
    rotated_img, rotated_geoms = rotate_sample(img, rel_boxes, 90, True)
    assert torch.all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)
    rotated_img, rotated_geoms = rotate_sample(img, rel_polys, 90, True)
    assert torch.all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)

    with pytest.raises(AssertionError):
        rotate_sample(img, boxes[None, ...], 90, False)


def test_random_rotate():
    rotator = RandomRotate(max_angle=10.0, expand=False)
    input_t = torch.ones((3, 50, 50), dtype=torch.float32)
    boxes = np.array([[15, 20, 35, 30]])
    r_img, _r_boxes = rotator(input_t, boxes)
    assert r_img.shape == input_t.shape

    rotator = RandomRotate(max_angle=10.0, expand=True)
    r_img, _r_boxes = rotator(input_t, boxes)
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
    assert np.all(c_boxes == np.array([15 - 12, 0, 35 - 12, 30 - 23])[None, ...])

    rel_boxes = np.array([
        [0.3, 0.4, 0.7, 0.6],
        [0.1, 0.2, 0.2, 0.4],
    ])
    crop_box = (0.24, 0.46, 1.0, 1.0)
    c_img, c_boxes = crop_detection(img, rel_boxes, crop_box)
    assert c_img.shape == (3, 26, 37)
    assert c_boxes.shape == (1, 4)
    assert np.abs(c_boxes - np.array([0.06 / 0.76, 0.0, 0.46 / 0.76, 0.14 / 0.54])[None, ...]).mean() < 1e-7

    # FP16
    img = torch.ones((3, 50, 50), dtype=torch.float16)
    c_img, _ = crop_detection(img, abs_boxes, crop_box)
    assert c_img.dtype == torch.float16

    with pytest.raises(AssertionError):
        crop_detection(img, abs_boxes, (2, 6, 24, 56))


@pytest.mark.parametrize(
    "target",
    [
        np.array([[15, 20, 35, 30]]),  # box
        np.array([[[15, 20], [35, 20], [35, 30], [15, 30]]]),  # polygon
    ],
)
def test_random_crop(target):
    cropper = RandomCrop(scale=(0.5, 1.0), ratio=(0.75, 1.33))
    input_t = torch.ones((3, 50, 50), dtype=torch.float32)
    img, target = cropper(input_t, target)
    # Check the scale
    assert img.shape[-1] * img.shape[-2] >= 0.4 * input_t.shape[-1] * input_t.shape[-2]
    # Check aspect ratio
    assert 0.65 <= img.shape[-2] / img.shape[-1] <= 1.6
    # Check the target
    assert np.all(target >= 0)
    if target.ndim == 2:
        assert np.all(target[:, [0, 2]] <= img.shape[-1]) and np.all(target[:, [1, 3]] <= img.shape[-2])
    else:
        assert np.all(target[..., 0] <= img.shape[-1]) and np.all(target[..., 1] <= img.shape[-2])


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


@pytest.mark.parametrize(
    "input_dtype,input_shape",
    [
        [torch.float32, (3, 32, 32)],
        [torch.uint8, (3, 32, 32)],
    ],
)
def test_gaussian_noise(input_dtype, input_shape):
    transform = GaussianNoise(0.0, 1.0)
    input_t = torch.rand(input_shape, dtype=torch.float32)
    if input_dtype == torch.uint8:
        input_t = (255 * input_t).round()
    input_t = input_t.to(dtype=input_dtype)
    transformed = transform(input_t)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == input_shape
    assert transformed.dtype == input_dtype
    assert torch.any(transformed != input_t)
    assert torch.all(transformed >= 0)
    if input_dtype == torch.uint8:
        assert torch.all(transformed <= 255)
    else:
        assert torch.all(transformed <= 1.0)


@pytest.mark.parametrize(
    "input_dtype, input_shape",
    [
        [torch.float32, (3, 32, 32)],
        [torch.uint8, (3, 32, 32)],
    ],
)
def test_gaussian_blur(input_dtype, input_shape):
    sigma_range = (0.5, 1.5)
    transform = GaussianBlur(sigma=sigma_range)

    input_t = torch.rand(input_shape, dtype=torch.float32)

    if input_dtype == torch.uint8:
        input_t = (255 * input_t).round().to(dtype=torch.uint8)

    blurred = transform(input_t)

    assert isinstance(blurred, torch.Tensor)
    assert blurred.shape == input_shape
    assert blurred.dtype == input_dtype

    if input_dtype == torch.uint8:
        assert torch.any(blurred != input_t)
        assert torch.all(blurred <= 255)
        assert torch.all(blurred >= 0)
    else:
        assert torch.any(blurred != input_t)
        assert torch.all(blurred <= 1.0)
        assert torch.all(blurred >= 0.0)


@pytest.mark.parametrize(
    "p,target",
    [
        [1, np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32)],
        [0, np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32)],
        [1, np.array([[[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]]], dtype=np.float32)],
        [0, np.array([[[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]]], dtype=np.float32)],
    ],
)
def test_randomhorizontalflip(p, target):
    # testing for 2 cases, with flip probability 1 and 0.
    transform = RandomHorizontalFlip(p)
    input_t = torch.ones((3, 32, 32), dtype=torch.float32)
    input_t[..., :16] = 0

    transformed, _target = transform(input_t, target)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == input_t.shape
    assert transformed.dtype == input_t.dtype
    # integrity check of targets
    assert isinstance(_target, np.ndarray)
    assert _target.dtype == np.float32
    if _target.ndim == 2:
        if p == 1:
            assert np.all(_target == np.array([[0.7, 0.1, 0.9, 0.4]], dtype=np.float32))
            assert torch.all(transformed.mean((0, 1)) == torch.tensor([1] * 16 + [0] * 16, dtype=torch.float32))
        elif p == 0:
            assert np.all(_target == np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32))
            assert torch.all(transformed.mean((0, 1)) == torch.tensor([0] * 16 + [1] * 16, dtype=torch.float32))
    else:
        if p == 1:
            assert np.all(_target == np.array([[[0.9, 0.1], [0.7, 0.1], [0.7, 0.4], [0.9, 0.4]]], dtype=np.float32))
            assert torch.all(transformed.mean((0, 1)) == torch.tensor([1] * 16 + [0] * 16, dtype=torch.float32))
        elif p == 0:
            assert np.all(_target == np.array([[[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]]], dtype=np.float32))
            assert torch.all(transformed.mean((0, 1)) == torch.tensor([0] * 16 + [1] * 16, dtype=torch.float32))


@pytest.mark.parametrize(
    "input_dtype,input_shape",
    [
        [torch.float32, (3, 32, 32)],
        [torch.uint8, (3, 32, 32)],
        [torch.float32, (3, 64, 32)],
        [torch.uint8, (3, 64, 32)],
    ],
)
def test_random_shadow(input_dtype, input_shape):
    transform = RandomShadow((0.2, 0.8))
    input_t = torch.ones(input_shape, dtype=torch.float32)
    if input_dtype == torch.uint8:
        input_t = (255 * input_t).round()
    input_t = input_t.to(dtype=input_dtype)
    transformed = transform(input_t)
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == input_shape
    assert transformed.dtype == input_dtype
    # The shadow will darken the picture
    assert input_t.float().mean() >= transformed.float().mean()
    assert torch.all(transformed >= 0)
    if input_dtype == torch.uint8:
        assert torch.all(transformed <= 255)
    else:
        assert torch.all(transformed <= 1.0)


@pytest.mark.parametrize(
    "p,preserve_aspect_ratio,symmetric_pad,target",
    [
        [1, True, False, np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32)],
        [0, True, False, np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32)],
        [1, True, False, np.array([[[0.1, 0.8], [0.3, 0.1], [0.3, 0.4], [0.8, 0.4]]], dtype=np.float32)],
        [0, True, False, np.array([[[0.1, 0.8], [0.3, 0.1], [0.3, 0.4], [0.8, 0.4]]], dtype=np.float32)],
        [1, 0.5, 0.5, np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32)],
        [0, 0.5, 0.5, np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32)],
    ],
)
def test_random_resize(p, preserve_aspect_ratio, symmetric_pad, target):
    transfo = RandomResize(
        scale_range=(0.3, 1.3), preserve_aspect_ratio=preserve_aspect_ratio, symmetric_pad=symmetric_pad, p=p
    )
    assert (
        repr(transfo)
        == f"RandomResize(scale_range=(0.3, 1.3), preserve_aspect_ratio={preserve_aspect_ratio}, symmetric_pad={symmetric_pad}, p={p})"  # noqa: E501
    )

    img = torch.rand((3, 64, 64))
    # Apply the transformation
    out_img, out_target = transfo(img, target)
    assert isinstance(out_img, torch.Tensor)
    assert isinstance(out_target, np.ndarray)
    # Resize is already well tested
    assert torch.all(out_img == img) if p == 0 else out_img.shape != img.shape
    assert out_target.shape == target.shape
