import math
import random

import numpy as np
import pytest
import torch
from torchvision.transforms.v2 import RandomGrayscale

from doctr.transforms import (
    ChannelShuffle,
    ColorInversion,
    GaussianBlur,
    GaussianNoise,
    ImageTorchvisionTransform,
    OneOf,
    RandomApply,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResize,
    RandomRotate,
    RandomShadow,
    Resize,
    SampleCompose,
)
from doctr.transforms.functional import crop_detection, rotate_sample
from doctr.utils import Sample


def test_resize():
    output_size = (32, 32)
    transfo = Resize(output_size)
    input_t = Sample(image=torch.ones((3, 64, 64), dtype=torch.float32))
    out = transfo(input_t).image

    assert torch.all(out == 1)
    assert out.shape[-2:] == output_size
    assert repr(transfo) == "Resize(output_size=(32, 32), interpolation='bilinear')"

    # Test return_padding_mask without aspect ratio
    transfo = Resize(output_size, return_padding_mask=True)
    data = transfo(input_t)
    out, mask = data.image, data.mask
    assert out.shape[-2:] == output_size
    assert mask.shape == output_size
    assert mask.dtype == torch.bool
    assert torch.all(mask == 0)

    # Test with preserve_aspect_ratio
    output_size = (32, 32)
    input_t = Sample(image=torch.ones((3, 32, 64), dtype=torch.float32))

    # Asymmetric padding
    transfo = Resize(output_size, preserve_aspect_ratio=True)
    out = transfo(input_t).image
    assert out.shape[-2:] == output_size
    assert not torch.all(out == 1)
    assert torch.all(out[:, -1] == 0) and torch.all(out[:, 0] == 1)

    # Asymmetric padding mask
    transfo = Resize(output_size, preserve_aspect_ratio=True, return_padding_mask=True)
    data = transfo(input_t)
    out, mask = data.image, data.mask
    assert mask.shape == output_size
    assert mask.dtype == torch.bool
    assert mask.any()
    assert torch.any(mask[:, -5:])
    assert torch.any(mask[:, 5:])

    # Symmetric padding
    transfo = Resize(32, preserve_aspect_ratio=True, symmetric_pad=True)
    out = transfo(input_t).image
    assert out.shape[-2:] == output_size
    assert torch.all(out[:, 0] == 0) and torch.all(out[:, -1] == 0)

    # Symmetric padding mask
    transfo = Resize(32, preserve_aspect_ratio=True, symmetric_pad=True, return_padding_mask=True)
    data = transfo(input_t)
    out, mask = data.image, data.mask
    assert mask.shape == output_size
    assert mask.dtype == torch.bool
    assert mask.any()
    assert torch.any(mask[:, :5])
    assert torch.any(mask[:, -5:])

    expected = "Resize(output_size=(32, 32), interpolation='bilinear', preserve_aspect_ratio=True, symmetric_pad=True)"
    assert repr(transfo) == expected

    # Test with inverse resize
    input_t = Sample(image=torch.ones((3, 64, 32), dtype=torch.float32))
    transfo = Resize(32, preserve_aspect_ratio=True, symmetric_pad=True)
    out = transfo(input_t).image
    assert out.shape[-2:] == (32, 32)

    # Test resize with same ratio
    transfo = Resize((32, 128), preserve_aspect_ratio=True)
    out = transfo(Sample(image=torch.ones((3, 16, 64), dtype=torch.float32))).image
    assert out.shape[-2:] == (32, 128)

    # Test with fp16 input
    transfo = Resize((32, 128), preserve_aspect_ratio=True)
    input_t = Sample(image=torch.ones((3, 64, 64), dtype=torch.float16))
    out = transfo(input_t).image
    assert out.dtype == torch.float16

    padding = [True, False]
    for symmetric_pad in padding:
        # Test with target boxes
        target_boxes = np.array([[0.1, 0.1, 0.3, 0.4], [0.2, 0.2, 0.8, 0.8]])
        transfo = Resize((64, 64), preserve_aspect_ratio=True, symmetric_pad=symmetric_pad, return_padding_mask=True)
        input_t = Sample(image=torch.ones((3, 32, 64), dtype=torch.float32), target=target_boxes)
        data = transfo(input_t)
        out, mask, new_target = data.image, data.mask, data.target

        assert out.shape[-2:] == (64, 64)
        assert new_target.shape == target_boxes.shape
        assert np.all((0 <= new_target) & (new_target <= 1))
        assert mask.shape == (64, 64)
        assert mask.dtype == torch.bool

        # Test with target polygons
        target_boxes = np.array([
            [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        ])
        transfo = Resize((64, 64), preserve_aspect_ratio=True, symmetric_pad=symmetric_pad, return_padding_mask=True)
        input_t = Sample(image=torch.ones((3, 32, 64), dtype=torch.float32), target=target_boxes)
        data = transfo(input_t)
        out, mask, new_target = data.image, data.mask, data.target

        assert out.shape[-2:] == (64, 64)
        assert new_target.shape == target_boxes.shape
        assert np.all((0 <= new_target) & (new_target <= 1))
        assert mask.shape == (64, 64)
        assert mask.dtype == torch.bool

    # Test with invalid target shape
    input_t = torch.ones((3, 32, 64), dtype=torch.float32)
    target = np.ones((2, 5))  # Invalid shape

    transfo = Resize((64, 64), preserve_aspect_ratio=True)
    with pytest.raises(AssertionError):
        transfo(Sample(image=input_t, target=target))

    # Test dict targets
    target_dict = {
        "boxes": np.array([[0.1, 0.1, 0.9, 0.9]], dtype=np.float32),
        "polygons": np.array([[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]], dtype=np.float32),
    }
    transfo = Resize(
        (64, 64),
        preserve_aspect_ratio=True,
        symmetric_pad=True,
    )
    new_target = transfo(Sample(image=input_t, target=target_dict)).target
    assert isinstance(new_target, dict)
    assert set(new_target.keys()) == {"boxes", "polygons"}
    assert new_target["boxes"].shape == (1, 4)
    assert new_target["polygons"].shape == (1, 4, 2)

    # Test return type combinations
    transfo = Resize((32, 32))
    out = transfo(Sample(image=input_t)).image
    assert isinstance(out, torch.Tensor)

    transfo = Resize((32, 32), return_padding_mask=True)
    out = transfo(Sample(image=input_t))
    assert isinstance(out, Sample)
    assert hasattr(out, "image") and hasattr(out, "mask")
    assert out.image.shape[-2:] == (32, 32)
    assert out.mask.shape[-2:] == (32, 32)

    transfo = Resize((32, 32), preserve_aspect_ratio=True)
    out = transfo(Sample(image=input_t, target=target_boxes))
    assert isinstance(out, Sample)
    assert hasattr(out, "image") and hasattr(out, "target")
    assert out.image.shape[-2:] == (32, 32)

    transfo = Resize(
        (32, 32),
        preserve_aspect_ratio=True,
        return_padding_mask=True,
    )

    out = transfo(Sample(image=input_t, target=target_boxes))
    assert isinstance(out, Sample)
    assert hasattr(out, "image") and hasattr(out, "mask") and hasattr(out, "target")
    assert out.image.shape[-2:] == (32, 32)
    assert out.mask.shape[-2:] == (32, 32)


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
    input_t = Sample(image=torch.ones((8, 3, 32, 32), dtype=torch.float32))
    out = transfo(input_t).image
    assert torch.all(out <= 1 - rgb_min + 1e-4)
    assert torch.all(out >= 0)

    input_t = Sample(image=torch.full((8, 3, 32, 32), 255, dtype=torch.uint8))
    out = transfo(input_t).image
    assert torch.all(out <= int(math.ceil(255 * (1 - rgb_min + 1e-4))))
    assert torch.all(out >= 0)

    # FP16
    input_t = Sample(image=torch.ones((8, 3, 32, 32), dtype=torch.float16))
    out = transfo(input_t).image
    assert out.dtype == torch.float16

    assert repr(transfo) == f"ColorInversion(min_val={rgb_min})"


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
    data = rotator(Sample(image=input_t, target=boxes))
    r_img = data.image
    assert r_img.shape == input_t.shape

    rotator = RandomRotate(max_angle=10.0, expand=True)
    data = rotator(Sample(image=input_t, target=boxes))
    r_img = data.image
    assert r_img.shape != input_t.shape

    assert repr(rotator) == "RandomRotate(max_angle=10.0, expand=True)"

    # Test dict targets
    dict_target = {
        "boxes": np.array([[15, 20, 35, 30]]),
        "polygons": np.array([[[15, 20], [35, 20], [35, 30], [15, 30]]]),
    }
    data = rotator(Sample(image=input_t, target=dict_target))
    r_img, r_targets = data.image, data.target
    assert isinstance(r_targets, dict)
    assert set(r_targets.keys()) == {"boxes", "polygons"}
    assert isinstance(r_targets["boxes"], np.ndarray)
    assert isinstance(r_targets["polygons"], np.ndarray)
    # Check rotated image
    assert r_img.ndim == input_t.ndim
    # Check boxes
    assert np.all(r_targets["boxes"] >= 0)
    if len(r_targets["boxes"]) > 0:
        assert r_targets["boxes"].shape[1] == 4
    # Check polygons
    assert np.all(r_targets["polygons"] >= 0)
    if len(r_targets["polygons"]) > 0:
        assert r_targets["polygons"].shape[1:] == (4, 2)

    # Empty dict targets
    empty_targets = {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "polygons": np.zeros((0, 4, 2), dtype=np.float32),
    }
    data = rotator(Sample(image=input_t, target=empty_targets))
    r_img, r_targets = data.image, data.target
    assert isinstance(r_targets, dict)
    assert r_targets["boxes"].shape == (0, 4)
    assert r_targets["polygons"].shape == (0, 4, 2)

    # FP16 (only on GPU)
    if torch.cuda.is_available():
        input_t = torch.ones((3, 50, 50), dtype=torch.float16).cuda()
        data = rotator(Sample(image=input_t, target=boxes))
        r_img = data.image
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
    assert repr(cropper) == "RandomCrop(scale=(0.5, 1.0), ratio=(0.75, 1.33))"
    input_t = torch.ones((3, 50, 50), dtype=torch.float32)

    sample = cropper(Sample(image=input_t, target=target))
    img, target = sample.image, sample.target
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

    # Test dict targets
    dict_target = {
        "boxes": np.array([[15, 20, 35, 30]]),
        "polygons": np.array([[[15, 20], [35, 20], [35, 30], [15, 30]]]),
    }
    sample = cropper(Sample(image=input_t, target=dict_target))
    img, cropped_targets = sample.image, sample.target
    assert isinstance(cropped_targets, dict)
    assert set(cropped_targets.keys()) == {"boxes", "polygons"}
    assert isinstance(cropped_targets["boxes"], np.ndarray)
    assert isinstance(cropped_targets["polygons"], np.ndarray)
    # Check cropped image properties
    assert img.shape[-1] * img.shape[-2] >= 0.4 * input_t.shape[-1] * input_t.shape[-2]
    assert 0.65 <= img.shape[-2] / img.shape[-1] <= 1.6
    # Check boxes
    assert np.all(cropped_targets["boxes"] >= 0)
    if len(cropped_targets["boxes"]) > 0:
        assert np.all(cropped_targets["boxes"][:, [0, 2]] <= img.shape[-1])
        assert np.all(cropped_targets["boxes"][:, [1, 3]] <= img.shape[-2])
    # Check polygons
    assert np.all(cropped_targets["polygons"] >= 0)
    if len(cropped_targets["polygons"]) > 0:
        assert np.all(cropped_targets["polygons"][..., 0] <= img.shape[-1])
        assert np.all(cropped_targets["polygons"][..., 1] <= img.shape[-2])


@pytest.mark.parametrize(
    "input_dtype, input_size",
    [
        [torch.float32, (3, 32, 32)],
        [torch.uint8, (3, 32, 32)],
    ],
)
def test_channel_shuffle(input_dtype, input_size):
    transfo = ChannelShuffle()
    input_t = Sample(image=torch.rand(input_size, dtype=torch.float32))
    if input_dtype == torch.uint8:
        input_t.image = (255 * input_t.image).round()
    input_t.image = input_t.image.to(dtype=input_dtype)
    out = transfo(input_t).image
    assert isinstance(out, torch.Tensor)
    assert out.shape == input_size
    assert out.dtype == input_dtype
    # Ensure that nothing has changed apart from channel order
    if input_dtype == torch.uint8:
        assert torch.all(input_t.image.sum(0) == out.sum(0))
    else:
        # Float approximation
        assert (input_t.image.sum(0) - out.sum(0)).abs().mean() < 1e-7


@pytest.mark.parametrize(
    "input_dtype,input_shape",
    [
        [torch.float32, (3, 32, 32)],
        [torch.uint8, (3, 32, 32)],
    ],
)
def test_gaussian_noise(input_dtype, input_shape):
    transform = GaussianNoise(0.0, 1.0)
    input_t = Sample(image=torch.rand(input_shape, dtype=torch.float32))
    if input_dtype == torch.uint8:
        input_t.image = (255 * input_t.image).round()
    input_t.image = input_t.image.to(dtype=input_dtype)
    transformed = transform(input_t).image
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == input_shape
    assert transformed.dtype == input_dtype
    assert torch.any(transformed != input_t.image)
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

    input_t = Sample(image=torch.rand(input_shape, dtype=torch.float32))

    if input_dtype == torch.uint8:
        input_t.image = (255 * input_t.image).round().to(dtype=torch.uint8)

    blurred = transform(input_t).image
    assert isinstance(blurred, torch.Tensor)
    assert blurred.shape == input_shape
    assert blurred.dtype == input_dtype

    if input_dtype == torch.uint8:
        assert torch.any(blurred != input_t.image)
        assert torch.all(blurred <= 255)
        assert torch.all(blurred >= 0)
    else:
        assert torch.any(blurred != input_t.image)
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

    data = transform(Sample(image=input_t, target=target))
    transformed, _target = data.image, data.target
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

    # Test dict targets
    dict_target = {
        "boxes": np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32),
        "polygons": np.array(
            [[[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]]],
            dtype=np.float32,
        ),
    }

    data = transform(Sample(image=input_t, target=dict_target))
    _target = data.target
    assert isinstance(_target, dict)
    assert set(_target.keys()) == {"boxes", "polygons"}
    assert _target["boxes"].dtype == np.float32
    assert _target["polygons"].dtype == np.float32
    if p == 1:
        assert np.all(_target["boxes"] == np.array([[0.7, 0.1, 0.9, 0.4]], dtype=np.float32))
        assert np.all(
            _target["polygons"]
            == np.array(
                [[[0.9, 0.1], [0.7, 0.1], [0.7, 0.4], [0.9, 0.4]]],
                dtype=np.float32,
            )
        )
    elif p == 0:
        assert np.all(_target["boxes"] == np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32))
        assert np.all(
            _target["polygons"]
            == np.array(
                [[[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]]],
                dtype=np.float32,
            )
        )


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
    input_t = Sample(image=torch.ones(input_shape, dtype=torch.float32))
    if input_dtype == torch.uint8:
        input_t.image = (255 * input_t.image).round()
    input_t.image = input_t.image.to(dtype=input_dtype)
    transformed = transform(input_t).image
    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == input_shape
    assert transformed.dtype == input_dtype
    # The shadow will darken the picture
    assert input_t.image.float().mean() >= transformed.float().mean()
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
    data = transfo(Sample(image=img, target=target))
    out_img, out_target = data.image, data.target
    assert isinstance(out_img, torch.Tensor)
    assert isinstance(out_target, np.ndarray)
    # Resize is already well tested
    assert torch.all(out_img == img) if p == 0 else out_img.shape != img.shape
    assert out_target.shape == target.shape


# ----------------------------------------------------------------------------
# End-to-end tests for SampleCompose with geometric and photometric transforms
# ----------------------------------------------------------------------------


def _make_pipeline():
    return SampleCompose([
        RandomHorizontalFlip(p=1.0),
        RandomRotate(max_angle=10.0, expand=False),
        RandomCrop(scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        RandomResize(
            scale_range=(0.8, 1.2),
            preserve_aspect_ratio=True,
            symmetric_pad=True,
            p=1.0,
        ),
        ColorInversion(min_val=0.7),
        GaussianNoise(mean=0.0, std=0.1),
        ChannelShuffle(),
        RandomShadow((0.2, 0.8)),
        RandomApply(RandomHorizontalFlip(p=1.0), p=1.0),
        OneOf([
            RandomRotate(max_angle=5.0, expand=False),
            RandomCrop(scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        ]),
        ImageTorchvisionTransform(RandomGrayscale(p=0.15)),
    ])


def test_samplecompose_end_to_end_boxes():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    input_t = torch.rand((3, 64, 64), dtype=torch.float32)
    targets = {
        "boxes": np.array(
            [
                [0.1, 0.1, 0.4, 0.4],
                [0.5, 0.5, 0.9, 0.9],
            ],
            dtype=np.float32,
        )
    }

    transforms = _make_pipeline()
    data = transforms(Sample(image=input_t, target=targets))
    out_img, out_targets = data.image, data.target

    # image checks
    assert isinstance(out_img, torch.Tensor)
    assert out_img.ndim == 3
    assert out_img.shape[0] == 3
    assert torch.all((out_img >= 0) & (out_img <= 1))

    # target checks
    assert isinstance(out_targets, dict)
    assert "boxes" in out_targets
    boxes = out_targets["boxes"]
    assert isinstance(boxes, np.ndarray)

    if len(boxes) > 0:
        # must stay boxes
        assert boxes.ndim == 2
        assert boxes.shape[1] == 4
        # geometry validity
        assert np.all(boxes[:, 2] >= boxes[:, 0])
        assert np.all(boxes[:, 3] >= boxes[:, 1])
        assert np.all(np.isfinite(boxes))
        assert np.all((boxes >= 0) & (boxes <= 1))

    # immutability check
    np.testing.assert_array_equal(
        targets["boxes"],
        np.array(
            [
                [0.1, 0.1, 0.4, 0.4],
                [0.5, 0.5, 0.9, 0.9],
            ],
            dtype=np.float32,
        ),
    )


def test_samplecompose_end_to_end_polygons():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    input_t = torch.rand((3, 64, 64), dtype=torch.float32)
    targets = {
        "polygons": np.array(
            [
                [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]],
                [[0.5, 0.5], [0.9, 0.5], [0.9, 0.9], [0.5, 0.9]],
            ],
            dtype=np.float32,
        )
    }

    transforms = _make_pipeline()
    data = transforms(Sample(image=input_t, target=targets))
    out_img, out_targets = data.image, data.target

    # image checks
    assert isinstance(out_img, torch.Tensor)
    assert out_img.ndim == 3
    assert out_img.shape[0] == 3
    assert torch.all((out_img >= 0) & (out_img <= 1))

    # target checks
    assert isinstance(out_targets, dict)
    assert "polygons" in out_targets
    polys = out_targets["polygons"]
    assert isinstance(polys, np.ndarray)

    if len(polys) > 0:
        assert polys.ndim == 3
        assert polys.shape[1:] == (4, 2)
        # geometry validity
        assert np.all(np.isfinite(polys))
        assert np.all((polys >= 0) & (polys <= 1))
        # ensure valid polygon structure (non-degenerate)
        assert np.all(np.linalg.norm(polys[:, 1] - polys[:, 0], axis=1) > 0)

    # immutability check
    np.testing.assert_array_equal(
        targets["polygons"],
        np.array(
            [
                [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]],
                [[0.5, 0.5], [0.9, 0.5], [0.9, 0.9], [0.5, 0.9]],
            ],
            dtype=np.float32,
        ),
    )
