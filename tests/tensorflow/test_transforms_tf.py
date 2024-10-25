import math

import numpy as np
import pytest
import tensorflow as tf

from doctr import transforms as T
from doctr.transforms.functional import crop_detection, rotate_sample


def test_resize():
    output_size = (32, 32)
    transfo = T.Resize(output_size)
    input_t = tf.cast(tf.fill([64, 64, 3], 1), dtype=tf.float32)
    out = transfo(input_t)

    assert tf.math.reduce_all(tf.math.abs(out - 1) < 1e-6)
    assert out.shape[:2] == output_size
    assert repr(transfo) == f"Resize(output_size={output_size}, method='bilinear')"

    transfo = T.Resize(output_size, preserve_aspect_ratio=True)
    input_t = tf.cast(tf.fill([32, 64, 3], 1), dtype=tf.float32)
    out = transfo(input_t)

    assert not tf.reduce_all(out == 1)
    # Asymetric padding
    assert tf.reduce_all(out[-1] == 0) and tf.math.reduce_all(tf.math.abs(out[0] - 1) < 1e-6)
    assert out.shape[:2] == output_size

    # Symetric padding
    transfo = T.Resize(output_size, preserve_aspect_ratio=True, symmetric_pad=True)
    assert repr(transfo) == (
        f"Resize(output_size={output_size}, method='bilinear', preserve_aspect_ratio=True, symmetric_pad=True)"
    )
    out = transfo(input_t)
    # Asymetric padding
    assert tf.reduce_all(out[-1] == 0) and tf.reduce_all(out[0] == 0)

    # Inverse aspect ratio
    input_t = tf.cast(tf.fill([64, 32, 3], 1), dtype=tf.float32)
    out = transfo(input_t)

    assert not tf.reduce_all(out == 1)
    assert out.shape[:2] == output_size

    # FP16
    input_t = tf.cast(tf.fill([64, 64, 3], 1), dtype=tf.float16)
    out = transfo(input_t)
    assert out.dtype == tf.float16

    # --- Test with target (bounding boxes) ---

    target_boxes = np.array([[0.1, 0.1, 0.9, 0.9], [0.2, 0.2, 0.8, 0.8]])
    output_size = (64, 64)

    transfo = T.Resize(output_size, preserve_aspect_ratio=True)
    input_t = tf.cast(tf.fill([64, 32, 3], 1), dtype=tf.float32)
    out, new_target = transfo(input_t, target_boxes)

    assert out.shape[:2] == output_size
    assert new_target.shape == target_boxes.shape
    assert np.all(new_target >= 0) and np.all(new_target <= 1)

    out = transfo(input_t)
    assert out.shape[:2] == output_size


def test_compose():
    output_size = (16, 16)
    transfo = T.Compose([T.Resize((32, 32)), T.Resize(output_size)])
    input_t = tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1)
    out = transfo(input_t)

    assert out.shape[:2] == output_size
    assert len(repr(transfo).split("\n")) == 6


@pytest.mark.parametrize(
    "input_shape",
    [
        [8, 32, 32, 3],
        [32, 32, 3],
        [32, 3],
    ],
)
def test_normalize(input_shape):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transfo = T.Normalize(mean, std)
    input_t = tf.cast(tf.fill(input_shape, 1), dtype=tf.float32)

    out = transfo(input_t)

    assert tf.reduce_all(out == 1)
    assert repr(transfo) == f"Normalize(mean={mean}, std={std})"

    # FP16
    input_t = tf.cast(tf.fill(input_shape, 1), dtype=tf.float16)
    out = transfo(input_t)
    assert out.dtype == tf.float16


def test_lambatransformation():
    transfo = T.LambdaTransformation(lambda x: x / 2)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], 1), dtype=tf.float32)
    out = transfo(input_t)

    assert tf.reduce_all(out == 0.5)


def test_togray():
    transfo = T.ToGray()
    r = tf.fill([8, 32, 32, 1], 0.2)
    g = tf.fill([8, 32, 32, 1], 0.6)
    b = tf.fill([8, 32, 32, 1], 0.7)
    input_t = tf.cast(tf.concat([r, g, b], axis=-1), dtype=tf.float32)
    out = transfo(input_t)

    assert tf.reduce_all(out <= 0.51)
    assert tf.reduce_all(out >= 0.49)

    # FP16
    input_t = tf.cast(tf.concat([r, g, b], axis=-1), dtype=tf.float16)
    out = transfo(input_t)
    assert out.dtype == tf.float16


@pytest.mark.parametrize(
    "rgb_min",
    [
        0.2,
        0.4,
        0.6,
    ],
)
def test_invert_colorize(rgb_min):
    transfo = T.ColorInversion(min_val=rgb_min)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], 1), dtype=tf.float32)
    out = transfo(input_t)
    assert tf.reduce_all(out <= 1 - rgb_min + 1e-4)
    assert tf.reduce_all(out >= 0)

    input_t = tf.cast(tf.fill([8, 32, 32, 3], 255), dtype=tf.uint8)
    out = transfo(input_t)
    assert tf.reduce_all(out <= int(math.ceil(255 * (1 - rgb_min))))
    assert tf.reduce_all(out >= 0)

    # FP16
    input_t = tf.cast(tf.fill([8, 32, 32, 3], 1), dtype=tf.float16)
    out = transfo(input_t)
    assert out.dtype == tf.float16


def test_brightness():
    transfo = T.RandomBrightness(max_delta=0.1)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], 0.5), dtype=tf.float32)
    out = transfo(input_t)

    assert tf.reduce_all(out >= 0.4)
    assert tf.reduce_all(out <= 0.6)

    # FP16
    input_t = tf.cast(tf.fill([8, 32, 32, 3], 0.5), dtype=tf.float16)
    out = transfo(input_t)
    assert out.dtype == tf.float16


def test_contrast():
    transfo = T.RandomContrast(delta=0.2)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], 0.5), dtype=tf.float32)
    out = transfo(input_t)

    assert tf.reduce_all(out == 0.5)

    # FP16
    if any(tf.config.list_physical_devices("GPU")):
        input_t = tf.cast(tf.fill([8, 32, 32, 3], 0.5), dtype=tf.float16)
        out = transfo(input_t)
        assert out.dtype == tf.float16


def test_saturation():
    transfo = T.RandomSaturation(delta=0.2)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], 0.5), dtype=tf.float32)
    input_t = tf.image.hsv_to_rgb(input_t)
    out = transfo(input_t)
    hsv = tf.image.rgb_to_hsv(out)

    assert tf.reduce_all(hsv[:, :, :, 1] >= 0.4)
    assert tf.reduce_all(hsv[:, :, :, 1] <= 0.6)

    # FP16
    if any(tf.config.list_physical_devices("GPU")):
        input_t = tf.cast(tf.fill([8, 32, 32, 3], 0.5), dtype=tf.float16)
        out = transfo(input_t)
        assert out.dtype == tf.float16


def test_hue():
    transfo = T.RandomHue(max_delta=0.2)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], 0.5), dtype=tf.float32)
    input_t = tf.image.hsv_to_rgb(input_t)
    out = transfo(input_t)
    hsv = tf.image.rgb_to_hsv(out)

    assert tf.reduce_all(hsv[:, :, :, 0] <= 0.7)
    assert tf.reduce_all(hsv[:, :, :, 0] >= 0.3)

    # FP16
    if any(tf.config.list_physical_devices("GPU")):
        input_t = tf.cast(tf.fill([8, 32, 32, 3], 0.5), dtype=tf.float16)
        out = transfo(input_t)
        assert out.dtype == tf.float16


def test_gamma():
    transfo = T.RandomGamma(min_gamma=1.0, max_gamma=2.0, min_gain=0.8, max_gain=1.0)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], 2.0), dtype=tf.float32)
    out = transfo(input_t)

    assert tf.reduce_all(out >= 1.6)
    assert tf.reduce_all(out <= 4.0)

    # FP16
    input_t = tf.cast(tf.fill([8, 32, 32, 3], 2.0), dtype=tf.float16)
    out = transfo(input_t)
    assert out.dtype == tf.float16


def test_jpegquality():
    transfo = T.RandomJpegQuality(min_quality=50)
    input_t = tf.cast(tf.fill([32, 32, 3], 1), dtype=tf.float32)
    out = transfo(input_t)
    assert out.shape == input_t.shape

    # FP16
    input_t = tf.cast(tf.fill([32, 32, 3], 1), dtype=tf.float16)
    out = transfo(input_t)
    assert out.dtype == tf.float16


def test_rotate_sample():
    img = tf.ones((200, 100, 3), dtype=tf.float32)
    boxes = np.array([0, 0, 100, 200])[None, ...]
    polys = np.stack((boxes[..., [0, 1]], boxes[..., [2, 1]], boxes[..., [2, 3]], boxes[..., [0, 3]]), axis=1)
    rel_boxes = np.array([0, 0, 1, 1], dtype=np.float32)[None, ...]
    rel_polys = np.stack(
        (rel_boxes[..., [0, 1]], rel_boxes[..., [2, 1]], rel_boxes[..., [2, 3]], rel_boxes[..., [0, 3]]), axis=1
    )

    # No angle
    rotated_img, rotated_geoms = rotate_sample(img, boxes, 0, False)
    assert tf.math.reduce_all(rotated_img == img) and np.all(rotated_geoms == rel_polys)
    rotated_img, rotated_geoms = rotate_sample(img, boxes, 0, True)
    assert tf.math.reduce_all(rotated_img == img) and np.all(rotated_geoms == rel_polys)
    rotated_img, rotated_geoms = rotate_sample(img, polys, 0, False)
    assert tf.math.reduce_all(rotated_img == img) and np.all(rotated_geoms == rel_polys)
    rotated_img, rotated_geoms = rotate_sample(img, polys, 0, True)
    assert tf.math.reduce_all(rotated_img == img) and np.all(rotated_geoms == rel_polys)

    # No expansion
    expected_img = np.zeros((200, 100, 3), dtype=np.float32)
    expected_img[50:150] = 1
    expected_img = tf.convert_to_tensor(expected_img)
    expected_polys = np.array([[0, 0.75], [0, 0.25], [1, 0.25], [1, 0.75]])[None, ...]
    rotated_img, rotated_geoms = rotate_sample(img, boxes, 90, False)
    assert tf.math.reduce_all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)
    rotated_img, rotated_geoms = rotate_sample(img, polys, 90, False)
    assert tf.math.reduce_all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)
    rotated_img, rotated_geoms = rotate_sample(img, rel_boxes, 90, False)
    assert tf.math.reduce_all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)
    rotated_img, rotated_geoms = rotate_sample(img, rel_polys, 90, False)
    assert tf.math.reduce_all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)

    # Expansion
    expected_img = tf.ones((100, 200, 3), dtype=tf.float32)
    expected_polys = np.array([[0, 1], [0, 0], [1, 0], [1, 1]], dtype=np.float32)[None, ...]
    rotated_img, rotated_geoms = rotate_sample(img, boxes, 90, True)
    assert tf.math.reduce_all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)
    rotated_img, rotated_geoms = rotate_sample(img, polys, 90, True)
    assert tf.math.reduce_all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)
    rotated_img, rotated_geoms = rotate_sample(img, rel_boxes, 90, True)
    assert tf.math.reduce_all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)
    rotated_img, rotated_geoms = rotate_sample(img, rel_polys, 90, True)
    assert tf.math.reduce_all(rotated_img == expected_img) and np.all(rotated_geoms == expected_polys)

    with pytest.raises(AssertionError):
        rotate_sample(img, boxes[None, ...], 90, False)


def test_random_rotate():
    rotator = T.RandomRotate(max_angle=10.0, expand=False)
    input_t = tf.ones((50, 50, 3), dtype=tf.float32)
    boxes = np.array([[15, 20, 35, 30]])
    r_img, _r_boxes = rotator(input_t, boxes)
    assert r_img.shape == input_t.shape

    rotator = T.RandomRotate(max_angle=10.0, expand=True)
    r_img, _r_boxes = rotator(input_t, boxes)
    assert r_img.shape != input_t.shape

    # FP16
    input_t = tf.ones((50, 50, 3), dtype=tf.float16)
    r_img, _ = rotator(input_t, boxes)
    assert r_img.dtype == tf.float16


def test_crop_detection():
    img = tf.ones((50, 50, 3), dtype=tf.float32)
    abs_boxes = np.array([
        [15, 20, 35, 30],
        [5, 10, 10, 20],
    ])
    crop_box = (12 / 50, 23 / 50, 1.0, 1.0)
    c_img, c_boxes = crop_detection(img, abs_boxes, crop_box)
    assert c_img.shape == (26, 37, 3)
    assert c_boxes.shape == (1, 4)
    assert np.all(c_boxes == np.array([15 - 12, 0, 35 - 12, 30 - 23])[None, ...])

    rel_boxes = np.array([
        [0.3, 0.4, 0.7, 0.6],
        [0.1, 0.2, 0.2, 0.4],
    ])
    c_img, c_boxes = crop_detection(img, rel_boxes, crop_box)
    assert c_img.shape == (26, 37, 3)
    assert c_boxes.shape == (1, 4)
    assert np.abs(c_boxes - np.array([0.06 / 0.76, 0.0, 0.46 / 0.76, 0.14 / 0.54])[None, ...]).mean() < 1e-7

    # FP16
    img = tf.ones((50, 50, 3), dtype=tf.float16)
    c_img, _ = crop_detection(img, rel_boxes, crop_box)
    assert c_img.dtype == tf.float16

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
    transfo = T.RandomCrop(scale=(0.5, 1.0), ratio=(0.75, 1.33))
    input_t = tf.ones((50, 50, 3), dtype=tf.float32)
    img, target = transfo(input_t, target)
    # Check the scale (take a margin)
    assert img.shape[0] * img.shape[1] >= 0.4 * input_t.shape[0] * input_t.shape[1]
    # Check aspect ratio (take a margin)
    assert 0.65 <= img.shape[0] / img.shape[1] <= 1.6
    # Check the target
    assert np.all(target >= 0)
    if target.ndim == 2:
        assert np.all(target[:, [0, 2]] <= img.shape[-1]) and np.all(target[:, [1, 3]] <= img.shape[-2])
    else:
        assert np.all(target[..., 0] <= img.shape[-1]) and np.all(target[..., 1] <= img.shape[-2])


def test_gaussian_blur():
    blur = T.GaussianBlur(3, (0.1, 3))
    input_t = np.ones((31, 31, 3), dtype=np.float32)
    input_t[15, 15] = 0
    blur_img = blur(tf.convert_to_tensor(input_t)).numpy()
    assert blur_img.shape == input_t.shape
    assert np.all(blur_img[15, 15] > 0)


@pytest.mark.parametrize(
    "input_dtype, input_size",
    [
        [tf.float32, (32, 32, 3)],
        [tf.uint8, (32, 32, 3)],
    ],
)
def test_channel_shuffle(input_dtype, input_size):
    transfo = T.ChannelShuffle()
    input_t = tf.random.uniform(input_size, dtype=tf.float32)
    if input_dtype == tf.uint8:
        input_t = tf.math.round(255 * input_t)
    input_t = tf.cast(input_t, dtype=input_dtype)
    out = transfo(input_t)
    assert isinstance(out, tf.Tensor)
    assert out.shape == input_size
    assert out.dtype == input_dtype
    # Ensure that nothing has changed apart from channel order
    assert tf.math.reduce_all(tf.math.reduce_sum(input_t, -1) == tf.math.reduce_sum(out, -1))


@pytest.mark.parametrize(
    "input_dtype,input_shape",
    [
        [tf.float32, (32, 32, 3)],
        [tf.uint8, (32, 32, 3)],
    ],
)
def test_gaussian_noise(input_dtype, input_shape):
    transform = T.GaussianNoise(0.0, 1.0)
    input_t = tf.random.uniform(input_shape, dtype=tf.float32)
    if input_dtype == tf.uint8:
        input_t = tf.math.round((255 * input_t))
    input_t = tf.cast(input_t, dtype=input_dtype)
    transformed = transform(input_t)
    assert isinstance(transformed, tf.Tensor)
    assert transformed.shape == input_shape
    assert transformed.dtype == input_dtype
    assert tf.math.reduce_any(transformed != input_t)
    assert tf.math.reduce_all(transformed >= 0)
    if input_dtype == tf.uint8:
        assert tf.reduce_all(transformed <= 255)
    else:
        assert tf.reduce_all(transformed <= 1.0)


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
    transform = T.RandomHorizontalFlip(p)
    input_t = np.ones((32, 32, 3))
    input_t[:, :16, :] = 0
    input_t = tf.convert_to_tensor(input_t)
    transformed, _target = transform(input_t, target)
    assert isinstance(transformed, tf.Tensor)
    assert transformed.shape == input_t.shape
    assert transformed.dtype == input_t.dtype
    # integrity check of targets
    assert isinstance(_target, np.ndarray)
    assert _target.dtype == np.float32
    if _target.ndim == 2:
        if p == 1:
            assert np.all(_target == np.array([[0.7, 0.1, 0.9, 0.4]], dtype=np.float32))
            assert tf.reduce_all(
                tf.math.reduce_mean(transformed, (0, 2)) == tf.constant([1] * 16 + [0] * 16, dtype=tf.float64)
            )
        elif p == 0:
            assert np.all(_target == np.array([[0.1, 0.1, 0.3, 0.4]], dtype=np.float32))
            assert tf.reduce_all(
                tf.math.reduce_mean(transformed, (0, 2)) == tf.constant([0] * 16 + [1] * 16, dtype=tf.float64)
            )
    else:
        if p == 1:
            assert np.all(_target == np.array([[[0.9, 0.1], [0.7, 0.1], [0.7, 0.4], [0.9, 0.4]]], dtype=np.float32))
            assert tf.reduce_all(
                tf.math.reduce_mean(transformed, (0, 2)) == tf.constant([1] * 16 + [0] * 16, dtype=tf.float64)
            )
        elif p == 0:
            assert np.all(_target == np.array([[[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]]], dtype=np.float32))
            assert tf.reduce_all(
                tf.math.reduce_mean(transformed, (0, 2)) == tf.constant([0] * 16 + [1] * 16, dtype=tf.float64)
            )


@pytest.mark.parametrize(
    "input_dtype,input_shape",
    [
        [tf.float32, (32, 32, 3)],
        [tf.uint8, (32, 32, 3)],
        [tf.float32, (64, 32, 3)],
        [tf.uint8, (64, 32, 3)],
    ],
)
def test_random_shadow(input_dtype, input_shape):
    transform = T.RandomShadow((0.2, 0.8))
    input_t = tf.random.uniform(input_shape, dtype=tf.float32)
    if input_dtype == tf.uint8:
        input_t = tf.math.round((255 * input_t))
    input_t = tf.cast(input_t, dtype=input_dtype)
    transformed = transform(input_t)
    assert isinstance(transformed, tf.Tensor)
    assert transformed.shape == input_shape
    assert transformed.dtype == input_dtype
    # The shadow will darken the picture
    assert tf.math.reduce_mean(input_t) >= tf.math.reduce_mean(transformed)
    assert tf.math.reduce_all(transformed >= 0)
    if input_dtype == tf.uint8:
        assert tf.reduce_all(transformed <= 255)
    else:
        assert tf.reduce_all(transformed <= 1.0)


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
    transfo = T.RandomResize(
        scale_range=(0.3, 1.3), preserve_aspect_ratio=preserve_aspect_ratio, symmetric_pad=symmetric_pad, p=p
    )
    assert (
        repr(transfo)
        == f"RandomResize(scale_range=(0.3, 1.3), preserve_aspect_ratio={preserve_aspect_ratio}, symmetric_pad={symmetric_pad}, p={p})"  # noqa: E501
    )

    img = tf.random.uniform((64, 64, 3))
    # Apply the transformation
    out_img, out_target = transfo(img, target)
    assert isinstance(out_img, tf.Tensor)
    assert isinstance(out_target, np.ndarray)
    # Resize is already well-tested
    assert tf.reduce_all(tf.equal(out_img, img)) if p == 0 else out_img.shape != img.shape
    assert out_target.shape == target.shape
