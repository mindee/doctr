import math

import numpy as np
import pytest
import tensorflow as tf

from doctr import transforms as T
from doctr.transforms.functional import crop_detection, rotate


def test_resize():
    output_size = (32, 32)
    transfo = T.Resize(output_size)
    input_t = tf.cast(tf.fill([64, 64, 3], 1), dtype=tf.float32)
    out = transfo(input_t)

    assert tf.reduce_all(out == 1)
    assert out.shape[:2] == output_size
    assert repr(transfo) == f"Resize(output_size={output_size}, method='bilinear')"

    transfo = T.Resize(output_size, preserve_aspect_ratio=True)
    input_t = tf.cast(tf.fill([32, 64, 3], 1), dtype=tf.float32)
    out = transfo(input_t)

    assert not tf.reduce_all(out == 1)
    # Asymetric padding
    assert tf.reduce_all(out[-1] == 0) and tf.reduce_all(out[0] == 1)
    assert out.shape[:2] == output_size

    # Symetric padding
    transfo = T.Resize(output_size, preserve_aspect_ratio=True, symmetric_pad=True)
    assert repr(transfo) == (f"Resize(output_size={output_size}, method='bilinear', "
                             f"preserve_aspect_ratio=True, symmetric_pad=True)")
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

    assert tf.reduce_all(out <= .51)
    assert tf.reduce_all(out >= .49)

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

    transfo = T.RandomBrightness(max_delta=.1)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], .5), dtype=tf.float32)
    out = transfo(input_t)

    assert tf.reduce_all(out >= .4)
    assert tf.reduce_all(out <= .6)

    # FP16
    input_t = tf.cast(tf.fill([8, 32, 32, 3], .5), dtype=tf.float16)
    out = transfo(input_t)
    assert out.dtype == tf.float16


def test_contrast():
    transfo = T.RandomContrast(delta=.2)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], .5), dtype=tf.float32)
    out = transfo(input_t)

    assert tf.reduce_all(out == .5)

    # FP16
    if any(tf.config.list_physical_devices('GPU')):
        input_t = tf.cast(tf.fill([8, 32, 32, 3], .5), dtype=tf.float16)
        out = transfo(input_t)
        assert out.dtype == tf.float16


def test_saturation():

    transfo = T.RandomSaturation(delta=.2)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], .5), dtype=tf.float32)
    input_t = tf.image.hsv_to_rgb(input_t)
    out = transfo(input_t)
    hsv = tf.image.rgb_to_hsv(out)

    assert tf.reduce_all(hsv[:, :, :, 1] >= .4)
    assert tf.reduce_all(hsv[:, :, :, 1] <= .6)

    # FP16
    if any(tf.config.list_physical_devices('GPU')):
        input_t = tf.cast(tf.fill([8, 32, 32, 3], .5), dtype=tf.float16)
        out = transfo(input_t)
        assert out.dtype == tf.float16


def test_hue():

    transfo = T.RandomHue(max_delta=.2)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], .5), dtype=tf.float32)
    input_t = tf.image.hsv_to_rgb(input_t)
    out = transfo(input_t)
    hsv = tf.image.rgb_to_hsv(out)

    assert tf.reduce_all(hsv[:, :, :, 0] <= .7)
    assert tf.reduce_all(hsv[:, :, :, 0] >= .3)

    # FP16
    if any(tf.config.list_physical_devices('GPU')):
        input_t = tf.cast(tf.fill([8, 32, 32, 3], .5), dtype=tf.float16)
        out = transfo(input_t)
        assert out.dtype == tf.float16


def test_gamma():

    transfo = T.RandomGamma(min_gamma=1., max_gamma=2., min_gain=.8, max_gain=1.)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], 2.), dtype=tf.float32)
    out = transfo(input_t)

    assert tf.reduce_all(out >= 1.6)
    assert tf.reduce_all(out <= 4.)

    # FP16
    input_t = tf.cast(tf.fill([8, 32, 32, 3], 2.), dtype=tf.float16)
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


def test_rotate():
    input_t = tf.ones((50, 50, 3), dtype=tf.float32)
    boxes = np.array([
        [15, 20, 35, 30]
    ])
    r_img, r_boxes = rotate(input_t, boxes, angle=12., expand=False)
    assert r_img.shape == (50, 50, 3)
    assert r_img[0, 0, 0] == 0.
    assert r_boxes.shape == (1, 4, 2)

    # Expand
    r_img, r_boxes = rotate(input_t, boxes, angle=12., expand=True)
    assert r_img.shape == (60, 60, 3)
    # With the expansion, there should be a maximum of 1 pixel of the initial image on the first row
    assert r_img[0, :, 0].numpy().sum() <= 1

    # Relative coords
    rel_boxes = np.array([[.3, .4, .7, .6]])
    r_img, r_boxes = rotate(input_t, rel_boxes, angle=90)
    assert r_boxes.shape == (1, 4, 2)
    assert np.isclose(r_boxes, np.asarray([[[0.4, 0.7], [0.4, 0.3], [0.6, 0.3], [0.6, 0.7]]])).all()

    # FP16
    input_t = tf.ones((50, 50, 3), dtype=tf.float16)
    r_img, _ = rotate(input_t, boxes, angle=12.)
    assert r_img.dtype == tf.float16


def test_random_rotate():
    rotator = T.RandomRotate(max_angle=10., expand=False)
    input_t = tf.ones((50, 50, 3), dtype=tf.float32)
    boxes = np.array([
        [15, 20, 35, 30]
    ])
    r_img, r_boxes = rotator(input_t, boxes)
    assert r_img.shape == input_t.shape

    rotator = T.RandomRotate(max_angle=10., expand=True)
    r_img, r_boxes = rotator(input_t, boxes)
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
    crop_box = (12 / 50, 23 / 50, 1., 1.)
    c_img, c_boxes = crop_detection(img, abs_boxes, crop_box)
    assert c_img.shape == (26, 37, 3)
    assert c_boxes.shape == (1, 4)
    rel_boxes = np.array([
        [.3, .4, .7, .6],
        [.1, .2, .2, .4],
    ])
    c_img, c_boxes = crop_detection(img, rel_boxes, crop_box)
    assert c_img.shape == (26, 37, 3)
    assert c_boxes.shape == (1, 4)

    # FP16
    img = tf.ones((50, 50, 3), dtype=tf.float16)
    c_img, _ = crop_detection(img, rel_boxes, crop_box)
    assert c_img.dtype == tf.float16


def test_random_crop():
    transfo = T.RandomCrop(scale=(0.5, 1.), ratio=(0.75, 1.33))
    input_t = tf.ones((50, 50, 3), dtype=tf.float32)
    boxes = np.array([
        [15, 20, 35, 30]
    ])
    img, target = transfo(input_t, dict(boxes=boxes))
    # Check the scale
    assert img.shape[0] * img.shape[1] >= 0.5 * input_t.shape[0] * input_t.shape[1]
    # Check aspect ratio
    assert 0.75 <= img.shape[0] / img.shape[1] <= 1.33
    # Check the target
    assert np.all(target['boxes'] >= 0)
    assert np.all(target['boxes'][:, [0, 2]] <= img.shape[1]) and np.all(target['boxes'][:, [1, 3]] <= img.shape[0])


def test_gaussian_blur():
    blur = T.GaussianBlur(3, (.1, 3))
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
    ]
)
def test_gaussian_noise(input_dtype, input_shape):
    transform = T.GaussianNoise(0., 1.)
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
        assert tf.reduce_all(transformed <= 1.)
