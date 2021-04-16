import pytest

import tensorflow as tf
from doctr import transforms as T


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


@pytest.mark.parametrize(
    "rgb_min",
    [
        0.2,
        0.4,
        0.6,
    ],
)
def test_invert_colorize(rgb_min):

    transfo = T.InvertColorize(r_min=rgb_min, g_min=rgb_min, b_min=rgb_min)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], 1), dtype=tf.float32)
    out = transfo(input_t)

    assert tf.reduce_all(out <= 1 - rgb_min)
    assert tf.reduce_all(out >= 0)


def test_brightness():

    transfo = T.Brightness(delta=.3)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], .5), dtype=tf.float32)
    out = transfo(input_t)

    assert tf.reduce_all(out == .8)


def test_contrast():

    transfo = T.Contrast(contrast_factor=1.3)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], .5), dtype=tf.float32)
    out = transfo(input_t)

    assert tf.reduce_all(out == .5)


def test_saturation():

    transfo = T.Saturation(saturation_factor=1.5)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], .5), dtype=tf.float32)
    input_t = tf.image.hsv_to_rgb(input_t)
    out = transfo(input_t)
    hsv = tf.image.rgb_to_hsv(out)

    assert tf.reduce_all(hsv[:, :, :, 1] == .75)


def test_hue():

    transfo = T.Hue(delta=.2)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], .5), dtype=tf.float32)
    input_t = tf.image.hsv_to_rgb(input_t)
    out = transfo(input_t)
    hsv = tf.image.rgb_to_hsv(out)

    assert tf.reduce_all(hsv[:, :, :, 0] == .7)


def test_gamma():

    transfo = T.Gamma(gamma=2., gain=2)
    input_t = tf.cast(tf.fill([8, 32, 32, 3], 2.), dtype=tf.float32)
    out = transfo(input_t)

    assert tf.reduce_all(out == 8.)


def test_jpegquality():

    transfo = T.JpegQuality(quality=50)
    input_t = tf.cast(tf.fill([32, 32, 3], 1), dtype=tf.float32)
    out = transfo(input_t)
    assert out.shape == input_t.shape
