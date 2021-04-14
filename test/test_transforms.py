import tensorflow as tf
from doctr import transforms as T


def test_resize():
    output_size = (32, 32)
    transfo = T.Resize(output_size)
    input_t = tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1)
    out = transfo(input_t)

    assert out.shape[:2] == output_size
    assert repr(transfo) == f"Resize(output_size={output_size}, method='bilinear')"


def test_compose():

    output_size = (16, 16)
    transfo = T.Compose([Resize((32, 32)), Resize(output_size)])
    input_t = tf.random.uniform(shape=[64, 64, 3], minval=0, maxval=1)
    out = transfo(input_t)

    assert out.shape[:2] == output_size
    assert len(repr(transfo).split("\n")) == 6
