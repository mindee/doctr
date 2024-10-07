import os

import pytest
import tensorflow as tf
from tensorflow.keras.applications import ResNet50

from doctr.models.classification import mobilenet_v3_small
from doctr.models.utils import (
    IntermediateLayerGetter,
    _bf16_to_float32,
    _copy_tensor,
    conv_sequence,
    load_pretrained_params,
)


def test_copy_tensor():
    x = tf.random.uniform(shape=[8], minval=0, maxval=1)
    m = _copy_tensor(x)
    assert m.device == x.device and m.dtype == x.dtype and m.shape == x.shape and tf.reduce_all(tf.equal(m, x))


def test_bf16_to_float32():
    x = tf.random.uniform(shape=[8], minval=0, maxval=1, dtype=tf.bfloat16)
    m = _bf16_to_float32(x)
    assert x.dtype == tf.bfloat16 and m.dtype == tf.float32 and tf.reduce_all(tf.equal(m, tf.cast(x, tf.float32)))


def test_load_pretrained_params(tmpdir_factory):
    model = mobilenet_v3_small(pretrained=False)
    # Retrieve this URL
    url = "https://doctr-static.mindee.com/models?id=v0.9.0/mobilenet_v3_small-3fcebad7.weights.h5&src=0"
    # Temp cache dir
    cache_dir = tmpdir_factory.mktemp("cache")
    # Pass an incorrect hash
    with pytest.raises(ValueError):
        load_pretrained_params(model, url, "mywronghash", cache_dir=str(cache_dir))
    # Let tit resolve the hash from the file name
    load_pretrained_params(model, url, cache_dir=str(cache_dir))
    # Check that the file was downloaded
    assert os.path.exists(cache_dir.join("models").join("mobilenet_v3_small-3fcebad7.weights.h5"))


def test_conv_sequence():
    assert len(conv_sequence(8, kernel_size=3)) == 1
    assert len(conv_sequence(8, "relu", kernel_size=3)) == 1
    assert len(conv_sequence(8, None, True, kernel_size=3)) == 2
    assert len(conv_sequence(8, "relu", True, kernel_size=3)) == 3


def test_intermediate_layer_getter():
    backbone = ResNet50(include_top=False, weights=None, pooling=None)
    feat_extractor = IntermediateLayerGetter(backbone, ["conv2_block3_out", "conv3_block4_out"])
    # Check num of output features
    input_tensor = tf.random.uniform(shape=[1, 224, 224, 3], minval=0, maxval=1)
    assert len(feat_extractor(input_tensor)) == 2

    # Repr
    assert repr(feat_extractor) == "IntermediateLayerGetter()"
