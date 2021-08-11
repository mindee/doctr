import pytest
import tensorflow as tf

from doctr.models import backbones


@pytest.mark.parametrize(
    "arch_name, top_implemented, input_shape, output_size",
    [
        ["vgg16_bn", False, (224, 224, 3), (7, 56, 512)],
        ["resnet31", False, (32, 128, 3), (4, 32, 512)],
        ["mobilenet_v3_small", False, (512, 512, 3), (16, 16, 576)],
        ["mobilenet_v3_large", False, (512, 512, 3), (16, 16, 960)],
    ],
)
def test_classification_architectures(arch_name, top_implemented, input_shape, output_size):
    # Model
    batch_size = 2
    model = backbones.__dict__[arch_name](pretrained=True, input_shape=input_shape)
    # Forward
    out = model(tf.random.uniform(shape=[batch_size, *input_shape], maxval=1, dtype=tf.float32))
    # Output checks
    assert isinstance(out, tf.Tensor)
    assert out.dtype == tf.float32
    assert out.numpy().shape == (batch_size, *output_size)
