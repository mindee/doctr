import pytest
import tensorflow as tf

from doctr.models import backbones


@pytest.mark.parametrize(
    "arch_name, small, top_implemented, input_shape, output_size",
    [
        ["vgg16_bn", None, False, (224, 224, 3), (7, 56, 512)],
        ["resnet31", None, False, (32, 128, 3), (4, 32, 512)],
        ["mobilenetv3", False, False, (512, 512, 3), (1, 1, 1280)],
        ["mobilenetv3", True, False, (512, 512, 3), (1, 1, 1280)],
    ],
)
def test_classification_architectures(arch_name, small, top_implemented, input_shape, output_size):
    # Model
    batch_size = 2
    kwargs = dict()
    if small is not None:
        kwargs["small"] = small
    model = backbones.__dict__[arch_name](**kwargs, pretrained=True)
    # Forward
    out = model(tf.random.uniform(shape=[batch_size, *input_shape], maxval=1, dtype=tf.float32))
    # Output checks
    assert isinstance(out, tf.Tensor)
    assert out.dtype == tf.float32
    assert out.numpy().shape == (batch_size, *output_size)
