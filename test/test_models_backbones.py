import pytest
import tensorflow as tf

from doctr.models import backbones


@pytest.mark.parametrize(
    "arch_name, top_implemented, input_shape, output_size",
    [
        ["vgg16_bn", False, (224, 224, 3), (7, 56, 512)],
        ["resnet31", False, (32, 128, 3), (4, 32, 512)],
    ],
)
def test_classification_architectures(arch_name, top_implemented, input_shape, output_size):
    # Head not implemented yet
    if not top_implemented:
        with pytest.raises(NotImplementedError):
            backbones.__dict__[arch_name](include_top=True)

    # Model
    batch_size = 2
    model = backbones.__dict__[arch_name](pretrained=True)
    # Forward
    out = model(tf.random.uniform(shape=[batch_size, *input_shape], maxval=1, dtype=tf.float32))
    # Output checks
    assert isinstance(out, tf.Tensor)
    assert out.numpy().shape == (batch_size, *output_size)
