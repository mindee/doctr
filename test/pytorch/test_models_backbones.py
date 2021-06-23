import pytest
import torch

from doctr.models import backbones


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size",
    [
        ["resnet31", (3, 32, 128), (512, 4, 32)],
    ],
)
def test_classification_architectures(arch_name, input_shape, output_size):
    # Model
    batch_size = 2
    model = backbones.__dict__[arch_name](pretrained=True)
    # Forward
    out = model(torch.rand((batch_size, *input_shape), dtype=tf.float32))
    # Output checks
    assert isinstance(out, torch.Tensor)
    assert out.numpy().shape == (batch_size, *output_size)
