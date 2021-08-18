import pytest
import torch

from doctr.models import backbones


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size",
    [
        ["vgg16_bn", (3, 224, 224), (512, 7, 56)],
        ["resnet31", (3, 32, 128), (512, 4, 32)],
        ["mobilenet_v3_small", (3, 32, 32), (123,)],
        ["mobilenet_v3_large", (3, 32, 32), (123,)],
    ],
)
def test_classification_architectures(arch_name, input_shape, output_size):
    # Model
    batch_size = 2
    model = backbones.__dict__[arch_name](pretrained=True).eval()
    # Forward
    with torch.no_grad():
        out = model(torch.rand((batch_size, *input_shape), dtype=torch.float32))
    # Output checks
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.numpy().shape == (batch_size, *output_size)
    # Check FP16
    if torch.cuda.is_available():
        model = model.half().cuda()
        with torch.no_grad():
            out = model(torch.rand((batch_size, *input_shape), dtype=torch.float16).cuda())
        assert out.dtype == torch.float16
