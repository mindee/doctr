import pytest
import torch
import numpy as np

from doctr.models.detection import differentiable_binarization_pt as db


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size, out_prob",
    [
        ["db_resnet34", (3, 1024, 1024), (1, 1024, 1024), True],
        ["db_resnet50", (3, 1024, 1024), (1, 1024, 1024), True],
        ["db_mobilenet_v3", (3, 1024, 1024), (1, 1024, 1024), True],
    ],
)
def test_detection_models(arch_name, input_shape, output_size, out_prob):
    batch_size = 2
    model = db.__dict__[arch_name](pretrained=False).eval()
    assert isinstance(model, torch.nn.Module)
    input_tensor = torch.rand((batch_size, *input_shape))

    with torch.no_grad():
        out = model(input_tensor, return_model_output=True)
    assert isinstance(out, dict)
    assert len(out) == 1
    # Check proba map
    assert out['out_map'].shape == (batch_size, *output_size)
    if out_prob:
        assert torch.all((out['out_map'] >= 0) & (out['out_map'] <= 1))
