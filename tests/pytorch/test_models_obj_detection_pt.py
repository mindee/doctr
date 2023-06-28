import pytest
import torch

from doctr.models import obj_detection


@pytest.mark.parametrize(
    "arch_name, input_shape, pretrained",
    [
        ["fasterrcnn_mobilenet_v3_large_fpn", (3, 512, 512), True],
        ["fasterrcnn_mobilenet_v3_large_fpn", (3, 512, 512), False],
    ],
)
def test_detection_models(arch_name, input_shape, pretrained):
    batch_size = 2
    model = obj_detection.__dict__[arch_name](pretrained=pretrained).eval()
    assert isinstance(model, torch.nn.Module)
    input_tensor = torch.rand((batch_size, *input_shape))
    if torch.cuda.is_available():
        model.cuda()
        input_tensor = input_tensor.cuda()
    out = model(input_tensor)
    assert isinstance(out, list) and all(isinstance(det, dict) for det in out)

    # Train mode
    model = model.train()
    target = [
        dict(boxes=torch.tensor([[0.5, 0.5, 1, 1]], dtype=torch.float32), labels=torch.tensor((0,), dtype=torch.long)),
        dict(boxes=torch.tensor([[0.5, 0.5, 1, 1]], dtype=torch.float32), labels=torch.tensor((0,), dtype=torch.long)),
    ]
    if torch.cuda.is_available():
        target = [{k: v.cuda() for k, v in t.items()} for t in target]
    out = model(input_tensor, target)
    assert isinstance(out, dict) and all(isinstance(v, torch.Tensor) for v in out.values())
