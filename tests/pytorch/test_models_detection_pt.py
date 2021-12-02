import numpy as np
import pytest
import torch

from doctr.models import detection
from doctr.models.detection.predictor import DetectionPredictor


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size, out_prob",
    [
        ["db_resnet34", (3, 1024, 1024), (1, 1024, 1024), True],
        ["db_resnet50", (3, 1024, 1024), (1, 1024, 1024), True],
        ["db_mobilenet_v3_large", (3, 1024, 1024), (1, 1024, 1024), True],
        ["linknet16", (3, 1024, 1024), (1, 1024, 1024), False],
    ],
)
def test_detection_models(arch_name, input_shape, output_size, out_prob):
    batch_size = 2
    model = detection.__dict__[arch_name](pretrained=False).eval()
    assert isinstance(model, torch.nn.Module)
    input_tensor = torch.rand((batch_size, *input_shape))
    target = [
        np.array([[.5, .5, 1, 1], [0.5, 0.5, .8, .8]], dtype=np.float32),
        np.array([[.5, .5, 1, 1], [0.5, 0.5, .8, .9]], dtype=np.float32),
    ]
    if torch.cuda.is_available():
        model.cuda()
        input_tensor = input_tensor.cuda()
    out = model(input_tensor, target, return_model_output=True, return_boxes=True)
    assert isinstance(out, dict)
    assert len(out) == 3
    # Check proba map
    assert out['out_map'].shape == (batch_size, *output_size)
    assert out['out_map'].dtype == torch.float32
    if out_prob:
        assert torch.all((out['out_map'] >= 0) & (out['out_map'] <= 1))
    # Check boxes
    for boxes in out['preds']:
        assert boxes.shape[1] == 5
        assert np.all(boxes[:, :2] < boxes[:, 2:4])
        assert np.all(boxes[:, :4] >= 0) and np.all(boxes[:, :4] <= 1)
    # Check loss
    assert isinstance(out['loss'], torch.Tensor)


@pytest.mark.parametrize(
    "arch_name",
    [
        "db_resnet34",
        "db_resnet50",
        "db_mobilenet_v3_large",
        "linknet16",
    ],
)
def test_detection_zoo(arch_name):
    # Model
    predictor = detection.zoo.detection_predictor(arch_name, pretrained=False)
    predictor.model.eval()
    # object check
    assert isinstance(predictor, DetectionPredictor)
    input_tensor = torch.rand((2, 3, 1024, 1024))
    if torch.cuda.is_available():
        predictor.model.cuda()
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        out = predictor(input_tensor)
    assert all(isinstance(boxes, np.ndarray) and boxes.shape[1] == 5 for boxes in out)
