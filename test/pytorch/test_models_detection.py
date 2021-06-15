import pytest
import torch
import numpy as np

from doctr.models import detection


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
    model = detection.__dict__[arch_name](pretrained=False).eval()
    assert isinstance(model, torch.nn.Module)
    input_tensor = torch.rand((batch_size, *input_shape))
    target = [
        dict(boxes=np.array([[.5, .5, 1, 1], [0.5, 0.5, .8, .8]], dtype=np.float32), flags=[True, False]),
        dict(boxes=np.array([[.5, .5, 1, 1], [0.5, 0.5, .8, .9]], dtype=np.float32), flags=[True, False])
    ]
    out = model(input_tensor, target, return_model_output=True, return_boxes=True)
    assert isinstance(out, dict)
    assert len(out) == 3
    # Check proba map
    assert out['out_map'].shape == (batch_size, *output_size)
    if out_prob:
        assert torch.all((out['out_map'] >= 0) & (out['out_map'] <= 1))
    # Check boxes
    for boxes in out['preds'][0]:
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
        "db_mobilenet_v3",
    ],
)
def test_detection_zoo(arch_name):
    # Model
    predictor = detection.zoo.detection_predictor(arch_name, pretrained=False)
    predictor.eval()
    # object check
    assert isinstance(predictor, detection.DetectionPredictor)
    input_tensor = torch.rand((2, 3, 1024, 1024))
    with torch.no_grad():
        out = predictor(input_tensor)
    assert all(isinstance(out_img, tuple) for out_img in out)
    all_boxes, _ = zip(*out)
    assert all(isinstance(boxes, np.ndarray) and boxes.shape[1] == 5 for boxes in all_boxes)


def test_detection_zoo_error():
    with pytest.raises(ValueError):
        _ = detection.zoo.detection_predictor("my_fancy_model", pretrained=False)
