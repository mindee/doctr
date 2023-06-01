import os
import tempfile

import numpy as np
import onnxruntime
import pytest
import torch

from doctr.file_utils import CLASS_NAME
from doctr.models import detection
from doctr.models.detection._utils import dilate, erode
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.utils import export_model_to_onnx


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size, out_prob",
    [
        ["db_resnet34", (3, 512, 512), (1, 512, 512), True],
        ["db_resnet50", (3, 512, 512), (1, 512, 512), True],
        ["db_mobilenet_v3_large", (3, 512, 512), (1, 512, 512), True],
        ["linknet_resnet18", (3, 512, 512), (1, 512, 512), False],
        ["linknet_resnet34", (3, 512, 512), (1, 512, 512), False],
        ["linknet_resnet50", (3, 512, 512), (1, 512, 512), False],
    ],
)
def test_detection_models(arch_name, input_shape, output_size, out_prob):
    batch_size = 2
    model = detection.__dict__[arch_name](pretrained=False).eval()
    assert isinstance(model, torch.nn.Module)
    input_tensor = torch.rand((batch_size, *input_shape))
    target = [
        {CLASS_NAME: np.array([[0.5, 0.5, 1, 1], [0.5, 0.5, 0.8, 0.8]], dtype=np.float32)},
        {CLASS_NAME: np.array([[0.5, 0.5, 1, 1], [0.5, 0.5, 0.8, 0.9]], dtype=np.float32)},
    ]
    if torch.cuda.is_available():
        model.cuda()
        input_tensor = input_tensor.cuda()
    out = model(input_tensor, target, return_model_output=True, return_preds=True)
    assert isinstance(out, dict)
    assert len(out) == 3
    # Check proba map
    assert out["out_map"].shape == (batch_size, *output_size)
    assert out["out_map"].dtype == torch.float32
    if out_prob:
        assert torch.all((out["out_map"] >= 0) & (out["out_map"] <= 1))
    # Check boxes
    for boxes_dict in out["preds"]:
        for boxes in boxes_dict.values():
            assert boxes.shape[1] == 5
            assert np.all(boxes[:, :2] < boxes[:, 2:4])
            assert np.all(boxes[:, :4] >= 0) and np.all(boxes[:, :4] <= 1)
    # Check loss
    assert isinstance(out["loss"], torch.Tensor)
    # Check the rotated case (same targets)
    target = [
        {
            CLASS_NAME: np.array(
                [[[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1]], [[0.5, 0.5], [0.8, 0.5], [0.8, 0.8], [0.5, 0.8]]],
                dtype=np.float32,
            )
        },
        {
            CLASS_NAME: np.array(
                [[[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1]], [[0.5, 0.5], [0.8, 0.5], [0.8, 0.9], [0.5, 0.9]]],
                dtype=np.float32,
            )
        },
    ]
    loss = model(input_tensor, target)["loss"]
    assert isinstance(loss, torch.Tensor) and ((loss - out["loss"]).abs() / loss).item() < 1e-1


@pytest.mark.parametrize(
    "arch_name",
    [
        "db_resnet34",
        "db_resnet50",
        "db_mobilenet_v3_large",
        "linknet_resnet18",
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
    assert all(isinstance(boxes, dict) for boxes in out)
    assert all(isinstance(boxes[CLASS_NAME], np.ndarray) and boxes[CLASS_NAME].shape[1] == 5 for boxes in out)


def test_erode():
    x = torch.zeros((1, 1, 3, 3))
    x[..., 1, 1] = 1
    expected = torch.zeros((1, 1, 3, 3))
    out = erode(x, 3)
    assert torch.equal(out, expected)


def test_dilate():
    x = torch.zeros((1, 1, 3, 3))
    x[..., 1, 1] = 1
    expected = torch.ones((1, 1, 3, 3))
    out = dilate(x, 3)
    assert torch.equal(out, expected)


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size",
    [
        ["db_resnet34", (3, 512, 512), (1, 512, 512)],
        ["db_resnet50", (3, 512, 512), (1, 512, 512)],
        ["db_mobilenet_v3_large", (3, 512, 512), (1, 512, 512)],
        ["linknet_resnet18", (3, 512, 512), (1, 512, 512)],
        ["linknet_resnet34", (3, 512, 512), (1, 512, 512)],
        ["linknet_resnet50", (3, 512, 512), (1, 512, 512)],
    ],
)
def test_models_onnx_export(arch_name, input_shape, output_size):
    # Model
    batch_size = 2
    model = detection.__dict__[arch_name](pretrained=True, exportable=True).eval()
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export
        model_path = export_model_to_onnx(model, model_name=os.path.join(tmpdir, "model"), dummy_input=dummy_input)
        assert os.path.exists(model_path)
        # Inference
        ort_session = onnxruntime.InferenceSession(
            os.path.join(tmpdir, "model.onnx"), providers=["CPUExecutionProvider"]
        )
        ort_outs = ort_session.run(["logits"], {"input": dummy_input.numpy()})
        assert isinstance(ort_outs, list) and len(ort_outs) == 1
        assert ort_outs[0].shape == (batch_size, *output_size)
