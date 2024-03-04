import os
import tempfile

import numpy as np
import onnxruntime
import pytest
import torch

from doctr.file_utils import CLASS_NAME
from doctr.models import detection
from doctr.models.detection._utils import dilate, erode
from doctr.models.detection.fast.pytorch import reparameterize
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.utils import export_model_to_onnx


@pytest.mark.parametrize("train_mode", [True, False])
@pytest.mark.parametrize(
    "arch_name, input_shape, output_size, out_prob",
    [
        ["db_resnet34", (3, 512, 512), (1, 512, 512), True],
        ["db_resnet50", (3, 512, 512), (1, 512, 512), True],
        ["db_mobilenet_v3_large", (3, 512, 512), (1, 512, 512), True],
        ["linknet_resnet18", (3, 512, 512), (1, 512, 512), True],
        ["linknet_resnet34", (3, 512, 512), (1, 512, 512), True],
        ["linknet_resnet50", (3, 512, 512), (1, 512, 512), True],
        ["fast_tiny", (3, 512, 512), (1, 512, 512), True],
        ["fast_tiny_rep", (3, 512, 512), (1, 512, 512), True],  # Reparameterized model
        ["fast_small", (3, 512, 512), (1, 512, 512), True],
        ["fast_base", (3, 512, 512), (1, 512, 512), True],
    ],
)
def test_detection_models(arch_name, input_shape, output_size, out_prob, train_mode):
    batch_size = 2
    if arch_name == "fast_tiny_rep":
        model = reparameterize(detection.fast_tiny(pretrained=True).eval())
        train_mode = False  # Reparameterized model is not trainable
    else:
        model = detection.__dict__[arch_name](pretrained=True)
        model = model.train() if train_mode else model.eval()
    assert isinstance(model, torch.nn.Module)
    input_tensor = torch.rand((batch_size, *input_shape))
    target = [
        {CLASS_NAME: np.array([[0.5, 0.5, 1, 1], [0.5, 0.5, 0.8, 0.8]], dtype=np.float32)},
        {CLASS_NAME: np.array([[0.5, 0.5, 1, 1], [0.5, 0.5, 0.8, 0.9]], dtype=np.float32)},
    ]
    if torch.cuda.is_available():
        model.cuda()
        input_tensor = input_tensor.cuda()
    out = model(input_tensor, target, return_model_output=True, return_preds=not train_mode)
    assert isinstance(out, dict)
    assert len(out) == 3 if not train_mode else len(out) == 2
    # Check proba map
    assert out["out_map"].shape == (batch_size, *output_size)
    assert out["out_map"].dtype == torch.float32
    if out_prob:
        assert torch.all((out["out_map"] >= 0) & (out["out_map"] <= 1))
    # Check boxes
    if not train_mode:
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
    assert isinstance(loss, torch.Tensor) and ((loss - out["loss"]).abs() / loss).item() < 1


@pytest.mark.parametrize(
    "arch_name",
    [
        "db_resnet34",
        "db_resnet50",
        "db_mobilenet_v3_large",
        "linknet_resnet18",
        "fast_tiny",
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
        out, seq_maps = predictor(input_tensor, return_maps=True)
    assert all(isinstance(boxes, dict) for boxes in out)
    assert all(isinstance(boxes[CLASS_NAME], np.ndarray) and boxes[CLASS_NAME].shape[1] == 5 for boxes in out)
    assert all(isinstance(seq_map, np.ndarray) for seq_map in seq_maps)
    assert all(seq_map.shape[:2] == (1024, 1024) for seq_map in seq_maps)
    # check that all values in the seq_maps are between 0 and 1
    assert all((seq_map >= 0).all() and (seq_map <= 1).all() for seq_map in seq_maps)


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
        ["fast_tiny", (3, 512, 512), (1, 512, 512)],
        ["fast_small", (3, 512, 512), (1, 512, 512)],
        ["fast_base", (3, 512, 512), (1, 512, 512)],
        ["fast_tiny_rep", (3, 512, 512), (1, 512, 512)],  # Reparameterized model
    ],
)
def test_models_onnx_export(arch_name, input_shape, output_size):
    # Model
    batch_size = 2
    if arch_name == "fast_tiny_rep":
        model = reparameterize(detection.fast_tiny(pretrained=True, exportable=True).eval())
    else:
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
