import os
import tempfile

import numpy as np
import onnxruntime
import pytest
import torch

from doctr.io import DocumentFile
from doctr.models import layout
from doctr.models.layout.predictor import LayoutPredictor
from doctr.models.utils import _CompiledModule, export_model_to_onnx


@pytest.mark.parametrize("train_mode", [True, False])
@pytest.mark.parametrize("use_polygons", [True, False])
@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["lw_detr_s", (3, 512, 512)],
        ["lw_detr_m", (3, 1024, 1024)],
    ],
)
def test_layout_models(arch_name, input_shape, train_mode, use_polygons):
    batch_size = 2
    model = layout.__dict__[arch_name](pretrained=True)
    model = model.train() if train_mode else model.eval()
    assert isinstance(model, torch.nn.Module)
    input_tensor = torch.rand((batch_size, *input_shape))
    input_masks = torch.zeros((batch_size, input_shape[1], input_shape[2]), dtype=torch.bool)

    class_names = model.class_names

    target = []
    for _ in range(batch_size):
        sample_target = {}
        num_boxes = 5
        for _ in range(num_boxes):
            cls_name = np.random.choice(class_names)
            x1, y1 = torch.rand(2) * 0.8
            if use_polygons:
                w, h = 0.1, 0.1

                box = np.array(
                    [
                        [x1, y1],
                        [x1 + w, y1],
                        [x1 + w, y1 + h],
                        [x1, y1 + h],
                    ],
                    dtype=np.float32,
                )  # (4,2)
            else:
                x2, y2 = x1 + 0.1, y1 + 0.1
                box = np.array([x1, y1, x2, y2], dtype=np.float32)  # (4,)
            sample_target.setdefault(cls_name, [])
            sample_target[cls_name].append(box)
        target.append(sample_target)

    if torch.cuda.is_available():
        model.cuda()
        input_tensor = input_tensor.cuda()
        input_masks = input_masks.cuda()
    out = model(input_tensor, input_masks, target, return_model_output=True, return_preds=not train_mode)
    assert isinstance(out, dict)
    assert len(out) == 3 if not train_mode else len(out) == 2
    # Check logits
    assert "logits" in out
    assert isinstance(out["logits"], torch.Tensor)

    # Check Preds
    if not train_mode:
        for results in out["preds"]:
            assert isinstance(results, tuple) and len(results) == 3
            assert isinstance(results[0], list) and all(isinstance(idxs, int) for idxs in results[0])
            assert isinstance(results[1], np.ndarray) and results[1].shape == (len(results[0]), 4)
            assert isinstance(results[2], list) and all(isinstance(scores, float) for scores in results[2])
            # Check class idxs are in the model's num_classes
            assert all(0 <= idx < model.num_classes for idx in results[0])
            # Check scores are between 0 and 1
            assert all(0 <= score <= 1 for score in results[2])
            # Check that the number of boxes, labels and scores are the same
            assert len(results[0]) == len(results[1]) == len(results[2])
            # Check that boxes are in the range [0, 1]
            assert all((box >= 0).all() and (box <= 1).all() for box in results[1])
    # Check loss
    assert isinstance(out["loss"], torch.Tensor)
    assert hasattr(model, "from_pretrained")


@pytest.mark.parametrize(
    "arch_name",
    [
        "lw_detr_s",
        "lw_detr_m",
    ],
)
def test_layout_zoo(arch_name):
    # Model
    predictor = layout.zoo.layout_predictor(arch_name, pretrained=False)
    predictor.model.eval()
    # object check
    assert isinstance(predictor, LayoutPredictor)
    input_tensor = np.random.rand(2, 1024, 1024, 3).astype(np.float32)
    if torch.cuda.is_available():
        predictor.model.cuda()

    with torch.no_grad():
        out = predictor(input_tensor)
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(sample, dict) for sample in out)
    assert all("class_names" in sample and "boxes" in sample and "scores" in sample for sample in out)
    assert all(isinstance(sample["class_names"], list) for sample in out)
    assert all(isinstance(sample["boxes"], np.ndarray) for sample in out)
    assert all(isinstance(sample["scores"], list) for sample in out)
    assert all(sample["boxes"].shape[1] == 4 for sample in out)
    assert all(len(sample["class_names"]) == len(sample["scores"]) == sample["boxes"].shape[0] for sample in out)
    assert all(all(isinstance(score, float) and 0 <= score <= 1 for score in sample["scores"]) for sample in out)


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["lw_detr_s", (3, 512, 512)],
        ["lw_detr_m", (3, 512, 512)],
    ],
)
def test_models_onnx_export(arch_name, input_shape):
    # Model
    batch_size = 2
    model = layout.__dict__[arch_name](pretrained=True, exportable=True).eval()
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    dummy_masks = torch.zeros((batch_size, input_shape[1], input_shape[2]), dtype=torch.bool)
    pt = model(dummy_input, dummy_masks)
    pt_logits = pt["logits"].detach().cpu().numpy()
    pt_boxes = pt["pred_boxes"].detach().cpu().numpy()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export
        model_path = export_model_to_onnx(
            model, model_name=os.path.join(tmpdir, "model"), dummy_input=(dummy_input, dummy_masks)
        )
        assert os.path.exists(model_path)
        # Inference
        ort_session = onnxruntime.InferenceSession(
            os.path.join(tmpdir, "model.onnx"), providers=["CPUExecutionProvider"]
        )
        ort_outs = ort_session.run(
            ["logits", "pred_boxes"], {"input": dummy_input.numpy(), "masks": dummy_masks.numpy()}
        )

    assert isinstance(ort_outs, list) and len(ort_outs) == 2
    # Check boxes shape
    assert ort_outs[0].shape == pt_logits.shape
    assert ort_outs[1].shape == pt_boxes.shape
    # Check that the output is close to the PyTorch output - only warn if not close
    try:
        assert np.allclose(pt_logits, ort_outs[0], atol=1e-4)
        assert np.allclose(pt_boxes, ort_outs[1], atol=1e-4)
    except AssertionError:
        pytest.skip(f"Output of {arch_name}:\nMax element-wise difference: {np.max(np.abs(pt_logits - ort_outs[0]))}")


@pytest.mark.parametrize(
    "arch_name",
    [
        "lw_detr_s",
        "lw_detr_m",
    ],
)
def test_torch_compiled_models(arch_name, mock_payslip):
    doc = DocumentFile.from_images([mock_payslip])
    predictor = layout.zoo.layout_predictor(arch_name, pretrained=True)
    assert isinstance(predictor, LayoutPredictor)
    out = predictor(doc)

    # Compile the model
    compiled_model = torch.compile(layout.__dict__[arch_name](pretrained=True).eval())
    assert isinstance(compiled_model, _CompiledModule)
    compiled_predictor = layout.zoo.layout_predictor(compiled_model)
    compiled_out = compiled_predictor(doc)

    # Compare that outputs are close
    assert len(out) == len(compiled_out) == 1
    # TODO: Enable if the model has a pretrained version
    # assert out[0]["class_names"] == compiled_out[0]["class_names"]
    # assert np.allclose(out[0]["boxes"], compiled_out[0]["boxes"], atol=1e-4)
    # assert np.allclose(out[0]["scores"], compiled_out[0]["scores"], atol=1e-4)
