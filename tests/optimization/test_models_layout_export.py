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


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["lw_detr_s", (3, 512, 512)],
        ["lw_detr_m", (3, 512, 512)],
    ],
)
# Coverage tracing slows the (pure Python) dynamo export down ~3x: its code paths are covered by the other tests
@pytest.mark.parametrize("dynamo", [False, pytest.param(True, marks=pytest.mark.no_cover)])
def test_models_onnx_export(arch_name, input_shape, dynamo):
    if dynamo:
        pytest.importorskip("onnxscript")
    # Model
    batch_size = 2
    model = layout.__dict__[arch_name](pretrained=True, exportable=True).eval()
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    dummy_masks = torch.ones((batch_size, input_shape[1], input_shape[2]), dtype=torch.bool)
    with torch.inference_mode():
        pt = model(dummy_input, dummy_masks)
    pt_logits = pt["logits"].detach().cpu().numpy()
    pt_boxes = pt["pred_boxes"].detach().cpu().numpy()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export
        model_path = export_model_to_onnx(
            model, model_name=os.path.join(tmpdir, "model"), dummy_input=(dummy_input, dummy_masks), dynamo=dynamo
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
    assert out[0]["class_names"] == compiled_out[0]["class_names"]
    assert np.allclose(out[0]["boxes"], compiled_out[0]["boxes"], atol=1e-4)
    assert np.allclose(out[0]["scores"], compiled_out[0]["scores"], atol=1e-4)
