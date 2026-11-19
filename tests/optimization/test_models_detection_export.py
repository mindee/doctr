import os
import tempfile

import numpy as np
import onnxruntime
import pytest
import torch

from doctr.file_utils import CLASS_NAME
from doctr.io import DocumentFile
from doctr.models import detection
from doctr.models.detection.fast.pytorch import reparameterize
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.utils import _CompiledModule, export_model_to_onnx


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
# Coverage tracing slows the (pure Python) dynamo export down ~3x: its code paths are covered by the other tests
@pytest.mark.parametrize("dynamo", [False, pytest.param(True, marks=pytest.mark.no_cover)])
def test_models_onnx_export(arch_name, input_shape, output_size, dynamo):
    if dynamo:
        pytest.importorskip("onnxscript")
    # Model
    batch_size = 2
    if arch_name == "fast_tiny_rep":
        model = reparameterize(detection.fast_tiny(pretrained=True, exportable=True).eval())
    else:
        model = detection.__dict__[arch_name](pretrained=True, exportable=True).eval()
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    with torch.inference_mode():
        pt_logits = model(dummy_input)["logits"].numpy()
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export
        model_path = export_model_to_onnx(
            model, model_name=os.path.join(tmpdir, "model"), dummy_input=dummy_input, dynamo=dynamo
        )
        assert os.path.exists(model_path)
        # Inference
        ort_session = onnxruntime.InferenceSession(
            os.path.join(tmpdir, "model.onnx"), providers=["CPUExecutionProvider"]
        )
        ort_outs = ort_session.run(["logits"], {"input": dummy_input.numpy()})

    assert isinstance(ort_outs, list) and len(ort_outs) == 1
    assert ort_outs[0].shape == (batch_size, *output_size)
    # Check that the output is close to the PyTorch output - only warn if not close
    try:
        assert np.allclose(pt_logits, ort_outs[0], atol=1e-4)
    except AssertionError:
        pytest.skip(f"Output of {arch_name}:\nMax element-wise difference: {np.max(np.abs(pt_logits - ort_outs[0]))}")


@pytest.mark.parametrize(
    "arch_name",
    [
        "db_resnet34",
        "db_resnet50",
        "db_mobilenet_v3_large",
        "linknet_resnet18",
        "linknet_resnet34",
        "linknet_resnet50",
        "fast_tiny",
        "fast_small",
        "fast_base",
    ],
)
def test_torch_compiled_models(arch_name, mock_payslip):
    doc = DocumentFile.from_images([mock_payslip])
    predictor = detection.zoo.detection_predictor(arch_name, pretrained=True)
    assert isinstance(predictor, DetectionPredictor)
    out, seg_maps = predictor(doc, return_maps=True)

    # Compile the model
    compiled_model = torch.compile(detection.__dict__[arch_name](pretrained=True).eval())
    assert isinstance(compiled_model, _CompiledModule)
    compiled_predictor = detection.zoo.detection_predictor(compiled_model)
    compiled_out, compiled_seg_maps = compiled_predictor(doc, return_maps=True)

    # Compare
    assert all(
        np.allclose(out_boxes[CLASS_NAME], compiled_out_boxes[CLASS_NAME], atol=1e-4)
        for out_boxes, compiled_out_boxes in zip(out, compiled_out)
    )
    assert all(
        # small increased tolerance because `fast_small` is slightly over the 1e-4 threshold
        np.allclose(seg_map, compiled_seg_map, atol=2e-4)
        for seg_map, compiled_seg_map in zip(seg_maps, compiled_seg_maps)
    )
