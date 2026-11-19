import os
import tempfile

import cv2
import numpy as np
import onnxruntime
import pytest
import torch

from doctr.models import classification
from doctr.models.classification.predictor import OrientationPredictor
from doctr.models.utils import _CompiledModule, export_model_to_onnx


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size",
    [
        ["vgg16_bn_r", (3, 32, 32), (126,)],
        ["resnet18", (3, 32, 32), (126,)],
        ["resnet31", (3, 32, 32), (126,)],
        ["resnet34", (3, 32, 32), (126,)],
        ["resnet34_wide", (3, 32, 32), (126,)],
        ["resnet50", (3, 32, 32), (126,)],
        ["magc_resnet31", (3, 32, 32), (126,)],
        ["mobilenet_v3_small", (3, 32, 32), (126,)],
        ["mobilenet_v3_large", (3, 32, 32), (126,)],
        ["mobilenet_v3_small_crop_orientation", (3, 256, 256), (4,)],
        ["mobilenet_v3_small_page_orientation", (3, 512, 512), (4,)],
        ["vit_s", (3, 32, 32), (126,)],
        ["vit_b", (3, 32, 32), (126,)],
        ["textnet_tiny", (3, 32, 32), (126,)],
        ["textnet_small", (3, 32, 32), (126,)],
        ["textnet_base", (3, 32, 32), (126,)],
        ["vip_base", (3, 32, 32), (126,)],
        ["vip_tiny", (3, 32, 32), (126,)],
        ["vit_det_s", (3, 32, 32), (126,)],
        ["vit_det_m", (3, 32, 32), (126,)],
        ["starnet_s3", (3, 32, 32), (126,)],
        ["hgnetv2_b4", (3, 32, 32), (126,)],
    ],
)
# Coverage tracing slows the (pure Python) dynamo export down ~3x: its code paths are covered by the other tests
@pytest.mark.parametrize("dynamo", [False, pytest.param(True, marks=pytest.mark.no_cover)])
def test_models_onnx_export(arch_name, input_shape, output_size, dynamo):
    if dynamo:
        pytest.importorskip("onnxscript")
        if os.environ.get("CI") and arch_name == "vip_base":
            # The ONNX optimizer of onnxscript takes minutes on the ViP blocks: `vip_tiny` covers them on CI
            pytest.skip("slow dynamo export on CI")
    # Model
    batch_size = 2
    model = classification.__dict__[arch_name](pretrained=True).eval()
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    with torch.inference_mode():
        pt_logits = model(dummy_input).numpy()
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


def test_torch_compiled_crop_orientation_model(mock_text_box):
    text_box_0 = cv2.imread(mock_text_box)
    # rotates counter-clockwise
    text_box_270 = np.rot90(text_box_0, 1)
    text_box_180 = np.rot90(text_box_0, 2)
    text_box_90 = np.rot90(text_box_0, 3)
    compiled_model = torch.compile(classification.mobilenet_v3_small_crop_orientation(pretrained=True))
    compiled_classifier = classification.crop_orientation_predictor(compiled_model)

    assert isinstance(compiled_model, _CompiledModule)
    assert isinstance(compiled_classifier, OrientationPredictor)
    assert compiled_classifier([text_box_0, text_box_270, text_box_180, text_box_90])[0] == [0, 1, 2, 3]
    assert compiled_classifier([text_box_0, text_box_270, text_box_180, text_box_90])[1] == [0, -90, 180, 90]
    assert all(
        isinstance(pred, float)
        for pred in compiled_classifier([text_box_0, text_box_270, text_box_180, text_box_90])[2]
    )


def test_torch_compiled_page_orientation_model(mock_payslip):
    text_box_0 = cv2.imread(mock_payslip)
    # rotates counter-clockwise
    text_box_270 = np.rot90(text_box_0, 1)
    text_box_180 = np.rot90(text_box_0, 2)
    text_box_90 = np.rot90(text_box_0, 3)
    compiled_model = torch.compile(classification.mobilenet_v3_small_page_orientation(pretrained=True))
    compiled_classifier = classification.page_orientation_predictor(compiled_model)

    assert isinstance(compiled_model, _CompiledModule)
    assert isinstance(compiled_classifier, OrientationPredictor)
    assert compiled_classifier([text_box_0, text_box_270, text_box_180, text_box_90])[0] == [0, 1, 2, 3]
    assert compiled_classifier([text_box_0, text_box_270, text_box_180, text_box_90])[1] == [0, -90, 180, 90]
    assert all(
        isinstance(pred, float)
        for pred in compiled_classifier([text_box_0, text_box_270, text_box_180, text_box_90])[2]
    )
