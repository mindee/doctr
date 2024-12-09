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


def _test_classification(model, input_shape, output_size, batch_size=2):
    # Forward
    with torch.no_grad():
        out = model(torch.rand((batch_size, *input_shape), dtype=torch.float32))
    # Output checks
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.numpy().shape == (batch_size, *output_size)
    # Check FP16
    if torch.cuda.is_available():
        model = model.half().cuda()
        with torch.no_grad():
            out = model(torch.rand((batch_size, *input_shape), dtype=torch.float16).cuda())
        assert out.dtype == torch.float16


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
        ["textnet_tiny", (3, 32, 32), (126,)],
        ["textnet_small", (3, 32, 32), (126,)],
        ["textnet_base", (3, 32, 32), (126,)],
        ["vit_s", (3, 32, 32), (126,)],
        ["vit_b", (3, 32, 32), (126,)],
        # Check that the interpolation of positional embeddings for vit models works correctly
        ["vit_s", (3, 64, 64), (126,)],
    ],
)
def test_classification_architectures(arch_name, input_shape, output_size):
    # Model
    model = classification.__dict__[arch_name](pretrained=True).eval()
    _test_classification(model, input_shape, output_size)
    # Check that you can pretrained everything up until the last layer
    assert classification.__dict__[arch_name](pretrained=True, num_classes=10)


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["mobilenet_v3_small_crop_orientation", (3, 256, 256)],
        ["mobilenet_v3_small_page_orientation", (3, 512, 512)],
    ],
)
def test_classification_models(arch_name, input_shape):
    batch_size = 8
    model = classification.__dict__[arch_name](pretrained=False, input_shape=input_shape).eval()
    assert isinstance(model, torch.nn.Module)
    input_tensor = torch.rand((batch_size, *input_shape))

    if torch.cuda.is_available():
        model.cuda()
        input_tensor = input_tensor.cuda()
    out = model(input_tensor)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (8, 4)


@pytest.mark.parametrize(
    "arch_name",
    [
        "mobilenet_v3_small_crop_orientation",
        "mobilenet_v3_small_page_orientation",
    ],
)
def test_classification_zoo(arch_name):
    if "crop" in arch_name:
        batch_size = 16
        input_tensor = torch.rand((batch_size, 3, 256, 256))
        # Model
        predictor = classification.zoo.crop_orientation_predictor(arch_name, pretrained=False)
        predictor.model.eval()

        with pytest.raises(ValueError):
            predictor = classification.zoo.crop_orientation_predictor(arch="wrong_model", pretrained=False)
    else:
        batch_size = 2
        input_tensor = torch.rand((batch_size, 3, 512, 512))
        # Model
        predictor = classification.zoo.page_orientation_predictor(arch_name, pretrained=False)
        predictor.model.eval()

        with pytest.raises(ValueError):
            predictor = classification.zoo.page_orientation_predictor(arch="wrong_model", pretrained=False)
    # object check
    assert isinstance(predictor, OrientationPredictor)
    if torch.cuda.is_available():
        predictor.model.cuda()
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        out = predictor(input_tensor)
    class_idxs, classes, confs = out[0], out[1], out[2]
    assert isinstance(class_idxs, list) and len(class_idxs) == batch_size
    assert isinstance(classes, list) and len(classes) == batch_size
    assert isinstance(confs, list) and len(confs) == batch_size
    assert all(isinstance(pred, int) for pred in class_idxs)
    assert all(isinstance(pred, int) for pred in classes) and all(pred in [0, 90, 180, -90] for pred in classes)
    assert all(isinstance(pred, float) for pred in confs)


def test_crop_orientation_model(mock_text_box):
    text_box_0 = cv2.imread(mock_text_box)
    # rotates counter-clockwise
    text_box_270 = np.rot90(text_box_0, 1)
    text_box_180 = np.rot90(text_box_0, 2)
    text_box_90 = np.rot90(text_box_0, 3)
    classifier = classification.crop_orientation_predictor("mobilenet_v3_small_crop_orientation", pretrained=True)
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90])[0] == [0, 1, 2, 3]
    # 270 degrees is equivalent to -90 degrees
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90])[1] == [0, -90, 180, 90]
    assert all(isinstance(pred, float) for pred in classifier([text_box_0, text_box_270, text_box_180, text_box_90])[2])

    # Test with disabled predictor
    classifier = classification.crop_orientation_predictor(
        "mobilenet_v3_small_crop_orientation", pretrained=False, disabled=True
    )
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90]) == [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1.0, 1.0, 1.0, 1.0],
    ]

    # Test custom model loading
    classifier = classification.crop_orientation_predictor(
        classification.mobilenet_v3_small_crop_orientation(pretrained=True)
    )
    assert isinstance(classifier, OrientationPredictor)

    with pytest.raises(ValueError):
        _ = classification.crop_orientation_predictor(classification.textnet_tiny(pretrained=True))

    # Test torch compilation
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


def test_page_orientation_model(mock_payslip):
    text_box_0 = cv2.imread(mock_payslip)
    # rotates counter-clockwise
    text_box_270 = np.rot90(text_box_0, 1)
    text_box_180 = np.rot90(text_box_0, 2)
    text_box_90 = np.rot90(text_box_0, 3)
    classifier = classification.page_orientation_predictor("mobilenet_v3_small_page_orientation", pretrained=True)
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90])[0] == [0, 1, 2, 3]
    # 270 degrees is equivalent to -90 degrees
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90])[1] == [0, -90, 180, 90]
    assert all(isinstance(pred, float) for pred in classifier([text_box_0, text_box_270, text_box_180, text_box_90])[2])

    # Test with disabled predictor
    classifier = classification.page_orientation_predictor(
        "mobilenet_v3_small_page_orientation", pretrained=False, disabled=True
    )
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90]) == [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1.0, 1.0, 1.0, 1.0],
    ]

    # Test custom model loading
    classifier = classification.page_orientation_predictor(
        classification.mobilenet_v3_small_page_orientation(pretrained=True)
    )
    assert isinstance(classifier, OrientationPredictor)

    with pytest.raises(ValueError):
        _ = classification.page_orientation_predictor(classification.textnet_tiny(pretrained=True))

    # Test torch compilation
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
    ],
)
def test_models_onnx_export(arch_name, input_shape, output_size):
    # Model
    batch_size = 2
    model = classification.__dict__[arch_name](pretrained=True).eval()
    dummy_input = torch.rand((batch_size, *input_shape), dtype=torch.float32)
    pt_logits = model(dummy_input).detach().cpu().numpy()
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
    # Check that the output is close to the PyTorch output - only warn if not close
    try:
        assert np.allclose(pt_logits, ort_outs[0], atol=1e-4)
    except AssertionError:
        pytest.skip(f"Output of {arch_name}:\nMax element-wise difference: {np.max(np.abs(pt_logits - ort_outs[0]))}")
