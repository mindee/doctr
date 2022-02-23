import cv2
import numpy as np
import pytest
import torch

from doctr.models import classification
from doctr.models.classification.predictor import CropOrientationPredictor


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
    ],
)
def test_classification_architectures(arch_name, input_shape, output_size):
    # Model
    batch_size = 2
    model = classification.__dict__[arch_name](pretrained=True).eval()
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
    "arch_name, input_shape",
    [
        ["mobilenet_v3_small_orientation", (3, 128, 128)],
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
        "mobilenet_v3_small_orientation",
    ],
)
def test_classification_zoo(arch_name):
    batch_size = 16
    # Model
    predictor = classification.zoo.crop_orientation_predictor(arch_name, pretrained=False)
    predictor.model.eval()
    # object check
    assert isinstance(predictor, CropOrientationPredictor)
    input_tensor = torch.rand((batch_size, 3, 128, 128))
    if torch.cuda.is_available():
        predictor.model.cuda()
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        out = predictor(input_tensor)
    out = predictor(input_tensor)
    assert isinstance(out, list) and len(out) == batch_size
    assert all(isinstance(pred, int) for pred in out)


def test_crop_orientation_model(mock_text_box):
    text_box_0 = cv2.imread(mock_text_box)
    text_box_90 = np.rot90(text_box_0, 1)
    text_box_180 = np.rot90(text_box_0, 2)
    text_box_270 = np.rot90(text_box_0, 3)
    classifier = classification.crop_orientation_predictor("mobilenet_v3_small_orientation", pretrained=True)
    assert classifier([text_box_0, text_box_90, text_box_180, text_box_270]) == [0, 1, 2, 3]
