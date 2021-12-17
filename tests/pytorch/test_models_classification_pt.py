import pytest
import torch

from doctr.models import classification
from doctr.models.classification.predictor import OrientationClassifier


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
def test_recognition_zoo(arch_name):
    batch_size = 16
    # Model
    predictor = classification.zoo.orientation_classifier(arch_name, pretrained=False)
    predictor.model.eval()
    # object check
    assert isinstance(predictor, OrientationClassifier)
    input_tensor = torch.rand((batch_size, 3, 128, 128))
    if torch.cuda.is_available():
        predictor.model.cuda()
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        out = predictor(input_tensor)
    out = predictor(input_tensor)
    assert isinstance(out, list) and len(out) == batch_size
    assert all(isinstance(pred, int) for pred in out)
