import cv2
import numpy as np
import pytest
import torch

from doctr.datasets import FONT_STYLE_ATTRIBUTES
from doctr.models import classification
from doctr.models.classification.predictor import FontStylePredictor, OrientationPredictor


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
        ["vip_base", (3, 32, 32), (126,)],
        ["vip_tiny", (3, 32, 32), (126,)],
        ["vit_det_s", (3, 32, 32), (126,)],
        ["vit_det_m", (3, 32, 32), (126,)],
        ["starnet_s3", (3, 32, 32), (126,)],
        ["hgnetv2_b4", (3, 32, 32), (126,)],
    ],
)
def test_classification_architectures(arch_name, input_shape, output_size):
    # Model
    model = classification.__dict__[arch_name](pretrained=True).eval()
    _test_classification(model, input_shape, output_size)
    # SAR, MASTER, ... rebuild their backbone with IntermediateLayerGetter, which iterates named_children() and so keeps
    # a module shared between top-level layers only once: every top-level layer must be its own module
    assert len(list(model.named_children())) == len(model._modules)
    # Check that you can pretrained everything up until the last layer
    assert classification.__dict__[arch_name](pretrained=True, num_classes=10)
    # Check from pretrained is a class method
    assert hasattr(model, "from_pretrained")


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
        input_tensor = np.random.randint(0, 255, (batch_size, 256, 256, 3), dtype=np.uint8)
        # Model
        predictor = classification.zoo.crop_orientation_predictor(arch_name, pretrained=False)
        predictor.model.eval()

        with pytest.raises(ValueError):
            predictor = classification.zoo.crop_orientation_predictor(arch="wrong_model", pretrained=False)
    else:
        batch_size = 2
        input_tensor = np.random.randint(0, 255, (batch_size, 512, 512, 3), dtype=np.uint8)
        # Model
        predictor = classification.zoo.page_orientation_predictor(arch_name, pretrained=False)
        predictor.model.eval()

        with pytest.raises(ValueError):
            predictor = classification.zoo.page_orientation_predictor(arch="wrong_model", pretrained=False)
    # object check
    assert isinstance(predictor, OrientationPredictor)
    if torch.cuda.is_available():
        predictor.model.cuda()

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
    assert classifier([]) == [[], [], []]

    # Test with disabled predictor
    classifier = classification.crop_orientation_predictor(
        "mobilenet_v3_small_crop_orientation", pretrained=False, disabled=True
    )
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90]) == [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    assert classifier([]) == [[], [], []]

    # Test custom model loading
    classifier = classification.crop_orientation_predictor(
        classification.mobilenet_v3_small_crop_orientation(pretrained=True)
    )
    assert isinstance(classifier, OrientationPredictor)

    with pytest.raises(ValueError):
        _ = classification.crop_orientation_predictor(classification.textnet_tiny(pretrained=True))


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
    assert classifier([]) == [[], [], []]

    # Test with disabled predictor
    classifier = classification.page_orientation_predictor(
        "mobilenet_v3_small_page_orientation", pretrained=False, disabled=True
    )
    assert classifier([text_box_0, text_box_270, text_box_180, text_box_90]) == [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1.0, 1.0, 1.0, 1.0],
    ]
    assert classifier([]) == [[], [], []]

    # Test custom model loading
    classifier = classification.page_orientation_predictor(
        classification.mobilenet_v3_small_page_orientation(pretrained=True)
    )
    assert isinstance(classifier, OrientationPredictor)

    with pytest.raises(ValueError):
        _ = classification.page_orientation_predictor(classification.textnet_tiny(pretrained=True))


def test_font_style_model(mock_text_box):
    num_outputs = sum(len(classes) for classes in FONT_STYLE_ATTRIBUTES.values())
    model = classification.mobilenet_v3_small_font_style(pretrained=False).eval()
    assert model.cfg["classes"] == FONT_STYLE_ATTRIBUTES
    assert model.cfg["num_classes"] == num_outputs
    _test_classification(model, (3, 64, 256), (num_outputs,))

    text_box = cv2.imread(mock_text_box)
    classifier = classification.font_style_predictor("mobilenet_v3_small_font_style", pretrained=True, batch_size=2)
    assert isinstance(classifier, FontStylePredictor)
    out = classifier([text_box, text_box[:, ::-1], text_box[: text_box.shape[0] // 2]])
    assert isinstance(out, list) and len(out) == 3
    for style in out:
        assert list(style) == list(FONT_STYLE_ATTRIBUTES)
        for attribute, pred in style.items():
            assert set(pred) == {"value", "confidence"}
            assert pred["value"] in FONT_STYLE_ATTRIBUTES[attribute]
            assert isinstance(pred["confidence"], float) and 0 <= pred["confidence"] <= 1
    assert classifier([]) == []
    # Chunked processing of an iterator gives the same predictions as a single pass
    chunked = classification.font_style_predictor(classifier.model, batch_size=2, chunk_size=2)
    assert chunked(iter([text_box, text_box[:, ::-1], text_box[: text_box.shape[0] // 2]])) == out

    with pytest.raises(ValueError):
        classifier([text_box[..., 0]])
    with pytest.raises(ValueError):
        classification.font_style_predictor(arch="wrong_model", pretrained=False)
    with pytest.raises(ValueError):
        classification.font_style_predictor(arch=42)
    # A model without grouped classes
    with pytest.raises(ValueError):
        classification.font_style_predictor(classification.mobilenet_v3_small_crop_orientation(pretrained=False))
    # The number of outputs must match the classes
    custom = classification.mobilenet_v3_small(pretrained=False, num_classes=4, classes={"bold": [False, True]})
    custom.cfg["input_shape"] = (3, 64, 256)
    with pytest.raises(ValueError):
        classification.font_style_predictor(custom)([text_box])
    # Another backbone trained on custom attributes
    custom = classification.resnet18(pretrained=False, num_classes=5, classes={"bold": [False, True], "u": [0, 1, 2]})
    custom.cfg["input_shape"] = (3, 64, 256)
    out = classification.font_style_predictor(custom)([text_box])
    assert list(out[0]) == ["bold", "u"]
    with pytest.raises(ValueError):
        FontStylePredictor(classifier.pre_processor, classifier.model, chunk_size=0)
    with pytest.raises(ValueError):
        FontStylePredictor(classifier.pre_processor, classifier.model, margin=-0.1)
    # The crop margin comes from the model configuration, and can be overridden
    assert classifier.margin == model.cfg["crop_margin"] > 0
    assert classification.font_style_predictor(classifier.model, margin=0.0).margin == 0.0
