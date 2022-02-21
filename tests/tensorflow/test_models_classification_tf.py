import cv2
import numpy as np
import pytest
import tensorflow as tf

from doctr.models import classification
from doctr.models.classification.predictor import CropOrientationPredictor


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size",
    [
        ["vgg16_bn_r", (32, 32, 3), (126,)],
        ["resnet18", (32, 32, 3), (126,)],
        ["resnet31", (32, 32, 3), (126,)],
        ["resnet34", (32, 32, 3), (126,)],
        ["resnet34_wide", (32, 32, 3), (126,)],
        ["resnet50", (32, 32, 3), (126,)],
        ["magc_resnet31", (32, 32, 3), (126,)],
        ["mobilenet_v3_small", (32, 32, 3), (126,)],
        ["mobilenet_v3_large", (32, 32, 3), (126,)],
    ],
)
def test_classification_architectures(arch_name, input_shape, output_size):
    # Model
    batch_size = 2
    tf.keras.backend.clear_session()
    model = classification.__dict__[arch_name](pretrained=True, include_top=True, input_shape=input_shape)
    # Forward
    out = model(tf.random.uniform(shape=[batch_size, *input_shape], maxval=1, dtype=tf.float32))
    # Output checks
    assert isinstance(out, tf.Tensor)
    assert out.dtype == tf.float32
    assert out.numpy().shape == (batch_size, *output_size)


@pytest.mark.parametrize(
    "arch_name, input_shape",
    [
        ["mobilenet_v3_small_orientation", (128, 128, 3)],
    ],
)
def test_classification_models(arch_name, input_shape):
    batch_size = 8
    reco_model = classification.__dict__[arch_name](pretrained=True, input_shape=input_shape)
    assert isinstance(reco_model, tf.keras.Model)
    input_tensor = tf.random.uniform(shape=[batch_size, *input_shape], minval=0, maxval=1)

    out = reco_model(input_tensor)
    assert isinstance(out, tf.Tensor)
    assert out.shape.as_list() == [8, 4]


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
    # object check
    assert isinstance(predictor, CropOrientationPredictor)
    input_tensor = tf.random.uniform(shape=[batch_size, 128, 128, 3], minval=0, maxval=1)
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
