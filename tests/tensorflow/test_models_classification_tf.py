import pytest
import tensorflow as tf

from doctr.models import classification
from doctr.models.classification.predictor import OrientationClassifier


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
    predictor = classification.zoo.orientation_classifier(arch_name, pretrained=False)
    # object check
    assert isinstance(predictor, OrientationClassifier)
    input_tensor = tf.random.uniform(shape=[batch_size, 128, 128, 3], minval=0, maxval=1)
    out = predictor(input_tensor)
    assert isinstance(out, list) and len(out) == batch_size
    assert all(isinstance(pred, int) for pred in out)
