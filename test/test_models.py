from doctr import models

import sys
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def _mock_model():
    _layers = [
        layers.Conv2D(8, 3, activation='relu', padding='same', input_shape=(224, 224, 3)),
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(10),
    ]
    return Sequential(_layers)


def test_convert_to_tflite():
    # Define a mock model
    mock_model = _mock_model()

    tflite_model = models.utils.convert_to_tflite(mock_model)
    assert isinstance(tflite_model, bytes)


def test_convert_to_fp16():
    # Define a mock model
    mock_model = _mock_model()

    fp16_model = models.utils.convert_to_fp16(mock_model)
    assert isinstance(fp16_model, bytes)


def test_quantize_model():
    # Define a mock model
    mock_model = _mock_model()

    int_model = models.utils.quantize_model(mock_model, (224, 224, 3))
    assert isinstance(int_model, bytes)
