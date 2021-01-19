import pytest
from io import BytesIO
import sys
import math
import requests
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from doctr import models
from doctr.documents import read_pdf
from test_documents import mock_pdf


@pytest.fixture(scope="module")
def mock_model():
    _layers = [
        layers.Conv2D(8, 3, activation='relu', padding='same', input_shape=(224, 224, 3)),
        layers.GlobalAveragePooling2D(),
        layers.Flatten(),
        layers.Dense(10),
    ]
    return Sequential(_layers)


@pytest.fixture(scope="module")
def test_convert_to_tflite(mock_model):
    serialized_model = models.utils.convert_to_tflite(mock_model)
    assert isinstance(serialized_model, bytes)
    return serialized_model


@pytest.fixture(scope="module")
def test_convert_to_fp16(mock_model):
    serialized_model = models.utils.convert_to_fp16(mock_model)
    assert isinstance(serialized_model, bytes)
    return serialized_model


@pytest.fixture(scope="module")
def test_quantize_model(mock_model):
    serialized_model = models.utils.quantize_model(mock_model, (224, 224, 3))
    assert isinstance(serialized_model, bytes)
    return serialized_model


def test_export_sizes(test_convert_to_tflite, test_convert_to_fp16, test_quantize_model):
    assert sys.getsizeof(test_convert_to_tflite) > sys.getsizeof(test_convert_to_fp16)
    assert sys.getsizeof(test_convert_to_fp16) > sys.getsizeof(test_quantize_model)


def test_preprocess_documents(mock_pdf):  # noqa: F811
    num_docs = 3
    batch_size = 4
    docs = [read_pdf(mock_pdf) for _ in range(num_docs)]
    processor = models.preprocessor.Preprocessor(output_size=(600, 600), batch_size=batch_size)
    batched_docs = processor(docs)

    # Number of batches
    assert len(batched_docs) == math.ceil(8 * num_docs / batch_size)
    # Total number of samples
    assert sum(batch.shape[0] for batch in batched_docs) == 8 * num_docs
    # Batch size
    assert all(batch.shape[0] == batch_size for batch in batched_docs[:-1])
    assert batched_docs[-1].shape[0] == batch_size if (8 * num_docs) % batch_size == 0 else (8 * num_docs) % batch_size
    # Data type
    assert all(batch.dtype == np.float32 for batch in batched_docs)
    # Image size
    assert all(batch.shape[1:] == (600, 600, 3) for batch in batched_docs)
