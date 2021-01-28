import pytest
from io import BytesIO

import tensorflow as tf
import numpy as np
import sys
import math
import requests

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from doctr.documents import read_pdf
from test_documents import mock_pdf
from doctr import models


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
def mock_mapping():
    return {
        "V": 0, "W": 1, ";": 2, "w": 3, "&": 4, "1": 5, "<": 6,
        "\u00fb": 7, "p": 8, "h": 9, "9": 10, "\u00f9": 11, "\u00d9": 12, "j": 13,
        "*": 14, "s": 15, "?": 16, ",": 17, "\u00ee": 18, "\u00d4": 19, "8": 20,
        "@": 21, "D": 22, ">": 23, "$": 24, "\u00db": 25, "k": 26, "{": 27, "I": 28,
        "F": 29, ":": 30, "O": 31, "\u00e0": 32, "a": 33, "\u00c0": 34, "v": 35, "X": 36,
        "[": 37, "\u00ea": 38, "M": 39, "q": 40, "5": 41, "\u00c2": 42, "G": 43, "\u00f4": 44,
        "\"": 45, "\u00e7": 46, "L": 47, "\u00e9": 48, "\u00ef": 49, "6": 50, "\u00ce": 51,
        "y": 52, "/": 53, "#": 54, "3": 55, "N": 56, "x": 57, "\u00c8": 58, "]": 59, "K": 60,
        "\u00a3": 61, "7": 62, "R": 63, "'": 64, "U": 65, "\u00e8": 66, "J": 67, "H": 68,
        "t": 69, "r": 70, "c": 71, "P": 72, ".": 73, "\u00cf": 74, "z": 75, "m": 76, "Z": 77,
        "}": 78, "0": 79, "(": 80, "\u00cb": 81, "b": 82, "\u00e2": 83, "-": 84, "B": 85, "T": 86,
        "\u00eb": 87, "%": 88, "\u20ac": 89, "E": 90, ")": 91, "i": 92, "_": 93, "Q": 94, "|": 95,
        "\u00c9": 96, "S": 97, "o": 98, "=": 99, "Y": 100, "A": 101, "4": 102, "e": 103, "n": 104,
        "u": 105, "g": 106, "!": 107, "2": 108, "l": 109, "f": 110, "+": 111, "\u00c7": 112,
        "C": 113, "d": 114
    }


@pytest.fixture(scope="module")
def test_convert_to_tflite(mock_model):
    serialized_model = models.export.convert_to_tflite(mock_model)
    assert isinstance(serialized_model, bytes)
    return serialized_model


@pytest.fixture(scope="module")
def test_convert_to_fp16(mock_model):
    serialized_model = models.export.convert_to_fp16(mock_model)
    assert isinstance(serialized_model, bytes)
    return serialized_model


@pytest.fixture(scope="module")
def test_quantize_model(mock_model):
    serialized_model = models.export.quantize_model(mock_model, (224, 224, 3))
    assert isinstance(serialized_model, bytes)
    return serialized_model


def test_export_sizes(test_convert_to_tflite, test_convert_to_fp16, test_quantize_model):
    assert sys.getsizeof(test_convert_to_tflite) > sys.getsizeof(test_convert_to_fp16)
    assert sys.getsizeof(test_convert_to_fp16) > sys.getsizeof(test_quantize_model)


def test_preprocess_documents(mock_pdf):  # noqa: F811
    num_docs = 3
    batch_size = 4
    docs = [read_pdf(mock_pdf) for _ in range(num_docs)]
    processor = models.PreProcessor(output_size=(600, 600), batch_size=batch_size)
    batched_docs = processor([page for doc in docs for page in doc])

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
    # Test with non-full last batch
    batch_size = 16
    processor = models.PreProcessor(output_size=(600, 600), batch_size=batch_size)
    batched_docs = processor([page for doc in docs for page in doc])
    assert batched_docs[-1].shape[0] == (8 * num_docs) % batch_size


def test_dbpostprocessor():
    postprocessor = models.DBPostProcessor()
    mock_batch = tf.random.uniform(shape=[8, 600, 600, 1], minval=0, maxval=1)
    out = postprocessor(mock_batch)
    # Batch composition
    assert isinstance(out, list)
    assert len(out) == 8
    assert all(isinstance(sample, np.ndarray) for sample in out)
    assert all(sample.shape[1] == 5 for sample in out)
    # Relative coords
    assert all(np.all(np.logical_and(sample[:4] > 0, sample[:4] < 1)) for sample in out)


def test_dbmodel():
    dbmodel = models.DBResNet50(input_size=(640, 640))
    dbinput = tf.random.uniform(shape=[8, 640, 640, 3], minval=0, maxval=1)
    # test prediction model
    dboutput_notrain = dbmodel(inputs=dbinput, training=False)
    assert isinstance(dboutput_notrain, tf.Tensor)
    assert isinstance(dbmodel, tf.keras.Model)
    assert dboutput_notrain.numpy().shape == (8, 640, 640, 1)
    assert np.all(dboutput_notrain.numpy() > 0) and np.all(dboutput_notrain.numpy() < 1)
    # test training model
    dboutput_train = dbmodel(inputs=dbinput, training=True)
    assert isinstance(dboutput_train, tuple)
    assert len(dboutput_train) == 3
    assert all(np.all(out_map.numpy() > 0) and np.all(out_map.numpy() < 1) for out_map in dboutput_train)
    # batch size
    assert all(out.numpy().shape == (8, 640, 640, 1) for out in dboutput_train)


def test_extract_crops(mock_pdf):  # noqa: F811
    doc_img = read_pdf(mock_pdf)[0]
    num_crops = 2
    boxes = np.array([[idx / num_crops, idx / num_crops, (idx + 1) / num_crops, (idx + 1) / num_crops]
                      for idx in range(num_crops)], dtype=np.float32)
    croped_imgs = models.extract_crops(doc_img, boxes)

    # Number of crops
    assert len(croped_imgs) == num_crops
    # Data type and shape
    assert all(isinstance(crop, np.ndarray) for crop in croped_imgs)
    assert all(crop.ndim == 3 for crop in croped_imgs)

    # Identity
    assert np.all(doc_img == models.extract_crops(doc_img, np.array([[0, 0, 1, 1]]))[0])

    # No box
    assert models.extract_crops(doc_img, np.zeros((0, 4))) == []


def test_crnn():
    crnn_model = models.CRNN(num_classes=30, input_size=(32, 128, 3), rnn_units=128)
    crnn_input = tf.random.uniform(shape=[8, 32, 128, 3], minval=0, maxval=1)
    crnn_out = crnn_model(inputs=crnn_input)
    assert isinstance(crnn_out, tf.Tensor)
    assert isinstance(crnn_model, tf.keras.Model)
    assert crnn_out.numpy().shape == (8, 32, 31)


def test_sar():
    sar_model = models.SAR(
        input_size=(64, 256, 3),
        rnn_units=512,
        embedding_units=512,
        attention_units=512,
        max_length=30,
        num_classes=110,
        num_decoder_layers=2
    )
    sar_input = tf.random.uniform(shape=[8, 64, 256, 3], minval=0, maxval=1)
    sar_out = sar_model(inputs=sar_input)
    assert isinstance(sar_out, tf.Tensor)
    assert isinstance(sar_model, tf.keras.Model)
    assert sar_out.numpy().shape == (8, 31, 111)


def test_ctc_decoder(mock_mapping):
    ctc_postprocessor = models.recognition.CTCPostProcessor(
        num_classes=len(mock_mapping),
        label_to_idx=mock_mapping
    )
    decoded = ctc_postprocessor(logits=tf.random.uniform(shape=[8, 30, 116], minval=0, maxval=1, dtype=tf.float32))
    assert isinstance(decoded, list)
    assert len(decoded) == 8
    assert all(len(word) <= 30 for word in decoded)


def test_detectionpredictor(mock_pdf):  # noqa: F811

    batch_size = 4
    predictor = models.DetectionPredictor(
        models.PreProcessor(output_size=(640, 640), batch_size=batch_size),
        models.DBResNet50(input_size=(640, 640)),
        models.DBPostProcessor()
    )

    pages = read_pdf(mock_pdf)
    out = predictor(pages)

    # The input PDF has 8 pages
    assert len(out) == 8


def test_recognitionpredictor(mock_pdf, mock_mapping):  # noqa: F811

    batch_size = 4
    predictor = models.RecognitionPredictor(
        models.PreProcessor(output_size=(32, 128), batch_size=batch_size),
        models.CRNN(num_classes=len(mock_mapping), input_size=(32, 128, 3)),
        models.CTCPostProcessor(num_classes=len(mock_mapping), label_to_idx=mock_mapping)
    )

    pages = read_pdf(mock_pdf)
    # Create bounding boxes
    boxes = np.array([[0, 0, 0.25, 0.25], [0.5, 0.5, 1., 1.]], dtype=np.float32)
    crops = models.extract_crops(pages[0], boxes)

    out = predictor(crops)

    # One prediction per crop
    assert len(out) == boxes.shape[0]
    assert all(isinstance(charseq, str) for charseq in out)


def test_ocrpredictor(mock_pdf, mock_mapping):  # noqa: F811

    num_docs = 3
    batch_size = 4
    predictor = models.OCRPredictor(
        models.DetectionPredictor(
            models.PreProcessor(output_size=(640, 640), batch_size=batch_size),
            models.DBResNet50(input_size=(640, 640), channels=128),
            models.DBPostProcessor()
        ),
        models.RecognitionPredictor(
            models.PreProcessor(output_size=(32, 128), batch_size=batch_size),
            models.CRNN(num_classes=len(mock_mapping), input_size=(32, 128, 3)),
            models.CTCPostProcessor(num_classes=len(mock_mapping), label_to_idx=mock_mapping)
        )
    )

    docs = [read_pdf(mock_pdf) for _ in range(num_docs)]
    out = predictor(docs)

    assert len(out) == num_docs
    # The input PDF has 8 pages
    assert all(len(doc) == 8 for doc in out)
    # Structure of page
    assert all(isinstance(page, list) for doc in out for page in doc)
    assert all(isinstance(elt, dict) for doc in out for page in doc for elt in page)
