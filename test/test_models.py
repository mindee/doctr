import pytest
import os
import numpy as np
import sys
import math
import warnings
import tensorflow as tf

# Ensure runnings tests on GPU doesn't run out of memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras import layers, Sequential

from doctr.documents import read_pdf, Document
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
def mock_vocab():
    return ('3K}7eé;5àÎYho]QwV6qU~W"XnbBvcADfËmy.9ÔpÛ*{CôïE%M4#ÈR:g@T$x?0î£|za1ù8,OG€P-kçHëÀÂ2É/ûIJ\'j'
            '(LNÙFut[)èZs+&°Sd=Ï!<â_Ç>rêi`l')


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
    if tf.__version__ < "2.4.0":
        assert sys.getsizeof(test_convert_to_fp16) >= sys.getsizeof(test_quantize_model)
    else:
        assert sys.getsizeof(test_convert_to_fp16) > sys.getsizeof(test_quantize_model)


def test_detpreprocessor(mock_pdf):  # noqa: F811
    num_docs = 3
    batch_size = 4
    docs = [read_pdf(mock_pdf) for _ in range(num_docs)]
    processor = models.DetectionPreProcessor(output_size=(512, 512), batch_size=batch_size)
    batched_docs = processor([page for doc in docs for page in doc])

    # Number of batches
    assert len(batched_docs) == math.ceil(8 * num_docs / batch_size)
    # Total number of samples
    assert sum(batch.shape[0] for batch in batched_docs) == 8 * num_docs
    # Batch size
    assert all(batch.shape[0] == batch_size for batch in batched_docs[:-1])
    assert batched_docs[-1].shape[0] == batch_size if (8 * num_docs) % batch_size == 0 else (8 * num_docs) % batch_size
    # Data type
    assert all(batch.dtype == tf.float32 for batch in batched_docs)
    # Image size
    assert all(batch.shape[1:] == (512, 512, 3) for batch in batched_docs)
    # Test with non-full last batch
    batch_size = 16
    processor = models.DetectionPreProcessor(output_size=(512, 512), batch_size=batch_size)
    batched_docs = processor([page for doc in docs for page in doc])
    assert batched_docs[-1].shape[0] == (8 * num_docs) % batch_size


def test_recopreprocessor(mock_pdf):  # noqa: F811
    num_docs = 3
    batch_size = 4
    docs = [read_pdf(mock_pdf) for _ in range(num_docs)]
    processor = models.RecognitionPreProcessor(output_size=(256, 128), batch_size=batch_size)
    batched_docs = processor([page for doc in docs for page in doc])

    # Number of batches
    assert len(batched_docs) == math.ceil(8 * num_docs / batch_size)
    # Total number of samples
    assert sum(batch.shape[0] for batch in batched_docs) == 8 * num_docs
    # Batch size
    assert all(batch.shape[0] == batch_size for batch in batched_docs[:-1])
    assert batched_docs[-1].shape[0] == batch_size if (8 * num_docs) % batch_size == 0 else (8 * num_docs) % batch_size
    # Data type
    assert all(batch.dtype == tf.float32 for batch in batched_docs)
    # Image size
    assert all(batch.shape[1:] == (256, 128, 3) for batch in batched_docs)
    # Test with non-full last batch
    batch_size = 16
    processor = models.RecognitionPreProcessor(output_size=(256, 128), batch_size=batch_size)
    batched_docs = processor([page for doc in docs for page in doc])
    assert batched_docs[-1].shape[0] == (8 * num_docs) % batch_size


def test_dbpostprocessor():
    postprocessor = models.DBPostProcessor()
    mock_batch = tf.random.uniform(shape=[2, 512, 512, 1], minval=0, maxval=1)
    out = postprocessor(mock_batch)
    # Batch composition
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(isinstance(sample, np.ndarray) for sample in out)
    assert all(sample.shape[1] == 5 for sample in out)
    # Relative coords
    assert all(np.all(np.logical_and(sample[:4] >= 0, sample[:4] <= 1)) for sample in out)


def test_db_resnet50():
    model = models.db_resnet50(pretrained=True)
    assert isinstance(model, tf.keras.Model)
    dbinput = tf.random.uniform(shape=[2, 1024, 1024, 3], minval=0, maxval=1)
    # test prediction model
    dboutput_notrain = model(dbinput)
    assert isinstance(dboutput_notrain, tf.Tensor)
    assert dboutput_notrain.numpy().shape == (2, 1024, 1024, 1)
    assert np.all(dboutput_notrain.numpy() > 0) and np.all(dboutput_notrain.numpy() < 1)
    # test training model
    dboutput_train = model(dbinput, training=True)
    assert isinstance(dboutput_train, tuple)
    assert len(dboutput_train) == 3
    assert all(np.all(np.logical_and(out_map.numpy() >= 0, out_map.numpy() <= 1)) for out_map in dboutput_train)
    # batch size
    assert all(out.numpy().shape == (2, 1024, 1024, 1) for out in dboutput_train)


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


@pytest.mark.parametrize(
    "arch_name, input_shape, output_size",
    [
        ["crnn_vgg16_bn", (32, 128, 3), (32, 119)],
        ["sar_vgg16_bn", (64, 256, 3), (41, 119)],
    ],
)
def test_recognition_architectures(arch_name, input_shape, output_size):
    batch_size = 8
    reco_model = models.__dict__[arch_name](pretrained=True, input_shape=input_shape)
    input_tensor = tf.random.uniform(shape=[batch_size, *input_shape], minval=0, maxval=1)
    out = reco_model(input_tensor)
    assert isinstance(out, tf.Tensor)
    assert isinstance(reco_model, tf.keras.Model)
    assert out.numpy().shape == (batch_size, *output_size)


@pytest.mark.parametrize(
    "post_processor, input_shape",
    [
        ["SARPostProcessor", [2, 30, 116]],
        ["CTCPostProcessor", [2, 30, 116]],
    ],
)
def test_reco_postprocessors(post_processor, input_shape, mock_vocab):
    processor = models.recognition.__dict__[post_processor](mock_vocab)
    decoded = processor(tf.random.uniform(shape=input_shape, minval=0, maxval=1, dtype=tf.float32))
    assert isinstance(decoded, list)
    assert len(decoded) == input_shape[0]
    assert all(len(word) <= input_shape[1] for word in decoded)


@pytest.fixture(scope="module")
def test_detectionpredictor(mock_pdf):  # noqa: F811

    batch_size = 4
    predictor = models.DetectionPredictor(
        models.DetectionPreProcessor(output_size=(512, 512), batch_size=batch_size),
        models.db_resnet50(input_shape=(512, 512, 3)),
        models.DBPostProcessor()
    )

    pages = read_pdf(mock_pdf)
    out = predictor(pages)

    # The input PDF has 8 pages
    assert len(out) == 8

    return predictor


@pytest.fixture(scope="module")
def test_recognitionpredictor(mock_pdf, mock_vocab):  # noqa: F811

    batch_size = 4
    predictor = models.RecognitionPredictor(
        models.RecognitionPreProcessor(output_size=(32, 128), batch_size=batch_size),
        models.crnn_vgg16_bn(vocab_size=len(mock_vocab), input_shape=(32, 128, 3)),
        models.CTCPostProcessor(mock_vocab)
    )

    pages = read_pdf(mock_pdf)
    # Create bounding boxes
    boxes = np.array([[0, 0, 0.25, 0.25], [0.5, 0.5, 1., 1.]], dtype=np.float32)
    crops = models.extract_crops(pages[0], boxes)

    out = predictor(crops)

    # One prediction per crop
    assert len(out) == boxes.shape[0]
    assert all(isinstance(charseq, str) for charseq in out)

    return predictor


def test_ocrpredictor(mock_pdf, test_detectionpredictor, test_recognitionpredictor):  # noqa: F811

    num_docs = 3
    predictor = models.OCRPredictor(
        test_detectionpredictor,
        test_recognitionpredictor
    )

    docs = [read_pdf(mock_pdf) for _ in range(num_docs)]
    out = predictor(docs)

    assert len(out) == num_docs
    # Document
    assert all(isinstance(doc, Document) for doc in out)
    # The input PDF has 8 pages
    assert all(len(doc.pages) == 8 for doc in out)


@pytest.mark.parametrize(
    "arch_name, top_implemented, input_shape, output_size",
    [
        ["vgg16_bn", False, (224, 224, 3), (7, 56, 512)],
        ["resnet31", False, (32, 128, 3), (4, 32, 512)],
    ],
)
def test_classification_architectures(arch_name, top_implemented, input_shape, output_size):
    # Head not implemented yet
    if not top_implemented:
        with pytest.raises(NotImplementedError):
            models.__dict__[arch_name](include_top=True)

    # Model
    batch_size = 2
    model = models.__dict__[arch_name](pretrained=True)
    # Forward
    out = model(tf.random.uniform(shape=[batch_size, *input_shape], maxval=1, dtype=tf.float32))
    # Output checks
    assert isinstance(out, tf.Tensor)
    assert out.numpy().shape == (batch_size, *output_size)


def test_load_pretrained_params(tmpdir_factory):

    model = Sequential([layers.Dense(8, activation='relu', input_shape=(4,)), layers.Dense(4)])
    # Retrieve this URL
    url = "https://srv-store1.gofile.io/download/0oRu0c/tmp_checkpoint-4a98e492.zip"
    # Temp cache dir
    cache_dir = tmpdir_factory.mktemp("cache")
    # Remove try except once files have been moved to github
    try:
        # Pass an incorrect hash
        with pytest.raises(ValueError):
            models.utils.load_pretrained_params(model, url, "mywronghash", cache_dir=str(cache_dir), internal_name='')
        # Let tit resolve the hash from the file name
        models.utils.load_pretrained_params(model, url, cache_dir=str(cache_dir), internal_name='')
        # Check that the file was downloaded & the archive extracted
        assert os.path.exists(cache_dir.join('models').join("tmp_checkpoint-4a98e492"))
        # Check that archive was deleted
        assert os.path.exists(cache_dir.join('models').join("tmp_checkpoint-4a98e492.zip"))
    except Exception as e:
        warnings.warn(e)


@pytest.mark.parametrize(
    "arch_name",
    [
        "ocr_db_sar",
    ],
)
def test_zoo_models(arch_name):
    # Model
    model = models.__dict__[arch_name](pretrained=True)
    # Output checks
    assert isinstance(model, models.OCRPredictor)
