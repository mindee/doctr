import pytest
import os
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from doctr.documents import read_pdf, Document
from test_documents import mock_pdf
from doctr import models


@pytest.fixture(scope="module")
def mock_vocab():
    return ('3K}7eé;5àÎYho]QwV6qU~W"XnbBvcADfËmy.9ÔpÛ*{CôïE%M4#ÈR:g@T$x?0î£|za1ù8,OG€P-kçHëÀÂ2É/ûIJ\'j'
            '(LNÙFut[)èZs+&°Sd=Ï!<â_Ç>rêi`l')


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
