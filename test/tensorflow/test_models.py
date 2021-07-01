import pytest
import math
import numpy as np
import tensorflow as tf

from doctr import models
from doctr.documents import Document, DocumentFile
from test_models_detection import test_detectionpredictor, test_rotated_detectionpredictor
from test_models_recognition import test_recognitionpredictor


def test_ocrpredictor(
    mock_pdf, test_detectionpredictor, test_recognitionpredictor, test_rotated_detectionpredictor  # noqa: F811
):

    predictor = models.OCRPredictor(
        test_detectionpredictor,
        test_recognitionpredictor
    )

    r_predictor = models.OCRPredictor(
        test_rotated_detectionpredictor,
        test_recognitionpredictor,
        rotated_bbox=True
    )

    doc = DocumentFile.from_pdf(mock_pdf).as_images()
    out = predictor(doc)
    r_out = r_predictor(doc)

    # Document
    assert isinstance(out, Document)
    assert isinstance(r_out, Document)

    # The input PDF has 8 pages
    assert len(out.pages) == 8
    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])


@pytest.mark.parametrize(
    "det_arch, reco_arch",
    [
        ["db_resnet50", "sar_vgg16_bn"],
        ["db_resnet50", "crnn_vgg16_bn"],
        ["db_resnet50", "sar_resnet31"],
        ["db_resnet50", "crnn_resnet31"],
    ],
)
def test_zoo_models(det_arch, reco_arch):
    # Model
    predictor = models.ocr_predictor(det_arch, reco_arch, pretrained=True)
    # Output checks
    assert isinstance(predictor, models.OCRPredictor)


def test_preprocessor(mock_pdf):
    preprocessor = models.PreProcessor(output_size=(1024, 1024), batch_size=2)
    input_tensor = tf.random.uniform(shape=[2, 512, 512, 3], minval=0, maxval=1)
    preprocessed = preprocessor(input_tensor)
    assert isinstance(preprocessed, list)
    for batch in preprocessed:
        assert batch.shape[0] == 2
        assert batch.shape[1] == batch.shape[2] == 1024

    with pytest.raises(AssertionError):
        _ = preprocessor(np.random.rand(1024, 1024, 3))

    num_docs = 3
    batch_size = 4
    docs = [DocumentFile.from_pdf(mock_pdf).as_images() for _ in range(num_docs)]
    processor = models.PreProcessor(output_size=(512, 512), batch_size=batch_size)
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
    processor = models.PreProcessor(output_size=(512, 512), batch_size=batch_size)
    batched_docs = processor([page for doc in docs for page in doc])
    assert batched_docs[-1].shape[0] == (8 * num_docs) % batch_size

    # Repr
    assert len(repr(processor).split('\n')) == 9

    # Assymetric
    processor = models.PreProcessor(output_size=(256, 128), batch_size=batch_size, preserve_aspect_ratio=True)
    batched_docs = processor([page for doc in docs for page in doc])
    # Image size
    assert all(batch.shape[1:] == (256, 128, 3) for batch in batched_docs)
