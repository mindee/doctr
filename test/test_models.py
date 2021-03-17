import pytest
import numpy as np
import tensorflow as tf

from doctr import models
from doctr.documents import read_pdf, Document
from test_models_detection import test_detectionpredictor
from test_models_recognition import test_recognitionpredictor


def test_extract_crops(mock_pdf):  # noqa: F811
    doc_img = read_pdf(mock_pdf)[0]
    num_crops = 2
    rel_boxes = np.array([[idx / num_crops, idx / num_crops, (idx + 1) / num_crops, (idx + 1) / num_crops]
                          for idx in range(num_crops)], dtype=np.float32)
    abs_boxes = np.array([[int(idx * doc_img.shape[1] / num_crops),
                           int(idx * doc_img.shape[0]) / num_crops,
                           int((idx + 1) * doc_img.shape[1] / num_crops),
                           int((idx + 1) * doc_img.shape[0] / num_crops)]
                          for idx in range(num_crops)], dtype=np.float32)

    with pytest.raises(AssertionError):
        models.extract_crops(doc_img, np.zeros((1, 5)))

    for boxes in (rel_boxes, abs_boxes):
        croped_imgs = models.extract_crops(doc_img, boxes)
        # Number of crops
        assert len(croped_imgs) == num_crops
        # Data type and shape
        assert all(isinstance(crop, np.ndarray) for crop in croped_imgs)
        assert all(crop.ndim == 3 for crop in croped_imgs)

    # Identity
    assert np.all(doc_img == models.extract_crops(doc_img, np.array([[0, 0, 1, 1]], dtype=np.float32))[0])

    # No box
    assert models.extract_crops(doc_img, np.zeros((0, 4))) == []


def test_documentbuilder():

    with pytest.raises(NotImplementedError):
        models.DocumentBuilder(resolve_blocks=True)

    words_per_page = 10
    num_pages = 2

    # Don't resolve lines
    doc_builder = models.DocumentBuilder()
    boxes = np.random.rand(words_per_page, 5)
    boxes[:2] *= boxes[2:4]

    out = doc_builder([boxes, boxes], ['hello'] * (num_pages * words_per_page), [num_pages], [(100, 200), (100, 200)])
    assert isinstance(out, list) and all(isinstance(doc, Document) for doc in out)
    assert len(out[0].pages) == num_pages
    # 1 Block & 1 line per page
    assert len(out[0].pages[0].blocks) == 1 and len(out[0].pages[0].blocks[0].lines) == 1
    assert len(out[0].pages[0].blocks[0].lines[0].words) == words_per_page

    # Resolve lines
    doc_builder = models.DocumentBuilder(resolve_lines=True)
    out = doc_builder([boxes, boxes], ['hello'] * (num_pages * words_per_page), [num_pages], [(100, 200), (100, 200)])

    # No detection
    boxes = np.zeros((0, 5))
    out = doc_builder([boxes, boxes], [], [num_pages], [(100, 200), (100, 200)])
    assert len(out[0].pages[0].blocks) == 0

    # Repr
    assert repr(doc_builder) == "DocumentBuilder(resolve_lines=True, paragraph_break=0.15)"


@pytest.mark.parametrize(
    "input_boxes, sorted_idxs",
    [
        [[[0, 0.5, 0.1, 0.6], [0, 0.3, 0.2, 0.4], [0, 0, 0.1, 0.1]], [2, 1, 0]],  # vertical
        [[[0.7, 0.5, 0.85, 0.6], [0.2, 0.3, 0.4, 0.4], [0, 0, 0.1, 0.1]], [2, 1, 0]],  # diagonal
        [[[0, 0.5, 0.1, 0.6], [0.15, 0.5, 0.25, 0.6], [0.5, 0.5, 0.6, 0.6]], [0, 1, 2]],  # same line, 2p
        [[[0, 0.5, 0.1, 0.6], [0.2, 0.48, 0.35, 0.58], [0.8, 0.52, 0.9, 0.63]], [0, 1, 2]],  # ~same line
        [[[0, 0.3, 0.4, 0.45], [0.5, 0.28, 0.75, 0.42], [0, 0.45, 0.1, 0.55]], [0, 1, 2]],  # 2 lines
        [[[0, 0.3, 0.4, 0.35], [0.75, 0.28, 0.95, 0.42], [0, 0.45, 0.1, 0.55]], [0, 1, 2]],  # 2 lines
    ],
)
def test_sort_boxes(input_boxes, sorted_idxs):

    doc_builder = models.DocumentBuilder()
    assert doc_builder._sort_boxes(np.asarray(input_boxes)).tolist() == sorted_idxs


@pytest.mark.parametrize(
    "input_boxes, sorted_idxs, lines",
    [
        [[[0, 0.5, 0.1, 0.6], [0, 0.3, 0.2, 0.4], [0, 0, 0.1, 0.1]], [2, 1, 0], [[2], [1], [0]]],  # vertical
        [[[0.7, 0.5, 0.85, 0.6], [0.2, 0.3, 0.4, 0.4], [0, 0, 0.1, 0.1]], [2, 1, 0], [[2], [1], [0]]],  # diagonal
        [[[0, 0.5, 0.1, 0.6], [0.15, 0.5, 0.25, 0.6], [0.5, 0.5, 0.6, 0.6]], [0, 1, 2], [[0, 1], [2]]],  # same line, 2p
        [[[0, 0.5, 0.1, 0.6], [0.2, 0.48, 0.35, 0.58], [0.8, 0.52, 0.9, 0.63]], [0, 1, 2], [[0, 1], [2]]],  # ~same line
        [[[0, 0.3, 0.4, 0.45], [0.5, 0.28, 0.75, 0.42], [0, 0.45, 0.1, 0.55]], [0, 1, 2], [[0, 1], [2]]],  # 2 lines
        [[[0, 0.3, 0.4, 0.35], [0.75, 0.28, 0.95, 0.42], [0, 0.45, 0.1, 0.55]], [0, 1, 2], [[0], [1], [2]]],  # 2 lines
    ],
)
def test_resolve_lines(input_boxes, sorted_idxs, lines):

    doc_builder = models.DocumentBuilder()
    assert doc_builder._resolve_lines(np.asarray(input_boxes), np.asarray(sorted_idxs)) == lines


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
    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([[input_page]])


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
        "ocr_db_sar_vgg",
        "ocr_db_crnn_vgg",
        "ocr_db_sar_resnet",
    ],
)
def test_zoo_models(arch_name):
    # Model
    predictor = models.__dict__[arch_name](pretrained=True)
    # Output checks
    assert isinstance(predictor, models.OCRPredictor)
