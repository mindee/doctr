import pytest
import numpy as np
import tensorflow as tf

from doctr import models
from doctr.documents import Document, DocumentFile
from test_models_detection import test_detectionpredictor
from test_models_recognition import test_recognitionpredictor


def test_extract_crops(mock_pdf):  # noqa: F811
    doc_img = DocumentFile.from_pdf(mock_pdf).as_images()[0]
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

    words_per_page = 10
    num_pages = 2

    # Don't resolve lines
    doc_builder = models.DocumentBuilder()
    boxes = np.random.rand(words_per_page, 5)
    boxes[:2] *= boxes[2:4]

    out = doc_builder([boxes, boxes], ['hello'] * (num_pages * words_per_page), [(100, 200), (100, 200)])
    assert isinstance(out, Document)
    assert len(out.pages) == num_pages
    # 1 Block & 1 line per page
    assert len(out.pages[0].blocks) == 1 and len(out.pages[0].blocks[0].lines) == 1
    assert len(out.pages[0].blocks[0].lines[0].words) == words_per_page

    # Resolve lines
    doc_builder = models.DocumentBuilder(resolve_lines=True, resolve_blocks=True)
    out = doc_builder([boxes, boxes], ['hello'] * (num_pages * words_per_page), [(100, 200), (100, 200)])

    # No detection
    boxes = np.zeros((0, 5))
    out = doc_builder([boxes, boxes], [], [(100, 200), (100, 200)])
    assert len(out.pages[0].blocks) == 0

    # Repr
    assert repr(doc_builder) == "DocumentBuilder(resolve_lines=True, resolve_blocks=True, paragraph_break=0.035)"


@pytest.mark.parametrize(
    "input_boxes, sorted_idxs",
    [
        [[[0, 0.5, 0.1, 0.6], [0, 0.3, 0.2, 0.4], [0, 0, 0.1, 0.1]], [2, 1, 0]],  # vertical
        [[[0.7, 0.5, 0.85, 0.6], [0.2, 0.3, 0.4, 0.4], [0, 0, 0.1, 0.1]], [2, 1, 0]],  # diagonal
        [[[0, 0.5, 0.1, 0.6], [0.15, 0.5, 0.25, 0.6], [0.5, 0.5, 0.6, 0.6]], [0, 1, 2]],  # same line, 2p
        [[[0, 0.5, 0.1, 0.6], [0.2, 0.49, 0.35, 0.59], [0.8, 0.52, 0.9, 0.63]], [0, 1, 2]],  # ~same line
        [[[0, 0.3, 0.4, 0.45], [0.5, 0.28, 0.75, 0.42], [0, 0.45, 0.1, 0.55]], [0, 1, 2]],  # 2 lines
        [[[0, 0.3, 0.4, 0.35], [0.75, 0.28, 0.95, 0.42], [0, 0.45, 0.1, 0.55]], [0, 1, 2]],  # 2 lines
    ],
)
def test_sort_boxes(input_boxes, sorted_idxs):

    doc_builder = models.DocumentBuilder()
    assert doc_builder._sort_boxes(np.asarray(input_boxes)).tolist() == sorted_idxs


@pytest.mark.parametrize(
    "input_boxes, lines",
    [
        [[[0, 0.5, 0.1, 0.6], [0, 0.3, 0.2, 0.4], [0, 0, 0.1, 0.1]], [[2], [1], [0]]],  # vertical
        [[[0.7, 0.5, 0.85, 0.6], [0.2, 0.3, 0.4, 0.4], [0, 0, 0.1, 0.1]], [[2], [1], [0]]],  # diagonal
        [[[0, 0.5, 0.14, 0.6], [0.15, 0.5, 0.25, 0.6], [0.5, 0.5, 0.6, 0.6]], [[0, 1], [2]]],  # same line, 2p
        [[[0, 0.5, 0.18, 0.6], [0.2, 0.48, 0.35, 0.58], [0.8, 0.52, 0.9, 0.63]], [[0, 1], [2]]],  # ~same line
        [[[0, 0.3, 0.48, 0.45], [0.5, 0.28, 0.75, 0.42], [0, 0.45, 0.1, 0.55]], [[0, 1], [2]]],  # 2 lines
        [[[0, 0.3, 0.4, 0.35], [0.75, 0.28, 0.95, 0.42], [0, 0.45, 0.1, 0.55]], [[0], [1], [2]]],  # 2 lines
    ],
)
def test_resolve_lines(input_boxes, lines):

    doc_builder = models.DocumentBuilder()
    assert doc_builder._resolve_lines(np.asarray(input_boxes)) == lines


def test_ocrpredictor(mock_pdf, test_detectionpredictor, test_recognitionpredictor):  # noqa: F811

    predictor = models.OCRPredictor(
        test_detectionpredictor,
        test_recognitionpredictor
    )

    doc = DocumentFile.from_pdf(mock_pdf).as_images()
    out = predictor(doc)

    # Document
    assert isinstance(out, Document)
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


def test_preprocessor():
    preprocessor = models.PreProcessor(output_size=(1024, 1024), batch_size=2)
    input_tensor = tf.random.uniform(shape=[2, 512, 512, 3], minval=0, maxval=1)
    preprocessed = preprocessor(input_tensor)
    assert isinstance(preprocessed, list)
    for batch in preprocessed:
        assert batch.shape[0] == 2
        assert batch.shape[1] == batch.shape[2] == 1024
