import numpy as np
import pytest
from test_models_detection_pt import test_detectionpredictor, test_rotated_detectionpredictor  # noqa: F401
from test_models_recognition_pt import test_recognitionpredictor

from doctr import models
from doctr.io import Document, DocumentFile
from doctr.models.predictor.pytorch import OCRPredictor


def test_ocrpredictor(
    mock_pdf, test_detectionpredictor, test_recognitionpredictor, test_rotated_detectionpredictor  # noqa: F811
):

    predictor = OCRPredictor(
        test_detectionpredictor,
        test_recognitionpredictor,
        assume_straight_pages=True,
        straighten_pages=False,
    )

    r_predictor = OCRPredictor(
        test_rotated_detectionpredictor,
        test_recognitionpredictor,
        assume_straight_pages=False,
        straighten_pages=False,
    )

    s_predictor = OCRPredictor(
        test_detectionpredictor,
        test_recognitionpredictor,
        assume_straight_pages=True,
        straighten_pages=True,
    )

    doc = DocumentFile.from_pdf(mock_pdf).as_images()
    out = predictor(doc)
    r_out = r_predictor(doc)
    s_out = s_predictor(doc)

    # Document
    assert isinstance(out, Document)
    assert isinstance(r_out, Document)
    assert isinstance(s_out, Document)

    # The input PDF has 8 pages
    assert len(out.pages) == 8
    assert len(r_out.pages) == 8
    assert len(s_out.pages) == 8
    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])


@pytest.mark.parametrize(
    "det_arch, reco_arch",
    [
        ["db_mobilenet_v3_large", "crnn_mobilenet_v3_large"],
    ],
)
def test_zoo_models(det_arch, reco_arch):
    # Model
    predictor = models.ocr_predictor(det_arch, reco_arch, pretrained=True)
    # Output checks
    assert isinstance(predictor, OCRPredictor)

    doc = [np.zeros((512, 512, 3), dtype=np.uint8)]
    out = predictor(doc)
    # Document
    assert isinstance(out, Document)

    # The input PDF has 8 pages
    assert len(out.pages) == 1
    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])
