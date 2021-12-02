import numpy as np
import pytest
from test_models_detection_pt import test_detectionpredictor_pt  # noqa: F401
from test_models_recognition_pt import test_recognitionpredictor_pt # noqa: F401

from doctr import models
from doctr.io import Document, DocumentFile
from doctr.models.predictor.pytorch import OCRPredictor


def test_ocrpredictor(
    mock_pdf, test_detectionpredictor_pt, test_recognitionpredictor_pt  # noqa: F811
):

    predictor = OCRPredictor(
        test_detectionpredictor_pt,
        test_recognitionpredictor_pt,
        assume_straight_pages=True,
        straighten_pages=False,
    )

    s_predictor = OCRPredictor(
        test_detectionpredictor_pt,
        test_recognitionpredictor_pt,
        assume_straight_pages=True,
        straighten_pages=True,
    )

    doc = DocumentFile.from_pdf(mock_pdf).as_images()
    out = predictor(doc)
    s_out = s_predictor(doc)

    # Document
    assert isinstance(out, Document)
    assert isinstance(s_out, Document)

    # The input PDF has 8 pages
    assert len(out.pages) == 8
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
