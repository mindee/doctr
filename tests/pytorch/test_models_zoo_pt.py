import numpy as np
import pytest

from doctr import models
from doctr.io import Document
from doctr.models.predictor import OCRPredictor


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
