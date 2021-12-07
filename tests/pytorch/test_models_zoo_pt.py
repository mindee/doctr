import numpy as np
import pytest

from doctr import models
from doctr.io import Document, DocumentFile
from doctr.models import detection, recognition
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.predictor import OCRPredictor
from doctr.models.preprocessor import PreProcessor
from doctr.models.recognition.predictor import RecognitionPredictor


def test_ocrpredictor(mock_pdf, mock_vocab):
    batch_size = 4
    detectionpredictor = DetectionPredictor(
        PreProcessor(output_size=(512, 512), batch_size=batch_size),
        detection.db_resnet50(pretrained=False).eval()
    )

    recognitionpredictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=batch_size, preserve_aspect_ratio=True),
        recognition.crnn_vgg16_bn(vocab=mock_vocab, input_shape=(32, 128, 3))
    )

    predictor = OCRPredictor(
        detectionpredictor,
        recognitionpredictor,
        assume_straight_pages=True,
        straighten_pages=False,
    )

    s_predictor = OCRPredictor(
        detectionpredictor,
        recognitionpredictor,
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
