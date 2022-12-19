import numpy as np
import pytest
from torch import nn

from doctr import models
from doctr.io import Document, DocumentFile
from doctr.io.elements import KIEDocument
from doctr.models import detection, recognition
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.kie_predictor import KIEPredictor
from doctr.models.predictor import OCRPredictor
from doctr.models.preprocessor import PreProcessor
from doctr.models.recognition.predictor import RecognitionPredictor


@pytest.mark.parametrize(
    "assume_straight_pages, straighten_pages",
    [
        [True, False],
        [False, False],
        [True, True],
    ],
)
def test_ocrpredictor(mock_pdf, mock_vocab, assume_straight_pages, straighten_pages):
    det_bsize = 4
    det_predictor = DetectionPredictor(
        PreProcessor(output_size=(512, 512), batch_size=det_bsize),
        detection.db_mobilenet_v3_large(
            pretrained=False,
            pretrained_backbone=False,
            assume_straight_pages=assume_straight_pages,
        ),
    )

    assert not det_predictor.model.training

    reco_bsize = 32
    reco_predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=reco_bsize, preserve_aspect_ratio=True),
        recognition.crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=mock_vocab),
    )

    assert not reco_predictor.model.training

    doc = DocumentFile.from_pdf(mock_pdf)

    predictor = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=assume_straight_pages,
        straighten_pages=straighten_pages,
        detect_orientation=True,
        detect_language=True,
    )

    if assume_straight_pages:
        assert predictor.crop_orientation_predictor is None
    else:
        assert isinstance(predictor.crop_orientation_predictor, nn.Module)

    out = predictor(doc)
    assert isinstance(out, Document)
    assert len(out.pages) == 2
    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])

    orientation = 0
    assert out.pages[0].orientation["value"] == orientation


@pytest.mark.parametrize(
    "assume_straight_pages, straighten_pages",
    [
        [True, False],
        [False, False],
        [True, True],
    ],
)
def test_kiepredictor(mock_pdf, mock_vocab, assume_straight_pages, straighten_pages):
    det_bsize = 4
    det_predictor = DetectionPredictor(
        PreProcessor(output_size=(512, 512), batch_size=det_bsize),
        detection.db_mobilenet_v3_large(
            pretrained=False,
            pretrained_backbone=False,
            assume_straight_pages=assume_straight_pages,
        ),
    )

    assert not det_predictor.model.training

    reco_bsize = 32
    reco_predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=reco_bsize, preserve_aspect_ratio=True),
        recognition.crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=mock_vocab),
    )

    assert not reco_predictor.model.training

    doc = DocumentFile.from_pdf(mock_pdf)

    predictor = KIEPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=assume_straight_pages,
        straighten_pages=straighten_pages,
        detect_orientation=True,
        detect_language=True,
    )

    if assume_straight_pages:
        assert predictor.crop_orientation_predictor is None
    else:
        assert isinstance(predictor.crop_orientation_predictor, nn.Module)

    out = predictor(doc)
    assert isinstance(out, Document)
    assert len(out.pages) == 2
    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])

    orientation = 0
    assert out.pages[0].orientation["value"] == orientation


def _test_predictor(predictor):
    # Output checks
    assert isinstance(predictor, OCRPredictor)

    doc = [np.zeros((512, 512, 3), dtype=np.uint8)]
    out = predictor(doc)
    # Document
    assert isinstance(out, Document)

    # The input doc has 1 page
    assert len(out.pages) == 1
    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])


def _test_kiepredictor(predictor):
    # Output checks
    assert isinstance(predictor, KIEPredictor)

    doc = [np.zeros((512, 512, 3), dtype=np.uint8)]
    out = predictor(doc)
    # Document
    assert isinstance(out, KIEDocument)

    # The input doc has 1 page
    assert len(out.pages) == 1
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
    _test_predictor(predictor)

    # passing model instance directly
    det_model = detection.__dict__[det_arch](pretrained=True)
    reco_model = recognition.__dict__[reco_arch](pretrained=True)
    predictor = models.ocr_predictor(det_model, reco_model)
    _test_predictor(predictor)

    # passing recognition model as detection model
    with pytest.raises(ValueError):
        models.ocr_predictor(det_arch=reco_model, pretrained=True)

    # passing detection model as recognition model
    with pytest.raises(ValueError):
        models.ocr_predictor(reco_arch=det_model, pretrained=True)

    # KIE predictor
    predictor = models.kie_predictor(det_arch, reco_arch, pretrained=True)
    _test_kiepredictor(predictor)

    # passing model instance directly
    det_model = detection.__dict__[det_arch](pretrained=True)
    reco_model = recognition.__dict__[reco_arch](pretrained=True)
    predictor = models.kie_predictor(det_model, reco_model)
    _test_kiepredictor(predictor)

    # passing recognition model as detection model
    with pytest.raises(ValueError):
        models.kie_predictor(det_arch=reco_model, pretrained=True)

    # passing detection model as recognition model
    with pytest.raises(ValueError):
        models.kie_predictor(reco_arch=det_model, pretrained=True)
