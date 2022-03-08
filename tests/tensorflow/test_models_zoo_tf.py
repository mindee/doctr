import numpy as np
import pytest

from doctr import models
from doctr.io import Document, DocumentFile
from doctr.models import detection, recognition
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.detection.zoo import detection_predictor
from doctr.models.predictor import OCRPredictor
from doctr.models.preprocessor import PreProcessor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.recognition.zoo import recognition_predictor
from doctr.utils.repr import NestedObject


@pytest.mark.parametrize(
    "assume_straight_pages, straighten_pages",
    [
        [True, False],
        [False, False],
        [True, True],
    ]
)
def test_ocrpredictor(mock_pdf, mock_vocab, assume_straight_pages, straighten_pages):
    det_bsize = 4
    det_predictor = DetectionPredictor(
        PreProcessor(output_size=(512, 512), batch_size=det_bsize),
        detection.db_mobilenet_v3_large(
            pretrained=True,
            pretrained_backbone=False,
            input_shape=(512, 512, 3),
            assume_straight_pages=assume_straight_pages,
        )
    )

    reco_bsize = 16
    reco_predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=reco_bsize, preserve_aspect_ratio=True),
        recognition.crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=mock_vocab)
    )

    doc = DocumentFile.from_pdf(mock_pdf)

    predictor = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=assume_straight_pages,
        straighten_pages=straighten_pages,
    )

    if assume_straight_pages:
        assert predictor.crop_orientation_predictor is None
    else:
        assert isinstance(predictor.crop_orientation_predictor, NestedObject)

    out = predictor(doc)
    assert isinstance(out, Document)
    assert len(out.pages) == 2
    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])


def test_trained_ocr_predictor(mock_tilted_payslip):
    doc = DocumentFile.from_images(mock_tilted_payslip)

    det_predictor = detection_predictor('db_resnet50', pretrained=True, batch_size=2, assume_straight_pages=True)
    reco_predictor = recognition_predictor('crnn_vgg16_bn', pretrained=True, batch_size=128)

    predictor = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=True,
        straighten_pages=True,
    )

    out = predictor(doc)

    assert out.pages[0].blocks[0].lines[0].words[0].value == 'Mr.'
    geometry_mr = np.array([[0.08844472, 0.35763523],
                            [0.11625107, 0.34320644],
                            [0.12588427, 0.35771032],
                            [0.09807791, 0.37213911]])
    assert np.allclose(np.array(out.pages[0].blocks[0].lines[0].words[0].geometry), geometry_mr)

    assert out.pages[0].blocks[1].lines[0].words[-1].value == 'revised'
    geometry_revised = np.array([[0.50422498, 0.19551784],
                                 [0.55741975, 0.16791493],
                                 [0.56705294, 0.18241881],
                                 [0.51385817, 0.21002172]])
    assert np.allclose(np.array(out.pages[0].blocks[1].lines[0].words[-1].geometry), geometry_revised)

    det_predictor = detection_predictor(
        'db_resnet50', pretrained=True, batch_size=2, assume_straight_pages=True,
        preserve_aspect_ratio=True, symmetric_pad=True
    )

    predictor = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=True,
        straighten_pages=True,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
    )

    out = predictor(doc)

    assert out.pages[0].blocks[0].lines[0].words[0].value == 'Mr.'


@pytest.mark.parametrize(
    "det_arch, reco_arch",
    [
        ["db_mobilenet_v3_large", "crnn_vgg16_bn"],
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

    # The input doc has 1 page
    assert len(out.pages) == 1
    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])
