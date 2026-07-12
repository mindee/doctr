import cv2
import numpy as np
import pytest
import torch
from torch import nn

from doctr import models
from doctr.file_utils import CLASS_NAME
from doctr.io import Document, DocumentFile
from doctr.io.elements import KIEDocument, LayoutElement, Table
from doctr.models import detection, layout, recognition
from doctr.models.classification import mobilenet_v3_small_crop_orientation, mobilenet_v3_small_page_orientation
from doctr.models.classification.zoo import crop_orientation_predictor, page_orientation_predictor
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.detection.zoo import detection_predictor
from doctr.models.kie_predictor import KIEPredictor
from doctr.models.layout.predictor import LayoutPredictor
from doctr.models.layout.zoo import layout_predictor
from doctr.models.predictor import OCRPredictor
from doctr.models.preprocessor import PreProcessor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.recognition.zoo import recognition_predictor
from doctr.models.table_structure.predictor import TablePredictor
from doctr.models.table_structure.zoo import table_predictor


# Create a dummy callback
class _DummyCallback:
    def __call__(self, loc_preds):
        return loc_preds


@pytest.mark.parametrize(
    "assume_straight_pages, straighten_pages, disable_page_orientation, disable_crop_orientation",
    [
        [True, False, False, False],
        [False, False, True, True],
        [True, True, False, False],
        [False, True, True, True],
        [True, False, True, False],
    ],
)
def test_ocrpredictor(
    mock_pdf, mock_vocab, assume_straight_pages, straighten_pages, disable_page_orientation, disable_crop_orientation
):
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
        resolve_blocks=True,
        resolve_lines=True,
        disable_page_orientation=disable_page_orientation,
        disable_crop_orientation=disable_crop_orientation,
    )

    assert (
        predictor._page_orientation_disabled if disable_page_orientation else not predictor._page_orientation_disabled
    )
    assert (
        predictor._crop_orientation_disabled if disable_crop_orientation else not predictor._crop_orientation_disabled
    )

    if assume_straight_pages:
        assert predictor.crop_orientation_predictor is None
        if predictor.detect_orientation or predictor.straighten_pages:
            assert isinstance(predictor.page_orientation_predictor, nn.Module)
        else:
            assert predictor.page_orientation_predictor is None
    else:
        assert isinstance(predictor.crop_orientation_predictor, nn.Module)
        assert isinstance(predictor.page_orientation_predictor, nn.Module)

    out = predictor(doc)
    assert isinstance(out, Document)
    assert len(out.pages) == 2
    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])

    orientation = 0
    assert out.pages[0].orientation["value"] == orientation

    # Test with custom orientation models
    custom_crop_orientation_model = mobilenet_v3_small_crop_orientation(pretrained=True)
    custom_page_orientation_model = mobilenet_v3_small_page_orientation(pretrained=True)

    if assume_straight_pages:
        if predictor.detect_orientation or predictor.straighten_pages:
            # Overwrite the default orientation models
            predictor.crop_orientation_predictor = crop_orientation_predictor(custom_crop_orientation_model)
            predictor.page_orientation_predictor = page_orientation_predictor(custom_page_orientation_model)
    else:
        # Overwrite the default orientation models
        predictor.crop_orientation_predictor = crop_orientation_predictor(custom_crop_orientation_model)
        predictor.page_orientation_predictor = page_orientation_predictor(custom_page_orientation_model)

    out = predictor(doc)
    orientation = 0
    assert out.pages[0].orientation["value"] == orientation


def test_ocrpredictor_layout(mock_pdf, mock_vocab, mock_payslip):
    det_predictor = DetectionPredictor(
        PreProcessor(output_size=(512, 512), batch_size=2),
        detection.db_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, assume_straight_pages=True),
    )
    reco_predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=32, preserve_aspect_ratio=True),
        recognition.crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=mock_vocab),
    )
    layout_pred = layout_predictor("lw_detr_s", pretrained=False)

    doc = DocumentFile.from_pdf(mock_pdf)

    # Without a layout predictor -> pages carry an empty layout
    predictor = OCRPredictor(det_predictor, reco_predictor, ignore_regions=["Picture", "Formula"])
    assert predictor.layout_predictor is None
    out = predictor(doc)
    assert all(page.layout == [] for page in out.pages)
    assert all(page.export()["layout"] == [] for page in out.pages)

    # With a layout predictor -> detected regions are attached to every page
    predictor = OCRPredictor(
        det_predictor, reco_predictor, layout_predictor=layout_pred, ignore_regions=["Picture", "Formula"]
    )
    assert isinstance(predictor.layout_predictor, LayoutPredictor)
    out = predictor(doc)
    assert isinstance(out, Document)
    for page in out.pages:
        assert isinstance(page.layout, list)
        assert all(isinstance(region, LayoutElement) for region in page.layout)
        # the layout is exported alongside the page
        exported = page.export()
        assert "layout" in exported
        assert exported["layout"] == [region.export() for region in page.layout]

    doc = DocumentFile.from_images(mock_payslip)

    det_predictor = detection_predictor(
        "fast_base",
        pretrained=True,
        batch_size=2,
        assume_straight_pages=True,
        symmetric_pad=True,
        preserve_aspect_ratio=False,
    )
    reco_predictor = recognition_predictor("crnn_vgg16_bn", pretrained=True, batch_size=128)

    predictor = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=True,
        straighten_pages=True,
        preserve_aspect_ratio=False,
        resolve_blocks=True,
        resolve_lines=True,
    )

    out = predictor(doc)

    assert out.pages[0].blocks[0].lines[0].words[0].value == "Mr."
    geometry_mr = np.array([[0.1083984375, 0.0634765625], [0.1494140625, 0.0859375]])
    assert np.allclose(np.array(out.pages[0].blocks[0].lines[0].words[0].geometry), geometry_mr, rtol=0.05)

    assert out.pages[0].blocks[1].lines[0].words[-1].value == "revised"
    geometry_revised = np.array([[0.7548828125, 0.126953125], [0.8388671875, 0.1484375]])
    assert np.allclose(np.array(out.pages[0].blocks[1].lines[0].words[-1].geometry), geometry_revised, rtol=0.05)

    det_predictor = detection_predictor(
        "fast_base",
        pretrained=True,
        batch_size=2,
        assume_straight_pages=True,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
    )

    predictor = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=True,
        straighten_pages=True,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
        resolve_blocks=True,
        resolve_lines=True,
        ignore_regions=["Picture", "Formula"],
    )
    # test hooks
    predictor.add_hook(_DummyCallback())

    out = predictor(doc)

    assert out.pages[0].blocks[0].lines[0].words[0].value == "Mr."


def test_ocrpredictor_tables(mock_pdf, mock_vocab):
    det_predictor = DetectionPredictor(
        PreProcessor(output_size=(512, 512), batch_size=2),
        detection.db_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, assume_straight_pages=True),
    )
    reco_predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=32, preserve_aspect_ratio=True),
        recognition.crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=mock_vocab),
    )
    layout_pred = layout_predictor("lw_detr_s", pretrained=False)
    table_pred = table_predictor("tablecenternet", pretrained=False)

    # A table predictor requires a layout predictor (tables are located with the layout model)
    with pytest.raises(ValueError):
        OCRPredictor(det_predictor, reco_predictor, table_predictor=table_pred)

    doc = DocumentFile.from_pdf(mock_pdf)

    # Without a table predictor -> pages carry an empty list of tables
    predictor = OCRPredictor(det_predictor, reco_predictor)
    assert predictor.table_predictor is None
    out = predictor(doc)
    assert all(page.tables == [] for page in out.pages)
    assert all(page.export()["tables"] == [] for page in out.pages)

    # With layout + table predictors -> structured tables are attached and exported
    predictor = OCRPredictor(det_predictor, reco_predictor, layout_predictor=layout_pred, table_predictor=table_pred)
    assert isinstance(predictor.layout_predictor, LayoutPredictor)
    assert isinstance(predictor.table_predictor, TablePredictor)
    out = predictor(doc)
    assert isinstance(out, Document)
    for page in out.pages:
        assert isinstance(page.tables, list)
        assert all(isinstance(t, Table) for t in page.tables)
        exported = page.export()
        assert "tables" in exported
        assert exported["tables"] == [t.export() for t in page.tables]


def test_ocrpredictor_tables_factory():
    # The factory exposes a single `detect_tables` flag, which also enables the layout model
    predictor = models.ocr_predictor("db_mobilenet_v3_large", "crnn_vgg16_bn", pretrained=False, detect_tables=True)
    assert isinstance(predictor.table_predictor, TablePredictor)
    assert isinstance(predictor.layout_predictor, LayoutPredictor)

    # No tables by default
    predictor = models.ocr_predictor("db_mobilenet_v3_large", "crnn_vgg16_bn", pretrained=False)
    assert predictor.table_predictor is None


def test_trained_ocr_predictor(mock_pdf, mock_vocab, mock_payslip):
    det_predictor = DetectionPredictor(
        PreProcessor(output_size=(512, 512), batch_size=2),
        detection.db_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, assume_straight_pages=True),
    )
    reco_predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=32, preserve_aspect_ratio=True),
        recognition.crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=mock_vocab),
    )
    layout_pred = layout_predictor("lw_detr_s", pretrained=True)

    doc = DocumentFile.from_pdf(mock_pdf)

    # Without a layout predictor -> pages carry an empty layout
    predictor = OCRPredictor(det_predictor, reco_predictor)
    assert predictor.layout_predictor is None
    out = predictor(doc)
    assert all(page.layout == [] for page in out.pages)
    assert all(page.export()["layout"] == [] for page in out.pages)

    # With a layout predictor -> detected regions are attached to every page
    predictor = OCRPredictor(det_predictor, reco_predictor, layout_predictor=layout_pred)
    assert isinstance(predictor.layout_predictor, LayoutPredictor)
    out = predictor(doc)
    assert isinstance(out, Document)
    for page in out.pages:
        assert isinstance(page.layout, list)
        assert all(isinstance(region, LayoutElement) for region in page.layout)
        # the layout is exported alongside the page
        exported = page.export()
        assert "layout" in exported
        assert exported["layout"] == [region.export() for region in page.layout]

    # Test KIE
    predictor = KIEPredictor(det_predictor, reco_predictor, layout_predictor=layout_pred)
    assert isinstance(predictor.layout_predictor, LayoutPredictor)
    out = predictor(doc)
    assert isinstance(out, KIEDocument)
    for page in out.pages:
        assert isinstance(page.layout, list)
        assert all(isinstance(region, LayoutElement) for region in page.layout)
        assert page.export()["layout"] == [region.export() for region in page.layout]

    doc = DocumentFile.from_images(mock_payslip)

    det_predictor = detection_predictor(
        "fast_base",
        pretrained=True,
        batch_size=2,
        assume_straight_pages=True,
        symmetric_pad=True,
        preserve_aspect_ratio=False,
    )
    reco_predictor = recognition_predictor("crnn_vgg16_bn", pretrained=True, batch_size=128)

    predictor = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=True,
        straighten_pages=True,
        preserve_aspect_ratio=False,
        resolve_blocks=True,
        resolve_lines=True,
    )

    out = predictor(doc)

    assert out.pages[0].blocks[0].lines[0].words[0].value == "Mr."
    geometry_mr = np.array([[0.1083984375, 0.0634765625], [0.1494140625, 0.0859375]])
    assert np.allclose(np.array(out.pages[0].blocks[0].lines[0].words[0].geometry), geometry_mr, rtol=0.05)

    assert out.pages[0].blocks[1].lines[0].words[-1].value == "revised"
    geometry_revised = np.array([[0.7548828125, 0.126953125], [0.8388671875, 0.1484375]])
    assert np.allclose(np.array(out.pages[0].blocks[1].lines[0].words[-1].geometry), geometry_revised, rtol=0.05)

    det_predictor = detection_predictor(
        "fast_base",
        pretrained=True,
        batch_size=2,
        assume_straight_pages=True,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
    )

    predictor = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=True,
        straighten_pages=True,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
        resolve_blocks=True,
        resolve_lines=True,
    )
    # test hooks
    predictor.add_hook(_DummyCallback())

    out = predictor(doc)

    assert out.pages[0].blocks[0].lines[0].words[0].value == "Mr."


@pytest.mark.parametrize(
    "assume_straight_pages, straighten_pages, disable_page_orientation, disable_crop_orientation",
    [
        [True, False, False, False],
        [False, False, True, True],
        [True, True, False, False],
        [False, True, True, True],
        [True, False, True, False],
    ],
)
def test_kiepredictor(
    mock_pdf, mock_vocab, assume_straight_pages, straighten_pages, disable_page_orientation, disable_crop_orientation
):
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
        resolve_blocks=True,
        resolve_lines=True,
        disable_page_orientation=disable_page_orientation,
        disable_crop_orientation=disable_crop_orientation,
    )

    assert (
        predictor._page_orientation_disabled if disable_page_orientation else not predictor._page_orientation_disabled
    )
    assert (
        predictor._crop_orientation_disabled if disable_crop_orientation else not predictor._crop_orientation_disabled
    )

    if assume_straight_pages:
        assert predictor.crop_orientation_predictor is None
        if predictor.detect_orientation or predictor.straighten_pages:
            assert isinstance(predictor.page_orientation_predictor, nn.Module)
        else:
            assert predictor.page_orientation_predictor is None
    else:
        assert isinstance(predictor.crop_orientation_predictor, nn.Module)
        assert isinstance(predictor.page_orientation_predictor, nn.Module)

    out = predictor(doc)
    assert isinstance(out, Document)
    assert len(out.pages) == 2
    # Dimension check
    with pytest.raises(ValueError):
        input_page = (255 * np.random.rand(1, 256, 512, 3)).astype(np.uint8)
        _ = predictor([input_page])

    orientation = 0
    assert out.pages[0].orientation["value"] == orientation

    # Test with custom orientation models
    custom_crop_orientation_model = mobilenet_v3_small_crop_orientation(pretrained=True)
    custom_page_orientation_model = mobilenet_v3_small_page_orientation(pretrained=True)

    if assume_straight_pages:
        if predictor.detect_orientation or predictor.straighten_pages:
            # Overwrite the default orientation models
            predictor.crop_orientation_predictor = crop_orientation_predictor(custom_crop_orientation_model)
            predictor.page_orientation_predictor = page_orientation_predictor(custom_page_orientation_model)
    else:
        # Overwrite the default orientation models
        predictor.crop_orientation_predictor = crop_orientation_predictor(custom_crop_orientation_model)
        predictor.page_orientation_predictor = page_orientation_predictor(custom_page_orientation_model)

    out = predictor(doc)
    orientation = 0
    assert out.pages[0].orientation["value"] == orientation


def test_trained_kie_predictor(mock_payslip):
    doc = DocumentFile.from_images(mock_payslip)

    det_predictor = detection_predictor(
        "fast_base",
        pretrained=True,
        batch_size=2,
        assume_straight_pages=True,
        symmetric_pad=True,
        preserve_aspect_ratio=False,
    )
    reco_predictor = recognition_predictor("crnn_vgg16_bn", pretrained=True, batch_size=128)

    predictor = KIEPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=True,
        straighten_pages=True,
        preserve_aspect_ratio=False,
        resolve_blocks=True,
        resolve_lines=True,
    )
    # test hooks
    predictor.add_hook(_DummyCallback())

    out = predictor(doc)

    assert isinstance(out, KIEDocument)
    assert out.pages[0].predictions[CLASS_NAME][0].value == "Mr."
    geometry_mr = np.array([[0.1083984375, 0.0634765625], [0.1494140625, 0.0859375]])
    assert np.allclose(np.array(out.pages[0].predictions[CLASS_NAME][0].geometry), geometry_mr, rtol=0.05)

    assert out.pages[0].predictions[CLASS_NAME][3].value == "revised"
    geometry_revised = np.array([[0.7548828125, 0.126953125], [0.8388671875, 0.1484375]])
    assert np.allclose(np.array(out.pages[0].predictions[CLASS_NAME][3].geometry), geometry_revised, rtol=0.05)

    det_predictor = detection_predictor(
        "fast_base",
        pretrained=True,
        batch_size=2,
        assume_straight_pages=True,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
    )

    predictor = KIEPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=True,
        straighten_pages=True,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
        resolve_blocks=True,
        resolve_lines=True,
    )

    out = predictor(doc)

    assert isinstance(out, KIEDocument)
    assert out.pages[0].predictions[CLASS_NAME][0].value == "Mr."


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

    # Layout-aware OCR predictor via the factory (detect_layout flag)
    predictor = models.ocr_predictor(det_arch, reco_arch, pretrained=True, detect_layout=True)
    assert isinstance(predictor.layout_predictor, LayoutPredictor)
    _test_predictor(predictor)

    # passing a (fine-tuned) layout model instance, like det/reco
    layout_model = layout.lw_detr_s(pretrained=False)
    predictor = models.ocr_predictor(det_arch, reco_arch, pretrained=True, detect_layout=True, layout_arch=layout_model)
    assert isinstance(predictor.layout_predictor, LayoutPredictor)
    assert predictor.layout_predictor.model is layout_model

    # disabled by default
    predictor = models.ocr_predictor(det_arch, reco_arch, pretrained=True)
    assert predictor.layout_predictor is None

    # Layout-aware KIE predictor via the factory
    predictor = models.kie_predictor(det_arch, reco_arch, pretrained=True, detect_layout=True)
    assert isinstance(predictor.layout_predictor, LayoutPredictor)
    _test_kiepredictor(predictor)


@pytest.mark.parametrize(
    "det_arch, reco_arch",
    [
        ["fast_base", "crnn_vgg16_bn"],
    ],
)
def test_end_to_end_torch_compile(det_arch, reco_arch, mock_payslip):
    doc = DocumentFile.from_images(mock_payslip)
    predictor = models.ocr_predictor(det_arch, reco_arch, pretrained=True, assume_straight_pages=False)
    out = predictor(doc)

    assert isinstance(out, Document)

    # Compile the models
    detection_model = torch.compile(detection.__dict__[det_arch](pretrained=True).eval())
    recognition_model = torch.compile(recognition.__dict__[reco_arch](pretrained=True).eval())
    crop_orientation_model = torch.compile(mobilenet_v3_small_crop_orientation(pretrained=True).eval())
    page_orientation_model = torch.compile(mobilenet_v3_small_page_orientation(pretrained=True).eval())

    predictor = models.ocr_predictor(detection_model, recognition_model, assume_straight_pages=False)
    # Set the orientation predictors
    # NOTE: only required for non-straight pages and non-disabled orientation classification
    predictor.crop_orientation_predictor = crop_orientation_predictor(crop_orientation_model)
    predictor.page_orientation_predictor = page_orientation_predictor(page_orientation_model)
    compiled_out = predictor(doc)

    # Check that the number of word detections is the same
    assert len(out.pages[0].blocks[0].lines[0].words) == len(compiled_out.pages[0].blocks[0].lines[0].words)
    # Check that the words are the same
    assert all(
        word.value == compiled_out.pages[0].blocks[0].lines[0].words[i].value
        for i, word in enumerate(out.pages[0].blocks[0].lines[0].words)
    )


# ---- preserve_original_coords tests ----


@pytest.fixture(scope="module")
def _preserve_coords_pred():
    from doctr.models import ocr_predictor

    return ocr_predictor(
        "db_resnet50",
        "crnn_vgg16_bn",
        pretrained=True,
        assume_straight_pages=False,
        straighten_pages=True,
        detect_orientation=True,
        preserve_original_coords=True,
    )


@pytest.fixture(scope="module")
def _preserve_coords_kie_pair():
    from doctr.models import kie_predictor

    pred_on = kie_predictor(
        "db_resnet50",
        "crnn_vgg16_bn",
        pretrained=True,
        assume_straight_pages=False,
        straighten_pages=True,
        detect_orientation=True,
        preserve_original_coords=True,
    )
    pred_off = kie_predictor(
        "db_resnet50",
        "crnn_vgg16_bn",
        pretrained=True,
        assume_straight_pages=False,
        straighten_pages=True,
        detect_orientation=True,
        preserve_original_coords=False,
    )
    return pred_on, pred_off


def test__remap_geometry():
    from doctr.io.elements import Word
    from doctr.models.predictor.base import _OCRPredictor

    oh, ow = 800, 600
    sw, sh = ow, oh  # square straightening: no dimension change

    # Identity matrix — relative coords unchanged when sw==ow and sh==oh
    m_inv = np.eye(3, dtype=np.float64)
    word = Word("test", 0.9, ((0.1, 0.1), (0.3, 0.2)), 0.8, {"value": 0, "confidence": 1.0})
    _OCRPredictor._remap_geometry(word, m_inv, sw, sh, oh, ow)
    g = np.array(word.geometry).reshape(-1, 2)
    assert np.allclose(g[0], [0.1, 0.1])
    assert np.allclose(g[1], [0.3, 0.2])

    # 4-point polygon with identity matrix
    poly = ((0.1, 0.1), (0.3, 0.1), (0.3, 0.2), (0.1, 0.2))
    word4 = Word("test", 0.9, poly, 0.8, {"value": 0, "confidence": 1.0})
    _OCRPredictor._remap_geometry(word4, m_inv, sw, sh, oh, ow)
    assert len(word4.geometry) == 4
    assert np.allclose(np.array(word4.geometry).reshape(-1, 2)[0], [0.1, 0.1])

    # Realistic 12° rotation via a real straighten_page m_inv
    h, w = oh, ow
    page_img = np.ones((h, w, 3), dtype=np.uint8) * 255
    from doctr.utils.geometry import straighten_page

    straightened, m_inv_rot = straighten_page(page_img, 12.0)
    sh, sw = straightened.shape[:2]
    word2 = Word("test", 0.9, ((0.1, 0.1), (0.3, 0.2)), 0.8, {"value": 0, "confidence": 1.0})
    orig_geo = np.array(word2.geometry).reshape(-1, 2).copy()
    _OCRPredictor._remap_geometry(word2, m_inv_rot, sw, sh, oh, ow)
    g2 = np.array(word2.geometry).reshape(-1, 2)
    assert not np.allclose(g2, orig_geo), "12° rotation must change geometry"


def test_remap_to_original_coords_walk():
    from doctr.io.elements import (
        Artefact,
        Block,
        Document,
        LayoutElement,
        Line,
        Page,
        Table,
        TableCell,
        Word,
    )
    from doctr.models.predictor.base import _OCRPredictor
    from doctr.utils.geometry import straighten_page

    oh, ow = 800, 600
    page_img = np.ones((oh, ow, 3), dtype=np.uint8) * 255

    # Build a realistic m_inv from the 12° straightening pipeline
    straightened, m_inv = straighten_page(page_img, 12.0)
    sh, sw = straightened.shape[:2]

    # Construct one of every geometry-bearing type
    word = Word("test", 0.9, ((0.1, 0.1), (0.3, 0.2)), 0.8, {"value": 0, "confidence": 1.0})
    artefact = Artefact("qr_code", 0.95, ((0.5, 0.5), (0.6, 0.6)))
    line = Line([word])
    block = Block([line], artefacts=[artefact])
    layout = LayoutElement("Text", 0.9, ((0.7, 0.1), (0.9, 0.3)))
    cell = TableCell("$100", 0.95, ((0.4, 0.4), (0.6, 0.5)), 0, 0, 0, 0)
    table = Table([cell], 1, 1, ((0.35, 0.35), (0.65, 0.55)), 0.9)

    page = Page(
        page=np.ones((oh, ow, 3), dtype=np.uint8) * 255,
        blocks=[block],
        page_idx=0,
        dimensions=(oh, ow),
        layout=[layout],
        tables=[table],
    )
    doc = Document([page])

    # Snapshot original geometries
    orig = {
        "word": np.array(word.geometry).reshape(-1, 2).copy(),
        "line": np.array(line.geometry).reshape(-1, 2).copy(),
        "block": np.array(block.geometry).reshape(-1, 2).copy(),
        "artefact": np.array(artefact.geometry).reshape(-1, 2).copy(),
        "layout": np.array(layout.geometry).reshape(-1, 2).copy(),
        "table": np.array(table.geometry).reshape(-1, 2).copy(),
        "cell": np.array(cell.geometry).reshape(-1, 2).copy(),
    }

    predictor = _OCRPredictor(assume_straight_pages=True, straighten_pages=False)
    predictor._remap_to_original_coords(doc, [(oh, ow)], [(sh, sw)], [m_inv])

    # Difference check: every geometry must have changed
    for key, orig_val in orig.items():
        obj = {
            "word": word,
            "line": line,
            "block": block,
            "artefact": artefact,
            "layout": layout,
            "table": table,
            "cell": cell,
        }[key]
        current = np.array(obj.geometry).reshape(-1, 2)
        assert not np.allclose(current, orig_val), f"{key} geometry unchanged after remap"

    # Containment: block envelope contains its word geometries
    block_poly = np.array(block.geometry).reshape(-1, 2)
    bx0, by0 = block_poly.min(axis=0)
    bx1, by1 = block_poly.max(axis=0)
    for word_obj in block.lines[0].words:
        wp = np.array(word_obj.geometry).reshape(-1, 2)
        assert wp[:, 0].min() >= bx0 - 1e-6
        assert wp[:, 1].min() >= by0 - 1e-6
        assert wp[:, 0].max() <= bx1 + 1e-6
        assert wp[:, 1].max() <= by1 + 1e-6


def test_preserve_original_coords_kie_smoke(_preserve_coords_kie_pair):
    h, w = 800, 600
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Sensitive", (50, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "information", (185, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    angle = 12
    m_skew = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    skewed = cv2.warpAffine(img, m_skew, (w, h), borderValue=(255, 255, 255))

    pred_on, pred_off = _preserve_coords_kie_pair
    result_on = pred_on([skewed])
    result_off = pred_off([skewed])

    def _collect_geoms(result):
        geoms = []
        for class_preds in result.pages[0].predictions.values():
            for pred in class_preds:
                geoms.append(np.array(pred.geometry).reshape(-1, 2))
        return geoms

    geoms_on = _collect_geoms(result_on)
    geoms_off = _collect_geoms(result_off)

    assert len(geoms_on) > 0 and len(geoms_off) > 0
    all_same = all(np.allclose(g_on, g_off) for g_on, g_off in zip(geoms_on, geoms_off))
    assert not all_same, "All geometries identical between flag-on and flag-off"


@pytest.mark.parametrize("angle, shape", [(12, (800, 600)), (-12, (800, 600)), (12, (600, 800)), (-12, (600, 800))])
def test_preserve_original_coords_roundtrip(angle, shape, _preserve_coords_pred):
    from shapely.geometry import Polygon

    h, w = shape
    words = [("Sensitive", 50, h // 2 - 2), ("information", 185, h // 2 - 2)]

    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for text, x, y in words:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    gt_orig = []
    for text, x, y in words:
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x0, x1 = max(0, x - 5), min(w, x + tw + 5)
        y0, y1 = max(0, y - th - 5), min(h, y + bl + 5)
        region = thresh[y0:y1, x0:x1]
        ys, xs = np.nonzero(region)
        gt_orig.append((x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1))

    m_skew = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    skewed = cv2.warpAffine(img, m_skew, (w, h), borderValue=(255, 255, 255))

    gt_skewed = []
    for gx1, gy1, gx2, gy2 in gt_orig:
        corners = np.array([[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        homo = np.column_stack([corners, ones])
        xs, ys = (homo @ m_skew.T)[:, 0], (homo @ m_skew.T)[:, 1]
        gt_skewed.append(np.column_stack([xs, ys]))

    result = _preserve_coords_pred([skewed])

    assert result.pages[0].page.shape[:2] == (h, w), "page.page should have original dimensions"
    assert result.pages[0].dimensions == (h, w), "page.dimensions should match original page shape"

    det_polys = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                g = np.array(word.geometry).reshape(-1, 2)
                det_polys.append(np.array([[g[i, 0] * w, g[i, 1] * h] for i in range(4)], dtype=np.float32))

    ious = []
    for det in det_polys:
        p_det = Polygon(det)
        best_iou = 0.0
        for polygon in gt_skewed:
            p_gt = Polygon(polygon)
            if not p_gt.is_valid or p_gt.area == 0:
                continue
            union = p_det.union(p_gt).area
            if union > 0:
                best_iou = max(best_iou, p_det.intersection(p_gt).area / union)
        ious.append(best_iou)

    mean_iou = float(np.mean(ious))
    assert mean_iou > 0.4, f"Mean IoU {mean_iou:.3f} below 0.4 threshold"


def test_preserve_original_coords_2point():
    from shapely.geometry import Polygon
    from doctr.models import ocr_predictor

    h, w = 800, 600
    words = [("Sensitive", 50, 298), ("information", 185, 298)]

    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for text, x, y in words:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    gt_orig = []
    for text, x, y in words:
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x0, x1 = max(0, x - 5), min(w, x + tw + 5)
        y0, y1 = max(0, y - th - 5), min(h, y + bl + 5)
        region = thresh[y0:y1, x0:x1]
        ys, xs = np.nonzero(region)
        gt_orig.append((x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1))

    angle = 12
    m_skew = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    skewed = cv2.warpAffine(img, m_skew, (w, h), borderValue=(255, 255, 255))

    gt_skewed = []
    for gx1, gy1, gx2, gy2 in gt_orig:
        corners = np.array([[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]], dtype=np.float32)
        ones = np.ones((4, 1), dtype=np.float32)
        homo = np.column_stack([corners, ones])
        xs, ys = (homo @ m_skew.T)[:, 0], (homo @ m_skew.T)[:, 1]
        gt_skewed.append(np.column_stack([xs, ys]))

    predictor = ocr_predictor(
        "db_resnet50",
        "crnn_vgg16_bn",
        pretrained=True,
        assume_straight_pages=True,
        straighten_pages=True,
        detect_orientation=True,
        preserve_original_coords=True,
    )
    result = predictor([skewed])

    det_polys = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            for word in line.words:
                g = np.array(word.geometry).reshape(-1, 2)
                assert g.shape[0] == 2, "2-point mode must return 2-point geometry"
                x0, y0 = g[0]
                x1, y1 = g[1]
                det_polys.append(
                    np.array([[x0 * w, y0 * h], [x1 * w, y0 * h], [x1 * w, y1 * h], [x0 * w, y1 * h]], dtype=np.float32)
                )

    ious = []
    for det in det_polys:
        p_det = Polygon(det)
        best_iou = 0.0
        for polygon in gt_skewed:
            p_gt = Polygon(polygon)
            if not p_gt.is_valid or p_gt.area == 0:
                continue
            union = p_det.union(p_gt).area
            if union > 0:
                best_iou = max(best_iou, p_det.intersection(p_gt).area / union)
        ious.append(best_iou)

    mean_iou = float(np.mean(ious))
    assert mean_iou > 0.4, f"Mean IoU {mean_iou:.3f} below 0.4 threshold"
