import cv2
import numpy as np
import pytest
import torch
from torch import nn

from doctr import models
from doctr.file_utils import CLASS_NAME
from doctr.io import Document, DocumentFile, FigureEncoder, crop_layout_region
from doctr.io.elements import KIEDocument, KIEPage, LayoutElement, Table
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


def test_predictors_on_empty_batch(mock_vocab):
    det_predictor = DetectionPredictor(
        PreProcessor(output_size=(512, 512), batch_size=2),
        detection.db_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, assume_straight_pages=True),
    )
    reco_predictor = RecognitionPredictor(
        PreProcessor(output_size=(32, 128), batch_size=32, preserve_aspect_ratio=True),
        recognition.crnn_vgg16_bn(pretrained=False, pretrained_backbone=False, vocab=mock_vocab),
    )

    # Detection keeps the shape its `return_maps` contract promises.
    assert det_predictor([]) == []
    assert det_predictor([], return_maps=True) == ([], [])

    # The recognition predictor already behaved; asserted here so the three
    # stay consistent if one of them is touched again.
    assert reco_predictor([]) == []

    for predictor, expected_type in (
        (OCRPredictor(det_predictor, reco_predictor, assume_straight_pages=True), Document),
        (KIEPredictor(det_predictor, reco_predictor, assume_straight_pages=True), KIEDocument),
    ):
        out = predictor([])
        # Exact type, not isinstance: KIEDocument subclasses Document, so an
        # isinstance check would not notice the KIE path degrading to the base
        # class and dropping the per-class prediction shape.
        assert type(out) is expected_type
        assert out.pages == []
        assert out.export() == {"pages": []}


@pytest.mark.parametrize("assume_straight_pages", [True, False])
@pytest.mark.parametrize("predictor_cls", [OCRPredictor, KIEPredictor])
def test_predictors_keep_objectness_scores_aligned(predictor_cls, assume_straight_pages):
    # The middle box lies on the right border of the page and has no width: its crop is empty,
    # so it is dropped before recognition
    if assume_straight_pages:
        boxes = np.array(
            [[0.1, 0.1, 0.3, 0.2, 0.9], [1.0, 0.5, 1.0, 0.6, 0.1], [0.5, 0.1, 0.7, 0.2, 0.8]],
            dtype=np.float32,
        )
    else:
        # Polygons with the objectness score in the last row
        boxes = np.array(
            [
                [[0.1, 0.1], [0.3, 0.1], [0.3, 0.2], [0.1, 0.2], [0.9, 0.9]],
                [[1.0, 0.5], [1.0, 0.5], [1.0, 0.6], [1.0, 0.6], [0.1, 0.1]],
                [[0.5, 0.1], [0.7, 0.1], [0.7, 0.2], [0.5, 0.2], [0.8, 0.8]],
            ],
            dtype=np.float32,
        )

    class _FixedDetPredictor(nn.Module):
        def forward(self, pages, **kwargs):
            return [{CLASS_NAME: boxes.copy()} for _ in pages]

    class _DummyRecoPredictor(nn.Module):
        def forward(self, crops, **kwargs):
            return [(f"word{idx}", 1.0) for idx in range(len(crops))]

    # Without page orientation, rotated boxes are cropped with `assume_horizontal=True`
    predictor = predictor_cls(
        _FixedDetPredictor(),
        _DummyRecoPredictor(),
        assume_straight_pages=assume_straight_pages,
        disable_page_orientation=True,
        disable_crop_orientation=True,
    )
    page = predictor([np.full((100, 200, 3), 255, dtype=np.uint8)]).pages[0]
    if predictor_cls is KIEPredictor:
        words = page.predictions[CLASS_NAME]
    else:
        words = [word for block in page.blocks for line in block.lines for word in line.words]

    # Each remaining word keeps the objectness score of its own box
    assert [word.geometry[0][0] for word in words] == pytest.approx([0.1, 0.5])
    assert [word.objectness_score for word in words] == pytest.approx([0.9, 0.8])


@pytest.mark.parametrize("assume_straight_pages", [True, False])
@pytest.mark.parametrize("predictor_cls", [OCRPredictor, KIEPredictor])
def test_predictors_hooks_run_before_cropping(predictor_cls, assume_straight_pages):
    # Each box covers a region of the page filled with its own gray level, read back by the recognition stub
    levels = [10, 100, 200]
    page = np.full((100, 200, 3), 255, dtype=np.uint8)
    straight_boxes = [[0.1, 0.1, 0.3, 0.2], [0.4, 0.1, 0.6, 0.2], [0.7, 0.1, 0.9, 0.2]]
    for (xmin, ymin, xmax, ymax), level in zip(straight_boxes, levels):
        page[int(ymin * 100) : int(ymax * 100), int(xmin * 200) : int(xmax * 200)] = level
    scores = [0.9, 0.8, 0.7]
    if assume_straight_pages:
        boxes = np.array([[*box, score] for box, score in zip(straight_boxes, scores)], dtype=np.float32)
    else:
        boxes = np.array(
            [
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [score, score]]
                for (x0, y0, x1, y1), score in zip(straight_boxes, scores)
            ],
            dtype=np.float32,
        )

    class _FixedDetPredictor(nn.Module):
        def forward(self, pages, **kwargs):
            return [{CLASS_NAME: boxes.copy()} for _ in pages]

    class _GrayLevelRecoPredictor(nn.Module):
        def forward(self, crops, **kwargs):
            return [(str(int(np.median(crop))), 1.0) for crop in crops]

    predictor = predictor_cls(
        _FixedDetPredictor(),
        _GrayLevelRecoPredictor(),
        assume_straight_pages=assume_straight_pages,
        disable_page_orientation=True,
        disable_crop_orientation=True,
    )

    def _words(out):
        page_out = out.pages[0]
        if predictor_cls is KIEPredictor:
            return page_out.predictions[CLASS_NAME]
        return [word for block in page_out.blocks for line in block.lines for word in line.words]

    # A hook removing the first box: only the other boxes are read, each with its own text and score
    if predictor_cls is KIEPredictor:
        predictor.add_hook(lambda preds: {name: [p[1:] for p in pages] for name, pages in preds.items()})
    else:
        predictor.add_hook(lambda preds: [p[1:] for p in preds])
    words = sorted(_words(predictor([page])), key=lambda word: np.asarray(word.geometry)[..., 0].min())
    assert [word.value for word in words] == ["100", "200"]
    assert [word.objectness_score for word in words] == pytest.approx([0.8, 0.7])

    # Hooks run in order: the second one only keeps the box of the third region
    if predictor_cls is KIEPredictor:
        predictor.add_hook(lambda preds: {name: [p[1:] for p in pages] for name, pages in preds.items()})
    else:
        predictor.add_hook(lambda preds: [p[1:] for p in preds])
    assert [word.value for word in _words(predictor([page]))] == ["200"]

    # Invalid outputs are reported
    predictor.hooks = [lambda preds: preds[0] if isinstance(preds, list) else list(preds.values())]
    with pytest.raises(ValueError):
        predictor([page])


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


def _custom_aggregation(scores):
    return float(np.percentile(scores, 25))


@pytest.mark.parametrize(
    "confidence_aggregation, expected",
    [
        # Default method of the recognition model
        [None, "min"],
        ["mean", "mean"],
        ["geometric_mean", "geometric_mean"],
        [_custom_aggregation, _custom_aggregation],
    ],
)
def test_ocr_predictor_confidence_aggregation(confidence_aggregation, expected):
    for factory in (models.ocr_predictor, models.kie_predictor):
        predictor = factory(
            "db_mobilenet_v3_large",
            "crnn_mobilenet_v3_small",
            pretrained=False,
            pretrained_backbone=False,
            confidence_aggregation=confidence_aggregation,
        )
        assert predictor.reco_predictor.model.postprocessor.confidence_aggregation == expected
        # The split crops keep their own aggregation
        assert predictor.reco_predictor.split_confidence_aggregation == "min"


def test_ocr_predictor_confidence_aggregation_model_instances(mock_payslip):
    reco_model = recognition.parseq(pretrained=True)
    predictor = models.ocr_predictor(
        detection.db_mobilenet_v3_large(pretrained=True), reco_model, confidence_aggregation="min"
    )
    assert reco_model.postprocessor.confidence_aggregation == "min"
    # None keeps the method of the model
    assert models.ocr_predictor("db_mobilenet_v3_large", reco_model).reco_predictor.model is reco_model
    assert reco_model.postprocessor.confidence_aggregation == "min"

    # End-to-end: the word confidences follow the aggregation method set on the recognition model
    doc = DocumentFile.from_images(mock_payslip)

    def word_confidences(method):
        reco_model.postprocessor.confidence_aggregation = method
        out = predictor(doc)
        return np.array([w.confidence for b in out.pages[0].blocks for line in b.lines for w in line.words])

    min_confs, max_confs = word_confidences("min"), word_confidences("max")
    mean_confs = word_confidences("mean")
    assert min_confs.size > 0
    assert np.all((min_confs >= 0) & (max_confs <= 1))
    assert np.all(min_confs <= mean_confs + 1e-6) and np.all(mean_confs <= max_confs + 1e-6)
    # The method is applied: at least one word has an uncertain character
    assert np.any(min_confs < max_confs - 1e-3)


def test_recognition_predictor_confidence_aggregation():
    # Architecture names
    reco_predictor = recognition_predictor("parseq", pretrained=False, confidence_aggregation="min")
    assert reco_predictor.model.postprocessor.confidence_aggregation == "min"
    # Model instances are modified
    reco_model = recognition.crnn_mobilenet_v3_small(pretrained=False, pretrained_backbone=False)
    assert recognition_predictor(reco_model, confidence_aggregation=np.median).model is reco_model
    assert reco_model.postprocessor.confidence_aggregation is np.median
    # None keeps the method of the model
    assert recognition_predictor(reco_model).model.postprocessor.confidence_aggregation is np.median
    with pytest.raises(ValueError, match="Unknown confidence aggregation"):
        recognition_predictor("crnn_mobilenet_v3_small", pretrained=False, confidence_aggregation="average")


@pytest.mark.parametrize(
    "detect_layout, detect_tables",
    [
        [False, False],
        [True, False],
        [False, True],
        [True, True],
    ],
)
def test_ocr_predictor_figures(mock_figure_page, tmp_path, detect_layout, detect_tables):
    predictor = models.ocr_predictor(pretrained=True, detect_layout=detect_layout, detect_tables=detect_tables)
    page = predictor(DocumentFile.from_images(mock_figure_page)).pages[0]
    exports = {images: page.export_as_markdown(images=images) for images in ("none", "placeholder", "embedded")}
    encoder = FigureEncoder("referenced", image_dir=tmp_path, path_prefix="assets")
    referenced = page.export_as_markdown(images=encoder)
    xml = page.export_as_xml()[0].decode()

    if not (detect_layout or detect_tables):
        # Without layout, the image modes change nothing
        assert page.layout == []
        assert len({*exports.values(), referenced}) == 1
        assert encoder.written == [] and 'class="ocr_photo"' not in xml
        return

    # The layout model (also run for the tables) finds the photograph
    figures = [item for item in page.items_in_reading_order(include_figures=True) if isinstance(item, LayoutElement)]
    assert [figure.type for figure in figures] == ["Picture"]
    assert page.tables == []
    assert xml.count('class="ocr_photo"') == 1
    assert exports["placeholder"].count("<!-- image -->") == 1
    assert exports["placeholder"].replace("<!-- image -->\n\n", "") == exports["none"]
    # With its pixels, the caption is the alt text and a line below the image
    caption = next(part for part in exports["none"].split("\n\n") if part.startswith("Figure 1"))
    assert exports["embedded"].count("](data:image/png;base64,") == 1
    assert f"![{caption}](data:image/png;base64," in exports["embedded"]
    assert exports["embedded"].count(caption) == 2
    assert f"![{caption}](assets/{encoder.written[0].name})\n\n*{caption}*" in referenced
    crop = cv2.imread(str(encoder.written[0]))
    assert abs(crop.shape[0] - 500) < 25 and abs(crop.shape[1] - 800) < 40
    assert exports["embedded"].split("\n\n")[:2] == exports["none"].split("\n\n")[:2]
    assert exports["embedded"].split("\n\n")[-1] == exports["none"].split("\n\n")[-1]


def test_ocr_predictor_figures_ignore_regions(mock_figure_page):
    # Ignored regions are only masked for the text detection: the figure keeps its pixels
    predictor = models.ocr_predictor(pretrained=True, detect_layout=True, ignore_regions=["Picture"])
    page = predictor(DocumentFile.from_images(mock_figure_page)).pages[0]
    figure = next(region for region in page.layout if region.type == "Picture")
    crop = crop_layout_region(page.page, figure.geometry)
    assert crop is not None and crop.any()
    assert page.export_as_markdown(images="embedded").count("](data:image/png;base64,") == 1


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


def test_ocr_predictor_straighten_with_preserve_original_coords(mock_tilted_payslip):
    doc = DocumentFile.from_images(mock_tilted_payslip)
    det_predictor = detection_predictor(
        "fast_base",
        pretrained=True,
        batch_size=2,
        assume_straight_pages=False,
        symmetric_pad=True,
        preserve_aspect_ratio=False,
    )
    reco_predictor = recognition_predictor("crnn_vgg16_bn", pretrained=True, batch_size=128)
    predictor_on = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=False,
        straighten_pages=True,
        detect_orientation=True,
        preserve_aspect_ratio=False,
        resolve_blocks=True,
        resolve_lines=True,
        preserve_original_coords=True,
    )
    predictor_off = OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=False,
        straighten_pages=True,
        detect_orientation=True,
        preserve_aspect_ratio=False,
        resolve_blocks=True,
        resolve_lines=True,
        preserve_original_coords=False,
    )
    out_on = predictor_on(doc)
    out_off = predictor_off(doc)
    assert len(out_on.pages[0].blocks) > 0
    assert len(out_off.pages[0].blocks) > 0
    geoms_on = [
        np.array(w.geometry).reshape(-1, 2).tolist()
        for block in out_on.pages[0].blocks
        for line in block.lines
        for w in line.words
    ]
    geoms_off = [
        np.array(w.geometry).reshape(-1, 2).tolist()
        for block in out_off.pages[0].blocks
        for line in block.lines
        for w in line.words
    ]
    assert geoms_on != geoms_off
    assert any(w.value == "Mr." for block in out_on.pages[0].blocks for line in block.lines for w in line.words)
    assert out_on.pages[0].page.shape[:2] == out_on.pages[0].dimensions
    assert out_on.pages[0].page.shape[:2] == doc[0].shape[:2]


def test_kie_predictor_straighten_with_preserve_original_coords(mock_tilted_payslip):
    doc = DocumentFile.from_images(mock_tilted_payslip)
    det_predictor = detection_predictor(
        "fast_base",
        pretrained=True,
        batch_size=2,
        assume_straight_pages=False,
        symmetric_pad=True,
        preserve_aspect_ratio=False,
    )
    reco_predictor = recognition_predictor("crnn_vgg16_bn", pretrained=True, batch_size=128)
    predictor_on = KIEPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=False,
        straighten_pages=True,
        detect_orientation=True,
        preserve_aspect_ratio=False,
        resolve_blocks=True,
        resolve_lines=True,
        preserve_original_coords=True,
    )
    predictor_off = KIEPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=False,
        straighten_pages=True,
        detect_orientation=True,
        preserve_aspect_ratio=False,
        resolve_blocks=True,
        resolve_lines=True,
        preserve_original_coords=False,
    )
    out_on = predictor_on(doc)
    out_off = predictor_off(doc)
    assert len(out_on.pages[0].predictions[CLASS_NAME]) > 0
    assert len(out_off.pages[0].predictions[CLASS_NAME]) > 0
    geoms_on = [np.array(p.geometry).reshape(-1, 2).tolist() for p in out_on.pages[0].predictions[CLASS_NAME]]
    geoms_off = [np.array(p.geometry).reshape(-1, 2).tolist() for p in out_off.pages[0].predictions[CLASS_NAME]]
    assert geoms_on != geoms_off
    assert out_on.pages[0].page.shape[:2] == out_on.pages[0].dimensions
    assert out_on.pages[0].page.shape[:2] == doc[0].shape[:2]


# Long documents: chunking, streaming & lazy page sources


class _ValueDetPredictor(nn.Module):
    """Fake detection: one box per page, and records the number of pages of each call"""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, pages, **kwargs):
        self.calls.append(len(pages))
        return [{CLASS_NAME: np.array([[0.1, 0.1, 0.9, 0.9, 0.9]], dtype=np.float32)} for _ in pages]


class _ValueRecoPredictor(nn.Module):
    """Fake recognition: reads the gray level of the crop, which identifies the page"""

    def forward(self, crops, **kwargs):
        return [(str(round(float(crop.mean()))), 1.0) for crop in crops]


def _page_words(page):
    if isinstance(page, KIEPage):
        return [word.value for word in page.predictions[CLASS_NAME]]
    return [word.value for block in page.blocks for line in block.lines for word in line.words]


def _value_pages(num_pages):
    return [np.full((60 + idx, 80, 3), 10 * idx, dtype=np.uint8) for idx in range(num_pages)]


def _value_predictor(predictor_cls, **kwargs):
    return predictor_cls(_ValueDetPredictor(), _ValueRecoPredictor(), assume_straight_pages=True, **kwargs)


@pytest.mark.parametrize(
    "predictor_cls, doc_cls",
    [(OCRPredictor, Document), (KIEPredictor, KIEDocument)],
)
def test_predictors_chunk_long_inputs(predictor_cls, doc_cls):
    pages = _value_pages(7)
    predictor = _value_predictor(predictor_cls, page_chunk_size=3)

    out = predictor(pages)
    assert type(out) is doc_cls
    # processed by chunks of at most 3 pages, the results being identical to a single pass
    assert predictor.det_predictor.calls == [3, 3, 1]
    assert [page.page_idx for page in out.pages] == list(range(7))
    assert [_page_words(page) for page in out.pages] == [[str(10 * idx)] for idx in range(7)]
    assert [page.dimensions for page in out.pages] == [page.shape[:2] for page in pages]
    # in-memory pages are referenced as-is
    assert all(page.page is image for page, image in zip(out.pages, pages))

    single_pass = _value_predictor(predictor_cls)
    assert single_pass(pages).export() == out.export()
    assert single_pass.det_predictor.calls == [7]

    # any iterable of pages, e.g. a generator, is supported
    assert predictor(page for page in pages).export() == out.export()
    assert predictor(tuple(pages)).export() == out.export()


@pytest.mark.parametrize("predictor_cls", [OCRPredictor, KIEPredictor])
def test_predictors_stream(predictor_cls):
    pages = _value_pages(5)
    predictor = _value_predictor(predictor_cls, page_chunk_size=4)

    stream = predictor.stream(pages, page_chunk_size=2, prefetch_pages=False)
    first = next(stream)
    # results are yielded as soon as their chunk is processed
    assert predictor.det_predictor.calls == [2]
    assert (first.page_idx, _page_words(first)) == (0, ["0"])
    rest = list(stream)
    assert [page.page_idx for page in rest] == [1, 2, 3, 4]
    assert predictor.det_predictor.calls == [2, 2, 1]

    # same results with prefetching, and the predictor default chunk size
    predictor.det_predictor.calls.clear()
    streamed = list(predictor.stream(iter(pages)))
    assert predictor.det_predictor.calls == [4, 1]
    assert [page.export() for page in streamed] == [page.export() for page in [first, *rest]]
    assert list(predictor.stream([])) == []


@pytest.mark.parametrize(
    "predictor_cls, doc_cls",
    [(OCRPredictor, Document), (KIEPredictor, KIEDocument)],
)
def test_predictors_lazy_source(tmp_path, predictor_cls, doc_cls):
    pages = _value_pages(5)
    for idx, page in enumerate(pages):
        cv2.imwrite(str(tmp_path / f"page_{idx}.png"), page)
    source = DocumentFile.from_images(tmp_path, lazy=True)

    predictor = _value_predictor(predictor_cls, page_chunk_size=2)
    out = predictor(source)
    assert type(out) is doc_cls
    assert predictor.det_predictor.calls == [2, 2, 1]
    assert out.export() == _value_predictor(predictor_cls)(pages).export()

    # the predictions only keep a reference to their page, decoded again on access
    assert all(type(page._page).__name__ == "_SourcePage" for page in out.pages)
    assert all(np.array_equal(page.page, image) for page, image in zip(out.pages, pages))
    assert type(predictor(source[:0])) is doc_cls


def test_predictors_lazy_source_transformed_pages(tmp_path):
    """Pages modified by the pipeline (e.g. straightened) cannot be read again from the source: they are kept
    losslessly compressed"""

    class _TransformingPredictor(OCRPredictor):
        def _predict_pages(self, pages, **kwargs):
            out = super()._predict_pages(pages, **kwargs)
            for page in out.pages:
                page.page = np.ascontiguousarray(page.page[::-1])
            return out

    pages = _value_pages(3)
    pages[1][:10] = 255
    for idx, page in enumerate(pages):
        cv2.imwrite(str(tmp_path / f"page_{idx}.png"), page)

    predictor = _TransformingPredictor(_ValueDetPredictor(), _ValueRecoPredictor(), page_chunk_size=2)
    out = predictor(DocumentFile.from_images(tmp_path, lazy=True))
    assert all(type(page._page).__name__ == "_EncodedImage" for page in out.pages)
    assert all(np.array_equal(page.page, image[::-1]) for page, image in zip(out.pages, pages))


def test_predictors_invalid_page_chunk_size():
    for predictor_cls in (OCRPredictor, KIEPredictor):
        with pytest.raises(ValueError):
            _value_predictor(predictor_cls, page_chunk_size=0)


@pytest.mark.parametrize("straighten_pages", [False, True])
def test_ocr_predictor_lazy_pdf(mock_pdf, straighten_pages):
    predictor = models.ocr_predictor(
        "db_mobilenet_v3_large",
        "crnn_mobilenet_v3_small",
        pretrained=True,
        straighten_pages=straighten_pages,
        page_chunk_size=1,
    )
    eager = predictor(DocumentFile.from_pdf(mock_pdf))
    lazy = predictor(DocumentFile.from_pdf(mock_pdf, lazy=True))

    assert lazy.export() == eager.export()
    assert lazy.render() == eager.render()
    assert all(np.array_equal(lazy_page.page, page.page) for lazy_page, page in zip(lazy.pages, eager.pages))
    assert len(lazy.export_as_markdown(images="embedded")) > 0
