# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

from .detection.zoo import detection_predictor
from .kie_predictor import KIEPredictor
from .layout.zoo import layout_predictor
from .predictor import OCRPredictor
from .recognition.zoo import recognition_predictor
from .table_structure.zoo import table_predictor

__all__ = ["ocr_predictor", "kie_predictor"]


def _predictor(
    det_arch: Any,
    reco_arch: Any,
    pretrained: bool,
    pretrained_backbone: bool = True,
    assume_straight_pages: bool = True,
    preserve_aspect_ratio: bool = True,
    symmetric_pad: bool = True,
    det_bs: int = 2,
    reco_bs: int = 128,
    detect_orientation: bool = False,
    straighten_pages: bool = False,
    detect_language: bool = False,
    detect_layout: bool = False,
    layout_arch: Any = "lw_detr_s",
    detect_tables: bool = False,
    **kwargs,
) -> OCRPredictor:
    # Detection
    det_predictor = detection_predictor(
        det_arch,
        pretrained=pretrained,
        pretrained_backbone=pretrained_backbone,
        batch_size=det_bs,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
    )

    # Recognition
    reco_predictor = recognition_predictor(
        reco_arch,
        pretrained=pretrained,
        pretrained_backbone=pretrained_backbone,
        batch_size=reco_bs,
    )

    # Layout - required for table detection, so build it whenever layout or tables are requested
    layout_pred = (
        layout_predictor(
            layout_arch,
            pretrained=pretrained,
            assume_straight_pages=assume_straight_pages,
            preserve_aspect_ratio=preserve_aspect_ratio,
            symmetric_pad=symmetric_pad,
            batch_size=det_bs,
        )
        if (detect_layout or detect_tables)
        else None
    )

    # Table structure - optional, applied on the cropped table regions found by the layout model
    table_pred = (
        table_predictor(
            "tablecenternet",
            pretrained=pretrained,
            batch_size=det_bs,
        )
        if detect_tables
        else None
    )

    return OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
        detect_orientation=detect_orientation,
        straighten_pages=straighten_pages,
        detect_language=detect_language,
        layout_predictor=layout_pred,
        table_predictor=table_pred,
        **kwargs,
    )


def ocr_predictor(
    det_arch: Any = "fast_base",
    reco_arch: Any = "crnn_vgg16_bn",
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    assume_straight_pages: bool = True,
    preserve_aspect_ratio: bool = True,
    symmetric_pad: bool = True,
    export_as_straight_boxes: bool = False,
    detect_orientation: bool = False,
    straighten_pages: bool = False,
    detect_language: bool = False,
    detect_layout: bool = False,
    layout_arch: Any = "lw_detr_s",
    detect_tables: bool = False,
    **kwargs: Any,
) -> OCRPredictor:
    """End-to-end OCR architecture using one model for localization, and another for text recognition.

    >>> import numpy as np
    >>> from doctr.models import ocr_predictor
    >>> model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)
    >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    >>> out = model([input_page])

    Args:
        det_arch: name of the detection architecture or the model itself to use
            (e.g. 'db_resnet50', 'db_mobilenet_v3_large')
        reco_arch: name of the recognition architecture or the model itself to use
            (e.g. 'crnn_vgg16_bn', 'sar_resnet31')
        pretrained: If True, returns a model pre-trained on our OCR dataset
        pretrained_backbone: If True, returns a model with a pretrained backbone
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        preserve_aspect_ratio: If True, pad the input document image to preserve the aspect ratio before
            running the detection model on it.
        symmetric_pad: if True, pad the image symmetrically instead of padding at the bottom-right.
        export_as_straight_boxes: when assume_straight_pages is set to False, export final predictions
            (potentially rotated) as straight bounding boxes.
        detect_orientation: if True, the estimated general page orientation will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        straighten_pages: if True, estimates the page general orientation
            based on the segmentation map median line orientation.
            Then, rotates page before passing it again to the deep learning detection module.
            Doing so will improve performances for documents with page-uniform rotations.
        detect_language: if True, the language prediction will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        detect_layout: if True, a layout detection model is run on each page and the detected regions are attached
            to each page.
            Doing so will slightly deteriorate the overall latency.
        layout_arch: name of the layout architecture or the model itself to use.
        detect_tables: if True, table regions found by the layout model are cropped and passed to a table
            structure model. Words falling inside a detected table are regrouped into a structured table
            (accessible via `page.tables`) and removed from the regular text output. This enables the layout
            model and slightly deteriorates the overall latency.
        kwargs: keyword args of `OCRPredictor`

    Returns:
        OCR predictor
    """
    return _predictor(
        det_arch,
        reco_arch,
        pretrained,
        pretrained_backbone=pretrained_backbone,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
        export_as_straight_boxes=export_as_straight_boxes,
        detect_orientation=detect_orientation,
        straighten_pages=straighten_pages,
        detect_language=detect_language,
        detect_layout=detect_layout,
        layout_arch=layout_arch,
        detect_tables=detect_tables,
        **kwargs,
    )


def _kie_predictor(
    det_arch: Any,
    reco_arch: Any,
    pretrained: bool,
    pretrained_backbone: bool = True,
    assume_straight_pages: bool = True,
    preserve_aspect_ratio: bool = True,
    symmetric_pad: bool = True,
    det_bs: int = 2,
    reco_bs: int = 128,
    detect_orientation: bool = False,
    straighten_pages: bool = False,
    detect_language: bool = False,
    detect_layout: bool = False,
    layout_arch: Any = "lw_detr_s",
    **kwargs,
) -> KIEPredictor:
    # Detection
    det_predictor = detection_predictor(
        det_arch,
        pretrained=pretrained,
        pretrained_backbone=pretrained_backbone,
        batch_size=det_bs,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
    )

    # Recognition
    reco_predictor = recognition_predictor(
        reco_arch,
        pretrained=pretrained,
        pretrained_backbone=pretrained_backbone,
        batch_size=reco_bs,
    )

    # Layout - optional
    layout_pred = (
        layout_predictor(
            layout_arch,
            pretrained=pretrained,
            assume_straight_pages=assume_straight_pages,
            preserve_aspect_ratio=preserve_aspect_ratio,
            symmetric_pad=symmetric_pad,
            batch_size=det_bs,
        )
        if detect_layout
        else None
    )

    return KIEPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
        detect_orientation=detect_orientation,
        straighten_pages=straighten_pages,
        detect_language=detect_language,
        layout_predictor=layout_pred,
        **kwargs,
    )


def kie_predictor(
    det_arch: Any = "fast_base",
    reco_arch: Any = "crnn_vgg16_bn",
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    assume_straight_pages: bool = True,
    preserve_aspect_ratio: bool = True,
    symmetric_pad: bool = True,
    export_as_straight_boxes: bool = False,
    detect_orientation: bool = False,
    straighten_pages: bool = False,
    detect_language: bool = False,
    detect_layout: bool = False,
    layout_arch: Any = "lw_detr_s",
    **kwargs: Any,
) -> KIEPredictor:
    """End-to-end KIE architecture using one model for localization, and another for text recognition.

    >>> import numpy as np
    >>> from doctr.models import kie_predictor
    >>> model = kie_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)
    >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    >>> out = model([input_page])

    Args:
        det_arch: name of the detection architecture or the model itself to use
            (e.g. 'db_resnet50', 'db_mobilenet_v3_large')
        reco_arch: name of the recognition architecture or the model itself to use
            (e.g. 'crnn_vgg16_bn', 'sar_resnet31')
        pretrained: If True, returns a model pre-trained on our OCR dataset
        pretrained_backbone: If True, returns a model with a pretrained backbone
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        preserve_aspect_ratio: If True, pad the input document image to preserve the aspect ratio before
            running the detection model on it.
        symmetric_pad: if True, pad the image symmetrically instead of padding at the bottom-right.
        export_as_straight_boxes: when assume_straight_pages is set to False, export final predictions
            (potentially rotated) as straight bounding boxes.
        detect_orientation: if True, the estimated general page orientation will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        straighten_pages: if True, estimates the page general orientation
            based on the segmentation map median line orientation.
            Then, rotates page before passing it again to the deep learning detection module.
            Doing so will improve performances for documents with page-uniform rotations.
        detect_language: if True, the language prediction will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        detect_layout: if True, a layout detection model is run on each page and the detected regions are attached
            to each page.
            Doing so will slightly deteriorate the overall latency.
        layout_arch: name of the layout architecture or the model itself to use.
        kwargs: keyword args of `OCRPredictor`

    Returns:
        KIE predictor
    """
    return _kie_predictor(
        det_arch,
        reco_arch,
        pretrained,
        pretrained_backbone=pretrained_backbone,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
        export_as_straight_boxes=export_as_straight_boxes,
        detect_orientation=detect_orientation,
        straighten_pages=straighten_pages,
        detect_language=detect_language,
        detect_layout=detect_layout,
        layout_arch=layout_arch,
        **kwargs,
    )
