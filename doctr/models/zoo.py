# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Any

from .detection.zoo import detection_predictor
from .predictor import OCRPredictor
from .recognition.zoo import recognition_predictor

__all__ = ["ocr_predictor"]


def _predictor(
    det_arch: str,
    reco_arch: str,
    pretrained: bool,
    assume_straight_pages: bool = True,
    preserve_aspect_ratio: bool = False,
    symmetric_pad: bool = True,
    det_bs: int = 2,
    reco_bs: int = 128,
    **kwargs,
) -> OCRPredictor:

    # Detection
    det_predictor = detection_predictor(
        det_arch,
        pretrained=pretrained,
        batch_size=det_bs,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
    )

    # Recognition
    reco_predictor = recognition_predictor(reco_arch, pretrained=pretrained, batch_size=reco_bs)

    return OCRPredictor(
        det_predictor,
        reco_predictor,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
        **kwargs
    )


def ocr_predictor(
    det_arch: str = 'db_resnet50',
    reco_arch: str = 'crnn_vgg16_bn',
    pretrained: bool = False,
    assume_straight_pages: bool = True,
    preserve_aspect_ratio: bool = False,
    symmetric_pad: bool = True,
    export_as_straight_boxes: bool = False,
    **kwargs: Any
) -> OCRPredictor:
    """End-to-end OCR architecture using one model for localization, and another for text recognition.

    >>> import numpy as np
    >>> from doctr.models import ocr_predictor
    >>> model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)
    >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    >>> out = model([input_page])

    Args:
        det_arch: name of the detection architecture to use (e.g. 'db_resnet50', 'db_mobilenet_v3_large')
        reco_arch: name of the recognition architecture to use (e.g. 'crnn_vgg16_bn', 'sar_resnet31')
        pretrained: If True, returns a model pre-trained on our OCR dataset
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        preserve_aspect_ratio: If True, pad the input document image to preserve the aspect ratio before
            running the detection model on it.
        symmetric_pad: if True, pad the image symmetrically instead of padding at the bottom-right.
        export_as_straight_boxes: when assume_straight_pages is set to False, export final predictions
            (potentially rotated) as straight bounding boxes.
        kwargs: keyword args of `OCRPredictor`

    Returns:
        OCR predictor
    """

    return _predictor(
        det_arch,
        reco_arch,
        pretrained,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
        export_as_straight_boxes=export_as_straight_boxes,
        **kwargs,
    )
