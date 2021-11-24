# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Any, List, Union

import numpy as np
import torch
from torch import nn

from doctr.io.elements import Document
from doctr.models._utils import estimate_orientation
from doctr.models.builder import DocumentBuilder
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.utils.geometry import rotate_boxes, rotate_image

from .base import _OCRPredictor

__all__ = ['OCRPredictor']


class OCRPredictor(nn.Module, _OCRPredictor):
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        det_predictor: detection module
        reco_predictor: recognition module
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        export_as_straight_boxes: when assume_straight_pages is set to False, export final predictions
            (potentially rotated) as straight bounding boxes.
        straighten_pages: if True, estimates the page general orientation based on the median line orientation.
            Then, rotates page before passing it to the deep learning modules. The final predictions will be remapped
            accordingly. Doing so will improve performances for documents with page-uniform rotations.

    """

    def __init__(
        self,
        det_predictor: DetectionPredictor,
        reco_predictor: RecognitionPredictor,
        assume_straight_pages: bool = True,
        export_as_straight_boxes: bool = False,
        straighten_pages: bool = False,
    ) -> None:

        super().__init__()
        self.det_predictor = det_predictor.eval()  # type: ignore[attr-defined]
        self.reco_predictor = reco_predictor.eval()  # type: ignore[attr-defined]
        self.doc_builder = DocumentBuilder(export_as_straight_boxes=export_as_straight_boxes)
        self.assume_straight_pages = assume_straight_pages
        self.straighten_pages = straighten_pages

    @torch.no_grad()
    def forward(
        self,
        pages: List[Union[np.ndarray, torch.Tensor]],
        **kwargs: Any,
    ) -> Document:

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        origin_page_shapes = [page.shape[:2] for page in pages]

        # Detect document rotation and rotate pages
        if self.straighten_pages:
            origin_page_orientations = [estimate_orientation(page) for page in pages]
            pages = [rotate_image(page, -angle, expand=True) for page, angle in zip(pages, origin_page_orientations)]

        # Localize text elements
        loc_preds = self.det_predictor(pages, **kwargs)
        # Check whether crop mode should be switched to channels first
        channels_last = len(pages) == 0 or isinstance(pages[0], np.ndarray)
        # Crop images
        crops, loc_preds = self._prepare_crops(
            pages, loc_preds, channels_last=channels_last, assume_straight_pages=self.assume_straight_pages
        )
        # Identify character sequences
        word_preds = self.reco_predictor([crop for page_crops in crops for crop in page_crops], **kwargs)

        boxes, text_preds = self._process_predictions(loc_preds, word_preds)

        # Rotate back pages and boxes while keeping original image size
        if self.straighten_pages:
            boxes = [rotate_boxes(page_boxes, angle, orig_shape=page.shape[:2], mask_shape=mask) for
                     page_boxes, page, angle, mask in zip(boxes, pages, origin_page_orientations, origin_page_shapes)]

        out = self.doc_builder(
            boxes,
            text_preds,
            [
                page.shape[:2] if channels_last else page.shape[-2:]  # type: ignore[misc]
                for page in pages
            ]
        )
        return out
