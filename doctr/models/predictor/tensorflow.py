# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import tensorflow as tf
from typing import List, Any, Union

from doctr.io.elements import Document
from doctr.utils.repr import NestedObject
from doctr.models.preprocessor import PreProcessor
from doctr.models.builder import DocumentBuilder
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.utils.geometry import rotate_image, rotate_boxes
from .._utils import extract_crops, extract_rcrops


__all__ = ['OCRPredictor']


class OCRPredictor(NestedObject):
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        det_predictor: detection module
        reco_predictor: recognition module
    """

    def __init__(
        self,
        det_predictor: DetectionPredictor,
        reco_predictor: RecognitionPredictor,
        rotated_bbox: bool = False
    ) -> None:

        super().__init__()
        self.det_predictor = det_predictor
        self.reco_predictor = reco_predictor
        self.doc_builder = DocumentBuilder(rotated_bbox=rotated_bbox)
        self.extract_crops_fn = extract_rcrops if rotated_bbox else extract_crops

    def __call__(
        self,
        pages: List[Union[np.ndarray, tf.Tensor]],
        **kwargs: Any,
    ) -> Document:

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        # Localize text elements
        boxes = self.det_predictor(pages, **kwargs)
        # Crop images, rotate page if necessary
        if self.doc_builder.rotated_bbox:
            crops = [
                crop for page, (_boxes, angle) in zip(pages, boxes) for crop in
                self.extract_crops_fn(  # type: ignore[operator]
                    rotate_image(page, -angle, False),
                    _boxes[:, :-1],
                )
            ]
        else:
            crops = [crop for page, (_boxes, _) in zip(pages, boxes) for crop in
                     self.extract_crops_fn(page, _boxes[:, :-1])]  # type: ignore[operator]
        # Avoid sending zero-sized crops
        is_kept = [all(s > 0 for s in crop.shape) for crop in crops]
        crops = [crop for crop, _kept in zip(crops, is_kept) if _kept]
        boxes = [box for box, _kept in zip(boxes, is_kept) if _kept]
        # Identify character sequences
        word_preds = self.reco_predictor(crops, **kwargs)

        # Rotate back boxes if necessary
        if len(boxes) > 0:
            boxes, angles = zip(*boxes)
            if self.doc_builder.rotated_bbox:
                boxes = [rotate_boxes(boxes_page, angle) for boxes_page, angle in zip(boxes, angles)]
        out = self.doc_builder(boxes, word_preds, [page.shape[:2] for page in pages])  # type: ignore[misc]
        return out
