# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
from typing import List, Tuple

from doctr.models.builder import DocumentBuilder
from doctr.utils.geometry import rotate_image, rotate_boxes
from .._utils import extract_crops, extract_rcrops


__all__ = ['_OCRPredictor']


class _OCRPredictor:
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        det_predictor: detection module
        reco_predictor: recognition module
    """

    doc_builder: DocumentBuilder

    @staticmethod
    def _generate_crops(
        pages: List[np.ndarray],
        loc_preds: List[Tuple[np.ndarray, float]],
        channels_last: bool,
        allow_rotated_boxes: bool = False
    ) -> List[List[np.ndarray]]:

        if allow_rotated_boxes:
            crops = [
                extract_rcrops(rotate_image(page, -angle, False), _boxes[:, :-1], channels_last=channels_last)
                for page, (_boxes, angle) in zip(pages, loc_preds)
            ]
        else:
            crops = [
                extract_crops(page, _boxes[:, :-1], channels_last=channels_last)
                for page, (_boxes, _) in zip(pages, loc_preds)
            ]

        return crops

    def _prepare_crops(
        self,
        pages: List[np.ndarray],
        loc_preds: List[Tuple[np.ndarray, float]],
        channels_last: bool,
    ) -> Tuple[List[List[np.ndarray]], List[Tuple[np.ndarray, float]]]:

        crops = self._generate_crops(pages, loc_preds, channels_last, self.doc_builder.rotated_bbox)

        # Avoid sending zero-sized crops
        is_kept = [[all(s > 0 for s in crop.shape) for crop in page_crops] for page_crops in crops]
        crops = [
            [crop for crop, _kept in zip(page_crops, page_kept) if _kept]
            for page_crops, page_kept in zip(crops, is_kept)
        ]
        loc_preds = [(_boxes[_kept], angle) for (_boxes, angle), _kept in zip(loc_preds, is_kept)]

        return crops, loc_preds

    @staticmethod
    def _process_predictions(
        loc_preds: List[Tuple[np.ndarray, float]],
        word_preds: List[Tuple[str, float]],
        allow_rotated_boxes: bool = False
    ) -> Tuple[List[np.ndarray], List[List[Tuple[str, float]]]]:

        boxes, text_preds = [], []
        if len(loc_preds) > 0:
            # Localization
            boxes, angles = zip(*loc_preds)
            # Rotate back boxes if necessary
            if allow_rotated_boxes:
                boxes = [rotate_boxes(page_boxes, angle) for page_boxes, angle in zip(boxes, angles)]
            # Text
            _idx = 0
            for page_boxes in boxes:
                text_preds.append(word_preds[_idx: _idx + page_boxes.shape[0]])
                _idx += page_boxes.shape[0]

        return boxes, text_preds
