# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List, Tuple

import numpy as np

from doctr.models.builder import DocumentBuilder

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
        loc_preds: List[np.ndarray],
        channels_last: bool,
        assume_straight_pages: bool = False,
    ) -> List[List[np.ndarray]]:

        extraction_fn = extract_crops if assume_straight_pages else extract_rcrops

        crops = [
            extraction_fn(
                page, _boxes[:, :-1] if assume_straight_pages else _boxes, channels_last=channels_last
            )  # type: ignore[operator]
            for page, _boxes in zip(pages, loc_preds)
        ]
        return crops

    @staticmethod
    def _prepare_crops(
        pages: List[np.ndarray],
        loc_preds: List[np.ndarray],
        channels_last: bool,
        assume_straight_pages: bool = False,
    ) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:

        crops = _OCRPredictor._generate_crops(pages, loc_preds, channels_last, assume_straight_pages)

        # Avoid sending zero-sized crops
        is_kept = [[all(s > 0 for s in crop.shape) for crop in page_crops] for page_crops in crops]
        crops = [
            [crop for crop, _kept in zip(page_crops, page_kept) if _kept]
            for page_crops, page_kept in zip(crops, is_kept)
        ]
        loc_preds = [_boxes[_kept] for _boxes, _kept in zip(loc_preds, is_kept)]

        return crops, loc_preds

    @staticmethod
    def _process_predictions(
        loc_preds: List[np.ndarray],
        word_preds: List[Tuple[str, float]],
    ) -> Tuple[List[np.ndarray], List[List[Tuple[str, float]]]]:

        text_preds = []
        if len(loc_preds) > 0:
            # Text
            _idx = 0
            for page_boxes in loc_preds:
                text_preds.append(word_preds[_idx: _idx + page_boxes.shape[0]])
                _idx += page_boxes.shape[0]

        return loc_preds, text_preds
