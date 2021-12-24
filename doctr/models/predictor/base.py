# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List, Tuple

import numpy as np

from doctr.models.builder import DocumentBuilder

from .._utils import extract_crops, extract_rcrops, rectify_crops, rectify_loc_preds
from ..classification import crop_orientation_predictor

__all__ = ['_OCRPredictor']


class _OCRPredictor:
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        det_predictor: detection module
        reco_predictor: recognition module
    """

    doc_builder: DocumentBuilder

    def __init__(self) -> None:
        self.crop_orientation_predictor = crop_orientation_predictor(pretrained=True)

    @staticmethod
    def _generate_crops(
        pages: List[np.ndarray],
        loc_preds: List[np.ndarray],
        channels_last: bool,
        assume_straight_pages: bool = False,
    ) -> List[List[np.ndarray]]:

        extraction_fn = extract_crops if assume_straight_pages else extract_rcrops

        crops = [
            extraction_fn(  # type: ignore[operator]
                page, _boxes[:, :-1] if assume_straight_pages else _boxes, channels_last=channels_last
            )
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

    def _rectify_crops(
        self,
        crops: List[List[np.ndarray]],
        loc_preds: List[np.ndarray],
    ) -> Tuple[List[List[np.ndarray]], List[np.ndarray]]:
        # Work at a page level
        orientations = [self.crop_orientation_predictor(page_crops) for page_crops in crops]
        rect_crops = [rectify_crops(page_crops, orientation) for page_crops, orientation in zip(crops, orientations)]
        rect_loc_preds = [
            rectify_loc_preds(page_loc_preds, orientation) for page_loc_preds, orientation
            in zip(loc_preds, orientations)
        ]
        return rect_crops, rect_loc_preds

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
                if page_boxes is None:
                    text_preds.append(None)
                text_preds.append(word_preds[_idx: _idx + page_boxes.shape[0]])
                _idx += page_boxes.shape[0]

        return loc_preds, text_preds
