# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Any, List, Optional, Tuple

import numpy as np

from doctr.models.builder import DocumentBuilder

from .._utils import extract_crops, extract_rcrops, rectify_crops, rectify_loc_preds
from ..classification import crop_orientation_predictor
from ..classification.predictor import CropOrientationPredictor

__all__ = ['_OCRPredictor']


class _OCRPredictor:
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        straighten_pages: if True, estimates the page general orientation based on the median line orientation.
            Then, rotates page before passing it to the deep learning modules. The final predictions will be remapped
            accordingly. Doing so will improve performances for documents with page-uniform rotations.
        kwargs: keyword args of `DocumentBuilder`
    """

    crop_orientation_predictor: Optional[CropOrientationPredictor]

    def __init__(
        self,
        assume_straight_pages: bool = True,
        straighten_pages: bool = False,
        **kwargs: Any,
    ) -> None:
        self.assume_straight_pages = assume_straight_pages
        self.straighten_pages = straighten_pages
        self.crop_orientation_predictor = None if assume_straight_pages else crop_orientation_predictor(pretrained=True)
        self.doc_builder = DocumentBuilder(**kwargs)

    @staticmethod
    def _generate_crops(
        pages: List[np.ndarray],
        loc_preds: List[np.ndarray],
        channels_last: bool,
        assume_straight_pages: bool = False,
    ) -> List[List[np.ndarray]]:

        extraction_fn = extract_crops if assume_straight_pages else extract_rcrops

        crops = [
            extraction_fn(page, _boxes[:, :4], channels_last=channels_last)  # type: ignore[operator]
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
        orientations = [self.crop_orientation_predictor(page_crops) for page_crops in crops]  # type: ignore[misc]
        rect_crops = [rectify_crops(page_crops, orientation) for page_crops, orientation in zip(crops, orientations)]
        rect_loc_preds = [
            rectify_loc_preds(page_loc_preds, orientation) if len(page_loc_preds) > 0 else page_loc_preds
            for page_loc_preds, orientation in zip(loc_preds, orientations)
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
                text_preds.append(word_preds[_idx: _idx + page_boxes.shape[0]])
                _idx += page_boxes.shape[0]

        return loc_preds, text_preds
