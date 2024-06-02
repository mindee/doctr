# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from doctr.models.builder import DocumentBuilder
from doctr.utils.geometry import extract_crops, extract_rcrops

from .._utils import rectify_crops, rectify_loc_preds
from ..classification import crop_orientation_predictor
from ..classification.predictor import OrientationPredictor

__all__ = ["_OCRPredictor"]


class _OCRPredictor:
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
    ----
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        straighten_pages: if True, estimates the page general orientation based on the median line orientation.
            Then, rotates page before passing it to the deep learning modules. The final predictions will be remapped
            accordingly. Doing so will improve performances for documents with page-uniform rotations.
        preserve_aspect_ratio: if True, resize preserving the aspect ratio (with padding)
        symmetric_pad: if True and preserve_aspect_ratio is True, pas the image symmetrically.
        **kwargs: keyword args of `DocumentBuilder`
    """

    crop_orientation_predictor: Optional[OrientationPredictor]

    def __init__(
        self,
        assume_straight_pages: bool = True,
        straighten_pages: bool = False,
        preserve_aspect_ratio: bool = True,
        symmetric_pad: bool = True,
        **kwargs: Any,
    ) -> None:
        self.assume_straight_pages = assume_straight_pages
        self.straighten_pages = straighten_pages
        self.crop_orientation_predictor = None if assume_straight_pages else crop_orientation_predictor(pretrained=True)
        self.doc_builder = DocumentBuilder(**kwargs)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.hooks: List[Callable] = []

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
    ) -> Tuple[List[List[np.ndarray]], List[np.ndarray], List[Tuple[int, float]]]:
        # Work at a page level
        orientations, classes, probs = zip(*[self.crop_orientation_predictor(page_crops) for page_crops in crops])  # type: ignore[misc]
        rect_crops = [rectify_crops(page_crops, orientation) for page_crops, orientation in zip(crops, orientations)]
        rect_loc_preds = [
            rectify_loc_preds(page_loc_preds, orientation) if len(page_loc_preds) > 0 else page_loc_preds
            for page_loc_preds, orientation in zip(loc_preds, orientations)
        ]
        # Flatten to list of tuples with (value, confidence)
        crop_orientations = [
            (orientation, prob)
            for page_classes, page_probs in zip(classes, probs)
            for orientation, prob in zip(page_classes, page_probs)
        ]
        return rect_crops, rect_loc_preds, crop_orientations  # type: ignore[return-value]

    @staticmethod
    def _process_predictions(
        loc_preds: List[np.ndarray],
        word_preds: List[Tuple[str, float]],
        crop_orientations: List[Dict[str, Any]],
    ) -> Tuple[List[np.ndarray], List[List[Tuple[str, float]]], List[List[Dict[str, Any]]]]:
        text_preds = []
        crop_orientation_preds = []
        if len(loc_preds) > 0:
            # Text & crop orientation predictions at page level
            _idx = 0
            for page_boxes in loc_preds:
                text_preds.append(word_preds[_idx : _idx + page_boxes.shape[0]])
                crop_orientation_preds.append(crop_orientations[_idx : _idx + page_boxes.shape[0]])
                _idx += page_boxes.shape[0]

        return loc_preds, text_preds, crop_orientation_preds

    def add_hook(self, hook: Callable) -> None:
        """Add a hook to the predictor

        Args:
        ----
            hook: a callable that takes as input the `loc_preds` and returns the modified `loc_preds`
        """
        self.hooks.append(hook)
