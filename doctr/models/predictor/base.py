# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Callable
from typing import Any

import numpy as np

from doctr.models.builder import DocumentBuilder
from doctr.utils.geometry import extract_crops, extract_rcrops, remove_image_padding, rotate_image

from .._utils import estimate_orientation, rectify_crops, rectify_loc_preds
from ..classification import crop_orientation_predictor, page_orientation_predictor
from ..classification.predictor import OrientationPredictor

__all__ = ["_OCRPredictor"]


class _OCRPredictor:
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        straighten_pages: if True, estimates the page general orientation based on the median line orientation.
            Then, rotates page before passing it to the deep learning modules. The final predictions will be remapped
            accordingly. Doing so will improve performances for documents with page-uniform rotations.
        preserve_aspect_ratio: if True, resize preserving the aspect ratio (with padding)
        symmetric_pad: if True and preserve_aspect_ratio is True, pas the image symmetrically.
        detect_orientation: if True, the estimated general page orientation will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        **kwargs: keyword args of `DocumentBuilder`
    """

    crop_orientation_predictor: OrientationPredictor | None
    page_orientation_predictor: OrientationPredictor | None

    def __init__(
        self,
        assume_straight_pages: bool = True,
        straighten_pages: bool = False,
        preserve_aspect_ratio: bool = True,
        symmetric_pad: bool = True,
        detect_orientation: bool = False,
        **kwargs: Any,
    ) -> None:
        self.assume_straight_pages = assume_straight_pages
        self.straighten_pages = straighten_pages
        self._page_orientation_disabled = kwargs.pop("disable_page_orientation", False)
        self._crop_orientation_disabled = kwargs.pop("disable_crop_orientation", False)
        self.crop_orientation_predictor = (
            None
            if assume_straight_pages
            else crop_orientation_predictor(pretrained=True, disabled=self._crop_orientation_disabled)
        )
        self.page_orientation_predictor = (
            page_orientation_predictor(pretrained=True, disabled=self._page_orientation_disabled)
            if detect_orientation or straighten_pages or not assume_straight_pages
            else None
        )
        self.doc_builder = DocumentBuilder(**kwargs)
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.symmetric_pad = symmetric_pad
        self.hooks: list[Callable] = []

    def _general_page_orientations(
        self,
        pages: list[np.ndarray],
    ) -> list[tuple[int, float]]:
        _, classes, probs = zip(self.page_orientation_predictor(pages))  # type: ignore[misc]
        # Flatten to list of tuples with (value, confidence)
        page_orientations = [
            (orientation, prob)
            for page_classes, page_probs in zip(classes, probs)
            for orientation, prob in zip(page_classes, page_probs)
        ]
        return page_orientations

    def _get_orientations(
        self, pages: list[np.ndarray], seg_maps: list[np.ndarray]
    ) -> tuple[list[tuple[int, float]], list[int]]:
        general_pages_orientations = self._general_page_orientations(pages)
        origin_page_orientations = [
            estimate_orientation(seq_map, general_orientation)
            for seq_map, general_orientation in zip(seg_maps, general_pages_orientations)
        ]
        return general_pages_orientations, origin_page_orientations

    def _straighten_pages(
        self,
        pages: list[np.ndarray],
        seg_maps: list[np.ndarray],
        general_pages_orientations: list[tuple[int, float]] | None = None,
        origin_pages_orientations: list[int] | None = None,
    ) -> list[np.ndarray]:
        general_pages_orientations = (
            general_pages_orientations if general_pages_orientations else self._general_page_orientations(pages)
        )
        origin_pages_orientations = (
            origin_pages_orientations
            if origin_pages_orientations
            else [
                estimate_orientation(seq_map, general_orientation)
                for seq_map, general_orientation in zip(seg_maps, general_pages_orientations)
            ]
        )
        return [
            # expand if height and width are not equal, then remove the padding
            remove_image_padding(rotate_image(page, angle, expand=page.shape[0] != page.shape[1]))
            for page, angle in zip(pages, origin_pages_orientations)
        ]

    @staticmethod
    def _generate_crops(
        pages: list[np.ndarray],
        loc_preds: list[np.ndarray],
        channels_last: bool,
        assume_straight_pages: bool = False,
        assume_horizontal: bool = False,
    ) -> list[list[np.ndarray]]:
        if assume_straight_pages:
            crops = [
                extract_crops(page, _boxes[:, :4], channels_last=channels_last)
                for page, _boxes in zip(pages, loc_preds)
            ]
        else:
            crops = [
                extract_rcrops(page, _boxes[:, :4], channels_last=channels_last, assume_horizontal=assume_horizontal)
                for page, _boxes in zip(pages, loc_preds)
            ]
        return crops

    @staticmethod
    def _prepare_crops(
        pages: list[np.ndarray],
        loc_preds: list[np.ndarray],
        channels_last: bool,
        assume_straight_pages: bool = False,
        assume_horizontal: bool = False,
    ) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
        crops = _OCRPredictor._generate_crops(pages, loc_preds, channels_last, assume_straight_pages, assume_horizontal)

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
        crops: list[list[np.ndarray]],
        loc_preds: list[np.ndarray],
    ) -> tuple[list[list[np.ndarray]], list[np.ndarray], list[tuple[int, float]]]:
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
        loc_preds: list[np.ndarray],
        word_preds: list[tuple[str, float]],
        crop_orientations: list[dict[str, Any]],
    ) -> tuple[list[np.ndarray], list[list[tuple[str, float]]], list[list[dict[str, Any]]]]:
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
            hook: a callable that takes as input the `loc_preds` and returns the modified `loc_preds`
        """
        self.hooks.append(hook)
