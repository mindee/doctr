# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, List, Union

import numpy as np
import tensorflow as tf

from doctr.io.elements import Document
from doctr.models._utils import estimate_orientation, get_language
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.utils.geometry import rotate_boxes, rotate_image
from doctr.utils.repr import NestedObject

from .base import _OCRPredictor

__all__ = ["OCRPredictor"]


class OCRPredictor(NestedObject, _OCRPredictor):
    """Implements an object able to localize and identify text elements in a set of documents

    Args:
        det_predictor: detection module
        reco_predictor: recognition module
        assume_straight_pages: if True, speeds up the inference by assuming you only pass straight pages
            without rotated textual elements.
        straighten_pages: if True, estimates the page general orientation based on the median line orientation.
            Then, rotates page before passing it to the deep learning modules. The final predictions will be remapped
            accordingly. Doing so will improve performances for documents with page-uniform rotations.
        detect_orientation: if True, the estimated general page orientation will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        detect_language: if True, the language prediction will be added to the predictions for each
            page. Doing so will slightly deteriorate the overall latency.
        kwargs: keyword args of `DocumentBuilder`
    """

    _children_names = ["det_predictor", "reco_predictor", "doc_builder"]

    def __init__(
        self,
        det_predictor: DetectionPredictor,
        reco_predictor: RecognitionPredictor,
        assume_straight_pages: bool = True,
        straighten_pages: bool = False,
        preserve_aspect_ratio: bool = False,
        symmetric_pad: bool = True,
        detect_orientation: bool = False,
        detect_language: bool = False,
        **kwargs: Any,
    ) -> None:
        self.det_predictor = det_predictor
        self.reco_predictor = reco_predictor
        _OCRPredictor.__init__(
            self, assume_straight_pages, straighten_pages, preserve_aspect_ratio, symmetric_pad, **kwargs
        )
        self.detect_orientation = detect_orientation
        self.detect_language = detect_language

    def __call__(
        self,
        pages: List[Union[np.ndarray, tf.Tensor]],
        **kwargs: Any,
    ) -> Document:
        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        origin_page_shapes = [page.shape[:2] for page in pages]

        # Detect document rotation and rotate pages
        if self.detect_orientation:
            origin_page_orientations = [estimate_orientation(page) for page in pages]
            orientations = [
                {"value": orientation_page, "confidence": 1.0} for orientation_page in origin_page_orientations
            ]
        else:
            orientations = None
        if self.straighten_pages:
            origin_page_orientations = (
                origin_page_orientations if self.detect_orientation else [estimate_orientation(page) for page in pages]
            )
            pages = [rotate_image(page, -angle, expand=True) for page, angle in zip(pages, origin_page_orientations)]

        # Localize text elements
        loc_preds_dict = self.det_predictor(pages, **kwargs)
        assert all(
            len(loc_pred) == 1 for loc_pred in loc_preds_dict
        ), "Detection Model in ocr_predictor should output only one class"

        loc_preds: List[np.ndarray] = [list(loc_pred.values())[0] for loc_pred in loc_preds_dict]

        # Rectify crops if aspect ratio
        loc_preds = self._remove_padding(pages, loc_preds)

        # Crop images
        crops, loc_preds = self._prepare_crops(
            pages, loc_preds, channels_last=True, assume_straight_pages=self.assume_straight_pages
        )
        # Rectify crop orientation
        if not self.assume_straight_pages:
            crops, loc_preds = self._rectify_crops(crops, loc_preds)

        # Identify character sequences
        word_preds = self.reco_predictor([crop for page_crops in crops for crop in page_crops], **kwargs)

        boxes, text_preds = self._process_predictions(loc_preds, word_preds)

        if self.detect_language:
            languages = [get_language(" ".join([item[0] for item in text_pred])) for text_pred in text_preds]
            languages_dict = [{"value": lang[0], "confidence": lang[1]} for lang in languages]
        else:
            languages_dict = None
        # Rotate back pages and boxes while keeping original image size
        if self.straighten_pages:
            boxes = [
                rotate_boxes(
                    page_boxes,
                    angle,
                    orig_shape=page.shape[:2] if isinstance(page, np.ndarray) else page.shape[-2:],
                    target_shape=mask,  # type: ignore[arg-type]
                )
                for page_boxes, page, angle, mask in zip(boxes, pages, origin_page_orientations, origin_page_shapes)
            ]

        out = self.doc_builder(
            boxes,
            text_preds,
            origin_page_shapes,  # type: ignore[arg-type]
            orientations,
            languages_dict,
        )
        return out
