# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

import numpy as np
import tensorflow as tf

from doctr.io.elements import Document
from doctr.models._utils import get_language, invert_data_structure
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.utils.geometry import detach_scores
from doctr.utils.repr import NestedObject

from .base import _KIEPredictor

__all__ = ["KIEPredictor"]


class KIEPredictor(NestedObject, _KIEPredictor):
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
        **kwargs: keyword args of `DocumentBuilder`
    """

    _children_names = ["det_predictor", "reco_predictor", "doc_builder"]

    def __init__(
        self,
        det_predictor: DetectionPredictor,
        reco_predictor: RecognitionPredictor,
        assume_straight_pages: bool = True,
        straighten_pages: bool = False,
        preserve_aspect_ratio: bool = True,
        symmetric_pad: bool = True,
        detect_orientation: bool = False,
        detect_language: bool = False,
        **kwargs: Any,
    ) -> None:
        self.det_predictor = det_predictor
        self.reco_predictor = reco_predictor
        _KIEPredictor.__init__(
            self,
            assume_straight_pages,
            straighten_pages,
            preserve_aspect_ratio,
            symmetric_pad,
            detect_orientation,
            **kwargs,
        )
        self.detect_orientation = detect_orientation
        self.detect_language = detect_language

    def __call__(
        self,
        pages: list[np.ndarray | tf.Tensor],
        **kwargs: Any,
    ) -> Document:
        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        origin_page_shapes = [page.shape[:2] for page in pages]

        # Localize text elements
        loc_preds, out_maps = self.det_predictor(pages, return_maps=True, **kwargs)

        # Detect document rotation and rotate pages
        seg_maps = [
            np.where(np.expand_dims(np.amax(out_map, axis=-1), axis=-1) > kwargs.get("bin_thresh", 0.3), 255, 0).astype(
                np.uint8
            )
            for out_map in out_maps
        ]
        if self.detect_orientation:
            general_pages_orientations, origin_pages_orientations = self._get_orientations(pages, seg_maps)
            orientations = [
                {"value": orientation_page, "confidence": None} for orientation_page in origin_pages_orientations
            ]
        else:
            orientations = None
            general_pages_orientations = None
            origin_pages_orientations = None
        if self.straighten_pages:
            pages = self._straighten_pages(pages, seg_maps, general_pages_orientations, origin_pages_orientations)
            # update page shapes after straightening
            origin_page_shapes = [page.shape[:2] for page in pages]

            # Forward again to get predictions on straight pages
            loc_preds = self.det_predictor(pages, **kwargs)

        dict_loc_preds: dict[str, list[np.ndarray]] = invert_data_structure(loc_preds)  # type: ignore

        # Detach objectness scores from loc_preds
        objectness_scores = {}
        for class_name, det_preds in dict_loc_preds.items():
            _loc_preds, _scores = detach_scores(det_preds)
            dict_loc_preds[class_name] = _loc_preds
            objectness_scores[class_name] = _scores

        # Apply hooks to loc_preds if any
        for hook in self.hooks:
            dict_loc_preds = hook(dict_loc_preds)

        # Crop images
        crops = {}
        for class_name in dict_loc_preds.keys():
            crops[class_name], dict_loc_preds[class_name] = self._prepare_crops(
                pages,
                dict_loc_preds[class_name],
                channels_last=True,
                assume_straight_pages=self.assume_straight_pages,
                assume_horizontal=self._page_orientation_disabled,
            )

        # Rectify crop orientation
        crop_orientations: Any = {}
        if not self.assume_straight_pages:
            for class_name in dict_loc_preds.keys():
                crops[class_name], dict_loc_preds[class_name], word_orientations = self._rectify_crops(
                    crops[class_name], dict_loc_preds[class_name]
                )
                crop_orientations[class_name] = [
                    {"value": orientation[0], "confidence": orientation[1]} for orientation in word_orientations
                ]

        # Identify character sequences
        word_preds = {
            k: self.reco_predictor([crop for page_crops in crop_value for crop in page_crops], **kwargs)
            for k, crop_value in crops.items()
        }
        if not crop_orientations:
            crop_orientations = {k: [{"value": 0, "confidence": None} for _ in word_preds[k]] for k in word_preds}

        boxes: dict = {}
        text_preds: dict = {}
        word_crop_orientations: dict = {}
        for class_name in dict_loc_preds.keys():
            boxes[class_name], text_preds[class_name], word_crop_orientations[class_name] = self._process_predictions(
                dict_loc_preds[class_name], word_preds[class_name], crop_orientations[class_name]
            )

        boxes_per_page: list[dict] = invert_data_structure(boxes)  # type: ignore[assignment]
        objectness_scores_per_page: list[dict] = invert_data_structure(objectness_scores)  # type: ignore[assignment]
        text_preds_per_page: list[dict] = invert_data_structure(text_preds)  # type: ignore[assignment]
        crop_orientations_per_page: list[dict] = invert_data_structure(word_crop_orientations)  # type: ignore[assignment]

        if self.detect_language:
            languages = [get_language(self.get_text(text_pred)) for text_pred in text_preds_per_page]
            languages_dict = [{"value": lang[0], "confidence": lang[1]} for lang in languages]
        else:
            languages_dict = None

        out = self.doc_builder(
            pages,
            boxes_per_page,
            objectness_scores_per_page,
            text_preds_per_page,
            origin_page_shapes,  # type: ignore[arg-type]
            crop_orientations_per_page,
            orientations,
            languages_dict,
        )
        return out

    @staticmethod
    def get_text(text_pred: dict) -> str:
        text = []
        for value in text_pred.values():
            text += [item[0] for item in value]

        return " ".join(text)
