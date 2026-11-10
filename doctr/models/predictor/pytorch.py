# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from doctr.io.elements import Document
from doctr.models._utils import get_language
from doctr.models.detection.predictor import DetectionPredictor
from doctr.models.layout.predictor import LayoutPredictor
from doctr.models.recognition.predictor import RecognitionPredictor
from doctr.models.table_structure.predictor import TablePredictor
from doctr.utils.geometry import detach_scores

from .base import _OCRPredictor

__all__ = ["OCRPredictor"]


class OCRPredictor(nn.Module, _OCRPredictor):
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
        preserve_original_coords: if True and straighten_pages is True, the returned bounding boxes are mapped back
            to the original page coordinate space. Useful for redaction and annotation.
        layout_predictor: optional layout detection module
        table_predictor: optional table structure recognition module. Requires `layout_predictor`: table
            regions are located by the layout model, cropped, and passed to this module. Words falling inside a
            detected table are regrouped into a structured table and removed from the regular text output.
        page_chunk_size: the maximum number of pages processed at once (32 by default) for lazy page sources
            (cf. `DocumentFile.from_pdf(..., lazy=True)`), iterators and :meth:`stream`, with a memory footprint that
            does not grow with the number of pages. Lists and tuples of pages are processed in a single pass.
        **kwargs: keyword args of `DocumentBuilder`
    """

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
        preserve_original_coords: bool = False,
        layout_predictor: LayoutPredictor | None = None,
        table_predictor: TablePredictor | None = None,
        **kwargs: Any,
    ) -> None:
        nn.Module.__init__(self)
        self.det_predictor = det_predictor.eval()
        self.reco_predictor = reco_predictor.eval()
        _OCRPredictor.__init__(
            self,
            assume_straight_pages,
            straighten_pages,
            preserve_aspect_ratio,
            symmetric_pad,
            detect_orientation,
            preserve_original_coords=preserve_original_coords,
            **kwargs,
        )
        self.detect_orientation = detect_orientation
        self.detect_language = detect_language
        self.layout_predictor = layout_predictor.eval() if layout_predictor is not None else None
        self.table_predictor = table_predictor.eval() if table_predictor is not None else None
        # Layout class label whose regions are cropped and passed to the table model
        self.table_class_name = "Table"
        if self.table_predictor is not None and self.layout_predictor is None:
            raise ValueError(
                "`table_predictor` requires a `layout_predictor`: tables are located with the layout model, "
                "cropped, and then passed to the table model."
            )

    @torch.inference_mode()
    def forward(
        self,
        pages: Sequence[np.ndarray] | Iterable[np.ndarray],
        **kwargs: Any,
    ) -> Document:
        """Predict on a document

        Args:
            pages: a list of pages (numpy arrays of shape H x W x C), processed in a single pass, or a lazy page
                source (cf. `DocumentFile.from_pdf(..., lazy=True)`) or any iterable of pages, processed chunk by
                chunk (`page_chunk_size` pages at a time) with a bounded memory footprint
            **kwargs: keyword arguments forwarded to the sub-predictors, and `on_error` for lazy page sources
                (cf. :meth:`stream`): "skip" leaves out the pages that cannot be read instead of raising an error

        Returns:
            the predicted document
        """
        return self._run(pages, **kwargs)

    @torch.inference_mode()
    def _predict_pages(
        self,
        pages: list[np.ndarray],
        **kwargs: Any,
    ) -> Document:
        if len(pages) == 0:
            return Document(pages=[])

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        origin_page_shapes = [page.shape[:2] for page in pages]

        if not self.straighten_pages:
            # Detect layout regions on the pages
            regions = self.layout_predictor(pages, **kwargs) if self.layout_predictor is not None else None
            det_pages = self._mask_regions(pages, regions)
        else:
            det_pages = pages

        # Localize text elements (segmentation maps are only materialized when actually consumed)
        if self.detect_orientation or self.straighten_pages:
            loc_preds, out_maps = self.det_predictor(det_pages, return_maps=True, **kwargs)
            bin_thresh = getattr(self.det_predictor.model.postprocessor, "bin_thresh")
            seg_maps = [((out_map > bin_thresh).astype(np.uint8) * 255) for out_map in out_maps]
        else:
            loc_preds = self.det_predictor(det_pages, **kwargs)

        # Detect document rotation and rotate pages
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
            _orig_shapes = origin_page_shapes
            _orig_pages = list(pages)
            pages, m_invs = self._straighten_pages(
                pages, seg_maps, general_pages_orientations, origin_pages_orientations
            )
            # update page shapes after straightening
            origin_page_shapes = [page.shape[:2] for page in pages]
            _straight_shapes = origin_page_shapes

            # Detect layout regions on the pages
            regions = self.layout_predictor(pages, **kwargs) if self.layout_predictor is not None else None
            det_pages = self._mask_regions(pages, regions)

            # Forward again to get predictions on straight pages
            loc_preds = self.det_predictor(det_pages, **kwargs)

        assert all(len(loc_pred) == 1 for loc_pred in loc_preds), (
            "Detection Model in ocr_predictor should output only one class"
        )

        loc_preds = [list(loc_pred.values())[0] for loc_pred in loc_preds]
        # Detach objectness scores from loc_preds
        loc_preds, objectness_scores = detach_scores(loc_preds)

        # Apply hooks to loc_preds if any, before the crops are extracted so that the hooked boxes are read
        loc_preds, objectness_scores = self._apply_hooks(loc_preds, objectness_scores)

        # Crop images
        crops, loc_preds, objectness_scores = self._prepare_crops(
            pages,
            loc_preds,
            objectness_scores,
            assume_straight_pages=self.assume_straight_pages,
            assume_horizontal=self._page_orientation_disabled,
        )

        # Rectify crop orientation and get crop orientation predictions
        crop_orientations: Any = []
        if not self.assume_straight_pages:
            crops, loc_preds, _crop_orientations = self._rectify_crops(crops, loc_preds)
            crop_orientations = [
                {"value": orientation[0], "confidence": orientation[1]} for orientation in _crop_orientations
            ]

        # Identify character sequences
        word_preds = self.reco_predictor([crop for page_crops in crops for crop in page_crops], **kwargs)
        if not crop_orientations:
            crop_orientations = [{"value": 0, "confidence": None} for _ in word_preds]

        boxes, text_preds, crop_orientations = self._process_predictions(loc_preds, word_preds, crop_orientations)

        if self.detect_language:
            languages = [get_language(" ".join([item[0] for item in text_pred])) for text_pred in text_preds]
            languages_dict = [{"value": lang[0], "confidence": lang[1]} for lang in languages]
        else:
            languages_dict = None

        # Recognize table structure: locate tables with the layout model, crop each one and run the table model
        tables = (
            self.table_predictor.predict_regions(pages, regions, boxes, self.table_class_name, **kwargs)
            if self.table_predictor is not None and regions is not None
            else None
        )

        out = self.doc_builder(
            pages,
            boxes,
            objectness_scores,
            text_preds,
            origin_page_shapes,
            crop_orientations,
            orientations,
            languages_dict,
            regions,
            tables,
        )

        if self.straighten_pages and self.preserve_original_coords:
            out = self._remap_to_original_coords(out, _orig_shapes, _straight_shapes, m_invs, _orig_pages)
        return out
