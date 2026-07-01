# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

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
        layout_predictor: optional layout detection module
        table_predictor: optional table structure recognition module. Requires `layout_predictor`: table
            regions are located by the layout model, cropped, and passed to this module. Words falling inside a
            detected table are regrouped into a structured table and removed from the regular text output.
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
        pages: list[np.ndarray],
        **kwargs: Any,
    ) -> Document:
        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        origin_page_shapes = [page.shape[:2] for page in pages]

        if not self.straighten_pages:
            # Detect layout regions on the pages
            regions = self.layout_predictor(pages, **kwargs) if self.layout_predictor is not None else None

        # Localize text elements
        loc_preds, out_maps = self.det_predictor(pages, return_maps=True, **kwargs)

        # Detect document rotation and rotate pages
        seg_maps = [
            np.where(out_map > getattr(self.det_predictor.model.postprocessor, "bin_thresh"), 255, 0).astype(np.uint8)
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

            # Detect layout regions on the pages
            regions = self.layout_predictor(pages, **kwargs) if self.layout_predictor is not None else None

            # Forward again to get predictions on straight pages
            loc_preds = self.det_predictor(pages, **kwargs)

        assert all(len(loc_pred) == 1 for loc_pred in loc_preds), (
            "Detection Model in ocr_predictor should output only one class"
        )

        loc_preds = [list(loc_pred.values())[0] for loc_pred in loc_preds]
        # Detach objectness scores from loc_preds
        loc_preds, objectness_scores = detach_scores(loc_preds)

        # Apply hooks to loc_preds if any
        for hook in self.hooks:
            loc_preds = hook(loc_preds)

        # Crop images
        crops, loc_preds = self._prepare_crops(
            pages,
            loc_preds,
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
            self._tables_from_regions(pages, regions, **kwargs)
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
        return out

    def _tables_from_regions(
        self,
        pages: list[np.ndarray],
        regions: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """Crop the table regions found by the layout model and run the table model on each crop.

        The table model is applied per cropped region, so a page naturally yields one structured table per
        detected `Table` region. Cell geometries are mapped back from crop-relative to page-relative coordinates.

        Args:
            pages: the (possibly straightened) page images
            regions: the per-page layout predictions `{"class_names", "boxes", "scores"}`
            **kwargs: keyword arguments forwarded to the table predictor

        Returns:
            a per-page list of table grids `{"cells": [...], "num_rows": int, "num_cols": int}` in page-relative
            coordinates
        """
        crops: list[np.ndarray] = []
        crop_meta: list[tuple[int, tuple[float, float, float, float]]] = []
        for p_idx, (page, region) in enumerate(zip(pages, regions)):
            if region is None:
                continue
            h, w = page.shape[:2]
            for cls_name, box in zip(region["class_names"], region["boxes"]):
                if cls_name != self.table_class_name:
                    continue
                pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
                x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
                x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
                # Relative box -> pixel crop (axis-aligned, clamped to the page)
                px0, py0 = max(0, int(round(x0 * w))), max(0, int(round(y0 * h)))
                px1, py1 = min(w, int(round(x1 * w))), min(h, int(round(y1 * h)))
                if px1 - px0 < 2 or py1 - py0 < 2:
                    continue
                crops.append(page[py0:py1, px0:px1])
                crop_meta.append((p_idx, (x0, y0, x1, y1)))

        tables_per_page: list[list[dict[str, Any]]] = [[] for _ in pages]
        if len(crops) == 0:
            return tables_per_page

        grids = self.table_predictor(crops, **kwargs)  # type: ignore[misc]
        for (p_idx, (x0, y0, x1, y1)), grid in zip(crop_meta, grids):
            region_w, region_h = (x1 - x0), (y1 - y0)
            remapped_cells: list[dict[str, Any]] = []
            for cell in grid["cells"]:
                poly = np.asarray(cell["geometry"], dtype=np.float32).reshape(-1, 2)
                poly[:, 0] = x0 + poly[:, 0] * region_w
                poly[:, 1] = y0 + poly[:, 1] * region_h
                new_cell = dict(cell)
                new_cell["geometry"] = poly.tolist()
                remapped_cells.append(new_cell)
            tables_per_page[p_idx].append({
                "cells": remapped_cells,
                "num_rows": grid["num_rows"],
                "num_cols": grid["num_cols"],
            })
        return tables_per_page
