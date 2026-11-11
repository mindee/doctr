# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Iterable, Sequence
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn

from doctr.models._utils import predict_by_chunks
from doctr.models.detection._utils import _remove_padding
from doctr.models.preprocessor import PreProcessor
from doctr.models.utils import set_device_and_dtype

from ..grid import as_polygons, fill_table_grid
from ..tablecenternet.base import _logic_from_geometry

__all__ = ["TablePredictor"]


def _crop_region(page: np.ndarray, region: Any) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Rectify a (page-relative) table region to an upright crop, with a margin of page context around it

    Returns:
        the crop, the perspective transform from page pixels to crop pixels, and the margin (pixels), None for a
        degenerate region
    """
    h, w = page.shape[:2]
    src = as_polygons([region])[0] * np.array([w, h], dtype=np.float32)
    crop_w = int(round(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3]))))
    crop_h = int(round(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1]))))
    if crop_w < 2 or crop_h < 2:
        return None
    pad = max(20, round(0.01 * max(crop_w, crop_h)))
    dst = np.array([[pad, pad], [pad + crop_w, pad], [pad + crop_w, pad + crop_h], [pad, pad + crop_h]], np.float32)
    transform = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(page, transform, (crop_w + 2 * pad, crop_h + 2 * pad), borderMode=cv2.BORDER_REPLICATE)
    return crop, transform, pad


class TablePredictor(nn.Module):
    """Implements an object able to recognize the cell structure of tables in a document.

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core table-structure-recognition architecture
        chunk_size: the maximum number of table images decoded and processed at once (32 by default). Longer
            inputs, lazy page sources (cf. `DocumentFile.from_images(..., lazy=True)`) and iterators are processed
            chunk by chunk, with a memory footprint that does not grow with the number of table images. It is rounded
            up to a multiple of the batch size, so that the predictions are the same as in a single pass.
    """

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: nn.Module,
        chunk_size: int = 32,
    ) -> None:
        super().__init__()
        self.pre_processor = pre_processor
        self.model = model.eval()
        if chunk_size < 1:
            raise ValueError("`chunk_size` must be a positive integer")
        self.chunk_size = chunk_size

    @torch.inference_mode()
    def forward(self, pages: Sequence[np.ndarray] | Iterable[np.ndarray], **kwargs: Any) -> list[dict[str, Any]]:
        """Recognize the cell structure of table images

        Args:
            pages: a list of table images (numpy arrays of shape H x W x C), a lazy page source
                (cf. `DocumentFile.from_images(..., lazy=True)`) or any iterable of images
            **kwargs: keyword arguments forwarded to the model

        Returns:
            the structure of each table: `{"cells", "num_rows", "num_cols"}`
        """
        return predict_by_chunks(
            lambda chunk: self._predict(chunk, **kwargs), pages, self.chunk_size, self.pre_processor.batch_size
        )

    @torch.inference_mode()
    def _predict(self, pages: list[np.ndarray], **kwargs: Any) -> list[dict[str, Any]]:
        if len(pages) == 0:
            return []
        # Extract parameters from the preprocessor
        preserve_aspect_ratio = self.pre_processor.resize.preserve_aspect_ratio
        symmetric_pad = self.pre_processor.resize.symmetric_pad
        assume_straight_pages = self.model.assume_straight_pages
        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(pages)
        _params = next(self.model.parameters())
        self.model, processed_batches = set_device_and_dtype(  # type: ignore[assignment]
            self.model, processed_batches, _params.device, _params.dtype
        )
        predicted_batches = [self.model(batch, return_preds=True, **kwargs) for batch in processed_batches]
        preds = [pred for batch in predicted_batches for pred in batch["preds"]]

        rectified = _remove_padding(
            pages,
            [{"polygons": pred["polygons"]} for pred in preds],
            preserve_aspect_ratio=preserve_aspect_ratio,
            symmetric_pad=symmetric_pad,
            assume_straight_pages=assume_straight_pages,  # type: ignore[arg-type]
        )

        results: list[dict[str, Any]] = []
        for pred, rect in zip(preds, rectified):
            polygons = rect["polygons"]  # * np.array([w, h], dtype=np.float32)  # relative -> absolute pixels
            scores, logical = pred["scores"], pred["logical"]
            cells, max_row, max_col = [], -1, -1
            for poly, score, lc in zip(polygons, scores, logical):
                start_col, end_col, start_row, end_row = (int(v) for v in lc)
                max_row, max_col = max(max_row, end_row), max(max_col, end_col)
                cells.append({
                    "geometry": poly.tolist(),  # 4 points (TL, TR, BR, BL) in relative coordinates
                    "score": float(score),
                    "row_start": start_row,
                    "row_end": end_row,
                    "col_start": start_col,
                    "col_end": end_col,
                })
            # logical coordinates are 0-indexed, so the table size is the largest index + 1 (0 if no cells)
            results.append({"cells": cells, "num_rows": max_row + 1, "num_cols": max_col + 1})
        return results

    @torch.inference_mode()
    def predict_regions(
        self,
        pages: list[np.ndarray],
        regions: list[dict[str, Any] | None],
        word_boxes: list[np.ndarray] | None = None,
        class_name: str = "Table",
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """Recognize the tables a layout model located on pages, and assign the words of the pages to their cells

        Each table region is rectified to an upright crop (so straight boxes and rotated polygons are both handled),
        with a margin of page context: the cells centered in this margin belong to the neighbouring content and are
        dropped. The words of the page lying in the region are assigned to the cells and complete the grid (cf.
        :func:`~doctr.models.table_structure.grid.fill_table_grid`), then the cells are mapped back to the page.

        Args:
            pages: the page images
            regions: the per-page layout predictions `{"class_names", "boxes", "scores"}` (None for none)
            word_boxes: the per-page word boxes, of shape (N, 4) or (N, 4, 2) in relative coordinates
            class_name: the layout class of the table regions
            **kwargs: keyword arguments forwarded to the model

        Returns:
            a per-page list of tables `{"cells": [...], "num_rows": int, "num_cols": int}` in page-relative
            coordinates, one per table region, each cell listing the indices of its words in the page (`"words"`) in
            reading order
        """
        crops, meta = [], []
        for p_idx, (page, region) in enumerate(zip(pages, regions)):
            region = region or {}
            for cls_name, box in zip(region.get("class_names", []), region.get("boxes", [])):
                cropped = _crop_region(page, box) if cls_name == class_name else None
                if cropped is not None:
                    crops.append(cropped[0])
                    meta.append((p_idx, cropped[1], cropped[2]))
        tables: list[list[dict[str, Any]]] = [[] for _ in pages]
        for (p_idx, transform, pad), crop, grid in zip(meta, crops, self(crops, **kwargs) if crops else []):
            words = None if word_boxes is None else word_boxes[p_idx]
            table = self._region_table(grid, crop.shape[:2], pages[p_idx].shape[:2], transform, pad, words)
            if table is not None:
                tables[p_idx].append(table)
        return tables

    def _region_table(
        self,
        grid: dict[str, Any],
        crop_shape: tuple[int, int],
        page_shape: tuple[int, int],
        transform: np.ndarray,
        pad: int,
        word_boxes: np.ndarray | None,
    ) -> dict[str, Any] | None:
        """Assemble the table of a region crop with the words of the page, in page-relative coordinates"""
        (crop_h, crop_w), (h, w) = crop_shape, page_shape
        if len(grid["cells"]) == 0:
            return None

        def in_region(points: np.ndarray) -> np.ndarray:
            return np.flatnonzero(np.all((points >= pad) & (points <= [crop_w - pad, crop_h - pad]), axis=1))

        # Cells in crop pixels, without the ones of the context margin
        polys = as_polygons([cell["geometry"] for cell in grid["cells"]]) * [crop_w, crop_h]
        keep = in_region(polys.mean(axis=1))
        if keep.size == 0:
            return None
        polys, cells = polys[keep], [grid["cells"][idx] for idx in keep]
        logic = np.array([[c["row_start"], c["row_end"], c["col_start"], c["col_end"]] for c in cells])
        geometric_logic = getattr(getattr(self.model, "postprocessor", None), "geometric_logic", False)
        if geometric_logic:
            # The dropped cells made grid lines of their own: index the rows / columns of the remaining ones again
            # (the logical coordinates of TableCenterNet are ordered (start_col, end_col, start_row, end_row))
            by_col = _logic_from_geometry(polys, logic[:, [2, 3, 0, 1]], split_tables=False)
            logic = by_col[:, [2, 3, 0, 1]]
        # The words of the region, in crop pixels
        word_idcs, words = np.zeros(0, dtype=np.int64), np.zeros((0, 4))
        if word_boxes is not None and len(word_boxes) > 0:
            pts = as_polygons(word_boxes) * np.array([w, h], dtype=np.float32)
            pts = cv2.perspectiveTransform(pts.reshape(1, -1, 2), transform).reshape(-1, 4, 2)
            word_idcs = in_region(pts.mean(axis=1))
            words = np.concatenate([pts[word_idcs].min(axis=1), pts[word_idcs].max(axis=1)], axis=1)
        boxes, cell_logic, cell_words, origin = fill_table_grid(
            np.concatenate([polys.min(axis=1), polys.max(axis=1)], axis=1), logic, words
        )
        # Crop pixels -> page-relative coordinates
        corners = boxes[:, [[0, 1], [2, 1], [2, 3], [0, 3]]].reshape(1, -1, 2).astype(np.float32)
        page_polys = cv2.perspectiveTransform(corners, np.linalg.inv(transform)).reshape(-1, 4, 2) / [w, h]
        return {
            "cells": [
                {
                    "geometry": poly.tolist(),
                    # The cells made from the words are not scored by the table model
                    "score": None if source < 0 else float(cells[source]["score"]),
                    "row_start": int(rs),
                    "row_end": int(re),
                    "col_start": int(cs),
                    "col_end": int(ce),
                    "words": [int(word_idcs[idx]) for idx in w_idcs],
                }
                for poly, (rs, re, cs, ce), w_idcs, source in zip(page_polys, cell_logic, cell_words, origin)
            ],
            "num_rows": int(cell_logic[:, 1].max()) + 1,
            "num_cols": int(cell_logic[:, 3].max()) + 1,
        }
