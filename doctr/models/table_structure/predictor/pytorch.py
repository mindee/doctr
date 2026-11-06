# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from doctr.models._utils import predict_by_chunks
from doctr.models.detection._utils import _remove_padding
from doctr.models.preprocessor import PreProcessor
from doctr.models.utils import set_device_and_dtype

__all__ = ["TablePredictor"]


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
