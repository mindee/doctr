# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Iterable, Sequence
from functools import partial
from typing import Any

import numpy as np
import torch
from torch import nn

from doctr.models._utils import merge_columns, merge_items, predict_by_chunks
from doctr.models.detection._utils import _remove_padding
from doctr.models.preprocessor import PreProcessor
from doctr.models.utils import set_device_and_dtype

__all__ = ["DetectionPredictor"]


class DetectionPredictor(nn.Module):
    """Implements an object able to localize text elements in a document

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
        chunk_size: the maximum number of pages decoded and processed at once (32 by default). Longer
            inputs, lazy page sources (cf. `DocumentFile.from_images(..., lazy=True)`) and iterators are processed
            chunk by chunk, with a memory footprint that does not grow with the number of pages. It is rounded
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
    def forward(
        self,
        pages: Sequence[np.ndarray] | Iterable[np.ndarray],
        return_maps: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, np.ndarray]] | tuple[list[dict[str, np.ndarray]], list[np.ndarray]]:
        """Localize the text elements of pages

        Args:
            pages: a list of pages (numpy arrays of shape H x W x C), a lazy page source
                (cf. `DocumentFile.from_pdf(..., lazy=True)`) or any iterable of pages
            return_maps: whether to also return the segmentation map of each page
            **kwargs: keyword arguments forwarded to the model

        Returns:
            the predictions of each page, and their segmentation maps if `return_maps` is True
        """
        return predict_by_chunks(
            partial(self._predict, return_maps=return_maps, **kwargs),
            pages,
            self.chunk_size,
            self.pre_processor.batch_size,
            merge=merge_columns if return_maps else merge_items,
        )

    @torch.inference_mode()
    def _predict(
        self,
        pages: list[np.ndarray],
        return_maps: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, np.ndarray]] | tuple[list[dict[str, np.ndarray]], list[np.ndarray]]:
        if len(pages) == 0:
            return ([], []) if return_maps else []

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
        predicted_batches = [
            self.model(batch, return_preds=True, return_model_output=return_maps, **kwargs)
            for batch in processed_batches
        ]
        # Remove padding from loc predictions
        preds = _remove_padding(
            pages,
            [pred for batch in predicted_batches for pred in batch["preds"]],
            preserve_aspect_ratio=preserve_aspect_ratio,
            symmetric_pad=symmetric_pad,
            assume_straight_pages=assume_straight_pages,  # type: ignore[arg-type]
        )

        if return_maps:
            seg_maps = [
                pred.permute(1, 2, 0).detach().cpu().numpy() for batch in predicted_batches for pred in batch["out_map"]
            ]
            return preds, seg_maps
        return preds
