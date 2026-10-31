# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from doctr.models._utils import ConfidenceAggregation, _resolve_confidence_aggregation, predict_by_chunks
from doctr.models.preprocessor import PreProcessor
from doctr.models.utils import set_device_and_dtype

from ._utils import remap_preds, split_crops

__all__ = ["RecognitionPredictor"]


class RecognitionPredictor(nn.Module):
    """Implements an object able to identify character sequences in images

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
        split_wide_crops: wether to use crop splitting for high aspect ratio crops
        chunk_size: the maximum number of crops decoded and processed at once (1024 by default). Longer
            inputs, lazy page sources (cf. `DocumentFile.from_images(..., lazy=True)`) and iterators are processed
            chunk by chunk, with a memory footprint that does not grow with the number of crops. It is rounded
            up to a multiple of the batch size, so that the predictions are the same as in a single pass.
    """

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: nn.Module,
        split_wide_crops: bool = True,
        chunk_size: int = 1024,
    ) -> None:
        super().__init__()
        self.pre_processor = pre_processor
        self.model = model.eval()
        if chunk_size < 1:
            raise ValueError("`chunk_size` must be a positive integer")
        self.chunk_size = chunk_size
        self.split_wide_crops = split_wide_crops
        self.critical_ar = 8  # Critical aspect ratio
        self.overlap_ratio = 0.5  # Ratio of overlap between neighboring crops
        self.target_ar = 6  # Target aspect ratio
        self.split_confidence_aggregation: ConfidenceAggregation = "min"  # Confidence aggregation for split crops

    @torch.inference_mode()
    def forward(
        self,
        crops: Sequence[np.ndarray] | Iterable[np.ndarray],
        **kwargs: Any,
    ) -> list[tuple[str, float]]:
        """Read the character sequence of word crops

        Args:
            crops: a list of word crops (numpy arrays of shape H x W x C), a lazy page source
                (cf. `DocumentFile.from_images(..., lazy=True)`) or any iterable of crops
            **kwargs: keyword arguments forwarded to the model

        Returns:
            the text and confidence of each crop
        """
        return predict_by_chunks(
            lambda chunk: self._predict(chunk, **kwargs), crops, self.chunk_size, self.pre_processor.batch_size
        )

    @torch.inference_mode()
    def _predict(
        self,
        crops: Sequence[np.ndarray],
        **kwargs: Any,
    ) -> list[tuple[str, float]]:
        if len(crops) == 0:
            return []
        _resolve_confidence_aggregation(self.split_confidence_aggregation)
        # Dimension check
        if any(crop.ndim != 3 for crop in crops):
            raise ValueError("incorrect input shape: all crops are expected to be multi-channel 2D images.")

        # Split crops that are too wide
        remapped = False
        if self.split_wide_crops:
            new_crops, crop_map, remapped = split_crops(
                crops,  # type: ignore[arg-type]
                self.critical_ar,
                self.target_ar,
                self.overlap_ratio,
            )
            if remapped:
                crops = new_crops

        # Resize & batch them
        processed_batches = self.pre_processor(crops)  # type: ignore[arg-type]

        # Forward it
        _params = next(self.model.parameters())
        self.model, processed_batches = set_device_and_dtype(  # type: ignore[assignment]
            self.model, processed_batches, _params.device, _params.dtype
        )
        raw = [self.model(batch, return_preds=True, **kwargs)["preds"] for batch in processed_batches]

        # Process outputs
        out = [charseq for batch in raw for charseq in batch]

        # Remap crops
        if self.split_wide_crops and remapped:
            out = remap_preds(out, crop_map, self.overlap_ratio, self.split_confidence_aggregation)

        return out
