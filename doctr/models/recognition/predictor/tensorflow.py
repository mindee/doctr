# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

import numpy as np
import tensorflow as tf

from doctr.models.preprocessor import PreProcessor
from doctr.utils.repr import NestedObject

from ..core import RecognitionModel
from ._utils import remap_preds, split_crops

__all__ = ["RecognitionPredictor"]


class RecognitionPredictor(NestedObject):
    """Implements an object able to identify character sequences in images

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
        split_wide_crops: wether to use crop splitting for high aspect ratio crops
    """

    _children_names: list[str] = ["pre_processor", "model"]

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: RecognitionModel,
        split_wide_crops: bool = True,
    ) -> None:
        super().__init__()
        self.pre_processor = pre_processor
        self.model = model
        self.split_wide_crops = split_wide_crops
        self.critical_ar = 8  # Critical aspect ratio
        self.dil_factor = 1.4  # Dilation factor to overlap the crops
        self.target_ar = 6  # Target aspect ratio

    def __call__(
        self,
        crops: list[np.ndarray | tf.Tensor],
        **kwargs: Any,
    ) -> list[tuple[str, float]]:
        if len(crops) == 0:
            return []
        # Dimension check
        if any(crop.ndim != 3 for crop in crops):
            raise ValueError("incorrect input shape: all crops are expected to be multi-channel 2D images.")

        # Split crops that are too wide
        remapped = False
        if self.split_wide_crops:
            new_crops, crop_map, remapped = split_crops(crops, self.critical_ar, self.target_ar, self.dil_factor)
            if remapped:
                crops = new_crops

        # Resize & batch them
        processed_batches = self.pre_processor(crops)

        # Forward it
        raw = [
            self.model(batch, return_preds=True, training=False, **kwargs)["preds"]  # type: ignore[operator]
            for batch in processed_batches
        ]

        # Process outputs
        out = [charseq for batch in raw for charseq in batch]

        # Remap crops
        if self.split_wide_crops and remapped:
            out = remap_preds(out, crop_map, self.dil_factor)

        return out
