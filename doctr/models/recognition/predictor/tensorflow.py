# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import tensorflow as tf
from typing import List, Any, Union, Tuple

from doctr.utils.repr import NestedObject
from doctr.models.preprocessor import PreProcessor
from ..core import RecognitionModel
from .base import split_crops, remap_preds


__all__ = ['RecognitionPredictor']


class RecognitionPredictor(NestedObject):
    """Implements an object able to identify character sequences in images

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
        split_wide_crops: wether to use crop splitting for high aspect ratio crops
    """

    _children_names: List[str] = ['pre_processor', 'model']

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
        self.target_ar = 4  # Target aspect ratio

    def __call__(
        self,
        crops: List[Union[np.ndarray, tf.Tensor]],
        **kwargs: Any,
    ) -> List[Tuple[str, float]]:

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
            self.model(batch, return_preds=True, training=False, **kwargs)['preds']  # type: ignore[operator]
            for batch in processed_batches
        ]

        # Process outputs
        out = [charseq for batch in raw for charseq in batch]

        # Remap crops
        if self.split_wide_crops and remapped:
            out = remap_preds(out, crop_map, self.dil_factor)

        return out
