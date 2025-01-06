# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from doctr.models.detection._utils import _remove_padding
from doctr.models.preprocessor import PreProcessor
from doctr.utils.repr import NestedObject

__all__ = ["DetectionPredictor"]


class DetectionPredictor(NestedObject):
    """Implements an object able to localize text elements in a document

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
    """

    _children_names: list[str] = ["pre_processor", "model"]

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: Model,
    ) -> None:
        self.pre_processor = pre_processor
        self.model = model

    def __call__(
        self,
        pages: list[np.ndarray | tf.Tensor],
        return_maps: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, np.ndarray]] | tuple[list[dict[str, np.ndarray]], list[np.ndarray]]:
        # Extract parameters from the preprocessor
        preserve_aspect_ratio = self.pre_processor.resize.preserve_aspect_ratio
        symmetric_pad = self.pre_processor.resize.symmetric_pad
        assume_straight_pages = self.model.assume_straight_pages

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(pages)
        predicted_batches = [
            self.model(batch, return_preds=True, return_model_output=True, training=False, **kwargs)
            for batch in processed_batches
        ]

        # Remove padding from loc predictions
        preds = _remove_padding(
            pages,
            [pred for batch in predicted_batches for pred in batch["preds"]],
            preserve_aspect_ratio=preserve_aspect_ratio,
            symmetric_pad=symmetric_pad,
            assume_straight_pages=assume_straight_pages,
        )

        if return_maps:
            seg_maps = [pred.numpy() for batch in predicted_batches for pred in batch["out_map"]]
            return preds, seg_maps
        return preds
