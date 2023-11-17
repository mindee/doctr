# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

from doctr.models.preprocessor import PreProcessor
from doctr.utils.repr import NestedObject

__all__ = ["DetectionPredictor"]


class DetectionPredictor(NestedObject):
    """Implements an object able to localize text elements in a document

    Args:
    ----
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
    """

    _children_names: List[str] = ["pre_processor", "model"]

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: keras.Model,
    ) -> None:
        self.pre_processor = pre_processor
        self.model = model

    def __call__(
        self,
        pages: List[Union[np.ndarray, tf.Tensor]],
        return_maps: bool = False,
        **kwargs: Any,
    ) -> Union[List[Dict[str, np.ndarray]], Tuple[List[Dict[str, np.ndarray]], List[np.ndarray]]]:
        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(pages)
        predicted_batches = [
            self.model(batch, return_preds=True, return_model_output=True, training=False, **kwargs)
            for batch in processed_batches
        ]

        preds = [pred for batch in predicted_batches for pred in batch["preds"]]
        if return_maps:
            seg_maps = [pred.numpy() for batch in predicted_batches for pred in batch["out_map"]]
            return preds, seg_maps
        return preds
