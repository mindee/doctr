# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Any, Union

from doctr.models.preprocessor import PreProcessor
from doctr.utils.repr import NestedObject


__all__ = ['DetectionPredictor']


class DetectionPredictor(NestedObject):
    """Implements an object able to localize text elements in a document

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
    """

    _children_names: List[str] = ['pre_processor', 'model']

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
        **kwargs: Any,
    ) -> List[np.ndarray]:

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(pages)
        predicted_batches = [
            self.model(batch, return_boxes=True, training=False, **kwargs)['preds']  # type:ignore[operator]
            for batch in processed_batches
        ]
        return [pred for batch in predicted_batches for pred in zip(*batch)]
