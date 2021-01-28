# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
from typing import Tuple, List
import numpy as np

from ..preprocessor import PreProcessor

__all__ = ['RecognitionPostProcessor', 'RecognitionModel', 'RecognitionPredictor']


class RecognitionModel(keras.Model):
    """Implements abstract RecognitionModel class

    Args:
        input_shape: shape (H, W) of the model inputs
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (640, 640, 3),
    ) -> None:
        super().__init__()
        self.input_size = input_size

    def call(
        self,
        inputs: tf.Tensor,
    ) -> tf.Tensor:
        raise NotImplementedError


class RecognitionPostProcessor:
    """Abstract class to postprocess the raw output of the model

    Args:
        min_size_box (int): minimal length (pix) to keep a box
        max_candidates (int): maximum boxes to consider in a single page
        box_thresh (float): minimal objectness score to consider a box
    """

    def __call__(
        self,
        x: List[tf.Tensor],
    ) -> List[str]:
        raise NotImplementedError


class RecognitionPredictor:
    """Implements an object able to identify character sequences in images

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
        post_processor: post process model outputs
    """

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: RecognitionModel,
        post_processor: RecognitionPostProcessor,
    ) -> None:

        self.pre_processor = pre_processor
        self.model = model
        self.post_processor = post_processor

    def __call__(
        self,
        crops: List[np.ndarray],
    ) -> List[str]:

        out = []
        if len(crops) > 0:
            # Resize & batch them
            processed_batches = self.pre_processor(crops)

            # Forward it
            out = [self.model(tf.convert_to_tensor(batch)) for batch in processed_batches]

            # Process outputs
            out = [charseq for batch in out for charseq in self.post_processor(batch)]

        return out
