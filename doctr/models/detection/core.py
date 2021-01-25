# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Union, List, Tuple, Optional, Any, Dict

__all__ = ['DetectionModel', 'PostProcessor']


class DetectionModel(keras.Model):
    """Implements abstract DetectionModel class

    Args:
        input_shape: shape (H, W) of the model inputs
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (600, 600),
    ) -> None:
        super().__init__()
        self.input_size = input_size

    def __call__(
        self,
        inputs: tf.Tensor,
        training: bool = False
    ) -> Union[List[tf.Tensor], tf.Tensor]:
        raise NotImplementedError


class PostProcessor:
    """Abstract class to postprocess the raw output of the model

    Args:
        min_size_box (int): minimal length (pix) to keep a box
        max_candidates (int): maximum boxes to consider in a single page
        box_thresh (float): minimal objectness score to consider a box
    """

    def __init__(
        self,
        min_size_box: int = 5,
        max_candidates: int = 100,
        box_thresh: float = 0.5,
    ) -> None:

        self.min_size_box = min_size_box
        self.max_candidates = max_candidates
        self.box_thresh = box_thresh

    def __call__(
        self,
        raw_pred: List[tf.Tensor],
    ) -> List[List[np.ndarray]]:
        raise NotImplementedError
