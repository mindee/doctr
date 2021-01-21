# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
from typing import Union, List, Tuple, Optional, Any, Dict

__all__ = ['DetectionModel']


class DetectionModel(keras.Model):
    """Implements abstract DetectionModel class

    Args:
        input_shape: shape (H, W) of the model inputs
    """

    def __init__(
        self,
        image_shape: Tuple[int, int] = (600, 600),
    ) -> None:
        super().__init__()
        self.image_shape = image_shape

    def __call__(
        self,
        inputs: tf.Tensor,
        training: bool = False
    ) -> Union[List[tf.Tensor], tf.Tensor]:
        raise NotImplementedError
