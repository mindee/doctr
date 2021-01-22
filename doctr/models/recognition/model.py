# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
from typing import Union, List, Tuple, Optional, Any, Dict

__all__ = ['RecognitionModel']


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

    def __call__(
        self,
        inputs: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        raise NotImplementedError
