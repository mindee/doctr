# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
from typing import Union, List, Tuple, Optional, Any, Dict

__all__ = ['DetectionModel']


class DetectionModel(keras.Model):
    """Implements abstract DetectionModel class

    """

    def __init__(
        self,
        shape: Tuple[int, int] = (600, 600),
    ) -> None:
        self.shape = shape

    def __call__(
        self,
        inputs: tf.Tensor,
        training: bool = False
    ) -> Tuple[keras.Model, keras.Model]:
        raise NotImplementedError
