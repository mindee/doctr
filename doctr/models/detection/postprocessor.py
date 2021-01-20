# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
import numpy as np
from typing import Union, List, Tuple, Optional, Any, Dict


__all__ = ['Postprocessor']


class Postprocessor:
    """Abstract class to postprocess the raw output of the model

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
