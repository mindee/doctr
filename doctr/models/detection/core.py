# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from typing import List, Any, Optional, Dict
from ..preprocessor import PreProcessor
from doctr.utils.repr import NestedObject


__all__ = ['DetectionModel', 'DetectionPostProcessor', 'DetectionPredictor']


class DetectionModel(keras.Model, NestedObject):
    """Implements abstract DetectionModel class"""

    def __init__(self, *args: Any, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def call(
        self,
        x: tf.Tensor,
        target: Optional[List[Dict[str, Any]]] = None,
        return_model_output: bool = False,
        return_boxes: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class DetectionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        min_size_box (int): minimal length (pix) to keep a box
        max_candidates (int): maximum boxes to consider in a single page
        box_thresh (float): minimal objectness score to consider a box
    """

    def __init__(
        self,
        box_thresh: float = 0.5,
        bin_thresh: float = 0.5
    ) -> None:

        self.box_thresh = box_thresh
        self.bin_thresh = bin_thresh

    def extra_repr(self) -> str:
        return f"box_thresh={self.box_thresh}, max_candidates={self.max_candidates}"

    @staticmethod
    def box_score(
        pred: np.ndarray,
        points: np.ndarray
    ) -> float:
        """Compute the confidence score for a polygon : mean of the p values on the polygon

        Args:
            pred (np.ndarray): p map returned by the model

        Returns:
            polygon objectness
        """
        h, w = pred.shape[:2]
        xmin = np.clip(np.floor(points[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(points[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(points[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(points[:, 1].max()).astype(np.int), 0, h - 1)

        return pred[ymin:ymax + 1, xmin:xmax + 1].mean()

    def bitmap_to_boxes(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
    ) -> List[List[float]]:
        raise NotImplementedError

    def __call__(
        self,
        proba_map: tf.Tensor,
    ) -> List[np.ndarray]:
        """Performs postprocessing for a list of model outputs

        Args:
            x: dictionary of the model output

        returns:
            list of N tensors (for each input sample), with each tensor of shape (*, 5).
        """

        proba_map = tf.squeeze(proba_map, axis=-1)  # remove last dim
        bitmap = tf.cast(proba_map > self.bin_thresh, tf.float32)

        proba_map = tf.unstack(proba_map, axis=0)
        bitmap = tf.unstack(bitmap, axis=0)

        boxes_batch = []
        # Kernel for opening, empirical law for ksize
        k_size = 1 + int(proba_map[0].shape[0] / 512)
        kernel = np.ones((k_size, k_size), np.uint8)

        for p_, bitmap_ in zip(proba_map, bitmap):
            p_ = p_.numpy()
            bitmap_ = bitmap_.numpy()
            # perform opening (erosion + dilatation)
            bitmap_ = cv2.morphologyEx(bitmap_, cv2.MORPH_OPEN, kernel)
            boxes = self.bitmap_to_boxes(pred=p_, bitmap=bitmap_)
            boxes_batch.append(boxes)

        return boxes_batch


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
        model: DetectionModel,
    ) -> None:

        self.pre_processor = pre_processor
        self.model = model

    def __call__(
        self,
        pages: List[np.ndarray],
        **kwargs: Any,
    ) -> List[np.ndarray]:

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(pages)
        out = [self.model(batch, return_boxes=True, **kwargs)['boxes'] for batch in processed_batches]
        out = [boxes for batch in out for boxes in batch]

        return out
