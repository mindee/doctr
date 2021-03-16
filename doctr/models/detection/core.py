# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Union, List, Tuple, Any, Optional, Dict
from ..preprocessor import PreProcessor
from doctr.utils.repr import NestedObject

__all__ = ['DetectionPreProcessor', 'DetectionModel', 'DetectionPostProcessor', 'DetectionPredictor']


class DetectionPreProcessor(PreProcessor):
    """Implements a detection preprocessor

        Example::
        >>> from doctr.documents import read_pdf
        >>> from doctr.models import RecoPreprocessor
        >>> processor = RecoPreprocessor(output_size=(600, 600), batch_size=8)
        >>> processed_doc = processor([read_pdf("path/to/your/doc.pdf")])

    Args:
        output_size: expected size of each page in format (H, W)
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
        interpolation: one of 'bilinear', 'nearest', 'bicubic', 'area', 'lanczos3', 'lanczos5'

    """

    def __init__(
        self,
        output_size: Tuple[int, int],
        batch_size: int = 1,
        mean: Tuple[float, float, float] = (.5, .5, .5),
        std: Tuple[float, float, float] = (1., 1., 1.),
        interpolation: str = 'bilinear'
    ) -> None:

        super().__init__(output_size, batch_size, mean, std, interpolation)

    def resize(
        self,
        x: tf.Tensor,
    ) -> tf.Tensor:
        """Resize images using tensorflow backend.

        Args:
            x: image as a tf.Tensor

        Returns:
            the processed image after being resized
        """

        return tf.image.resize(x, self.output_size, method=self.interpolation)


class DetectionModel(keras.Model, NestedObject):
    """Implements abstract DetectionModel class"""

    def __init__(self, *args: Any, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def call(
        self,
        x: tf.Tensor,
        **kwargs: Any,
    ) -> Union[List[tf.Tensor], tf.Tensor]:
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
        min_size_box: int = 5,
        max_candidates: int = 100,
        box_thresh: float = 0.5,
    ) -> None:

        self.min_size_box = min_size_box
        self.max_candidates = max_candidates
        self.box_thresh = box_thresh

    def extra_repr(self) -> str:
        return f"box_thresh={self.box_thresh}, max_candidates={self.max_candidates}"

    def __call__(
        self,
        raw_pred: List[tf.Tensor],
    ) -> List[List[np.ndarray]]:
        raise NotImplementedError


class DetectionPredictor(NestedObject):
    """Implements an object able to localize text elements in a document

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
        post_processor: post process model outputs
    """

    _children_names: List[str] = ['pre_processor', 'model', 'post_processor']

    def __init__(
        self,
        pre_processor: DetectionPreProcessor,
        model: DetectionModel,
        post_processor: DetectionPostProcessor,
    ) -> None:

        self.pre_processor = pre_processor
        self.model = model
        self.post_processor = post_processor

    def __call__(
        self,
        pages: List[np.ndarray],
        **kwargs: Any,
    ) -> List[np.ndarray]:

        # Dimension check
        if any(page.ndim != 3 for page in pages):
            raise ValueError("incorrect input shape: all pages are expected to be multi-channel 2D images.")

        processed_batches = self.pre_processor(pages)
        out = [self.model(batch, **kwargs) for batch in processed_batches]
        out = [self.post_processor(batch) for batch in out]
        out = [boxes for batch in out for boxes in batch]

        return out
