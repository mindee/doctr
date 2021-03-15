# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
from typing import Tuple, List, Any, Optional, Dict
import numpy as np

from ..preprocessor import PreProcessor
from doctr.utils.repr import NestedObject

__all__ = ['RecognitionPreProcessor', 'RecognitionPostProcessor', 'RecognitionModel', 'RecognitionPredictor']


class RecognitionPreProcessor(PreProcessor):
    """Implements a recognition preprocessor

    Example::
        >>> from doctr.documents import read_pdf
        >>> from doctr.models import RecoPreprocessor
        >>> processor = RecoPreprocessor(output_size=(128, 256), batch_size=8)
        >>> processed_doc = processor([read_pdf("path/to/your/doc.pdf")])

    Args:
        output_size: expected size of each page in format (H, W)
        batch_size: the size of page batches
        mean: mean value of the training distribution by channel
        std: standard deviation of the training distribution by channel
    """

    def __init__(
        self,
        output_size: Tuple[int, int],
        batch_size: int = 32,
        mean: Tuple[float, float, float] = (.5, .5, .5),
        std: Tuple[float, float, float] = (1., 1., 1.),
        interpolation: str = 'bilinear',
    ) -> None:

        super().__init__(output_size, batch_size, mean, std, interpolation)

    def resize(
        self,
        x: tf.Tensor,
    ) -> tf.Tensor:
        """Resize images using tensorflow backend.
        The images is resized to (output_height, width) where width is computed as follow :
            - If (preserving aspect-ratio width) output_height/image_height * image_width < output__width :
                resize to (output_height, output_height/image_height * image_width)
            - Else :
                resize to (output_height, output_width)
        Pads the image source with 0 to the right to match target_width and to the bottom to match target_height

        Args:
            x: image as a tf.Tensor

        Returns:
            the processed image after being resized
        """

        # Preserve aspect ratio during resizing
        resized = tf.image.resize(x, self.output_size, method=self.interpolation, preserve_aspect_ratio=True)
        # Pad on the side that is still too small
        padded = tf.image.pad_to_bounding_box(resized, 0, 0, *self.output_size)

        return padded


class RecognitionModel(keras.Model, NestedObject):
    """Implements abstract RecognitionModel class"""

    def __init__(self, *args: Any, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg

    def call(
        self,
        x: tf.Tensor,
        **kwargs: Any,
    ) -> tf.Tensor:
        raise NotImplementedError


class RecognitionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        vocab: string containing the ordered sequence of supported characters
        ignore_case: if True, ignore case of letters
        ignore_accents: if True, ignore accents of letters
    """

    def __init__(
        self,
        vocab: str,
        ignore_case: bool = False,
        ignore_accents: bool = False
    ) -> None:

        self.vocab = vocab
        self._embedding = tf.constant(list(self.vocab) + ['<eos>'], dtype=tf.string)
        self.ignore_case = ignore_case
        self.ignore_accents = ignore_accents

    def extra_repr(self) -> str:
        return f"vocab_size={len(self.vocab)}"

    def __call__(
        self,
        x: List[tf.Tensor],
    ) -> List[str]:
        raise NotImplementedError


class RecognitionPredictor(NestedObject):
    """Implements an object able to identify character sequences in images

    Args:
        pre_processor: transform inputs for easier batched model inference
        model: core detection architecture
        post_processor: post process model outputs
    """

    _children_names: List[str] = ['pre_processor', 'model', 'post_processor']

    def __init__(
        self,
        pre_processor: RecognitionPreProcessor,
        model: RecognitionModel,
        post_processor: RecognitionPostProcessor,
    ) -> None:

        self.pre_processor = pre_processor
        self.model = model
        self.post_processor = post_processor

    def __call__(
        self,
        crops: List[np.ndarray],
        **kwargs: Any,
    ) -> List[str]:

        out = []
        if len(crops) > 0:
            # Dimension check
            if any(crop.ndim != 3 for crop in crops):
                raise ValueError("incorrect input shape: all crops are expected to be multi-channel 2D images.")

            # Resize & batch them
            processed_batches = self.pre_processor(crops)

            # Forward it
            out = [self.model(batch, **kwargs) for batch in processed_batches]

            # Process outputs
            out = [charseq for batch in out for charseq in self.post_processor(batch)]

        return out
