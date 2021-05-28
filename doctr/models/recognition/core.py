# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow import keras
from typing import Tuple, List, Any, Optional, Dict
import numpy as np

from ..preprocessor import PreProcessor
from doctr.utils.repr import NestedObject
from doctr.datasets import encode_sequences


__all__ = ['RecognitionPostProcessor', 'RecognitionModel', 'RecognitionPredictor']


class RecognitionModel(keras.Model, NestedObject):
    """Implements abstract RecognitionModel class"""

    def __init__(self, *args: Any, vocab: str, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.vocab = vocab
        self.cfg = cfg

    def compute_target(
        self,
        gts: List[str],
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Encode a list of gts sequences into a tf tensor and gives the corresponding*
        sequence lengths.

        Args:
            gts: list of ground-truth labels

        Returns:
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(
            sequences=gts,
            vocab=self.vocab,
            target_size=self.max_length,
            eos=len(self.vocab)
        )
        tf_encoded = tf.cast(encoded, tf.int64)
        seq_len = [len(word) for word in gts]
        tf_seq_len = tf.cast(seq_len, tf.int64)
        return tf_encoded, tf_seq_len

    def call(
        self,
        x: tf.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class RecognitionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(
        self,
        vocab: str,
    ) -> None:

        self.vocab = vocab
        self._embedding = tf.constant(list(self.vocab) + ['<eos>'], dtype=tf.string)

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
    """

    _children_names: List[str] = ['pre_processor', 'model']

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: RecognitionModel,
    ) -> None:

        self.pre_processor = pre_processor
        self.model = model

    def __call__(
        self,
        crops: List[np.ndarray],
        **kwargs: Any,
    ) -> List[Tuple[str, float]]:

        out = []
        if len(crops) > 0:
            # Dimension check
            if any(crop.ndim != 3 for crop in crops):
                raise ValueError("incorrect input shape: all crops are expected to be multi-channel 2D images.")

            # Resize & batch them
            processed_batches = self.pre_processor(crops)

            # Forward it
            raw = [self.model(batch, return_preds=True, **kwargs)['preds'] for batch in processed_batches]

            # Process outputs
            out = [charseq for batch in raw for charseq in batch]

        return out
