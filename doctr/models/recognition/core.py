# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Tuple, List, Any
import numpy as np

from ..preprocessor import PreProcessor
from doctr.utils.repr import NestedObject
from doctr.datasets import encode_sequences


__all__ = ['RecognitionPostProcessor', 'RecognitionModel', 'RecognitionPredictor']


class RecognitionModel(NestedObject):
    """Implements abstract RecognitionModel class"""

    vocab: str
    max_length: int

    def compute_target(
        self,
        gts: List[str],
    ) -> Tuple[np.ndarray, List[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
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
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


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
        self._embedding = list(self.vocab) + ['<eos>']

    def extra_repr(self) -> str:
        return f"vocab_size={len(self.vocab)}"


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
            raw = [
                self.model(batch, return_preds=True, **kwargs)['preds']  # type: ignore[operator]
                for batch in processed_batches
            ]

            # Process outputs
            out = [charseq for batch in raw for charseq in batch]

        return out
