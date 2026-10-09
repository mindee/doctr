# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import numpy as np

from doctr.datasets import encode_sequences
from doctr.models._utils import (
    ConfidenceAggregation,
    _confidence_aggregation_repr,
    _resolve_confidence_aggregation,
)
from doctr.utils.repr import NestedObject

__all__ = ["RecognitionPostProcessor", "RecognitionModel"]


class RecognitionModel(NestedObject):
    """Implements abstract RecognitionModel class"""

    vocab: str
    max_length: int

    def build_target(
        self,
        gts: list[str],
    ) -> tuple[np.ndarray, list[int]]:
        """Encode a list of gts sequences into a np array and gives the corresponding*
        sequence lengths.

        Args:
            gts: list of ground-truth labels

        Returns:
            A tuple of 2 tensors: Encoded labels and sequence lengths (for each entry of the batch)
        """
        encoded = encode_sequences(sequences=gts, vocab=self.vocab, target_size=self.max_length, eos=len(self.vocab))
        seq_len = [len(word) for word in gts]
        return encoded, seq_len


class RecognitionPostProcessor(NestedObject):
    """Abstract class to postprocess the raw output of the model

    Args:
        vocab: string containing the ordered sequence of supported characters
        confidence_aggregation: aggregation method of the character probabilities into the word confidence:
            "mean", "min", "max", "median", "geometric_mean", "harmonic_mean" or a callable
    """

    def __init__(
        self,
        vocab: str,
        confidence_aggregation: ConfidenceAggregation = "mean",
    ) -> None:
        _resolve_confidence_aggregation(confidence_aggregation)
        self.vocab = vocab
        self.confidence_aggregation = confidence_aggregation
        self._embedding = list(self.vocab) + ["<eos>"]

    def extra_repr(self) -> str:
        return (
            f"vocab_size={len(self.vocab)}, "
            f"confidence_aggregation={_confidence_aggregation_repr(self.confidence_aggregation)}"
        )
