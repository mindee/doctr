# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Callable, Literal, Union

import numpy as np

from doctr.datasets import encode_sequences
from doctr.utils.repr import NestedObject

__all__ = ["RecognitionPostProcessor", "RecognitionModel", "aggregate_confidence", "ConfidenceAggregation"]

# Type alias for confidence aggregation methods
ConfidenceAggregation = Union[Literal["mean", "geometric_mean", "harmonic_mean", "min", "max"], Callable[[np.ndarray], float]]


def aggregate_confidence(
    probs: np.ndarray,
    method: ConfidenceAggregation = "mean",
) -> float:
    """Aggregate character-level confidence scores into a word-level confidence score.

    Args:
        probs: Array of character-level confidence scores (values between 0 and 1)
        method: Aggregation method to use. Can be one of:
            - "mean": Arithmetic mean (default)
            - "geometric_mean": Geometric mean (more sensitive to low values)
            - "harmonic_mean": Harmonic mean (even more sensitive to low values)
            - "min": Minimum confidence (most conservative)
            - "max": Maximum confidence (most optimistic)
            - A callable that takes an ndarray and returns a float

    Returns:
        Aggregated confidence score as a float between 0 and 1
    """
    if len(probs) == 0:
        return 0.0

    # Convert to numpy if needed and ensure float type
    probs = np.asarray(probs, dtype=np.float64)

    # Clip to valid probability range
    probs = np.clip(probs, 0.0, 1.0)

    if callable(method):
        return float(method(probs))

    if method == "mean":
        return float(np.mean(probs))
    elif method == "geometric_mean":
        # Use log-sum-exp trick for numerical stability
        # geometric_mean = exp(mean(log(probs)))
        # Handle zeros by replacing with small epsilon
        safe_probs = np.where(probs > 0, probs, 1e-10)
        return float(np.exp(np.mean(np.log(safe_probs))))
    elif method == "harmonic_mean":
        # harmonic_mean = n / sum(1/probs)
        # Handle zeros by replacing with small epsilon
        safe_probs = np.where(probs > 0, probs, 1e-10)
        return float(len(safe_probs) / np.sum(1.0 / safe_probs))
    elif method == "min":
        return float(np.min(probs))
    elif method == "max":
        return float(np.max(probs))
    else:
        raise ValueError(f"Unknown aggregation method: {method}. Expected one of 'mean', 'geometric_mean', 'harmonic_mean', 'min', 'max', or a callable.")


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
        confidence_aggregation: method to aggregate character-level confidence scores into word-level confidence.
            Can be "mean", "geometric_mean", "harmonic_mean", "min", "max", or a custom callable.
    """

    def __init__(
        self,
        vocab: str,
        confidence_aggregation: ConfidenceAggregation = "mean",
    ) -> None:
        self.vocab = vocab
        self.confidence_aggregation = confidence_aggregation
        self._embedding = list(self.vocab) + ["<eos>"]

    def extra_repr(self) -> str:
        agg_repr = self.confidence_aggregation if isinstance(self.confidence_aggregation, str) else "custom"
        return f"vocab_size={len(self.vocab)}, confidence_aggregation='{agg_repr}'"
