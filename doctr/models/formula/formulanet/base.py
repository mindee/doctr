# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Sequence

import numpy as np

from doctr.datasets import encode_sequences, normalize_latex
from doctr.utils.repr import NestedObject

from ..._utils import ConfidenceAggregation, _confidence_aggregation_repr, _resolve_confidence_aggregation

__all__ = ["_FormulaNet", "_FormulaNetPostProcessor"]


class _FormulaNet:
    vocab: list[str]
    max_length: int

    def build_target(self, gts: list[str]) -> tuple[np.ndarray, list[int]]:
        r"""Encode LaTeX labels into token sequences `[SOS, tokens..., EOS, PAD...]`

        The labels are tokenized first (cf. :func:`doctr.datasets.normalize_latex`), so they can be given either raw
        (e.g. `\frac{a}{b}`) or normalized (e.g. `\frac { a } { b }`).

        Args:
            gts: list of ground-truth LaTeX labels

        Returns:
            the encoded sequences, of shape (N, L), and the number of tokens of each label
        """
        normalized = [normalize_latex(gt) for gt in gts]
        encoded = encode_sequences(
            sequences=normalized,
            vocab=self.vocab,
            target_size=self.max_length + 2,  # SOS + tokens + EOS
            eos=len(self.vocab),
            sos=len(self.vocab) + 1,
            pad=len(self.vocab) + 2,
            dynamic_seq_length=True,
        )
        seq_len = [len(label.split()) for label in normalized]
        return encoded, seq_len


class _FormulaNetPostProcessor(NestedObject):
    """Turns the token scores of a formula recognition model into LaTeX strings

    Args:
        vocab: the list of LaTeX tokens of the model
        confidence_aggregation: aggregation method of the token probabilities into the formula confidence:
            "mean", "min", "max", "median", "geometric_mean", "harmonic_mean" or a callable
    """

    def __init__(
        self,
        vocab: Sequence[str],
        confidence_aggregation: ConfidenceAggregation = "mean",
    ) -> None:
        _resolve_confidence_aggregation(confidence_aggregation)
        self.vocab = list(vocab)
        self.confidence_aggregation = confidence_aggregation

    def extra_repr(self) -> str:
        return (
            f"vocab_size={len(self.vocab)}, "
            f"confidence_aggregation={_confidence_aggregation_repr(self.confidence_aggregation)}"
        )
