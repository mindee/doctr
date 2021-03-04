# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import List

__all__ = ['ExactMatch']


class ExactMatch:
    """Implements exact match metric (word-level accuracy) for recognition task

    Args:
        ignore_case: if true, ignore letter case when computing metric
        ignore_accents: if true, ignore accents errors when computing metrics"""

    def __init__(
        self,
        ignore_case: bool = False,
        ignore_accents: bool = False,
    ) -> None:

        self.matches = 0
        self.total = 0
        self.ignore_case = ignore_case
        self.ignore_accents = ignore_accents

    @staticmethod
    def remove_accent(input_string: str) -> str:
        """Removes all accents (¨^çéè...) from input_string

        Args:
            input_string: character sequence with accents

        Returns:
            character sequence without accents"""

        raise NotImplementedError

    def update_state(
        self,
        gt: List[str],
        pred: List[str],
    ) -> None:
        """Update the state of the metric with new predictions

        Args:
            gt: list of groung-truth character sequences
            pred: list of predicted character sequences"""

        if len(gt) != len(pred):
            raise AssertionError("prediction size does not match with ground-truth labels size")

        for pred_word, gt_word in zip(pred, gt):
            if self.ignore_accents:
                gt_word = remove_accent(gt_word)
                pred_word = remove_accent(pred_word)

            if self.ignore_case:
                gt_word = gt_word.lower()
                pred_word = pred_word.lower()

            if pred_word == gt_word:
                self.matches += 1

        self.total += len(gt)

    def result(self) -> float:
        """Gives the result of the metric

        Returns:
            metric result"""

        return self.matches / self.total
