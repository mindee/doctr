# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Tuple, List, Any
import numpy as np
from Levenshtein import distance
from numpy.core.numeric import Inf

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

            splitted_crops = []
            splitted_idxs = []
            for crop in crops:
                h, w = crop.shape[:2]
                aspect_ratio = w / h
                if aspect_ratio > 8:
                    # Determine the number of crops, reference aspect ratio = 4 = 128/32
                    n_crops = int(aspect_ratio // 4)
                    # Find the new widths, additional 20% to overlap crops
                    new_width = int(1.4 * w / n_crops)
                    new_centers = [int((w / n_crops)*(1 / 2 + i)) for i in range(n_crops)]
                    # Crop
                    splitted_idxs.append([len(splitted_crops) + i for i in range(n_crops)]) 
                    splitted_crops.extend([crop[:, max(0, int(new_centers[i] - new_width / 2)):min(w-1, int(new_centers[i] + new_width / 2)), :] for i in range(n_crops)])
                else:
                    splitted_crops.append(crop)

            # Resize & batch them
            processed_batches = self.pre_processor(splitted_crops)

            # Forward it
            raw = [
                self.model(batch, return_preds=True, **kwargs)['preds']  # type: ignore[operator]
                for batch in processed_batches
            ]

            # Process outputs
            out = [charseq for batch in raw for charseq in batch]

            # Find splitted crops and merged back the predictions
            if len(splitted_idxs):
                merged_out = []
                out_idx = 0
                for splitted_list in splitted_idxs:
                    while out_idx < splitted_list[0]:
                        merged_out.append(out[out_idx])
                        out_idx += 1
                    print([out[i][0] for i in splitted_list])
                    merged = compute_overlap_multi([out[i][0] for i in splitted_list])
                    print(merged)
                    print()
                    merged_score = min([out[i][1] for i in splitted_list])
                    merged_out.append((merged, merged_score))
                    out_idx += len(splitted_list)
                # Append last outputs (after the last splitted box)
                while out_idx < len(out):
                    merged_out.append(out[out_idx])
                    out_idx += 1
                return merged_out

        return out


def compute_overlap(a: str, b: str, dil_factor: float = 1.4) -> str:
    """Compute the (best) overlap between 2 character sequences and merge them.
    
    Args:
        a: first char seq, suffix should be close to b's prefix.
        b: second char seq, prefix should be close to a's suffix.
        dil_factor: dilation factor of the boxes to overlap, should be > 1

    Returns:
        A merged character sequence.
    """
    seq_len = min(len(a), len(b))
    if seq_len == 0:  # One sequence is empty, return the other
        if len(a) == 0:
            return b
        else:
            return a
    min_score, index = 1, 0  # No overlap, just concatenate

    scores = [distance(a[-i:], b[:i]) / i for i in range(1, seq_len)]

    # Edge case for split in the middle of repetitions: if it starts with 2 or more 0
    if (scores[0], scores[1]) == (0, 0):
        # Compute the n_overlap (number of commmon chars)
        n_overlap = round(len(b) * (dil_factor - 1) / dil_factor)
        # Find the number of consecutive zeros
        n_zeros = 0
        for i, score in enumerate(scores):
            if score == 0:
                n_zeros += 1  # Impossible to have a zeros after a non-zeros in that case
        min_score, index = 0, min(n_zeros, n_overlap)

    else:  # Common case: choose the min score index
        for i, score in enumerate(scores):
            if score < min_score:
                min_score, index = score, i + 1  # Add one because first index is an overlap of 1 char

    # Merge with correct overlap
    if index == 0:
        return a + b
    return a[:-1] + b[index-1:]


def compute_overlap_multi(string_list: List[str]) -> str:
    """Wrapper for the resursive version of compute_overlap
    Compute the merged string from a list of strings:

    For instance, compute_overlap_multi(['abc', 'bcdef', 'difghi', 'aijkl']) returns 'abcdefghijkl'
    """
    def compute_overlap_rec(a: str, string_list: List[str]) -> str:
        # Recursive version of compute_overlap
        if len(string_list) == 1:
            return compute_overlap(a, string_list[0])
        return compute_overlap_rec(compute_overlap(a, string_list[0]), string_list[1:])

    return compute_overlap_rec("", string_list)
