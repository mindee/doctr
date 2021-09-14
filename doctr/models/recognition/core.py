# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Tuple, List, Any
import numpy as np

from ..preprocessor import PreProcessor
from doctr.utils.repr import NestedObject
from doctr.datasets import encode_sequences
from .utils import merge_multi_strings
from ...file_utils import is_tf_available


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
        use_crop_splitting: wether to use crop splitting for high aspect ratio crops
    """

    _children_names: List[str] = ['pre_processor', 'model']

    def __init__(
        self,
        pre_processor: PreProcessor,
        model: RecognitionModel,
        use_crop_splitting: bool = True,
    ) -> None:

        self.pre_processor = pre_processor
        self.model = model
        self.use_crop_splitting = use_crop_splitting
        self.critical_ar = 8  # Critical aspect ratio
        self.dil_factor = 1.4  # Dilation factor to overlap the crops

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

            if self.use_crop_splitting:
                splitted_crops: List[np.ndarray] = []
                splitted_idxs: List[List[int]] = []
                for crop in crops:
                    channels_last = isinstance(crops[0], np.ndarray) or is_tf_available()
                    h, w = crop.shape[:2] if channels_last else crop.shape[-2:]
                    aspect_ratio = w / h
                    if aspect_ratio > self.critical_ar:
                        # Determine the number of crops, reference aspect ratio = 4 = 128 / 32
                        n_crops = int(aspect_ratio // 4)
                        # Find the new widths, additional dilation factor to overlap crops
                        width = self.dil_factor * w / n_crops
                        centers = [(w / n_crops) * (1 / 2 + i) for i in range(n_crops)]
                        # Crop and keep track of indexes
                        splitted_idxs.append([len(splitted_crops) + i for i in range(n_crops)])
                        if channels_last:
                            splitted_crops.extend([
                                crop[
                                    :,
                                    max(0, int(round(center - width / 2))): min(w - 1, int(round(center + width / 2))),
                                    :
                                ]
                                for center in centers
                            ])
                        else:
                            splitted_crops.extend([
                                crop[
                                    :,
                                    :,
                                    max(0, int(round(center - width / 2))): min(w - 1, int(round(center + width / 2)))
                                ]
                                for center in centers
                            ])
                    else:  # Append whole text box
                        splitted_crops.append(crop)

                # Resize & batch them
                processed_batches = self.pre_processor(splitted_crops)

            else:
                processed_batches = self.pre_processor(crops)

            # Forward it
            raw = [
                self.model(batch, return_preds=True, **kwargs)['preds']  # type: ignore[operator]
                for batch in processed_batches
            ]

            # Process outputs
            out = [charseq for batch in raw for charseq in batch]

            if self.use_crop_splitting:
                # Find if crops were splitted, and if so merge back the predictions
                if len(splitted_idxs) > 0:
                    merged_out = []
                    out_idx = 0
                    for splitted_list in splitted_idxs:
                        # Iterate over words to reconstruct
                        merged_out.extend(out[out_idx: splitted_list[0]])
                        out_idx = splitted_list[0]
                        # Merge splitted words
                        merged = merge_multi_strings([out[i][0] for i in splitted_list], self.dil_factor)
                        merged_score = min([out[i][1] for i in splitted_list])
                        merged_out.append((merged, merged_score))
                        out_idx += len(splitted_list)
                    # Append last unsplitted words after the last reconstructed word
                    merged_out.extend(out[out_idx: len(out)])
                    return merged_out

        return out
