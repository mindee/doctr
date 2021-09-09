# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Tuple, List, Any
import numpy as np
from numpy.core.numeric import full
from numpy.lib import index_tricks

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
                    new_width = int(0.6 * w)
                    splitted = [crop[:, :new_width, :], crop[:, -new_width:, :]]
                    # Add the 2 new boxes in the splitted_crops and keep track of indexes
                    splitted_idxs.append((len(splitted_crops), len(splitted_crops)+1))
                    splitted_crops.extend(splitted)
                    
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

                # Merge predictions
                def overlap(a, b):
                    return max(i for i in range(len(b)+1) if a.endswith(b[:i]))

                merged_out = []
                merged = False
                for i, charseq in enumerate(out):
                    if (i, i+1) in splitted_idxs:
                        a, b = out[i][0], out[i+1][0]
                        print(a)
                        print(b)
                        min_conf = min(out[i][1], out[i+1][1])
                        full_seq = a + b[overlap(a, b):]
                        merged_out.append((full_seq, min_conf))
                        print(full_seq)
                        print()
                        merged = True
                    else:
                        if merged == True:
                            merged = False
                            continue
                        merged_out.append(out[i])
                return merged_out

        return out
