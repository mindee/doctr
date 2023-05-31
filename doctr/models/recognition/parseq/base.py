# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import List, Tuple

import numpy as np

from ....datasets import encode_sequences
from ..core import RecognitionPostProcessor
import math
from abc import ABC, abstractmethod

from typing import Optional, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules import transformer

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR



class _PARSeq:
    vocab: str
    max_length: int

    def build_target(
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
            eos=len(self.vocab),
            sos=len(self.vocab) + 1,
            pad=len(self.vocab) + 2,
        )
        seq_len = [len(word) for word in gts]
        return encoded, seq_len

class _PARSeqPostProcessor(RecognitionPostProcessor):
    """Abstract class to postprocess the raw output of the model

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __init__(
        self,
        vocab: str,
    ) -> None:
        super().__init__(vocab)
        self._embedding = list(vocab) + ["<eos>", "<sos>", "<pad>"]
        
   
