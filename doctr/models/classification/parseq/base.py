# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from abc import ABC, abstractmethod

import re
from abc import ABC, abstractmethod
from itertools import groupby
from typing import List, Optional, Tuple




class CharsetAdapter:
    """Transforms labels according to the target charset."""

    def __init__(self, target_charset) -> None:
        super().__init__()
        self.lowercase_only = target_charset == target_charset.lower()
        self.uppercase_only = target_charset == target_charset.upper()
        self.unsupported = f'[^{re.escape(target_charset)}]'

    def __call__(self, label):
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()
        # Remove unsupported characters
        label = re.sub(self.unsupported, '', label)
        return label


class BaseTokenizer(ABC):

    def __init__(self, charset: str, specials_first: tuple = (), specials_last: tuple = ()) -> None:
        self._itos = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> List[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens

    @abstractmethod
    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        """Encode a batch of labels to a representation suitable for the model.

        Args:
            labels: List of labels. Each can be of arbitrary length.
            device: Create tensor on this device.

        Returns:
            Batched tensor representation padded to the max label length. Shape: N, L
        """
        raise NotImplementedError

    @abstractmethod
    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        """Internal method which performs the necessary filtering prior to decoding."""
        raise NotImplementedError

    def decode(self, token_dists: Tensor, raw: bool = False) -> Tuple[List[str], List[Tensor]]:
        """Decode a batch of token distributions.

        Args:
            token_dists: softmax probabilities over the token distribution. Shape: N, L, C
            raw: return unprocessed labels (will return list of list of strings)

        Returns:
            list of string labels (arbitrary length) and
            their corresponding sequence probabilities as a list of Tensors
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)  # greedy selection
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs


class Tokenizer(BaseTokenizer):
    BOS = '[B]'
    EOS = '[E]'
    PAD = '[P]'

    def __init__(self, charset: str) -> None:
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        super().__init__(charset, specials_first, specials_last)
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        batch = [torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
                 for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)  # Nothing to truncate.
        # Truncate after EOS
        ids = ids[:eos_idx]
        probs = probs[:eos_idx + 1]  # but include prob. for EOS (if it exists)
        return probs, ids


class CTCTokenizer(BaseTokenizer):
    BLANK = '[B]'

    def __init__(self, charset: str) -> None:
        # BLANK uses index == 0 by default
        super().__init__(charset, specials_first=(self.BLANK,))
        self.blank_id = self._stoi[self.BLANK]

    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        # We use a padded representation since we don't want to use CUDNN's CTC implementation
        batch = [torch.as_tensor(self._tok2ids(y), dtype=torch.long, device=device) for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.blank_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> Tuple[Tensor, List[int]]:
        # Best path decoding:
        ids = list(zip(*groupby(ids.tolist())))[0]  # Remove duplicate tokens
        ids = [x for x in ids if x != self.blank_id]  # Remove BLANKs
        # `probs` is just pass-through since all positions are considered part of the path
        return probs, ids



class BatchResult:
    num_samples: int
    correct: int
    ned: float
    confidence: float
    label_length: int
    loss: Tensor
    loss_numel: int


class BaseSystem(ABC,nn.Module):

    def __init__(self, tokenizer: BaseTokenizer, charset_test: str,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.charset_adapter = CharsetAdapter(charset_test)
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_pct = warmup_pct
        self.weight_decay = weight_decay

    @abstractmethod
    def forward(self, images: Tensor, max_length: Optional[int] = None) -> Tensor:
        """Inference

        Args:
            images: Batch of images. Shape: N, Ch, H, W
            max_length: Max sequence length of the output. If None, will use default.

        Returns:
            logits: N, L, C (L = sequence length, C = number of classes, typically len(charset_train) + num specials)
        """
        raise NotImplementedError

    @abstractmethod
    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        """Like forward(), but also computes the loss (calls forward() internally).

        Args:
            images: Batch of images. Shape: N, Ch, H, W
            labels: Text labels of the images

        Returns:
            logits: N, L, C (L = sequence length, C = number of classes, typically len(charset_train) + num specials)
            loss: mean loss for the batch
            loss_numel: number of elements the loss was calculated from
        """
        raise NotImplementedError

    def configure_optimizers(self):
        agb = self.trainer.accumulate_grad_batches
        # Linear scaling so that the effective learning rate is constant regardless of the number of GPUs used with DDP.
        lr_scale = agb * math.sqrt(self.trainer.num_devices) * self.batch_size / 256.
        lr = lr_scale * self.lr
        optim = create_optimizer_v2(self, 'adamw', lr, self.weight_decay)
        sched = OneCycleLR(optim, lr, self.trainer.estimated_stepping_batches, pct_start=self.warmup_pct,
                           cycle_momentum=False)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched, 'interval': 'step'}}

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer, optimizer_idx: int):
        optimizer.zero_grad(set_to_none=True)

    def _eval_step(self, batch, validation: bool):
        images, labels = batch

        correct = 0
        total = 0
        ned = 0
        confidence = 0
        label_length = 0
        if validation:
            logits, loss, loss_numel = self.forward_logits_loss(images, labels)
        else:
            # At test-time, we shouldn't specify a max_label_length because the test-time charset used
            # might be different from the train-time charset. max_label_length in eval_logits_loss() is computed
            # based on the transformed label, which could be wrong if the actual gt label contains characters existing
            # in the train-time charset but not in the test-time charset. For example, "aishahaleyes.blogspot.com"
            # is exactly 25 characters, but if processed by CharsetAdapter for the 36-char set, it becomes 23 characters
            # long only, which sets max_label_length = 23. This will cause the model prediction to be truncated.
            logits = self.forward(images)
            loss = loss_numel = None  # Only used for validation; not needed at test-time.

        probs = logits.softmax(-1)
        preds, probs = self.tokenizer.decode(probs)
        for pred, prob, gt in zip(preds, probs, labels):
            confidence += prob.prod().item()
            pred = self.charset_adapter(pred)
            # Follow ICDAR 2019 definition of N.E.D.
            ned += edit_distance(pred, gt) / max(len(pred), len(gt))
            if pred == gt:
                correct += 1
            total += 1
            label_length += len(pred)
        return dict(output=BatchResult(total, correct, ned, confidence, label_length, loss, loss_numel))

    @staticmethod
    def _aggregate_results(outputs) -> Tuple[float, float, float]:
        if not outputs:
            return 0., 0., 0.
        total_loss = 0
        total_loss_numel = 0
        total_n_correct = 0
        total_norm_ED = 0
        total_size = 0
        for result in outputs:
            result = result['output']
            total_loss += result.loss_numel * result.loss
            total_loss_numel += result.loss_numel
            total_n_correct += result.correct
            total_norm_ED += result.ned
            total_size += result.num_samples
        acc = total_n_correct / total_size
        ned = (1 - total_norm_ED / total_size)
        loss = total_loss / total_loss_numel
        return acc, ned, loss

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, True)

    def validation_epoch_end(self, outputs) -> None:
        acc, ned, loss = self._aggregate_results(outputs)
        self.log('val_accuracy', 100 * acc, sync_dist=True)
        self.log('val_NED', 100 * ned, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)
        self.log('hp_metric', acc, sync_dist=True)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, False)


class CrossEntropySystem(BaseSystem):

    def __init__(self, charset_train: str, charset_test: str,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float) -> None:
        tokenizer = Tokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id

    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Discard <bos>
        max_len = targets.shape[1] - 1  # exclude <eos> from count
        logits = self.forward(images, max_len)
        loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.pad_id)
        loss_numel = (targets != self.pad_id).sum()
        return logits, loss, loss_numel


class CTCSystem(BaseSystem):

    def __init__(self, charset_train: str, charset_test: str,
                 batch_size: int, lr: float, warmup_pct: float, weight_decay: float) -> None:
        tokenizer = CTCTokenizer(charset_train)
        super().__init__(tokenizer, charset_test, batch_size, lr, warmup_pct, weight_decay)
        self.blank_id = tokenizer.blank_id

    def forward_logits_loss(self, images: Tensor, labels: List[str]) -> Tuple[Tensor, Tensor, int]:
        targets = self.tokenizer.encode(labels, self.device)
        logits = self.forward(images)
        log_probs = logits.log_softmax(-1).transpose(0, 1)  # swap batch and seq. dims
        T, N, _ = log_probs.shape
        input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long, device=self.device)
        target_lengths = torch.as_tensor(list(map(len, labels)), dtype=torch.long, device=self.device)
        loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=self.blank_id, zero_infinity=True)
        return logits, loss, N
