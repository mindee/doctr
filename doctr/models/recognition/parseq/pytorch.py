# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from copy import deepcopy
from itertools import permutations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.datasets import VOCABS
from doctr.models.modules.transformer import MultiHeadAttention, PositionwiseFeedForward

from ...classification import vit_s
from ...utils.pytorch import _bf16_to_float32, load_pretrained_params
from .base import _PARSeq, _PARSeqPostProcessor

__all__ = ["PARSeq", "parseq"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "parseq": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/parseq-56125471.pt&src=0",
    },
}


class CharEmbedding(nn.Module):
    """Implements the character embedding module

    Args:
    ----
        vocab_size: size of the vocabulary
        d_model: dimension of the model
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(self.d_model) * self.embedding(x)


class PARSeqDecoder(nn.Module):
    """Implements decoder module of the PARSeq model

    Args:
    ----
        d_model: dimension of the model
        num_heads: number of attention heads
        ffd: dimension of the feed forward layer
        ffd_ratio: depth multiplier for the feed forward layer
        dropout: dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 12,
        ffd: int = 2048,
        ffd_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.cross_attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.position_feed_forward = PositionwiseFeedForward(d_model, ffd * ffd_ratio, dropout, nn.GELU())

        self.attention_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.cross_attention_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.query_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.content_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.output_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.attention_dropout = nn.Dropout(dropout)
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward(
        self,
        target,
        content,
        memory,
        target_mask: Optional[torch.Tensor] = None,
    ):
        query_norm = self.query_norm(target)
        content_norm = self.content_norm(content)
        target = target.clone() + self.attention_dropout(
            self.attention(query_norm, content_norm, content_norm, mask=target_mask)
        )
        target = target.clone() + self.cross_attention_dropout(
            self.cross_attention(self.query_norm(target), memory, memory)
        )
        target = target.clone() + self.feed_forward_dropout(self.position_feed_forward(self.feed_forward_norm(target)))
        return self.output_norm(target)


class PARSeq(_PARSeq, nn.Module):
    """Implements a PARSeq architecture as described in `"Scene Text Recognition
    with Permuted Autoregressive Sequence Models" <https://arxiv.org/pdf/2207.06966>`_.
    Slightly modified implementation based on the official Pytorch implementation: <https://github.com/baudm/parseq/tree/main`_.

    Args:
    ----
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        embedding_units: number of embedding units
        max_length: maximum word length handled by the model
        dropout_prob: dropout probability for the decoder
        dec_num_heads: number of attention heads in the decoder
        dec_ff_dim: dimension of the feed forward layer in the decoder
        dec_ffd_ratio: depth multiplier for the feed forward layer in the decoder
        input_shape: input shape of the image
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    def __init__(
        self,
        feature_extractor,
        vocab: str,
        embedding_units: int,
        max_length: int = 32,  # different from the paper
        dropout_prob: float = 0.1,
        dec_num_heads: int = 12,
        dec_ff_dim: int = 384,  # we use it from the original implementation instead of 2048
        dec_ffd_ratio: int = 4,
        input_shape: Tuple[int, int, int] = (3, 32, 128),
        exportable: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length
        self.vocab_size = len(vocab)
        self.rng = np.random.default_rng()

        self.feat_extractor = feature_extractor
        self.decoder = PARSeqDecoder(embedding_units, dec_num_heads, dec_ff_dim, dec_ffd_ratio, dropout_prob)
        self.head = nn.Linear(embedding_units, self.vocab_size + 1)  # +1 for EOS
        self.embed = CharEmbedding(self.vocab_size + 3, embedding_units)  # +3 for SOS, EOS, PAD

        self.pos_queries = nn.Parameter(torch.Tensor(1, self.max_length + 1, embedding_units))  # +1 for EOS
        self.dropout = nn.Dropout(p=dropout_prob)

        self.postprocessor = PARSeqPostProcessor(vocab=self.vocab)

        nn.init.trunc_normal_(self.pos_queries, std=0.02)
        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith("feat_extractor."):
                continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def generate_permutations(self, seqlen: torch.Tensor) -> torch.Tensor:
        # Generates permutations of the target sequence.
        # Borrowed from https://github.com/baudm/parseq/blob/main/strhub/models/parseq/system.py
        # with small modifications

        max_num_chars = int(seqlen.max().item())  # get longest sequence length in batch
        perms = [torch.arange(max_num_chars, device=seqlen.device)]

        max_perms = math.factorial(max_num_chars) // 2
        num_gen_perms = min(3, max_perms)
        if max_num_chars < 5:
            # Pool of permutations to sample from. We only need the first half (if complementary option is selected)
            # Special handling for max_num_chars == 4 which correctly divides the pool into the flipped halves
            if max_num_chars == 4:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)), device=seqlen.device)[
                selector
            ]
            # If the forward permutation is always selected, no need to add it to the pool for sampling
            perm_pool = perm_pool[1:]
            final_perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(final_perms), replace=False)
                final_perms = torch.cat([final_perms, perm_pool[i]])
        else:
            perms.extend(
                [torch.randperm(max_num_chars, device=seqlen.device) for _ in range(num_gen_perms - len(perms))]
            )
            final_perms = torch.stack(perms)

        comp = final_perms.flip(-1)
        final_perms = torch.stack([final_perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)

        sos_idx = torch.zeros(len(final_perms), 1, device=seqlen.device)
        eos_idx = torch.full((len(final_perms), 1), max_num_chars + 1, device=seqlen.device)
        combined = torch.cat([sos_idx, final_perms + 1, eos_idx], dim=1).int()  # type: ignore
        if len(combined) > 1:
            combined[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=seqlen.device)
        return combined

    def generate_permutations_attention_masks(self, permutation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate source and target mask for the decoder attention.
        sz = permutation.shape[0]
        mask = torch.ones((sz, sz), device=permutation.device)

        for i in range(sz):
            query_idx = permutation[i]
            masked_keys = permutation[i + 1 :]
            mask[query_idx, masked_keys] = 0.0
        source_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=permutation.device)] = 0.0
        target_mask = mask[1:, :-1]

        return source_mask.int(), target_mask.int()

    def decode(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        target_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add positional information to the target sequence and pass it through the decoder."""
        batch_size, sequence_length = target.shape
        # apply positional information to the target sequence excluding the SOS token
        null_ctx = self.embed(target[:, :1])
        content = self.pos_queries[:, : sequence_length - 1] + self.embed(target[:, 1:])
        content = self.dropout(torch.cat([null_ctx, content], dim=1))
        if target_query is None:
            target_query = self.pos_queries[:, :sequence_length].expand(batch_size, -1, -1)
        target_query = self.dropout(target_query)
        return self.decoder(target_query, content, memory, target_mask)

    def decode_autoregressive(self, features: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        """Generate predictions for the given features."""
        max_length = max_len if max_len is not None else self.max_length
        max_length = min(max_length, self.max_length) + 1
        # Padding symbol + SOS at the beginning
        ys = torch.full(
            (features.size(0), max_length), self.vocab_size + 2, dtype=torch.long, device=features.device
        )  # pad
        ys[:, 0] = self.vocab_size + 1  # SOS token
        pos_queries = self.pos_queries[:, :max_length].expand(features.size(0), -1, -1)
        # Create query mask for the decoder attention
        query_mask = (
            torch.tril(torch.ones((max_length, max_length), device=features.device), diagonal=0).to(dtype=torch.bool)
        ).int()

        pos_logits = []
        for i in range(max_length):
            # Decode one token at a time without providing information about the future tokens
            tgt_out = self.decode(
                ys[:, : i + 1],
                features,
                query_mask[i : i + 1, : i + 1],
                target_query=pos_queries[:, i : i + 1],
            )
            pos_prob = self.head(tgt_out)
            pos_logits.append(pos_prob)

            if i + 1 < max_length:
                # Update with the next token
                ys[:, i + 1] = pos_prob.squeeze().argmax(-1)

                # Stop decoding if all sequences have reached the EOS token
                if max_len is None and (ys == self.vocab_size).any(dim=-1).all():  # type: ignore[attr-defined]
                    break

        logits = torch.cat(pos_logits, dim=1)  # (N, max_length, vocab_size + 1)

        # One refine iteration
        # Update query mask
        query_mask[torch.triu(torch.ones(max_length, max_length, dtype=torch.bool, device=features.device), 2)] = 1

        # Prepare target input for 1 refine iteration
        sos = torch.full((features.size(0), 1), self.vocab_size + 1, dtype=torch.long, device=features.device)
        ys = torch.cat([sos, logits[:, :-1].argmax(-1)], dim=1)

        # Create padding mask for refined target input maskes all behind EOS token as False
        # (N, 1, 1, max_length)
        target_pad_mask = ~((ys == self.vocab_size).int().cumsum(-1) > 0).unsqueeze(1).unsqueeze(1)  # type: ignore[attr-defined]
        mask = (target_pad_mask.bool() & query_mask[:, : ys.shape[1]].bool()).int()
        logits = self.head(self.decode(ys, features, mask, target_query=pos_queries))

        return logits  # (N, max_length, vocab_size + 1)

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        features = self.feat_extractor(x)["features"]  # (batch_size, patches_seqlen, d_model)
        # remove cls token
        features = features[:, 1:, :]

        if self.training and target is None:
            raise ValueError("Need to provide labels during training")

        if target is not None:
            # Build target tensor
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long).to(x.device), torch.tensor(_seq_len).to(x.device)
            gt = gt[:, : int(seq_len.max().item()) + 2]  # slice up to the max length of the batch + 2 (SOS + EOS)

            if self.training:
                # Generate permutations for the target sequences
                tgt_perms = self.generate_permutations(seq_len)

                gt_in = gt[:, :-1]  # remove EOS token from longest target sequence
                gt_out = gt[:, 1:]  # remove SOS token
                # Create padding mask for target input
                # [True, True, True, ..., False, False, False] -> False is masked
                padding_mask = ~(
                    ((gt_in == self.vocab_size + 2) | (gt_in == self.vocab_size)).int().cumsum(-1) > 0
                ).unsqueeze(1).unsqueeze(1)  # (N, 1, 1, seq_len)

                loss = torch.tensor(0.0, device=features.device)
                loss_numel: Union[int, float] = 0
                n = (gt_out != self.vocab_size + 2).sum().item()
                for i, perm in enumerate(tgt_perms):
                    _, target_mask = self.generate_permutations_attention_masks(perm)  # (seq_len, seq_len)
                    # combine both masks
                    mask = (target_mask.bool() & padding_mask.bool()).int()  # (N, 1, seq_len, seq_len)

                    logits = self.head(self.decode(gt_in, features, mask)).flatten(end_dim=1)
                    loss += n * F.cross_entropy(logits, gt_out.flatten(), ignore_index=self.vocab_size + 2)
                    loss_numel += n
                    # After the second iteration (i.e. done with canonical and reverse orderings),
                    # remove the [EOS] tokens for the succeeding perms
                    if i == 1:
                        gt_out = torch.where(gt_out == self.vocab_size, self.vocab_size + 2, gt_out)
                        n = (gt_out != self.vocab_size + 2).sum().item()

                loss /= loss_numel

            else:
                gt = gt[:, 1:]  # remove SOS token
                max_len = gt.shape[1] - 1  # exclude EOS token
                logits = self.decode_autoregressive(features, max_len)
                loss = F.cross_entropy(logits.flatten(end_dim=1), gt.flatten(), ignore_index=self.vocab_size + 2)
        else:
            logits = self.decode_autoregressive(features)

        logits = _bf16_to_float32(logits)

        out: Dict[str, Any] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output:
            out["out_map"] = logits

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(logits)

        if target is not None:
            out["loss"] = loss

        return out


class PARSeqPostProcessor(_PARSeqPostProcessor):
    """Post processor for PARSeq architecture

    Args:
    ----
        vocab: string containing the ordered sequence of supported characters
    """

    def __call__(
        self,
        logits: torch.Tensor,
    ) -> List[Tuple[str, float]]:
        # compute pred with argmax for attention models
        out_idxs = logits.argmax(-1)
        preds_prob = torch.softmax(logits, -1).max(dim=-1)[0]

        # Manual decoding
        word_values = [
            "".join(self._embedding[idx] for idx in encoded_seq).split("<eos>")[0]
            for encoded_seq in out_idxs.cpu().numpy()
        ]
        # compute probabilties for each word up to the EOS token
        probs = [
            preds_prob[i, : len(word)].clip(0, 1).mean().item() if word else 0.0 for i, word in enumerate(word_values)
        ]

        return list(zip(word_values, probs))


def _parseq(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    layer: str,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> PARSeq:
    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])
    patch_size = kwargs.get("patch_size", (4, 8))

    kwargs["vocab"] = _cfg["vocab"]
    kwargs["input_shape"] = _cfg["input_shape"]

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        # NOTE: we don't use a pretrained backbone for non-rectangular patches to avoid the pos embed mismatch
        backbone_fn(False, input_shape=_cfg["input_shape"], patch_size=patch_size),  # type: ignore[call-arg]
        {layer: "features"},
    )

    kwargs.pop("patch_size", None)
    kwargs.pop("pretrained_backbone", None)

    # Build the model
    model = PARSeq(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if _cfg["vocab"] != default_cfgs[arch]["vocab"] else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def parseq(pretrained: bool = False, **kwargs: Any) -> PARSeq:
    """PARSeq architecture from
    `"Scene Text Recognition with Permuted Autoregressive Sequence Models" <https://arxiv.org/pdf/2207.06966>`_.

    >>> import torch
    >>> from doctr.models import parseq
    >>> model = parseq(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 128))
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
        **kwargs: keyword arguments of the PARSeq architecture

    Returns:
    -------
        text recognition architecture
    """
    return _parseq(
        "parseq",
        pretrained,
        vit_s,
        "1",
        embedding_units=384,
        patch_size=(4, 8),
        ignore_keys=["embed.embedding.weight", "head.weight", "head.bias"],
        **kwargs,
    )
