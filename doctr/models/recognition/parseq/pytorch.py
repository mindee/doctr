# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from copy import deepcopy
from itertools import permutations
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.datasets import VOCABS
from doctr.models.modules.transformer import MultiHeadAttention, PositionwiseFeedForward

from ...classification import vit_s
from ...utils.pytorch import load_pretrained_params
from .base import _PARSeq, _PARSeqPostProcessor

__all__ = ["PARSeq", "parseq"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "parseq": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url": None,
    },
}


class CharEmbedding(nn.Module):
    """Implements the character embedding module

    Args:
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
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        #self.position_feed_forward = PositionwiseFeedForward(d_model, ffd * ffd_ratio, dropout, nn.GELU())
        self.position_feed_forward = nn.Linear(d_model, d_model*ffd_ratio )
        self.linear2 = nn.Linear(d_model*ffd_ratio, d_model)
        
        self.attention_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.cross_attention_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.query_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.content_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.feed_forward_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.output_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.attention_dropout = nn.Dropout(dropout)
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)
        self.activation = F.gelu
        self.dropout3 = nn.Dropout(dropout)
        
        
    def forward(self, query, content, memory, 
    query_mask: Optional[Tensor] = None, 
    content_mask: Optional[Tensor] = None,
    content_key_padding_mask: Optional[Tensor] = None):


        query_norm = self.query_norm(query)
        content_norm = self.content_norm(content)
        if query_mask is not None:
            query_mask = query_mask.bool()

        query = query.clone() + self.attention_dropout(
        
            self.attention(query=query_norm, key=content_norm, value=content_norm, attn_mask=query_mask,key_padding_mask=content_key_padding_mask)[0]
        )
        query = query.clone() + self.cross_attention_dropout(
        
            self.cross_attention(self.norm1(query), memory, memory)[0]
        )
        query = query.clone() +self.dropout3(self.linear2(self.feed_forward_dropout(self.activation(self.position_feed_forward(self.feed_forward_norm(query))))))
        return self.output_norm(query)


class PARSeq(_PARSeq, nn.Module):
    """Implements a PARSeq architecture as described in `"Scene Text Recognition
    with Permuted Autoregressive Sequence Models" <https://arxiv.org/pdf/2207.06966>`_.
    Slightly modified implementation based on the official Pytorch implementation: <https://github.com/baudm/parseq/tree/main`_.

    Args:
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
        dec_ff_dim: int = 2048,
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
    def gen_tgt_perms(self, tgt):
        """Generate shared permutations for the whole batch.
           This works because the same attention mask can be used for the shorter sequences
           because of the padding mask.
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=tgt.device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=tgt.device)]
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)//2

        num_gen_perms = min(3, max_perms)
        # For 4-char sequences and shorter, we generate all permutations and sample from the pool to avoid collisions
        # Note that this code path might NEVER get executed since the labels in a mini-batch typically exceed 4 chars.
        if max_num_chars < 5:
            # Pool of permutations to sample from. We only need the first half (if complementary option is selected)
            # Special handling for max_num_chars == 4 which correctly divides the pool into the flipped halves
            if max_num_chars == 4:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(list(permutations(range(max_num_chars), max_num_chars)), device=tgt.device)[selector]
            # If the forward permutation is always selected, no need to add it to the pool for sampling
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend([torch.randperm(max_num_chars, device=tgt.device) for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)

        comp = perms.flip(-1)
        # Stack in such a way that the pairs are next to each other.
        perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        # NOTE:
        # The only meaningful way of permuting the EOS position is by moving it one character position at a time.
        # However, since the number of permutations = T! and number of EOS positions = T + 1, the number of possible EOS
        # positions will always be much less than the number of permutations (unless a low perm_num is set).
        # Thus, it would be simpler to just train EOS using the full and null contexts rather than trying to evenly
        # distribute it across the chosen number of permutations.
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for the reverse direction. This does two things:
        # 1. Reverse context for the characters
        # 2. Null context for [EOS] (required for learning to predict [EOS] in NAR mode)
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=tgt.device)
        return perms
        #return F.pad(
        #    perms, (0, self.max_length + 1 - perms.shape[-1]), value=max_num_chars + 1
        #)  # (num_perms, self.max_length + 1)
        

    def generate_attn_masks(self, perm):
        """Generate attention masks given a sequence permutation (includes pos. for bos and eos tokens)
        :param perm: the permutation sequence. i = 0 is always the BOS
        :return: lookahead attention masks
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), device=perm.device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] =0.0
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=perm.device)] = 0.0 
        query_mask = mask[1:, :-1]
        return content_mask, query_mask


    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[Tensor] = None,
               tgt_padding_mask: Optional[Tensor] = None, tgt_query: Optional[Tensor] = None,
               tgt_query_mask: Optional[Tensor] = None):
        N, L = tgt.shape
        # <bos> stands for the null context. We only supply position information for characters after <bos>.
        null_ctx = self.embed(tgt[:, :1])
        tgt_emb = self.pos_queries[:, :L - 1] + self.embed(tgt[:, 1:])
        tgt_emb = self.dropout(torch.cat([null_ctx, tgt_emb], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :L].expand(N, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, tgt_emb, memory, tgt_query_mask, tgt_mask, tgt_padding_mask)


    def decode_autoregressive(self, features: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        """Generate predictions for the given features."""
        # Padding symbol + SOS at the beginning
        max_length = max_len if max_len is not None else self.max_length
        max_length = min(max_length, self.max_length) + 1
        ys = torch.full(
            (features.size(0), max_length), self.vocab_size + 2, dtype=torch.long, device=features.device
        )  # pad
        ys[:, 0] = self.vocab_size + 1  # SOS token
        pos_queries = self.pos_queries[:, :max_length].expand(features.size(0), -1, -1)
        # Create query mask for the decoder attention
        tgt_mask = query_mask = (
            torch.tril(torch.ones((max_length, max_length), device=features.device), diagonal=0).to(dtype=torch.bool)
        ).int()

        pos_logits = []
        for i in range(max_length):
            # Decode one token at a time without providing information about the future tokens
            tgt_out = self.decode(
                ys[:, : i + 1],
                features,
                tgt_mask[:i+1, :i+1],
                tgt_query_mask=query_mask[i : i + 1, : i + 1],
                tgt_query=pos_queries[:, i : i + 1],
            )
            pos_prob = self.head(tgt_out)
            pos_logits.append(pos_prob)

            if i + 1 < max_length:
                # Update with the next token
                ys[:, i + 1] = pos_prob.squeeze().argmax(-1)

                # Stop decoding if all sequences have reached the EOS token
                if max_len is None and (ys == self.vocab_size).any(dim=-1).all():
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
        target_pad_mask = ~((ys == self.vocab_size).int().cumsum(-1) > 0).unsqueeze(1).unsqueeze(1)
        mask = (target_pad_mask.bool() & query_mask[:, : ys.shape[1]].bool()).int()
        logits = self.head(self.decode(ys, features, mask, tgt_query=pos_queries))
        return logits

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
    
        features = self.feat_extractor(x)["features"]  # (batch_size, patches_seqlen, d_model)
        
        
        if self.training and target is None:
            raise ValueError("Need to provide labels during training")

        if target is not None:
            # Build target tensor

            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long).to(x.device), torch.tensor(_seq_len).to(x.device)
            if self.training:

                tgt_perms = self.gen_tgt_perms(gt)

                gt_in = gt[:,:-1]
                gt_out = gt[:,1:]

                tgt_padding_mask = (gt_in == self.vocab_size + 2) | (gt_in == self.vocab_size)
                loss = None
                loss_numel = 0
                n = (gt_out != self.vocab_size+2).sum().item()
                for i,perm in enumerate(tgt_perms):

                    target_mask, source_mask = self.generate_attn_masks(perm) 
                    logits = self.head(self.decode(gt_in, features, target_mask, tgt_padding_mask, tgt_query_mask=source_mask))

                    if loss is None:
                    
                        loss = self.compute_loss(logits, gt_out, seq_len, ignore_index=self.vocab_size + 2)
                    else:
                        loss += n * self.compute_loss(logits, gt_out, seq_len, ignore_index=self.vocab_size + 2)
                    loss_numel += n
                    
                    if i == 1:
                        gt_out = torch.where(gt_out == self.vocab_size, self.vocab_size+2, gt_out)
                        n = (gt_out != self.vocab_size+2).sum().item()
                loss = loss / loss_numel
            else:
                logits = self.decode_autoregressive(features, max_len=int(seq_len.max().item()))
                gt_out = gt_out[:, : logits.shape[1]]
                loss = self.compute_loss(logits, gt_out, seq_len, ignore_index=self.vocab_size + 2)
                
        else:
            logits = self.decode_autoregressive(features)

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

    @staticmethod
    
    def compute_loss(
        model_output: torch.Tensor,
        gt: torch.Tensor,
        seq_len: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        
        input_len = model_output.shape[1]
        cce = F.cross_entropy(
            model_output.permute(0, 2, 1), gt, ignore_index=ignore_index
        )
        return cce.mean()

    """
    def compute_loss(
        model_output: torch.Tensor,
        gt: torch.Tensor,
        seq_len: torch.Tensor,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        
        # Input length : number of steps
        input_len = model_output.shape[1]
        # Add one for additional <eos> token (sos disappear in shift!)
        seq_len = seq_len + 1
        # Compute loss: don't forget to shift gt! Otherwise the model learns to output the gt[t-1]!
        # The "masked" first gt char is <sos>. Delete last logit of the model output.
        cce = F.cross_entropy(
            model_output[:, :-1, :].permute(0, 2, 1), gt[:, 1:], reduction="none", ignore_index=ignore_index
        )
        # Compute mask, remove 1 timestep here as well
        mask_2d = torch.arange(input_len - 1, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0

        ce_loss = cce.sum(1) / seq_len.to(dtype=model_output.dtype)
        return ce_loss.mean()
"""

class PARSeqPostProcessor(_PARSeqPostProcessor):
    """Post processor for PARSeq architecture

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __call__(
        self,
        logits: torch.Tensor,
    ) -> List[Tuple[str, float]]:
        # compute pred with argmax for attention models
        out_idxs = logits.argmax(-1)
        # N x L
        probs = torch.gather(torch.softmax(logits, -1), -1, out_idxs.unsqueeze(-1)).squeeze(-1)
        # Take the minimum confidence of the sequence
        probs = probs.min(dim=1).values.detach().cpu()

        # Manual decoding
        word_values = [
            "".join(self._embedding[idx] for idx in encoded_seq).split("<eos>")[0]
            for encoded_seq in out_idxs.cpu().numpy()
        ]

        return list(zip(word_values, probs.numpy().tolist()))


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
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
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
