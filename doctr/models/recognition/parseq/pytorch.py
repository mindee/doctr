# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
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
        max_length: int = 25,
        dropout_prob: int = 0.1,
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
        self.max_length = max_length + 3  # +3 for SOS, EOS and PAD
        self.vocab_size = len(vocab)
        self.rng = np.random.default_rng()

        self.feat_extractor = feature_extractor
        self.decoder = PARSeqDecoder(embedding_units, dec_num_heads, dec_ff_dim, dec_ffd_ratio, dropout_prob)
        self.head = nn.Linear(embedding_units, self.vocab_size + 1)  # +1 for EOS
        self.text_embed = CharEmbedding(self.vocab_size + 3, embedding_units)

        self.pos_queries = nn.Parameter(torch.Tensor(1, self.max_length, embedding_units))
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
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def generate_permutations(self, target: torch.Tensor) -> torch.Tensor:
        # Generates permutations of the target sequence.
        # Modified from https://github.com/baudm/parseq/blob/main/strhub/models/parseq/system.py"""

        max_num_chars = target.shape[1] - 2
        perms = [torch.arange(max_num_chars, device=target.device)]

        max_perms = math.factorial(max_num_chars) // 2
        num_gen_perms = min(3, max_perms)
        perms.extend([torch.randperm(max_num_chars, device=target.device) for _ in range(num_gen_perms - len(perms))])
        perms = torch.stack(perms)

        comp = perms.flip(-1)
        perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)

        bos_idx = torch.zeros(len(perms), 1, device=perms.device)
        eos_idx = torch.full((len(perms), 1), max_num_chars + 1, device=perms.device)
        return torch.cat([bos_idx, perms + 1, eos_idx], dim=1).int()  # (num_perms, max_length + 1)

    def generate_permutation_attention_masks(self, permutation: torch.Tensor) -> torch.Tensor:
        # Generate query mask for the decoder attention.
        sz = permutation.shape[0]
        mask = torch.ones((sz, sz), device=permutation.device)

        for i in range(sz):
            query_idx = permutation[i]
            masked_keys = permutation[i + 1 :]
            mask[query_idx, masked_keys] = 0.0
        mask[torch.eye(sz, dtype=torch.bool, device=permutation.device)] = 0.0
        query_mask = mask[1:, :-1]

        return query_mask.int()

    def decode(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size, sequence_length = target.shape
        # apply positional information to the target sequence excluding the SOS token
        null_ctx = self.text_embed(target[:, :1])
        content = self.pos_queries[:, : sequence_length - 1] + self.text_embed(target[:, 1:])
        content = self.dropout(torch.cat([null_ctx, content], dim=1))
        if tgt_query is None:
            tgt_query = self.pos_queries[:, :sequence_length].expand(batch_size, -1, -1)
        tgt_query = self.dropout(tgt_query)
        return self.decoder(tgt_query, content, memory, tgt_mask)

    def decode_predictions(self, features: torch.Tensor) -> torch.Tensor:
        pos_queries = self.pos_queries[:, : self.max_length].expand(features.size(0), -1, -1)
        # Create query mask for the decoder attention
        query_mask = (
            torch.tril(torch.ones((self.max_length, self.max_length), device=features.device), diagonal=0).to(
                dtype=torch.bool
            )
        ).int()
        # Initialize target input tensor with SOS token
        ys = torch.full((features.size(0), self.max_length), self.vocab_size + 1, dtype=torch.long, device=features.device)
        ys[:, 0] = self.vocab_size + 1  # SOS token

        logits = []
        for i in range(self.max_length):
            j = i + 1  # next token index

            # Efficient decoding: Input the context up to the ith token using one query at a time
            tgt_out = self.decode(
                ys[:, :j],
                features,
                query_mask[i:j, :j],
                tgt_query=pos_queries[:, i:j],
            )

            # Obtain the next token probability in the output's ith token position
            p_i = self.head(tgt_out)
            logits.append(p_i)

            if j < self.max_length:
                # Greedy decode: Add the next token index to the target input
                ys[:, j] = p_i.squeeze().argmax(-1)

                # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                if (ys == self.vocab_size + 2).any(dim=-1).all():
                    break

        logits = torch.cat(logits, dim=1)

        # Update query mask
        # query_mask[
        #    torch.triu(torch.ones(self.max_length, self.max_length, dtype=torch.bool, device=features.device), 2)
        # ] = 0

        # Prepare target input for 1 refine iteration
        # bos = torch.full((features.size(0), 1), self.vocab_size - 1, dtype=torch.long, device=features.device)
        # ys = torch.cat([bos, logits[:, :-1].argmax(-1)], dim=1)

        # Create padding mask for refined target input
        # target_pad_mask = (ys != self.vocab_size -2).unsqueeze(1).unsqueeze(1)  # (N, 1, 1, max_length)
        # mask = (target_pad_mask & query_mask[:, : ys.shape[1]]).int()
        # logits = self.head(self.decode(ys, features, mask, tgt_query=pos_queries))

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

            # Generate permutations of the target sequences
            tgt_perms = self.generate_permutations(gt)
            target = gt[:, :-1]

            # Create padding mask for target input
            # [True, True, True, ..., False, False, False] -> False is masked
            target_pad_mask = (target != self.vocab_size + 2).unsqueeze(1).unsqueeze(1)  # (N, 1, 1, max_length)
            # TODO: train on more data this part is tricky check the combined mask again
            # TODO: without the padding mask it works (because to less dummy data)
            # TODO: https://github.com/pytorch/pytorch/blob/eb0971cfe9b05940978bed73d6e2b43aea49fc84/torch/nn/modules/activation.py#LL1247C5-L1286C38

            for perm in tgt_perms:
                # Generate attention masks for the permutations
                query_mask = self.generate_permutation_attention_masks(perm)
                print(query_mask)
                # combine target padding mask and query mask
                mask = (query_mask & target_pad_mask).int()  # TODO
                logits = self.head(self.decode(target, features, mask))
        else:
            logits = self.decode_predictions(features)

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
            out["loss"] = self.compute_loss(logits, gt, seq_len)

        return out

    @staticmethod
    def compute_loss(
        model_output: torch.Tensor,
        gt: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """Compute categorical cross-entropy loss for the model.
        Sequences are masked after the EOS character.

        Args:
            model_output: predicted logits of the model
            gt: the encoded tensor with gt labels
            seq_len: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        # Input length : number of steps
        input_len = model_output.shape[1]
        # Add one for additional <eos> token (sos disappear in shift!)
        seq_len = seq_len + 1
        # Compute loss: don't forget to shift gt! Otherwise the model learns to output the gt[t-1]!
        # The "masked" first gt char is <sos>. Delete last logit of the model output.
        cce = F.cross_entropy(model_output[:, :-1, :].permute(0, 2, 1), gt[:, 1:-1], reduction="none")
        # Compute mask, remove 1 timestep here as well
        mask_2d = torch.arange(input_len - 1, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0

        ce_loss = cce.sum(1) / seq_len.to(dtype=model_output.dtype)
        return ce_loss.mean()


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
    pretrained_backbone: bool = True,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> PARSeq:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])

    kwargs["vocab"] = _cfg["vocab"]
    kwargs["input_shape"] = _cfg["input_shape"]

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(pretrained_backbone, input_shape=_cfg["input_shape"]),  # type: ignore[call-arg]
        {layer: "features"},
    )

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
        ignore_keys=["head.weight", "head.bias"], # TODO: add embedding weights
        **kwargs,
    )
