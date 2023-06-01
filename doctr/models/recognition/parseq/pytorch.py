# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.datasets import VOCABS
from doctr.models.modules.transformer import MultiHeadAttention, PositionwiseFeedForward, Decoder, PositionalEncoding

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
        self.output_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.attention_dropout = nn.Dropout(dropout)
        self.cross_attention_dropout = nn.Dropout(dropout)
        self.feed_forward_dropout = nn.Dropout(dropout)

    def forward_stream(
        self,
        target: torch.Tensor,
        normalized_target: torch.Tensor,
        kv_target: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
    ):
        target = target.clone() + self.attention_dropout(self.attention(normalized_target, kv_target, kv_target, mask=tgt_mask))
        target = target.clone() + self.cross_attention_dropout(self.cross_attention(self.attention_norm(target), memory, memory))
        target = target.clone() + self.feed_forward_dropout(self.position_feed_forward(self.cross_attention_norm(target)))
        return target

    def forward(self, query, content, memory, query_mask: Optional[torch.Tensor] = None):
        query_norm = self.query_norm(query)
        content_norm = self.content_norm(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask)
        return self.output_norm(query)


class PARSeq(_PARSeq, nn.Module):
    """Implements a PARSeq architecture as described in `"Scene Text Recognition
    with Permuted Autoregressive Sequence Models" <https://arxiv.org/pdf/2207.06966>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        embedding_units: number of embedding units
        max_length: maximum word length handled by the model
        dropout_prob: dropout probability of the encoder LSTM
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
        input_shape: Tuple[int, int, int] = (3, 32, 128),
        exportable: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length + 3  # Add 1 step for EOS, 1 for SOS, 1 for PAD
        self.vocab_size = len(vocab)

        self.feat_extractor = feature_extractor
        self.embed_tgt = CharEmbedding(self.vocab_size, embedding_units)
        self.decoder = PARSeqDecoder(embedding_units)

        self.pos_queries = nn.Parameter(torch.Tensor(1, self.max_length, embedding_units))
        self.dropout = nn.Dropout(0.1)

        self.head = nn.Linear(embedding_units, self.vocab_size + 3)

        self.postprocessor = PARSeqPostProcessor(vocab=self.vocab)

    def make_source_and_target_mask(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # borrowed and slightly modified from  https://github.com/wenwenyu/MASTER-pytorch
        # NOTE: nn.TransformerDecoder takes the inverse from this implementation
        # [True, True, True, ..., False, False, False] -> False is masked
        target_pad_mask = (target != self.vocab_size + 2).unsqueeze(1).unsqueeze(1)  # (N, 1, 1, max_length)
        target_length = target.size(1)
        # sub mask filled diagonal with True = see and False = masked (max_length, max_length)
        # NOTE: onnxruntime tril/triu works only with float currently (onnxruntime 1.11.1 - opset 14)
        target_sub_mask = torch.tril(torch.ones((target_length, target_length), device=source.device), diagonal=0).to(
            dtype=torch.bool
        )
        # source mask filled with ones (max_length, positional_encoded_seq_len)
        source_mask = torch.ones((target_length, source.size(1)), dtype=torch.uint8, device=source.device)
        # combine the two masks into one (N, 1, max_length, max_length)
        target_mask = target_pad_mask & target_sub_mask
        return source_mask, target_mask.int()

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        features = self.feat_extractor(x)["features"]  # (batch_size, patches_seqlen, d_model)
        # add positional encoding to features
        features = self.positional_encoding(features)
        self.positions = self.pos_queries[:, :self.max_length].expand(features.size(0), -1, -1)

        if self.training and target is None:
            raise ValueError("Need to provide labels during training")

        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long), torch.tensor(_seq_len)
            gt, seq_len = gt.to(x.device), seq_len.to(x.device)
            # Compute source mask and target mask
            source_mask, target_mask = self.make_source_and_target_mask(features, gt)
            # TODO train stuff
            output = self.decoder(gt, features, source_mask, target_mask)
            # Compute logits
            logits = self.head(output)
        else:
            logits = self.decode(features)


        # TODO: decoding -> decode_ar looks like the MASTER model positionwise decoding

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

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decode function for prediction

        Args:
            encoded: input tensor

        Return:
            A Tuple of torch.Tensor: predictions, logits
        """
        b = encoded.size(0)


        # Padding symbol + SOS at the beginning
        ys = torch.full((b, self.max_length), self.vocab_size + 2, dtype=torch.long, device=encoded.device)  # pad
        ys[:, 0] = self.vocab_size + 1  # sos

        # Final dimension include EOS/SOS/PAD
        for i in range(self.max_length - 1):
            source_mask, target_mask = self.make_source_and_target_mask(encoded, ys)
            output = self.decoder(ys, encoded, source_mask, target_mask)
            logits = self.head(output)
            prob = torch.softmax(logits, dim=-1)
            next_token = torch.max(prob, dim=-1).indices
            # update ys with the next token and ignore the first token (SOS)
            ys[:, i + 1] = next_token[:, i]

        # Shape (N, max_length, vocab_size + 1)
        return logits

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
        cce = F.cross_entropy(model_output[:, :-1, :].permute(0, 2, 1), gt[:, 1:], reduction="none")
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
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )
