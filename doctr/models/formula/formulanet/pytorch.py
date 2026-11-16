# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.datasets import LATEX_TOKENS

from ..._utils import ConfidenceAggregation, aggregate_confidence
from ...classification import hgnetv2_b4
from ...utils import _bf16_to_float32, load_pretrained_params
from .base import _FormulaNet, _FormulaNetPostProcessor

__all__ = ["FormulaNet", "formulanet"]

default_cfgs: dict[str, dict[str, Any]] = {
    "formulanet": {
        "mean": (0.7931, 0.7931, 0.7931),
        "std": (0.1738, 0.1738, 0.1738),
        "input_shape": (3, 256, 768),
        "vocab": LATEX_TOKENS,
        "url": "https://github.com/mindee/doctr/releases/download/v1.1.0/formulanet-f7fcd308.pt",
    },
}


def _sincos_2d(height: int, width: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Fixed 2D sinusoidal positional encoding of a feature map, half of the channels for each axis

    Returns:
        the encoding, of shape (height * width, dim)
    """
    quarter = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter, device=device, dtype=torch.float32) / quarter))
    ys = torch.arange(height, device=device, dtype=torch.float32)[:, None] * omega[None]  # (H, D/4)
    xs = torch.arange(width, device=device, dtype=torch.float32)[:, None] * omega[None]  # (W, D/4)
    pos_y = torch.cat([ys.sin(), ys.cos()], dim=1)[:, None].expand(height, width, 2 * quarter)
    pos_x = torch.cat([xs.sin(), xs.cos()], dim=1)[None].expand(height, width, 2 * quarter)
    pos = torch.cat([pos_y, pos_x], dim=-1).reshape(height * width, 4 * quarter)
    return F.pad(pos, (0, dim - 4 * quarter)).to(dtype)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with separate key / value projections, to cache them during decoding

    Args:
        d_model: dimension of the model
        num_heads: number of attention heads
        dropout: dropout rate of the attention weights
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (N, L, D) -> (N, H, L, D / H)
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def project_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project a sequence into the keys and values of the attention, of shape (N, H, L, D / H)"""
        return self._split_heads(self.k_proj(x)), self._split_heads(self.v_proj(x))

    def forward(
        self, query: torch.Tensor, key_value: tuple[torch.Tensor, torch.Tensor], is_causal: bool = False
    ) -> torch.Tensor:
        q = self._split_heads(self.q_proj(query))
        out = F.scaled_dot_product_attention(
            q, *key_value, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
        )
        return self.out_proj(out.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1))


class FormulaDecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer (causal self-attention, cross-attention on the image features, MLP)

    Args:
        d_model: dimension of the model
        num_heads: number of attention heads
        ffn_dim: hidden dimension of the feed-forward network
        dropout: dropout rate
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn_dim, d_model)
        )
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory_kv: tuple[torch.Tensor, torch.Tensor],
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Decode a sequence (causally), or the next tokens of a sequence given the cached keys / values

        Args:
            x: the embedded tokens, of shape (N, L, D)
            memory_kv: the keys and values of the image features (cf. `MultiHeadAttention.project_kv`)
            cache: the keys and values of the previous tokens, None for the start of the sequence

        Returns:
            the decoded tokens, and the keys and values of the sequence decoded so far
        """
        h = self.self_attn_norm(x)
        k, v = self.self_attn.project_kv(h)
        if cache is not None:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        # With a cache, the new tokens attend to all the previous ones (decoding one token at a time)
        x = x + self.dropout(self.self_attn(h, (k, v), is_causal=cache is None))
        x = x + self.dropout(self.cross_attn(self.cross_attn_norm(x), memory_kv))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, (k, v)


class FormulaNet(_FormulaNet, nn.Module):
    """Implements a formula recognition model: a convolutional encoder (HGNetV2) whose last feature map is flattened
    into a sequence, and a small autoregressive transformer decoder (MBart-style) predicting LaTeX tokens, as in
    `"PP-FormulaNet" <https://arxiv.org/abs/2503.18382>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: the list of LaTeX tokens
        d_model: dimension of the decoder
        num_layers: number of decoder layers
        num_heads: number of attention heads of the decoder
        ffn_dim: hidden dimension of the feed-forward networks of the decoder
        max_length: maximum number of tokens of a formula: size of the positional embeddings and maximum length of the
            training labels
        max_decode_length: maximum number of tokens generated by the greedy decoding (at most `max_length`). The
            decoding stops once every formula of a batch is complete: this caps the time spent on a formula the model
            fails to end (e.g. a repetition loop), which holds up its whole batch
        dropout_prob: dropout probability of the decoder
        feat_channels: number of channels of the feature map of the backbone
        input_shape: input shape of the image
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
        confidence_aggregation: aggregation method of the token probabilities into the formula confidence
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        vocab: Sequence[str],
        d_model: int = 384,
        num_layers: int = 4,
        num_heads: int = 16,
        ffn_dim: int = 1536,
        max_length: int = 4096,
        max_decode_length: int = 2048,
        dropout_prob: float = 0.1,
        feat_channels: int = 2048,
        input_shape: tuple[int, int, int] = (3, 256, 768),
        exportable: bool = False,
        cfg: dict[str, Any] | None = None,
        confidence_aggregation: ConfidenceAggregation = "mean",
    ) -> None:
        super().__init__()
        self.vocab = list(vocab)
        self.vocab_size = len(self.vocab)
        self.max_length = max_length
        self.max_decode_length = max_decode_length
        self.d_model = d_model
        self.exportable = exportable
        self.cfg = cfg
        # Special tokens, after the vocab
        self.eos, self.sos, self.pad = self.vocab_size, self.vocab_size + 1, self.vocab_size + 2

        self.feat_extractor = feature_extractor
        self.feat_proj = nn.Linear(feat_channels, d_model)
        self.feat_norm = nn.LayerNorm(d_model)

        self.embed = nn.Embedding(self.vocab_size + 3, d_model, padding_idx=self.pad)  # +3 for EOS, SOS, PAD
        self.pos_embed = nn.Embedding(max_length + 1, d_model)  # SOS + tokens
        self.embed_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)
        self.layers = nn.ModuleList([
            FormulaDecoderLayer(d_model, num_heads, ffn_dim, dropout_prob) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.vocab_size + 1)  # +1 for EOS

        self.postprocessor = FormulaNetPostProcessor(vocab=self.vocab, confidence_aggregation=confidence_aggregation)

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

    def from_pretrained(self, path_or_url: str, **kwargs: Any) -> None:
        """Load pretrained parameters onto the model

        Args:
            path_or_url: the path or URL to the model parameters (checkpoint)
            **kwargs: additional arguments to be passed to `doctr.models.utils.load_pretrained_params`
        """
        load_pretrained_params(self, path_or_url, **kwargs)

    def _decoder_layers(self) -> list[FormulaDecoderLayer]:
        return cast(list[FormulaDecoderLayer], list(self.layers))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into a sequence of features, of shape (N, H / 32 * W / 32, d_model)"""
        features = self.feat_extractor(x)["features"]  # (N, C, H, W)
        _, _, height, width = features.shape
        memory = self.feat_proj(features.flatten(2).transpose(1, 2))
        memory = memory + _sincos_2d(height, width, self.d_model, memory.device, memory.dtype)
        return self.feat_norm(memory)

    def _embed_tokens(self, tokens: torch.Tensor, start: int = 0) -> torch.Tensor:
        positions = torch.arange(start, start + tokens.shape[1], device=tokens.device)
        x = math.sqrt(self.d_model) * self.embed(tokens) + self.pos_embed(positions)[None]
        return self.dropout(self.embed_norm(x))

    def decode(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Decode full token sequences in parallel (teacher forcing)

        Args:
            tokens: the input tokens, starting with SOS, of shape (N, L)
            memory: the encoded images

        Returns:
            the logits of the next token at each position, of shape (N, L, vocab_size + 1)
        """
        x = self._embed_tokens(tokens)
        for layer in self._decoder_layers():
            x, _ = layer(x, layer.cross_attn.project_kv(memory))
        return self.head(self.norm(x))

    # The greedy decoding is not differentiable (argmax), and tracking the gradients of the growing key / value caches
    # would take a memory quadratic in the number of tokens: the training relies on `decode` (teacher forcing)
    @torch.no_grad()
    def decode_greedy(self, memory: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
        """Generate token sequences greedily, one token at a time, caching the keys / values of the decoded tokens

        Args:
            memory: the encoded images
            max_len: maximum number of tokens to generate (`max_decode_length` by default), at most `max_length`

        Returns:
            the logits of the generated tokens, of shape (N, L, vocab_size + 1)
        """
        steps = min(self.max_decode_length if max_len is None else max_len, self.max_length) + 1  # tokens + EOS
        memory_kv = [layer.cross_attn.project_kv(memory) for layer in self._decoder_layers()]
        caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.layers)
        token = torch.full((memory.shape[0], 1), self.sos, dtype=torch.long, device=memory.device)
        done = torch.zeros(memory.shape[0], dtype=torch.bool, device=memory.device)
        logits = []
        for step in range(steps):
            x = self._embed_tokens(token, start=step)
            for idx, layer in enumerate(self.layers):
                x, caches[idx] = layer(x, memory_kv[idx], caches[idx])
            step_logits = self.head(self.norm(x))
            logits.append(step_logits)
            token = step_logits.argmax(-1)
            # Stop once every sequence has produced EOS
            # NOTE: `break` isn't correctly translated to ONNX so the export always runs every step
            if not self.exportable:
                done |= token[:, 0] == self.eos
                if bool(done.all()):
                    break
        return torch.cat(logits, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        target: list[str] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> dict[str, Any]:
        memory = self.encode(x)

        if self.training and target is None:
            raise ValueError("Need to provide labels during training")

        loss = None
        if target is not None:
            # Teacher forcing: predict each token from the previous ones
            _gt, _seq_len = self.build_target(target)
            gt = torch.from_numpy(_gt).to(dtype=torch.long, device=x.device)
            gt = gt[:, : max(_seq_len, default=0) + 2]  # SOS + longest sequence + EOS
            logits = self.decode(gt[:, :-1], memory)
            loss = F.cross_entropy(logits.flatten(end_dim=1), gt[:, 1:].flatten(), ignore_index=self.pad)

        if target is None or return_preds:
            logits = self.decode_greedy(memory)

        logits = _bf16_to_float32(logits)

        out: dict[str, Any] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output:
            out["out_map"] = logits

        if target is None or return_preds:
            # Disable for torch.compile compatibility
            @torch.compiler.disable
            def _postprocess(logits: torch.Tensor) -> list[tuple[str, float]]:
                return self.postprocessor(logits)

            out["preds"] = _postprocess(logits)

        if loss is not None:
            out["loss"] = loss

        return out


class FormulaNetPostProcessor(_FormulaNetPostProcessor):
    """Post processor for FormulaNet architecture

    Args:
        vocab: the list of LaTeX tokens of the model
        confidence_aggregation: aggregation method of the token probabilities into the formula confidence
    """

    def __call__(self, logits: torch.Tensor) -> list[tuple[str, float]]:
        out_idxs = logits.argmax(-1).detach().cpu().numpy()
        probs = torch.softmax(logits, -1).max(dim=-1)[0].detach().float().cpu().numpy()
        eos = len(self.vocab)

        results: list[tuple[str, float]] = []
        for idxs, token_probs in zip(out_idxs, probs):
            ends = np.flatnonzero(idxs == eos)
            length = int(ends[0]) if ends.size > 0 else len(idxs)
            latex = " ".join(self.vocab[idx] for idx in idxs[:length])
            results.append((latex, aggregate_confidence(token_probs[:length], self.confidence_aggregation)))
        return results


def _formulanet(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[..., nn.Module],
    layer: str,
    pretrained_backbone: bool = True,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> FormulaNet:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = list(kwargs.get("vocab", _cfg["vocab"]))
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])

    kwargs["vocab"] = _cfg["vocab"]
    kwargs["input_shape"] = _cfg["input_shape"]

    # Feature extractor: the classification model (head included, to load its pretrained weights as they are), of which
    # only the layers up to `layer` are kept
    backbone = backbone_fn(pretrained_backbone)
    feat_extractor = IntermediateLayerGetter(backbone, {layer: "features"})

    # Build the model
    feat_channels = cast(list[int], backbone.channels)[-1]
    model = FormulaNet(feat_extractor, feat_channels=feat_channels, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of tokens is not the same as the number of tokens of the pretrained model =>
        # remove the embedding and the last layer weights
        _ignore_keys = ignore_keys if _cfg["vocab"] != default_cfgs[arch]["vocab"] else None
        model.from_pretrained(default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def formulanet(pretrained: bool = False, **kwargs: Any) -> FormulaNet:
    """FormulaNet: a HGNetV2-B4 encoder with a transformer decoder predicting LaTeX tokens, following
    `"PP-FormulaNet: Bridging Accuracy and Efficiency in Advanced Formula Recognition"
    <https://arxiv.org/abs/2503.18382>`_.

    >>> import torch
    >>> from doctr.models import formulanet
    >>> model = formulanet(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 256, 768))
    >>> out = model(input_tensor)

    Args:
        pretrained: If True, returns a model pre-trained on a formula recognition dataset
        **kwargs: keyword arguments of the FormulaNet architecture

    Returns:
        formula recognition architecture
    """
    return _formulanet(
        "formulanet",
        pretrained,
        hgnetv2_b4,
        "4",
        ignore_keys=["embed.weight", "head.weight", "head.bias"],
        **kwargs,
    )
