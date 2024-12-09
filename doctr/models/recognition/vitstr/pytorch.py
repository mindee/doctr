# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.datasets import VOCABS

from ...classification import vit_b, vit_s
from ...utils.pytorch import _bf16_to_float32, load_pretrained_params
from .base import _ViTSTR, _ViTSTRPostProcessor

__all__ = ["ViTSTR", "vitstr_small", "vitstr_base"]

default_cfgs: dict[str, dict[str, Any]] = {
    "vitstr_small": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/vitstr_small-fcd12655.pt&src=0",
    },
    "vitstr_base": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url": "https://doctr-static.mindee.com/models?id=v0.7.0/vitstr_base-50b21df2.pt&src=0",
    },
}


class ViTSTR(_ViTSTR, nn.Module):
    """Implements a ViTSTR architecture as described in `"Vision Transformer for Fast and
    Efficient Scene Text Recognition" <https://arxiv.org/pdf/2105.08582.pdf>`_.

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
        max_length: int = 32,  # different from paper
        input_shape: tuple[int, int, int] = (3, 32, 128),  # different from paper
        exportable: bool = False,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length + 2  # +2 for SOS and EOS

        self.feat_extractor = feature_extractor
        self.head = nn.Linear(embedding_units, len(self.vocab) + 1)  # +1 for EOS

        self.postprocessor = ViTSTRPostProcessor(vocab=self.vocab)

    def forward(
        self,
        x: torch.Tensor,
        target: list[str] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> dict[str, Any]:
        features = self.feat_extractor(x)["features"]  # (batch_size, patches_seqlen, d_model)

        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long), torch.tensor(_seq_len)
            gt, seq_len = gt.to(x.device), seq_len.to(x.device)

        if self.training and target is None:
            raise ValueError("Need to provide labels during training")

        # borrowed from : https://github.com/baudm/parseq/blob/main/strhub/models/vitstr/model.py
        features = features[:, : self.max_length]  # (batch_size, max_length, d_model)
        B, N, E = features.size()
        features = features.reshape(B * N, E)
        logits = self.head(features).view(B, N, len(self.vocab) + 1)  # (batch_size, max_length, vocab + 1)
        decoded_features = _bf16_to_float32(logits[:, 1:])  # remove cls_token

        out: dict[str, Any] = {}
        if self.exportable:
            out["logits"] = decoded_features
            return out

        if return_model_output:
            out["out_map"] = decoded_features

        if target is None or return_preds:
            # Disable for torch.compile compatibility
            @torch.compiler.disable  # type: ignore[attr-defined]
            def _postprocess(decoded_features: torch.Tensor) -> list[tuple[str, float]]:
                return self.postprocessor(decoded_features)

            # Post-process boxes
            out["preds"] = _postprocess(decoded_features)

        if target is not None:
            out["loss"] = self.compute_loss(decoded_features, gt, seq_len)

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
        seq_len = seq_len + 1  # type: ignore[assignment]
        # Compute loss: don't forget to shift gt! Otherwise the model learns to output the gt[t-1]!
        # The "masked" first gt char is <sos>.
        cce = F.cross_entropy(model_output.permute(0, 2, 1), gt[:, 1:], reduction="none")
        # Compute mask
        mask_2d = torch.arange(input_len, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0

        ce_loss = cce.sum(1) / seq_len.to(dtype=model_output.dtype)
        return ce_loss.mean()


class ViTSTRPostProcessor(_ViTSTRPostProcessor):
    """Post processor for ViTSTR architecture

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __call__(
        self,
        logits: torch.Tensor,
    ) -> list[tuple[str, float]]:
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


def _vitstr(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    layer: str,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> ViTSTR:
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
    model = ViTSTR(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if _cfg["vocab"] != default_cfgs[arch]["vocab"] else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def vitstr_small(pretrained: bool = False, **kwargs: Any) -> ViTSTR:
    """ViTSTR-Small as described in `"Vision Transformer for Fast and Efficient Scene Text Recognition"
    <https://arxiv.org/pdf/2105.08582.pdf>`_.

    >>> import torch
    >>> from doctr.models import vitstr_small
    >>> model = vitstr_small(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 128))
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
        kwargs: keyword arguments of the ViTSTR architecture

    Returns:
        text recognition architecture
    """
    return _vitstr(
        "vitstr_small",
        pretrained,
        vit_s,
        "1",
        embedding_units=384,
        patch_size=(4, 8),
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )


def vitstr_base(pretrained: bool = False, **kwargs: Any) -> ViTSTR:
    """ViTSTR-Base as described in `"Vision Transformer for Fast and Efficient Scene Text Recognition"
    <https://arxiv.org/pdf/2105.08582.pdf>`_.

    >>> import torch
    >>> from doctr.models import vitstr_base
    >>> model = vitstr_base(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 128))
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
        kwargs: keyword arguments of the ViTSTR architecture

    Returns:
        text recognition architecture
    """
    return _vitstr(
        "vitstr_base",
        pretrained,
        vit_b,
        "1",
        embedding_units=768,
        patch_size=(4, 8),
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )
