# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from collections.abc import Callable
from copy import deepcopy
from itertools import groupby
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter

from doctr.datasets import VOCABS, decode_sequence

from ...classification import vip_tiny
from ...utils import _bf16_to_float32, load_pretrained_params
from ..core import RecognitionModel, RecognitionPostProcessor

__all__ = ["VIPTR", "viptr_tiny"]


default_cfgs: dict[str, dict[str, Any]] = {
    "viptr_tiny": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url": "https://doctr-static.mindee.com/models?id=v0.11.0/viptr_tiny-1cb2515e.pt&src=0",
    },
}


class VIPTRPostProcessor(RecognitionPostProcessor):
    """Postprocess raw prediction of the model (logits) to a list of words using CTC decoding

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    @staticmethod
    def ctc_best_path(
        logits: torch.Tensor,
        vocab: str = VOCABS["french"],
        blank: int = 0,
    ) -> list[tuple[str, float]]:
        """Implements best path decoding as shown by Graves (Dissertation, p63), highly inspired from
        <https://github.com/githubharald/CTCDecoder>`_.

        Args:
            logits: model output, shape: N x T x C
            vocab: vocabulary to use
            blank: index of blank label

        Returns:
            A list of tuples: (word, confidence)
        """
        # Gather the most confident characters, and assign the smallest conf among those to the sequence prob
        probs = F.softmax(logits, dim=-1).max(dim=-1).values.min(dim=1).values

        # collapse best path (using itertools.groupby), map to chars, join char list to string
        words = [
            decode_sequence([k for k, _ in groupby(seq.tolist()) if k != blank], vocab)
            for seq in torch.argmax(logits, dim=-1)
        ]

        return list(zip(words, probs.tolist()))

    def __call__(self, logits: torch.Tensor) -> list[tuple[str, float]]:
        """Performs decoding of raw output with CTC and decoding of CTC predictions
        with label_to_idx mapping dictionary

        Args:
            logits: raw output of the model, shape (N, C + 1, seq_len)

        Returns:
            A tuple of 2 lists: a list of str (words) and a list of float (probs)

        """
        # Decode CTC
        return self.ctc_best_path(logits=logits, vocab=self.vocab, blank=len(self.vocab))


class VIPTR(RecognitionModel, nn.Module):
    """Implements a VIPTR architecture as described in `"A Vision Permutable Extractor for Fast and Efficient
    Scene Text Recognition" <https://arxiv.org/abs/2401.10110>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        input_shape: input shape of the image
        exportable: onnx exportable returns only logits
        cfg: configuration dictionary
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        vocab: str,
        input_shape: tuple[int, int, int] = (3, 32, 128),
        exportable: bool = False,
        cfg: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = 32
        self.vocab_size = len(vocab)

        self.feat_extractor = feature_extractor
        with torch.inference_mode():
            embedding_units = self.feat_extractor(torch.zeros((1, *input_shape)))["features"].shape[-1]

        self.postprocessor = VIPTRPostProcessor(vocab=self.vocab)
        self.head = nn.Linear(embedding_units, len(self.vocab) + 1)  # +1 for PAD

        for n, m in self.named_modules():
            # Don't override the initialization of the backbone
            if n.startswith("feat_extractor."):
                continue
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def from_pretrained(self, path_or_url: str, **kwargs: Any) -> None:
        """Load pretrained parameters onto the model

        Args:
            path_or_url: the path or URL to the model parameters (checkpoint)
            **kwargs: additional arguments to be passed to `doctr.models.utils.load_pretrained_params`
        """
        load_pretrained_params(self, path_or_url, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        target: list[str] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> dict[str, Any]:
        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long), torch.tensor(_seq_len)
            gt, seq_len = gt.to(x.device), seq_len.to(x.device)

        if self.training and target is None:
            raise ValueError("Need to provide labels during training")

        features = self.feat_extractor(x)["features"]  # (B, max_len, embed_dim)
        B, N, E = features.size()
        logits = self.head(features).view(B, N, len(self.vocab) + 1)

        decoded_features = _bf16_to_float32(logits)

        out: dict[str, Any] = {}
        if self.exportable:
            out["logits"] = decoded_features
            return out

        if return_model_output:
            out["out_map"] = decoded_features

        if target is None or return_preds:
            # Disable for torch.compile compatibility
            @torch.compiler.disable
            def _postprocess(decoded_features: torch.Tensor) -> list[tuple[str, float]]:
                return self.postprocessor(decoded_features)

            # Post-process boxes
            out["preds"] = _postprocess(decoded_features)

        if target is not None:
            out["loss"] = self.compute_loss(decoded_features, gt, seq_len, len(self.vocab))

        return out

    @staticmethod
    def compute_loss(
        model_output: torch.Tensor,
        gt: torch.Tensor,
        seq_len: torch.Tensor,
        blank_idx: int = 0,
    ) -> torch.Tensor:
        """Compute CTC loss for the model.

        Args:
            model_output: predicted logits of the model
            gt: ground truth tensor
            seq_len: sequence lengths of the ground truth
            blank_idx: index of the blank label

        Returns:
            The loss of the model on the batch
        """
        batch_len = model_output.shape[0]
        input_length = model_output.shape[1] * torch.ones(size=(batch_len,), dtype=torch.int32)
        # N x T x C -> T x N x C
        logits = model_output.permute(1, 0, 2)
        probs = F.log_softmax(logits, dim=-1)
        ctc_loss = F.ctc_loss(
            probs,
            gt,
            input_length,
            seq_len,
            blank_idx,
            zero_infinity=True,
        )

        return ctc_loss


def _viptr(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    layer: str,
    pretrained_backbone: bool = True,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> VIPTR:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(pretrained_backbone, input_shape=_cfg["input_shape"]),  # type: ignore[call-arg]
        {layer: "features"},
    )

    kwargs["vocab"] = _cfg["vocab"]
    kwargs["input_shape"] = _cfg["input_shape"]

    model = VIPTR(feat_extractor, cfg=_cfg, **kwargs)

    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if _cfg["vocab"] != default_cfgs[arch]["vocab"] else None
        model.from_pretrained(default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def viptr_tiny(pretrained: bool = False, **kwargs: Any) -> VIPTR:
    """VIPTR-Tiny as described in `"A Vision Permutable Extractor for Fast and Efficient Scene Text Recognition"
    <https://arxiv.org/abs/2401.10110>`_.

    >>> import torch
    >>> from doctr.models import viptr_tiny
    >>> model = viptr_tiny(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 128))
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
        **kwargs: keyword arguments of the VIPTR architecture

    Returns:
        VIPTR: a VIPTR model instance
    """
    return _viptr(
        "viptr_tiny",
        pretrained,
        vip_tiny,
        "5",
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )
