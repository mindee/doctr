from copy import deepcopy
from itertools import groupby
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from doctr.datasets import VOCABS, decode_sequence

from ...classification import vip_base, vip_tiny
from ...classification.vip import default_cfgs  # Explicit import instead of '*'
from ...utils.pytorch import _bf16_to_float32, load_pretrained_params
from .base import _VIPTR, _VIPTRPostProcessor

__all__ = ["VIPTRPostProcessor", "VIPTR", "viptr_base", "viptr_tiny"]


class VIPTRPostProcessor(_VIPTRPostProcessor):
    """Postprocess raw prediction of the model (logits) to a list of words using CTC decoding."""

    @staticmethod
    def ctc_best_path(
        logits: torch.Tensor,
        vocab: str,
        blank: int = 0,
    ) -> List[Tuple[str, float]]:
        """
        Implements best path decoding as shown by Graves (Dissertation, p63).

        Args:
            logits: model output, shape: N x T x C
            vocab: vocabulary string
            blank: index of blank label

        Returns:
            List of (word, confidence) for each sample
        """
        # Compute a "confidence" for each sequence
        probs = F.softmax(logits, dim=-1).max(dim=-1).values.min(dim=1).values

        # Collapse best path (itertools.groupby), map indices->chars, join to string
        preds_indices = torch.argmax(logits, dim=-1)
        words = [decode_sequence([k for k, _ in groupby(seq.tolist()) if k != blank], vocab) for seq in preds_indices]

        return list(zip(words, probs.tolist()))

    def __call__(self, logits: torch.Tensor) -> List[Tuple[str, float]]:
        """
        Decodes CTC logits into strings plus confidence.

        Args:
            logits: raw output of shape (N, T, C)

        Returns:
            List of (word, confidence)
        """
        return self.ctc_best_path(logits=logits, vocab=self.vocab, blank=0)


class VIPTR(_VIPTR, nn.Module):
    """
    Main VIPTR model class for text recognition.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        vocab: str,
        embedding_units: int = 384,
        input_shape: Tuple[int, int, int] = (3, 32, 128),
        output_channel: int = 192,
        cfg: Optional[Dict[str, Any]] = None,
        exportable: bool = False,
        max_length: int = 32,
    ):
        """
        Args:
            feature_extractor: backbone feature extractor
            vocab: string containing all supported characters
            embedding_units: dimension of embeddings
            input_shape: (C, H, W)
            output_channel: optional final output channels
            cfg: optional config
            exportable: whether to mark the model export-friendly
            max_length: max sequence length
        """
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length
        self.vocab_size = len(vocab)

        self.feat_extractor = feature_extractor
        self.postprocessor = VIPTRPostProcessor(vocab=self.vocab)
        self.head = nn.Linear(embedding_units, len(self.vocab) + 1)  # +1 for PAD

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass of the VIPTR model.

        Args:
            x: input image batch of shape (N, C, H, W)
            target: list of strings, the labels for training
            return_model_output: whether to include raw logits
            return_preds: whether to decode predictions

        Returns:
            Dictionary containing model outputs, potential loss, etc.
        """
        features = self.feat_extractor(x).contiguous()  # e.g. (B, seq, embed)
        if features.size(-1) != (len(self.vocab) + 1):
            # If the final classification head wasn't done by the extractor
            features = features[:, : self.max_length]
            B, N, E = features.size()
            features = features.reshape(B * N, E)
            logits = self.head(features).view(B, N, len(self.vocab) + 1)
        else:
            # classification layer was applied inside the feature extractor
            logits = features

        decoded_features = _bf16_to_float32(logits)

        if self.training and target is None:
            raise ValueError("Need to provide labels during training.")

        if target is not None:
            # Build target tensor
            _gt, _seq_len = self.build_target(target)
            gt = torch.from_numpy(_gt).to(dtype=torch.long, device=x.device)
            seq_len = torch.tensor(_seq_len, device=x.device)
            gt = gt[:, : (int(seq_len.max().item()) + 1)]

        out: Dict[str, Any] = {}
        if self.exportable:
            out["logits"] = decoded_features
            return out

        if return_model_output:
            out["out_map"] = decoded_features

        if target is None or return_preds:
            # disable for torch.compile compatibility
            @torch.compiler.disable  # type: ignore[attr-defined]
            def _postprocess(decoded: torch.Tensor) -> List[Tuple[str, float]]:
                return self.postprocessor(decoded)

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
        """
        Compute CTC loss for the model.

        Args:
            model_output: predicted logits (N, T, C)
            gt: ground-truth indices
            seq_len: the length of each label sequence

        Returns:
            The averaged CTC loss
        """
        batch_len = model_output.shape[0]
        input_length = model_output.shape[1] * torch.ones(size=(batch_len,), dtype=torch.int32)
        # (N, T, C) -> (T, N, C)
        logits = model_output.permute(1, 0, 2)
        probs = F.log_softmax(logits, dim=-1)

        ctc_loss = F.ctc_loss(
            probs,
            gt,
            input_length,
            torch.tensor(seq_len, dtype=torch.int, device=gt.device),
            zero_infinity=True,
        )
        return ctc_loss


def _viptr(
    arch: str,
    pretrained: bool,
    backbone_fn: Callable[[bool], nn.Module],
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> VIPTR:
    """
    Internal constructor for the VIPTR model.

    Args:
        arch: architecture key, e.g. 'viptr_tiny'
        pretrained: load pretrained weights?
        backbone_fn: a callable that returns the backbone model
        ignore_keys: list of checkpoint keys to ignore
        **kwargs: additional arguments
    """
    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    vocab = kwargs.get("vocab", _cfg.get("classes", VOCABS["french"]))
    kwargs["vocab"] = vocab
    kwargs["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])
    _cfg["vocab"] = vocab
    _cfg["input_shape"] = kwargs["input_shape"]
    include_top = kwargs.get("include_top", False)
    # out_dim = kwargs.get("out_dim", 384)
    # num_classes = kwargs.get("num_classes", len(vocab) + 1)
    # input_shape = kwargs.get("input_shape", (3, 32, 32))
    # Feature extractor
    feat_extractor = backbone_fn(include_top=include_top)
    model = VIPTR(feat_extractor, cfg=_cfg, **kwargs)

    # Load pretrained parameters
    if pretrained:
        # If user changed vocab, ignore last layer weights
        base_vocab = list(default_cfgs[arch]["classes"])
        _ignore_keys = ignore_keys if vocab != base_vocab else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def viptr_base(pretrained: bool = False, **kwargs: Any) -> VIPTR:
    """
    Construct a VIPTR-Base model.

    Args:
        pretrained: whether to load pretrained weights
        **kwargs: extra parameters for the VIPTR builder

    Returns:
        VIPTR: a VIPTR model instance
    """
    return _viptr(
        "vip_base",
        pretrained,
        vip_base,
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )


def viptr_tiny(pretrained: bool = False, **kwargs: Any) -> VIPTR:
    """
    Construct a VIPTR-Tiny model.

    Args:
        pretrained: whether to load pretrained weights
        **kwargs: extra parameters for the VIPTR builder

    Returns:
        VIPTR: a VIPTR model instance
    """
    return _viptr(
        "vip_tiny",
        pretrained,
        vip_tiny,
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )
