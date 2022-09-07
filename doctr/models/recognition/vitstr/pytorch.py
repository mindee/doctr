# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from doctr.datasets import VOCABS

from ...modules import VisionTransformer
from ...utils.pytorch import load_pretrained_params
from ..core import RecognitionModel, RecognitionPostProcessor

__all__ = ["ViTSTR", "vitstr"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "vitstr": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 128),
        "vocab": VOCABS["french"],
        "url": None,
    },
}


class ViTSTR(nn.Module, RecognitionModel):
    """Implements a ViTSTR architecture as described in `"Vision Transformer for Fast and
    Efficient Scene Text Recognition" <https://arxiv.org/pdf/2105.08582.pdf>`_.

    Args:
        vocab: vocabulary used for encoding
        embedding_units: number of embedding units
        max_length: maximum word length handled by the model
        dropout_prob: dropout probability of the encoder LSTM
        input_shape: input shape of the image
        patch_size: size of the patches
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    def __init__(
        self,
        vocab: str,
        embedding_units: int = 384,
        max_length: int = 25,
        dropout_prob: float = 0.0,
        input_shape: Tuple[int, int, int] = (3, 32, 128),  # different from paper
        patch_size: Tuple[int, int] = (4, 8),  # different from paper to match our size
        exportable: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg

        self.max_length = max_length + 1  # Add 1 timestep for EOS after the longest word

        self.feat_extractor = VisionTransformer(
            img_size=input_shape[1:],
            patch_size=patch_size,
            d_model=embedding_units,
            num_layers=12,
            num_heads=6,
            dropout=dropout_prob,
        )
        self.head = nn.Linear(embedding_units, len(self.vocab) + 1)

        self.postprocessor = ViTSTRPostProcessor(vocab=self.vocab)

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
    ) -> Dict[str, Any]:

        features = self.feat_extractor(x)  # (batch_size, seq_len, d_model)

        if target is not None:
            _gt, _seq_len = self.build_target(target)
            gt, seq_len = torch.from_numpy(_gt).to(dtype=torch.long), torch.tensor(_seq_len)
            gt, seq_len = gt.to(x.device), seq_len.to(x.device)

        if self.training and target is None:
            raise ValueError("Need to provide labels during training")

        # borrowed from : https://github.com/baudm/parseq/blob/main/strhub/models/vitstr/model.py
        features = features[:, : self.max_length + 1]  # add 1 for unused cls token (ViT)
        # batch, seqlen, embedding_size
        B, N, E = features.size()
        features = features.reshape(B * N, E)
        logits = self.head(features).view(B, N, len(self.vocab) + 1)  # (batch, seqlen, vocab + 1)
        decoded_features = logits[:, 1:]  # remove cls_token

        out: Dict[str, Any] = {}
        if self.exportable:
            out["logits"] = decoded_features
            return out

        if return_model_output:
            out["out_map"] = decoded_features

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(decoded_features)

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
        # Input length : number of timesteps
        input_len = model_output.shape[1]
        # Add one for additional <eos> token
        seq_len = seq_len + 1
        # Compute loss
        # (N, L, vocab_size + 1)
        cce = F.cross_entropy(model_output.permute(0, 2, 1), gt, reduction="none")
        mask_2d = torch.arange(input_len, device=model_output.device)[None, :] >= seq_len[:, None]
        cce[mask_2d] = 0

        ce_loss = cce.sum(1) / seq_len.to(dtype=model_output.dtype)
        return ce_loss.mean()


class ViTSTRPostProcessor(RecognitionPostProcessor):
    """Post processor for ViTSTR architectures

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
            for encoded_seq in out_idxs.detach().cpu().numpy()
        ]

        return list(zip(word_values, probs.numpy().tolist()))


def _vitstr(
    arch: str,
    pretrained: bool,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> ViTSTR:

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])

    kwargs["vocab"] = _cfg["vocab"]
    kwargs["input_shape"] = _cfg["input_shape"]

    # Build the model
    model = ViTSTR(cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if _cfg["vocab"] != default_cfgs[arch]["vocab"] else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def vitstr(pretrained: bool = False, **kwargs: Any) -> ViTSTR:
    """ViTSTR as described in `"Vision Transformer for Fast and Efficient Scene Text Recognition"
    <https://arxiv.org/pdf/2105.08582.pdf>`_.

    >>> import torch
    >>> from doctr.models import vitstr
    >>> model = vitstr(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 128))
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _vitstr(
        "vitstr",
        pretrained,
        ignore_keys=["head.weight", "head.bias"],
        **kwargs,
    )
