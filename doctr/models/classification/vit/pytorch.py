# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any

import torch
from torch import nn

from doctr.datasets import VOCABS
from doctr.models.modules.transformer import EncoderBlock
from doctr.models.modules.vision_transformer.pytorch import PatchEmbedding

from ...utils.pytorch import load_pretrained_params

__all__ = ["vit_s", "vit_b"]


default_cfgs: dict[str, dict[str, Any]] = {
    "vit_s": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.6.0/vit_s-5d05442d.pt&src=0",
    },
    "vit_b": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.6.0/vit_b-0fbef167.pt&src=0",
    },
}


class ClassifierHead(nn.Module):
    """Classifier head for Vision Transformer

    Args:
        in_channels: number of input channels
        num_classes: number of output classes
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self.head = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, num_classes) cls token
        return self.head(x[:, 0])


class VisionTransformer(nn.Sequential):
    """VisionTransformer architecture as described in
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_.

    Args:
        d_model: dimension of the transformer layers
        num_layers: number of transformer layers
        num_heads: number of attention heads
        ffd_ratio: multiplier for the hidden dimension of the feedforward layer
        patch_size: size of the patches
        input_shape: size of the input image
        dropout: dropout rate
        num_classes: number of output classes
        include_top: whether the classifier head should be instantiated
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffd_ratio: int,
        patch_size: tuple[int, int] = (4, 4),
        input_shape: tuple[int, int, int] = (3, 32, 32),
        dropout: float = 0.0,
        num_classes: int = 1000,
        include_top: bool = True,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        _layers: list[nn.Module] = [
            PatchEmbedding(input_shape, d_model, patch_size),
            EncoderBlock(num_layers, num_heads, d_model, d_model * ffd_ratio, dropout, nn.GELU()),
        ]
        if include_top:
            _layers.append(ClassifierHead(d_model, num_classes))

        super().__init__(*_layers)
        self.cfg = cfg


def _vit(
    arch: str,
    pretrained: bool,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> VisionTransformer:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["input_shape"] = kwargs.get("input_shape", default_cfgs[arch]["input_shape"])
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["input_shape"] = kwargs["input_shape"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = VisionTransformer(cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def vit_s(pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
    """VisionTransformer-S architecture
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_. Patches: (H, W) -> (H/8, W/8)

    NOTE: unofficial config used in ViTSTR and ParSeq

    >>> import torch
    >>> from doctr.models import vit_s
    >>> model = vit_s(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the VisionTransformer architecture

    Returns:
        A feature extractor model
    """
    return _vit(
        "vit_s",
        pretrained,
        d_model=384,
        num_layers=12,
        num_heads=6,
        ffd_ratio=4,
        ignore_keys=["2.head.weight", "2.head.bias"],
        **kwargs,
    )


def vit_b(pretrained: bool = False, **kwargs: Any) -> VisionTransformer:
    """VisionTransformer-B architecture as described in
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
    <https://arxiv.org/pdf/2010.11929.pdf>`_. Patches: (H, W) -> (H/8, W/8)

    >>> import torch
    >>> from doctr.models import vit_b
    >>> model = vit_b(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the VisionTransformer architecture

    Returns:
        A feature extractor model
    """
    return _vit(
        "vit_b",
        pretrained,
        d_model=768,
        num_layers=12,
        num_heads=12,
        ffd_ratio=4,
        ignore_keys=["2.head.weight", "2.head.bias"],
        **kwargs,
    )
