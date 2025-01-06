# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import math
from copy import deepcopy
from functools import partial
from typing import Any

import torch
from torch import nn

from doctr.datasets import VOCABS

from ...utils.pytorch import load_pretrained_params
from ..resnet.pytorch import ResNet

__all__ = ["magc_resnet31"]


default_cfgs: dict[str, dict[str, Any]] = {
    "magc_resnet31": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.4.1/magc_resnet31-857391d8.pt&src=0",
    },
}


class MAGC(nn.Module):
    """Implements the Multi-Aspect Global Context Attention, as described in
    <https://arxiv.org/pdf/1910.02562.pdf>`_.

    Args:
        inplanes: input channels
        headers: number of headers to split channels
        attn_scale: if True, re-scale attention to counteract the variance distibutions
        ratio: bottleneck ratio
        **kwargs
    """

    def __init__(
        self,
        inplanes: int,
        headers: int = 8,
        attn_scale: bool = False,
        ratio: float = 0.0625,  # bottleneck ratio of 1/16 as described in paper
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.headers = headers
        self.inplanes = inplanes
        self.attn_scale = attn_scale
        self.planes = int(inplanes * ratio)

        self.single_header_inplanes = int(inplanes / headers)

        self.conv_mask = nn.Conv2d(self.single_header_inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        self.transform = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = inputs.size()
        # (N * headers, C / headers, H , W)
        x = inputs.view(batch * self.headers, self.single_header_inplanes, height, width)
        shortcut = x
        # (N * headers, C / headers, H * W)
        shortcut = shortcut.view(batch * self.headers, self.single_header_inplanes, height * width)

        # (N * headers, 1, H, W)
        context_mask = self.conv_mask(x)
        # (N * headers, H * W)
        context_mask = context_mask.view(batch * self.headers, -1)

        # scale variance
        if self.attn_scale and self.headers > 1:
            context_mask = context_mask / math.sqrt(self.single_header_inplanes)

        # (N * headers, H * W)
        context_mask = self.softmax(context_mask)

        # (N * headers, C / headers)
        context = (shortcut * context_mask.unsqueeze(1)).sum(-1)

        # (N, C, 1, 1)
        context = context.view(batch, self.headers * self.single_header_inplanes, 1, 1)

        # Transform: B, C, 1, 1 ->  B, C, 1, 1
        transformed = self.transform(context)
        return inputs + transformed


def _magc_resnet(
    arch: str,
    pretrained: bool,
    num_blocks: list[int],
    output_channels: list[int],
    stage_stride: list[int],
    stage_conv: list[bool],
    stage_pooling: list[tuple[int, int] | None],
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> ResNet:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = ResNet(
        num_blocks,
        output_channels,
        stage_stride,
        stage_conv,
        stage_pooling,
        attn_module=partial(MAGC, headers=8, attn_scale=True),
        cfg=_cfg,
        **kwargs,
    )
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def magc_resnet31(pretrained: bool = False, **kwargs: Any) -> ResNet:
    """Resnet31 architecture with Multi-Aspect Global Context Attention as described in
    `"MASTER: Multi-Aspect Non-local Network for Scene Text Recognition",
    <https://arxiv.org/pdf/1910.02562.pdf>`_.

    >>> import torch
    >>> from doctr.models import magc_resnet31
    >>> model = magc_resnet31(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 224, 224), dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the ResNet architecture

    Returns:
        A feature extractor model
    """
    return _magc_resnet(
        "magc_resnet31",
        pretrained,
        [1, 2, 5, 3],
        [256, 256, 512, 512],
        [1, 1, 1, 1],
        [True] * 4,
        [(2, 2), (2, 1), None, None],
        origin_stem=False,
        stem_channels=128,
        ignore_keys=["13.weight", "13.bias"],
        **kwargs,
    )
