# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import math
from typing import Any, Dict, Tuple

import torch
from torch import nn

from ...utils import conv_sequence_pt, load_pretrained_params
from ..resnet import resnet_stage

__all__ = ['MAGCResNet', 'magc_resnet31']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'magc_resnet31': {
        'num_blocks': (1, 2, 5, 3),
        'output_channels': (256, 512, 512, 512),
        'url': None
    },
}


class MAGC(nn.Module):
    """Implements the Multi-Aspect Global Context Attention, as described in
    <https://arxiv.org/pdf/1910.02562.pdf>`_.

    Args:
        inplanes: input channels
        headers: number of headers to split channels
        att_scale: if True, re-scale attention to counteract the variance distibutions
        ratio: bottleneck ratio
        **kwargs
    """

    def __init__(
        self,
        inplanes: int,
        headers: int = 8,
        att_scale: bool = False,
        ratio: float = 0.0625,  # bottleneck ratio of 1/16 as described in paper
    ) -> None:
        super().__init__()

        self.headers = headers
        self.inplanes = inplanes
        self.att_scale = att_scale
        self.planes = int(inplanes * ratio)

        self.single_header_inplanes = int(inplanes / headers)

        self.conv_mask = nn.Conv2d(self.single_header_inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        self.transform = nn.Sequential(
            nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
            nn.LayerNorm([self.planes, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
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
        if self.att_scale and self.headers > 1:
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


class MAGCResNet(nn.Sequential):

    """Implements the modified resnet with MAGC layers, as described in paper.

    Args:
        num_blocks: number of residual blocks in each stage
        output_channels: number of output channels in each stage
        headers: number of header to split channels in MAGC layers
    """

    def __init__(
        self,
        num_blocks: Tuple[int, int, int, int],
        output_channels: Tuple[int, int, int, int],
        headers: int = 8,
    ) -> None:
        _layers = [
            # conv_1x
            *conv_sequence_pt(3, 64, relu=True, bn=True, kernel_size=3, padding=1),
            *conv_sequence_pt(64, 128, relu=True, bn=True, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # Stage 1
            nn.Sequential(
                *resnet_stage(128, output_channels[0], num_blocks[0]),
                MAGC(output_channels[0], headers=headers, att_scale=True),
                *conv_sequence_pt(output_channels[0], output_channels[0], True, True, kernel_size=3, padding=1),
            ),
            nn.MaxPool2d(2),
            # Stage 2
            nn.Sequential(
                *resnet_stage(output_channels[0], output_channels[1], num_blocks[1]),
                MAGC(output_channels[1], headers=headers, att_scale=True),
                *conv_sequence_pt(output_channels[1], output_channels[1], True, True, kernel_size=3, padding=1),
            ),
            nn.MaxPool2d((2, 1)),
            # Stage 3
            nn.Sequential(
                *resnet_stage(output_channels[1], output_channels[2], num_blocks[2]),
                MAGC(output_channels[2], headers=headers, att_scale=True),
                *conv_sequence_pt(output_channels[2], output_channels[2], True, True, kernel_size=3, padding=1),
            ),
            # Stage 4
            nn.Sequential(
                *resnet_stage(output_channels[2], output_channels[3], num_blocks[3]),
                MAGC(output_channels[3], headers=headers, att_scale=True),
                *conv_sequence_pt(output_channels[3], output_channels[3], True, True, kernel_size=3, padding=1),
            ),
        ]
        super().__init__(*_layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def _magc_resnet(arch: str, pretrained: bool, **kwargs: Any) -> MAGCResNet:

    # Build the model
    model = MAGCResNet(
        default_cfgs[arch]['num_blocks'],
        default_cfgs[arch]['output_channels'],
        **kwargs,
    )
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def magc_resnet31(pretrained: bool = False, **kwargs: Any) -> MAGCResNet:
    """Resnet31 architecture with Multi-Aspect Global Context Attention as described in
    `"MASTER: Multi-Aspect Non-local Network for Scene Text Recognition",
    <https://arxiv.org/pdf/1910.02562.pdf>`_.

    Example::
        >>> import torch
        >>> from doctr.models import magc_resnet31
        >>> model = magc_resnet31(pretrained=False)
        >>> input_tensor = torch.rand((1, 3, 224, 224), dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A feature extractor model
    """

    return _magc_resnet('magc_resnet31', pretrained, **kwargs)
