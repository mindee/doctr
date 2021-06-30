# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


from torch import nn
from torchvision.models.resnet import BasicBlock

from typing import Tuple, Dict, Any, List
from ...utils import conv_sequence_pt, load_pretrained_params

__all__ = ['ResNet', 'resnet31', 'resnet_stage']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'resnet31': {'num_blocks': (1, 2, 5, 3), 'output_channels': (256, 256, 512, 512),
                 'url': None},
}


def resnet_stage(in_channels: int, out_channels: int, num_blocks: int) -> List[nn.Module]:
    _layers: List[nn.Module] = []

    in_chan = in_channels
    for _ in range(num_blocks):
        downsample = None
        if in_chan != out_channels:
            downsample = nn.Sequential(*conv_sequence_pt(in_chan, out_channels, False, True, kernel_size=1))

        _layers.append(BasicBlock(in_chan, out_channels, downsample=downsample))
        in_chan = out_channels

    return _layers


class ResNet(nn.Sequential):
    """Implements a ResNet-31 architecture from `"Show, Attend and Read:A Simple and Strong Baseline for Irregular
    Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Args:
        num_blocks: number of residual blocks in each stage
        output_channels: number of output channels in each stage
    """

    def __init__(
        self,
        num_blocks: Tuple[int, int, int, int],
        output_channels: Tuple[int, int, int, int],
    ) -> None:

        _layers: List[nn.Module] = [
            *conv_sequence_pt(3, 64, True, True, kernel_size=3, padding=1),
            *conv_sequence_pt(64, 128, True, True, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            # Stage 1
            nn.Sequential(
                *resnet_stage(128, output_channels[0], num_blocks[0]),
                *conv_sequence_pt(output_channels[0], output_channels[0], True, True, kernel_size=3, padding=1),
            ),
            nn.MaxPool2d(2),
            # Stage 2
            nn.Sequential(
                *resnet_stage(output_channels[0], output_channels[1], num_blocks[1]),
                *conv_sequence_pt(output_channels[1], output_channels[1], True, True, kernel_size=3, padding=1),
            ),
            nn.MaxPool2d((2, 1)),
            # Stage 3
            nn.Sequential(
                *resnet_stage(output_channels[1], output_channels[2], num_blocks[2]),
                *conv_sequence_pt(output_channels[2], output_channels[2], True, True, kernel_size=3, padding=1),
            ),
            # Stage 4
            nn.Sequential(
                *resnet_stage(output_channels[2], output_channels[3], num_blocks[3]),
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


def _resnet(arch: str, pretrained: bool) -> ResNet:

    # Build the model
    model = ResNet(
        default_cfgs[arch]['num_blocks'],
        default_cfgs[arch]['output_channels'],
    )
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def resnet31(pretrained: bool = False) -> ResNet:
    """Resnet31 architecture with rectangular pooling windows as described in
    `"Show, Attend and Read:A Simple and Strong Baseline for Irregular Text Recognition",
    <https://arxiv.org/pdf/1811.00751.pdf>`_. Downsizing: (H, W) --> (H/8, W/4)

    Example::
        >>> import torch
        >>> from doctr.models import resnet31
        >>> model = resnet31(pretrained=False)
        >>> input_tensor = torch.rand((1, 3, 224, 224), dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A resnet31 model
    """

    return _resnet('resnet31', pretrained)
