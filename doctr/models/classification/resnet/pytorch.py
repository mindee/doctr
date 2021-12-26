# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


from typing import Any, Callable, Dict, List, Optional, Tuple

from torch import nn
from torchvision.models.resnet import BasicBlock

from doctr.datasets import VOCABS

from ...utils import conv_sequence_pt, load_pretrained_params

__all__ = ['ResNet', 'resnet31', 'resnet_stage']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'resnet31': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 32),
        'classes': list(VOCABS['french']),
        'url': 'https://github.com/mindee/doctr/releases/download/v0.4.1/resnet31-1056cc5c.pt',
    },
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
        num_blocks: number of resnet block in each stage
        output_channels: number of channels in each stage
        stage_conv: whether to add a conv_sequence after each stage
        stage_pooling: pooling to add after each stage (if None, no pooling)
        attn_module: attention module to use in each stage
        include_top: whether the classifier head should be instantiated
        num_classes: number of output classes
    """

    def __init__(
        self,
        num_blocks: List[int],
        output_channels: List[int],
        stage_conv: List[bool],
        stage_pooling: List[Optional[Tuple[int, int]]],
        attn_module: Optional[Callable[[int], nn.Module]] = None,
        include_top: bool = True,
        num_classes: int = 1000,
    ) -> None:

        _layers: List[nn.Module] = [
            *conv_sequence_pt(3, 64, True, True, kernel_size=3, padding=1),
            *conv_sequence_pt(64, 128, True, True, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
        ]
        in_chans = [128] + output_channels[:-1]
        for in_chan, out_chan, n_blocks, conv, pool in zip(in_chans, output_channels, num_blocks, stage_conv,
                                                           stage_pooling):
            _stage = resnet_stage(in_chan, out_chan, n_blocks)
            if attn_module is not None:
                _stage.append(attn_module(out_chan))
            if conv:
                _stage.extend(conv_sequence_pt(out_chan, out_chan, True, True, kernel_size=3, padding=1))
            if pool is not None:
                _stage.append(nn.MaxPool2d(pool))
            _layers.append(nn.Sequential(*_stage))

        if include_top:
            _layers.extend([
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(output_channels[-1], num_classes, bias=True),
            ])

        super().__init__(*_layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def _resnet(
    arch: str,
    pretrained: bool,
    num_blocks: List[int],
    output_channels: List[int],
    stage_conv: List[bool],
    stage_pooling: List[Optional[Tuple[int, int]]],
    **kwargs: Any,
) -> ResNet:

    kwargs['num_classes'] = kwargs.get('num_classes', len(default_cfgs[arch]['classes']))

    # Build the model
    model = ResNet(num_blocks, output_channels, stage_conv, stage_pooling, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def resnet31(pretrained: bool = False, **kwargs: Any) -> ResNet:
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

    return _resnet(
        'resnet31',
        pretrained,
        [1, 2, 5, 3],
        [256, 256, 512, 512],
        [True] * 4,
        [(2, 2), (2, 1), None, None],
        **kwargs,
    )
