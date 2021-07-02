# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from torch import nn
from torchvision.models import vgg as tv_vgg
from typing import Dict, Any

from ...utils import load_pretrained_params


__all__ = ['vgg16_bn']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'vgg16_bn': {'tv_base': 'vgg16_bn', 'rect_pools': [23, 33, 43],
                 'url': None},
}


def _vgg(arch: str, pretrained: bool, **kwargs: Any) -> tv_vgg.VGG:

    # Build the model
    model = tv_vgg.__dict__[default_cfgs[arch]['tv_base']](pretrained=False).features
    for idx in default_cfgs[arch]['rect_pools']:
        model[idx] = nn.MaxPool2d((2, 1))
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def vgg16_bn(pretrained: bool = False, **kwargs: Any) -> tv_vgg.VGG:
    """VGG-16 architecture as described in `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/pdf/1409.1556.pdf>`_, modified by adding batch normalization and rectangular pooling.

    Example::
        >>> import torch
        >>> from doctr.models import vgg16_bn
        >>> model = vgg16_bn(pretrained=False)
        >>> input_tensor = torch.rand((1, 3, 224, 224), dtype=torch.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        VGG feature extractor
    """

    return _vgg('vgg16_bn', pretrained, **kwargs)
