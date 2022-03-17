# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Any, Dict

from torch import nn
from torchvision.models import vgg as tv_vgg

from doctr.datasets import VOCABS

from ...utils import load_pretrained_params

__all__ = ['vgg16_bn_r']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'vgg16_bn_r': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 32),
        'classes': list(VOCABS['french']),
        'url': 'https://github.com/mindee/doctr/releases/download/v0.4.1/vgg16_bn_r-d108c19c.pt',
    },
}


def _vgg(
    arch: str,
    pretrained: bool,
    tv_arch: str,
    num_rect_pools: int = 3,
    **kwargs: Any
) -> tv_vgg.VGG:

    kwargs['num_classes'] = kwargs.get('num_classes', len(default_cfgs[arch]['classes']))

    # Build the model
    model = tv_vgg.__dict__[tv_arch](**kwargs)
    # List the MaxPool2d
    pool_idcs = [idx for idx, m in enumerate(model.features) if isinstance(m, nn.MaxPool2d)]
    # Replace their kernel with rectangular ones
    for idx in pool_idcs[-num_rect_pools:]:
        model.features[idx] = nn.MaxPool2d((2, 1))
    # Patch average pool & classification head
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Linear(512, kwargs['num_classes'])
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def vgg16_bn_r(pretrained: bool = False, **kwargs: Any) -> tv_vgg.VGG:
    """VGG-16 architecture as described in `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/pdf/1409.1556.pdf>`_, modified by adding batch normalization, rectangular pooling and a simpler
    classification head.

    >>> import torch
    >>> from doctr.models import vgg16_bn_r
    >>> model = vgg16_bn_r(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        VGG feature extractor
    """

    return _vgg('vgg16_bn_r', pretrained, 'vgg16_bn', 3, **kwargs)
