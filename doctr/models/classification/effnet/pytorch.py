# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional

from torch import nn
from torchvision.models import efficientnet as tv_effnet

from doctr.datasets import VOCABS

from ...utils import load_pretrained_params

__all__ = ['efficientnet_b0', 'efficientnet_b3']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'efficientnet_b0': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 32),
        'classes': list(VOCABS['french']),
    },
    'efficientnet_b3': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 32),
        'classes': list(VOCABS['french']),
    },
}


def _efficientnet(
    arch: str,
    pretrained: bool,
    tv_arch: str,
    num_rect_pools: int = 3,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any
) -> tv_effnet.EfficientNet:

    kwargs['num_classes'] = kwargs.get('num_classes', len(default_cfgs[arch]['classes']))
    kwargs['classes'] = kwargs.get('classes', default_cfgs[arch]['classes'])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg['num_classes'] = kwargs['num_classes']
    _cfg['classes'] = kwargs['classes']
    kwargs.pop('classes')

    # Build the model
    model = tv_effnet.__dict__[tv_arch](pretrained = True)
    # model.features[0][0].stride = (2, 1)
    # model.features[2][0].block[1][0].stride = (2, 1)
    if kwargs.get("vert_stride", None):
        vert_stride = 1
    else:
        vert_stride = 2 
    model.features[3][0].block[1][0].stride = (vert_stride, 1)
    model.features[4][0].block[1][0].stride = (vert_stride, 1)
    model.features[6][0].block[1][0].stride = (vert_stride, 1)
    # print(model.features)
    # print(model)
    # Replace their kernel with rectangular ones


    # model.classifier = nn.Linear(512, kwargs['num_classes'])
    # Load pretrained parameters
    model.cfg = _cfg
    # print(model)
    return model


def efficientnet_b0(pretrained: bool = False, **kwargs: Any) -> tv_effnet.EfficientNet:
    return _efficientnet(
        'efficientnet_b0',
        pretrained,
        'efficientnet_b0',
        3,
        **kwargs,
    )

def efficientnet_b3(pretrained: bool = False, **kwargs: Any) -> tv_effnet.EfficientNet:
    return _efficientnet(
        'efficientnet_b3',
        pretrained,
        'efficientnet_b3',
        3,
        **kwargs,
    )
# convnext_tiny(pretrained = True)