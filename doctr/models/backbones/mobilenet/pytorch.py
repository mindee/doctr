# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Greatly inspired by https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv3.py

from torchvision.models import mobilenetv3
from typing import Any, Dict
from doctr.datasets import VOCABS
from ...utils import load_pretrained_params


__all__ = ["mobilenet_v3_small", "mobilenet_v3_small_r", "mobilenet_v3_large", "mobilenet_v3_large_r"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    'mobilenet_v3_large': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 32),
        'vocab': VOCABS['legacy_french'],
        'url': 'https://github.com/mindee/doctr/releases/download/v0.3.0/mobilenet_v3_large-a0aea820.pt',
    },
    'mobilenet_v3_large_r': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 32),
        'rect_stride': ['features.4.block.1.0', 'features.7.block.1.0', 'features.13.block.1.0'],
        'vocab': VOCABS['french'],
        'url': None,
    },
    'mobilenet_v3_small': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 32),
        'vocab': VOCABS['legacy_french'],
        'url': 'https://github.com/mindee/doctr/releases/download/v0.3.0/mobilenet_v3_small-69c7267d.pt',
    },
    'mobilenet_v3_small_r': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 32, 32),
        'rect_stride': ['features.2.block.1.0', 'features.4.block.1.0', 'features.9.block.1.0'],
        'vocab': VOCABS['french'],
        'url': None,
    },
}


def _mobilenet_v3(
    arch: str,
    pretrained: bool,
    **kwargs: Any
) -> mobilenetv3.MobileNetV3:

    kwargs['num_classes'] = kwargs.get('num_classes', len(default_cfgs[arch]['vocab']))

    if arch.startswith("mobilenet_v3_small"):
        model = mobilenetv3.mobilenet_v3_small(**kwargs)
    else:
        model = mobilenetv3.mobilenet_v3_large(**kwargs)

    # Rectangular strides
    if isinstance(default_cfgs[arch].get('rect_stride'), list):
        for layer_name in default_cfgs[arch]['rect_stride']:
            m = model
            for child in layer_name.split('.'):
                m = getattr(m, child)
            m.stride = (2, 1)

    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def mobilenet_v3_small(pretrained: bool = False, **kwargs: Any) -> mobilenetv3.MobileNetV3:
    """MobileNetV3-Small architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_.

    Example::
        >>> import torch
        >>> from doctr.models import mobilenet_v3_small
        >>> model = mobilenetv3_small(pretrained=False)
        >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        a torch.nn.Module
    """

    return _mobilenet_v3('mobilenet_v3_small', pretrained, **kwargs)


def mobilenet_v3_small_r(pretrained: bool = False, **kwargs: Any) -> mobilenetv3.MobileNetV3:
    """MobileNetV3-Small architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_, with rectangular pooling.

    Example::
        >>> import torch
        >>> from doctr.models import mobilenet_v3_small_r
        >>> model = mobilenet_v3_small_r(pretrained=False)
        >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        a torch.nn.Module
    """

    return _mobilenet_v3('mobilenet_v3_small_r', pretrained, **kwargs)


def mobilenet_v3_large(pretrained: bool = False, **kwargs: Any) -> mobilenetv3.MobileNetV3:
    """MobileNetV3-Large architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_.

    Example::
        >>> import torch
        >>> from doctr.models import mobilenetv3_large
        >>> model = mobilenetv3_large(pretrained=False)
        >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        a torch.nn.Module
    """
    return _mobilenet_v3('mobilenet_v3_large', pretrained, **kwargs)


def mobilenet_v3_large_r(pretrained: bool = False, **kwargs: Any) -> mobilenetv3.MobileNetV3:
    """MobileNetV3-Large architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_, with rectangular pooling.

    Example::
        >>> import torch
        >>> from doctr.models import mobilenet_v3_large_r
        >>> model = mobilenet_v3_large_r(pretrained=False)
        >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        a torch.nn.Module
    """
    return _mobilenet_v3('mobilenet_v3_large_r', pretrained, **kwargs)
