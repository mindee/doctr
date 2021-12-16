# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Greatly inspired by https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv3.py

from typing import Any, Dict, Tuple

from torchvision.models import mobilenetv3
from copy import deepcopy

from ...utils import load_pretrained_params

__all__ = ["classif_mobilenet_v3_small"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    'classif_mobilenet_v3_small': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (3, 128, 128),
        'url': ''
    },
}

def _classif_mobilenet_v3_small(
    arch: str,
    pretrained: bool,
    input_shape: Tuple[int, int, int] = None,
    **kwargs: Any,
) -> mobilenetv3.MobileNetV3:

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']

    kwargs['input_shape'] = _cfg['input_shape']
    # Build the model
    model = mobilenetv3.mobilenet_v3_small(input_shape=input_shape, num_classes=4)

    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, _cfg['url'])

    return model


def classif_mobilenet_v3_small(pretrained: bool = False, **kwargs: Any) -> mobilenetv3.MobileNetV3:
    """MobileNetV3-Small architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import mobilenetv3_large
        >>> model = mobilenetv3_small(pretrained=False)
        >>> input_tensor = tf.random.uniform(shape=[1, 512, 512, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        a keras.Model
    """

    return _classif_mobilenet_v3_small('mobilenet_v3_small', pretrained, **kwargs)
