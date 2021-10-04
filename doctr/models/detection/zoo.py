# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Any

from doctr.file_utils import is_tf_available, is_torch_available
from .predictor import DetectionPredictor
from ..preprocessor import PreProcessor
from .. import detection


__all__ = ["detection_predictor"]


if is_tf_available():
    ARCHS = ['db_resnet50', 'db_mobilenet_v3_large', 'linknet16']
elif is_torch_available():
    ARCHS = ['db_resnet34', 'db_resnet50', 'db_mobilenet_v3_large', 'linknet16']


def _predictor(arch: str, pretrained: bool, **kwargs: Any) -> DetectionPredictor:

    if arch not in ARCHS:
        raise ValueError(f"unknown architecture '{arch}'")

    # Detection
    _model = detection.__dict__[arch](pretrained=pretrained)
    kwargs['mean'] = kwargs.get('mean', _model.cfg['mean'])
    kwargs['std'] = kwargs.get('std', _model.cfg['std'])
    kwargs['batch_size'] = kwargs.get('batch_size', 1)
    predictor = DetectionPredictor(
        PreProcessor(_model.cfg['input_shape'][:-1] if is_tf_available() else _model.cfg['input_shape'][1:], **kwargs),
        _model
    )
    return predictor


def detection_predictor(arch: str = 'db_resnet50', pretrained: bool = False, **kwargs: Any) -> DetectionPredictor:
    """Text detection architecture.

    Example::
        >>> import numpy as np
        >>> from doctr.models import detection_predictor
        >>> model = detection_predictor(arch='db_resnet50', pretrained=True)
        >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
        >>> out = model([input_page])

    Args:
        arch: name of the architecture to use (e.g. 'db_resnet50')
        pretrained: If True, returns a model pre-trained on our text detection dataset

    Returns:
        Detection predictor
    """

    return _predictor(arch, pretrained, **kwargs)
