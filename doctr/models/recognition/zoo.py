# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Any

from doctr import is_tf_available
from .predictor import RecognitionPredictor
from doctr.models.preprocessor import PreProcessor
from .. import recognition


__all__ = ["recognition_predictor"]


ARCHS = ['crnn_vgg16_bn', 'crnn_mobilenet_v3_small', 'crnn_mobilenet_v3_large', 'sar_resnet31', 'master']


def _predictor(arch: str, pretrained: bool, **kwargs: Any) -> RecognitionPredictor:

    if arch not in ARCHS:
        raise ValueError(f"unknown architecture '{arch}'")

    _model = recognition.__dict__[arch](pretrained=pretrained)
    kwargs['mean'] = kwargs.get('mean', _model.cfg['mean'])
    kwargs['std'] = kwargs.get('std', _model.cfg['std'])
    kwargs['batch_size'] = kwargs.get('batch_size', 32)
    input_shape = _model.cfg['input_shape'][:2] if is_tf_available() else _model.cfg['input_shape'][-2:]
    predictor = RecognitionPredictor(
        PreProcessor(input_shape, preserve_aspect_ratio=True, **kwargs),
        _model
    )

    return predictor


def recognition_predictor(arch: str = 'crnn_vgg16_bn', pretrained: bool = False, **kwargs: Any) -> RecognitionPredictor:
    """Text recognition architecture.

    Example::
        >>> import numpy as np
        >>> from doctr.models import recognition_predictor
        >>> model = recognition_predictor(pretrained=True)
        >>> input_page = (255 * np.random.rand(32, 128, 3)).astype(np.uint8)
        >>> out = model([input_page])

    Args:
        arch: name of the architecture to use (e.g. 'crnn_vgg16_bn')
        pretrained: If True, returns a model pre-trained on our text recognition dataset

    Returns:
        Recognition predictor
    """

    return _predictor(arch, pretrained, **kwargs)
