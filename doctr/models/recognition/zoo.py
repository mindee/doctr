# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Dict, Any
from .core import RecognitionPredictor, RecognitionPreProcessor
from .. import recognition


__all__ = ["crnn_vgg16_bn_predictor", "sar_vgg16_bn_predictor"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    'crnn_vgg16_bn_predictor': {'model': 'crnn_vgg16_bn', 'post_processor': 'CTCPostProcessor'},
    'sar_vgg16_bn_predictor': {'model': 'sar_vgg16_bn', 'post_processor': 'SARPostProcessor'},
}


def _predictor(arch: str, pretrained: bool, **kwargs: Any) -> RecognitionPredictor:

    _model = recognition.__dict__[default_cfgs[arch]['model']](pretrained=pretrained)
    predictor = RecognitionPredictor(
        RecognitionPreProcessor(output_size=_model.cfg['input_shape'][:2], **kwargs),
        _model,
        recognition.__dict__[default_cfgs[arch]['post_processor']](_model.cfg['vocab'])
    )

    return predictor


def crnn_vgg16_bn_predictor(pretrained: bool = False, **kwargs: Any) -> RecognitionPredictor:
    """Text detection architecture using a DBNet with a ResNet-50 backbone.

    Example::
        >>> import numpy as np
        >>> from doctr.models import crnn_vgg16_bn_predictor
        >>> model = crnn_vgg16_bn_predictor(pretrained=False)
        >>> input_page = (255 * np.random.rand(1, 512, 128, 3)).astype(np.uint8)
        >>> out = model([input_page])

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        Recognition predictor
    """

    return _predictor('crnn_vgg16_bn_predictor', pretrained, **kwargs)


def sar_vgg16_bn_predictor(pretrained: bool = False, **kwargs: Any) -> RecognitionPredictor:
    """Text detection architecture using a DBNet with a ResNet-50 backbone.

    Example::
        >>> import numpy as np
        >>> from doctr.models import sar_vgg16_bn_predictor
        >>> model = sar_vgg16_bn_predictor(pretrained=False)
        >>> input_page = (255 * np.random.rand(1, 512, 128, 3)).astype(np.uint8)
        >>> out = model([input_page])

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        Recognition predictor
    """

    return _predictor('sar_vgg16_bn_predictor', pretrained, **kwargs)
