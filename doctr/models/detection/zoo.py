# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Dict, Any
from .core import DetectionPredictor, DetectionPreProcessor
from .. import detection


__all__ = ["db_resnet50_predictor"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    'db_resnet50_predictor': {'model': 'db_resnet50', 'post_processor': 'DBPostProcessor'},
}


def _predictor(arch: str, pretrained: bool, **kwargs: Any) -> DetectionPredictor:
    # Detection
    _model = detection.__dict__[default_cfgs[arch]['model']](pretrained=pretrained)
    kwargs['mean'] = kwargs.get('mean', _model.cfg['mean'])
    kwargs['std'] = kwargs.get('std', _model.cfg['std'])
    predictor = DetectionPredictor(
        DetectionPreProcessor(output_size=_model.cfg['input_shape'][:2], **kwargs),
        _model,
        detection.__dict__[default_cfgs[arch]['post_processor']]()
    )
    return predictor


def db_resnet50_predictor(pretrained: bool = False, **kwargs: Any) -> DetectionPredictor:
    """Text detection architecture using a DBNet with a ResNet-50 backbone.

    Example::
        >>> import numpy as np
        >>> from doctr.models import db_resnet50_predictor
        >>> model = db_resnet50_predictor(pretrained=True)
        >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
        >>> out = model([input_page])

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        Detection predictor
    """

    return _predictor('db_resnet50_predictor', pretrained, **kwargs)
