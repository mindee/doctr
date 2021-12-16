# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Any

from doctr.file_utils import is_tf_available, is_torch_available

from .. import classification
from ..preprocessor import PreProcessor
from .predictor import OrientationClassifier

__all__ = ["orientation_classifier"]


if is_tf_available():
    ARCHS = ['classif_mobilenet_v3_small']
elif is_torch_available():
    ARCHS = ['classif_mobilenet_v3_small']


def _predictor(
    arch: str,
    pretrained: bool,
    **kwargs: Any
) -> OrientationClassifier:

    if arch not in ARCHS:
        raise ValueError(f"unknown architecture '{arch}'")

    # Load directly classifier from backbone
    _model = classification.__dict__[arch](pretrained=pretrained)
    kwargs['mean'] = kwargs.get('mean', _model.cfg['mean'])
    kwargs['std'] = kwargs.get('std', _model.cfg['std'])
    kwargs['batch_size'] = kwargs.get('batch_size', 64)
    input_shape = _model.cfg['input_shape'][:-1] if is_tf_available() else _model.cfg['input_shape'][1:]
    predictor = OrientationClassifier(
        PreProcessor(input_shape, preserve_aspect_ratio=True, symmetric_pad=True, **kwargs),
        _model
    )
    return predictor


def detection_predictor(
    arch: str = 'classif_mobilenet_v3_small',
    pretrained: bool = False,
    **kwargs: Any
) -> OrientationClassifier:
    """Orientation classification architecture.

    Example::
        >>> import numpy as np
        >>> from doctr.models import orientation_classifier
        >>> model = orientaztion_classifier(arch='classif_mobilenet_v3_small', pretrained=True)
        >>> input_crop = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
        >>> out = model([input_crop])

    Args:
        arch: name of the architecture to use (e.g. 'mobilenet_v3_small')
        pretrained: If True, returns a model pre-trained on our recognition crops dataset

    Returns:
        Orientation classifier
    """

    return _predictor(arch, pretrained, **kwargs)
