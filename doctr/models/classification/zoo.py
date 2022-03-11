# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Any

from doctr.file_utils import is_tf_available, is_torch_available

from .. import classification
from ..preprocessor import PreProcessor
from .predictor import CropOrientationPredictor

__all__ = ["crop_orientation_predictor"]


if is_tf_available():
    ARCHS = ['mobilenet_v3_small_orientation']
elif is_torch_available():
    ARCHS = ['mobilenet_v3_small_orientation']


def _crop_orientation_predictor(
    arch: str,
    pretrained: bool,
    **kwargs: Any
) -> CropOrientationPredictor:

    if arch not in ARCHS:
        raise ValueError(f"unknown architecture '{arch}'")

    # Load directly classifier from backbone
    _model = classification.__dict__[arch](pretrained=pretrained)
    kwargs['mean'] = kwargs.get('mean', _model.cfg['mean'])
    kwargs['std'] = kwargs.get('std', _model.cfg['std'])
    kwargs['batch_size'] = kwargs.get('batch_size', 64)
    input_shape = _model.cfg['input_shape'][:-1] if is_tf_available() else _model.cfg['input_shape'][1:]
    predictor = CropOrientationPredictor(
        PreProcessor(input_shape, preserve_aspect_ratio=True, symmetric_pad=True, **kwargs),
        _model
    )
    return predictor


def crop_orientation_predictor(
    arch: str = 'mobilenet_v3_small_orientation',
    pretrained: bool = False,
    **kwargs: Any
) -> CropOrientationPredictor:
    """Orientation classification architecture.

    >>> import numpy as np
    >>> from doctr.models import crop_orientation_predictor
    >>> model = crop_orientation_predictor(arch='classif_mobilenet_v3_small', pretrained=True)
    >>> input_crop = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    >>> out = model([input_crop])

    Args:
        arch: name of the architecture to use (e.g. 'mobilenet_v3_small')
        pretrained: If True, returns a model pre-trained on our recognition crops dataset

    Returns:
        CropOrientationPredictor
    """

    return _crop_orientation_predictor(arch, pretrained, **kwargs)
