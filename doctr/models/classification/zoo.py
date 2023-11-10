# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, List

from doctr.file_utils import is_tf_available

from .. import classification
from ..preprocessor import PreProcessor
from .predictor import CropOrientationPredictor

__all__ = ["crop_orientation_predictor"]

ARCHS: List[str] = [
    "magc_resnet31",
    "mobilenet_v3_small",
    "mobilenet_v3_small_r",
    "mobilenet_v3_large",
    "mobilenet_v3_large_r",
    "resnet18",
    "resnet31",
    "resnet34",
    "resnet50",
    "resnet34_wide",
    "vgg16_bn_r",
    "vit_s",
    "vit_b",
]
ORIENTATION_ARCHS: List[str] = ["mobilenet_v3_small_orientation"]


def _crop_orientation_predictor(arch: str, pretrained: bool, **kwargs: Any) -> CropOrientationPredictor:
    if arch not in ORIENTATION_ARCHS:
        raise ValueError(f"unknown architecture '{arch}'")

    # Load directly classifier from backbone
    _model = classification.__dict__[arch](pretrained=pretrained)
    kwargs["mean"] = kwargs.get("mean", _model.cfg["mean"])
    kwargs["std"] = kwargs.get("std", _model.cfg["std"])
    kwargs["batch_size"] = kwargs.get("batch_size", 64)
    input_shape = _model.cfg["input_shape"][:-1] if is_tf_available() else _model.cfg["input_shape"][1:]
    predictor = CropOrientationPredictor(
        PreProcessor(input_shape, preserve_aspect_ratio=True, symmetric_pad=True, **kwargs), _model
    )
    return predictor


def crop_orientation_predictor(
    arch: str = "mobilenet_v3_small_orientation", pretrained: bool = False, **kwargs: Any
) -> CropOrientationPredictor:
    """Orientation classification architecture.

    >>> import numpy as np
    >>> from doctr.models import crop_orientation_predictor
    >>> model = crop_orientation_predictor(arch='classif_mobilenet_v3_small', pretrained=True)
    >>> input_crop = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    >>> out = model([input_crop])

    Args:
    ----
        arch: name of the architecture to use (e.g. 'mobilenet_v3_small')
        pretrained: If True, returns a model pre-trained on our recognition crops dataset
        **kwargs: keyword arguments to be passed to the CropOrientationPredictor

    Returns:
    -------
        CropOrientationPredictor
    """
    return _crop_orientation_predictor(arch, pretrained, **kwargs)
