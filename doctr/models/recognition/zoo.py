# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, List

from doctr.file_utils import is_tf_available
from doctr.models.preprocessor import PreProcessor

from .. import recognition
from .predictor import RecognitionPredictor

__all__ = ["recognition_predictor"]


ARCHS: List[str] = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "sar_resnet31",
    "master",
    "vitstr_small",
    "vitstr_base",
    "parseq",
]


def _predictor(arch: Any, pretrained: bool, **kwargs: Any) -> RecognitionPredictor:
    if isinstance(arch, str):
        if arch not in ARCHS:
            raise ValueError(f"unknown architecture '{arch}'")

        _model = recognition.__dict__[arch](
            pretrained=pretrained, pretrained_backbone=kwargs.get("pretrained_backbone", True)
        )
    else:
        if not isinstance(
            arch, (recognition.CRNN, recognition.SAR, recognition.MASTER, recognition.ViTSTR, recognition.PARSeq)
        ):
            raise ValueError(f"unknown architecture: {type(arch)}")
        _model = arch

    kwargs.pop("pretrained_backbone", None)

    kwargs["mean"] = kwargs.get("mean", _model.cfg["mean"])
    kwargs["std"] = kwargs.get("std", _model.cfg["std"])
    kwargs["batch_size"] = kwargs.get("batch_size", 32)
    input_shape = _model.cfg["input_shape"][:2] if is_tf_available() else _model.cfg["input_shape"][-2:]
    predictor = RecognitionPredictor(PreProcessor(input_shape, preserve_aspect_ratio=True, **kwargs), _model)

    return predictor


def recognition_predictor(arch: Any = "crnn_vgg16_bn", pretrained: bool = False, **kwargs: Any) -> RecognitionPredictor:
    """Text recognition architecture.

    Example::
        >>> import numpy as np
        >>> from doctr.models import recognition_predictor
        >>> model = recognition_predictor(pretrained=True)
        >>> input_page = (255 * np.random.rand(32, 128, 3)).astype(np.uint8)
        >>> out = model([input_page])

    Args:
    ----
        arch: name of the architecture or model itself to use (e.g. 'crnn_vgg16_bn')
        pretrained: If True, returns a model pre-trained on our text recognition dataset
        **kwargs: optional parameters to be passed to the architecture

    Returns:
    -------
        Recognition predictor
    """
    return _predictor(arch, pretrained, **kwargs)
