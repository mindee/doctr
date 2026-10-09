# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

from doctr.models.preprocessor import PreProcessor
from doctr.models.utils import _CompiledModule

from .. import recognition
from .._utils import ConfidenceAggregation, _resolve_confidence_aggregation
from .predictor import RecognitionPredictor

__all__ = ["recognition_predictor"]


ARCHS: list[str] = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "sar_resnet31",
    "master",
    "vitstr_small",
    "vitstr_base",
    "parseq",
    "viptr_tiny",
]


def _predictor(
    arch: Any,
    pretrained: bool,
    confidence_aggregation: ConfidenceAggregation | None = None,
    **kwargs: Any,
) -> RecognitionPredictor:
    if confidence_aggregation is not None:
        _resolve_confidence_aggregation(confidence_aggregation)

    if isinstance(arch, str):
        if arch not in ARCHS:
            raise ValueError(f"unknown architecture '{arch}'")

        _model = recognition.__dict__[arch](
            pretrained=pretrained, pretrained_backbone=kwargs.get("pretrained_backbone", True)
        )
    else:
        # Adding the type for torch compiled models to the allowed architectures
        allowed_archs = [
            recognition.CRNN,
            recognition.SAR,
            recognition.MASTER,
            recognition.ViTSTR,
            recognition.PARSeq,
            recognition.VIPTR,
            _CompiledModule,
        ]

        if not isinstance(arch, tuple(allowed_archs)):
            raise ValueError(f"unknown architecture: {type(arch)}")
        _model = arch

    if confidence_aggregation is not None:
        _model.postprocessor.confidence_aggregation = confidence_aggregation

    kwargs.pop("pretrained_backbone", None)

    kwargs["mean"] = kwargs.get("mean", _model.cfg["mean"])
    kwargs["std"] = kwargs.get("std", _model.cfg["std"])
    kwargs["batch_size"] = kwargs.get("batch_size", 128)
    input_shape = _model.cfg["input_shape"][-2:]
    predictor = RecognitionPredictor(PreProcessor(input_shape, preserve_aspect_ratio=True, **kwargs), _model)

    return predictor


def recognition_predictor(
    arch: Any = "crnn_vgg16_bn",
    pretrained: bool = False,
    symmetric_pad: bool = False,
    batch_size: int = 128,
    confidence_aggregation: ConfidenceAggregation | None = None,
    **kwargs: Any,
) -> RecognitionPredictor:
    """Text recognition architecture.

    >>> import numpy as np
    >>> from doctr.models import recognition_predictor
    >>> model = recognition_predictor(pretrained=True)
    >>> input_page = (255 * np.random.rand(32, 128, 3)).astype(np.uint8)
    >>> out = model([input_page])

    Args:
        arch: name of the architecture or model itself to use (e.g. 'crnn_vgg16_bn')
        pretrained: If True, returns a model pre-trained on our text recognition dataset
        symmetric_pad: if True, pad the image symmetrically instead of padding at the bottom-right
        batch_size: number of samples the model processes in parallel
        confidence_aggregation: aggregation method of the character probabilities into the word confidence:
            "mean", "min", "max", "median", "geometric_mean", "harmonic_mean" or a callable. If None, the one of
            the model is kept ("min" for CRNN, VIPTR, SAR and MASTER, "mean" for ViTSTR and PARSeq by default)
        **kwargs: optional parameters to be passed to the architecture

    Returns:
        Recognition predictor
    """
    return _predictor(
        arch=arch,
        pretrained=pretrained,
        symmetric_pad=symmetric_pad,
        batch_size=batch_size,
        confidence_aggregation=confidence_aggregation,
        **kwargs,
    )
