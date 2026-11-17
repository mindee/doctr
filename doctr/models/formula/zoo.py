# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

from doctr.models.utils import _CompiledModule

from .. import formula
from .._utils import ConfidenceAggregation, _resolve_confidence_aggregation
from ..preprocessor import PreProcessor
from .predictor import FormulaPredictor

__all__ = ["formula_predictor"]

ARCHS: list[str] = ["formulanet"]


def _predictor(
    arch: Any,
    pretrained: bool,
    confidence_aggregation: ConfidenceAggregation | None = None,
    max_decode_length: int | None = None,
    **kwargs: Any,
) -> FormulaPredictor:
    if confidence_aggregation is not None:
        _resolve_confidence_aggregation(confidence_aggregation)

    if isinstance(arch, str):
        if arch not in ARCHS:
            raise ValueError(f"unknown architecture '{arch}'")
        _model = formula.__dict__[arch](
            pretrained=pretrained, pretrained_backbone=kwargs.get("pretrained_backbone", True)
        )
    else:
        # Adding the type for torch compiled models to the allowed architectures
        allowed_archs = [formula.FormulaNet, _CompiledModule]
        if not isinstance(arch, tuple(allowed_archs)):
            raise ValueError(f"unknown architecture: {type(arch)}")
        _model = arch

    if confidence_aggregation is not None:
        _model.postprocessor.confidence_aggregation = confidence_aggregation
    if max_decode_length is not None:
        _model.max_decode_length = max_decode_length

    kwargs.pop("pretrained_backbone", None)
    kwargs["mean"] = kwargs.get("mean", _model.cfg["mean"])
    kwargs["std"] = kwargs.get("std", _model.cfg["std"])
    kwargs["batch_size"] = kwargs.get("batch_size", 16)
    chunk_size = kwargs.pop("chunk_size", 128)
    # Same resizing as during training: aspect ratio preserved, padded at the bottom-right
    kwargs.setdefault("symmetric_pad", False)
    predictor = FormulaPredictor(
        PreProcessor(_model.cfg["input_shape"][1:], preserve_aspect_ratio=True, **kwargs),
        _model,
        chunk_size=chunk_size,
    )
    return predictor


def formula_predictor(
    arch: Any = "formulanet",
    pretrained: bool = False,
    batch_size: int = 16,
    confidence_aggregation: ConfidenceAggregation | None = None,
    max_decode_length: int | None = None,
    **kwargs: Any,
) -> FormulaPredictor:
    """Formula recognition architecture: turns images of mathematical formulas into LaTeX strings.

    >>> import numpy as np
    >>> from doctr.models import formula_predictor
    >>> model = formula_predictor(arch='formulanet', pretrained=True)
    >>> input_crop = (255 * np.random.rand(64, 256, 3)).astype(np.uint8)
    >>> out = model([input_crop])

    Args:
        arch: name of the architecture or model itself to use (e.g. 'formulanet')
        pretrained: If True, returns a model pre-trained on a formula recognition dataset
        batch_size: number of samples the model processes in parallel
        confidence_aggregation: aggregation method of the token probabilities into the formula confidence:
            "mean", "min", "max", "median", "geometric_mean", "harmonic_mean" or a callable. If None, the one of
            the model is kept ("mean" by default)
        max_decode_length: maximum number of tokens generated per formula. If None, the one of the model is kept
            (2048 by default). A model instance passed as `arch` is modified.
        chunk_size: maximum number of formula images decoded and processed at once (128 by default): longer inputs and
            lazy page sources (cf. `DocumentFile.from_images(..., lazy=True)`) are processed chunk by chunk
        **kwargs: optional keyword arguments passed to the architecture

    Returns:
        Formula recognition predictor
    """
    return _predictor(
        arch=arch,
        pretrained=pretrained,
        batch_size=batch_size,
        confidence_aggregation=confidence_aggregation,
        max_decode_length=max_decode_length,
        **kwargs,
    )
