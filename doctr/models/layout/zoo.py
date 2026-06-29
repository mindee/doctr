# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any

from doctr.models.utils import _CompiledModule

from .. import layout
from ..preprocessor import PreProcessor
from .predictor import LayoutPredictor

__all__ = ["layout_predictor"]

ARCHS: list[str]

ARCHS = ["lw_detr_s", "lw_detr_m"]


def _predictor(arch: Any, pretrained: bool, assume_straight_pages: bool = True, **kwargs: Any) -> LayoutPredictor:
    if isinstance(arch, str):
        if arch not in ARCHS:
            raise ValueError(f"unknown architecture '{arch}'")

        _model = layout.__dict__[arch](
            pretrained=pretrained,
            assume_straight_pages=assume_straight_pages,
        )
    else:
        # Adding the type for torch compiled models to the allowed architectures
        allowed_archs = [layout.LWDETR, _CompiledModule]

        if not isinstance(arch, tuple(allowed_archs)):
            raise ValueError(f"unknown architecture: {type(arch)}")
        _model = arch

    kwargs.pop("pretrained_backbone", None)

    kwargs["mean"] = kwargs.get("mean", _model.cfg["mean"])
    kwargs["std"] = kwargs.get("std", _model.cfg["std"])
    kwargs["batch_size"] = kwargs.get("batch_size", 2)
    predictor = LayoutPredictor(
        PreProcessor(_model.cfg["input_shape"][1:], **kwargs),
        _model,
    )
    return predictor


def layout_predictor(
    arch: Any = "lw_detr_s",
    pretrained: bool = False,
    assume_straight_pages: bool = True,
    preserve_aspect_ratio: bool = True,
    symmetric_pad: bool = True,
    batch_size: int = 2,
    **kwargs: Any,
) -> LayoutPredictor:
    """Layout prediction architecture.

    >>> import numpy as np
    >>> from doctr.models import layout_predictor
    >>> model = layout_predictor(arch='lw_detr_s', pretrained=True)
    >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
    >>> out = model([input_page])

    Args:
        arch: name of the architecture or model itself to use (e.g. 'lw_detr_s')
        pretrained: If True, returns a model pre-trained on our layout prediction dataset
        assume_straight_pages: If True, fit straight boxes to the page
        preserve_aspect_ratio: If True, pad the input document image to preserve the aspect ratio before
            running the detection model on it
        symmetric_pad: if True, pad the image symmetrically instead of padding at the bottom-right
        batch_size: number of samples the model processes in parallel
        **kwargs: optional keyword arguments passed to the architecture

    Returns:
        Layout predictor
    """
    return _predictor(
        arch=arch,
        pretrained=pretrained,
        assume_straight_pages=assume_straight_pages,
        preserve_aspect_ratio=preserve_aspect_ratio,
        symmetric_pad=symmetric_pad,
        batch_size=batch_size,
        **kwargs,
    )
