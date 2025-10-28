# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from collections.abc import Callable

import torch

from doctr.models import kie_predictor, ocr_predictor

from .schemas import DetectionIn, KIEIn, OCRIn, RecognitionIn


def _move_to_device(predictor: Callable) -> Callable:
    """Move the predictor to the desired device

    Args:
        predictor: the predictor to move

    Returns:
        Callable: the predictor moved to the desired device
    """
    return predictor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def init_predictor(request: KIEIn | OCRIn | RecognitionIn | DetectionIn) -> Callable:
    """Initialize the predictor based on the request

    Args:
        request: input request

    Returns:
        Callable: the predictor
    """
    params = request.model_dump()
    bin_thresh = params.pop("bin_thresh", None)
    box_thresh = params.pop("box_thresh", None)
    if isinstance(request, (OCRIn, RecognitionIn, DetectionIn)):
        predictor = ocr_predictor(pretrained=True, **params)
        predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
        predictor.det_predictor.model.postprocessor.box_thresh = box_thresh
        if isinstance(request, DetectionIn):
            return _move_to_device(predictor.det_predictor)
        elif isinstance(request, RecognitionIn):
            return _move_to_device(predictor.reco_predictor)
        return _move_to_device(predictor)
    elif isinstance(request, KIEIn):
        predictor = kie_predictor(pretrained=True, **params)
        predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
        predictor.det_predictor.model.postprocessor.box_thresh = box_thresh
        return _move_to_device(predictor)
