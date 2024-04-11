# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from typing import Callable, Union

from doctr.models import kie_predictor, ocr_predictor

from .schemas import DetectionIn, KIEIn, OCRIn, RecognitionIn


def init_predictor(request: Union[KIEIn, OCRIn, RecognitionIn, DetectionIn]) -> Callable:
    """Initialize the predictor based on the request

    Args:
    ----
        request: input request

    Returns:
    -------
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
            return predictor.det_predictor
        elif isinstance(request, RecognitionIn):
            return predictor.reco_predictor
        return predictor
    elif isinstance(request, KIEIn):
        predictor = kie_predictor(pretrained=True, **params)
        predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
        predictor.det_predictor.model.postprocessor.box_thresh = box_thresh
        return predictor
