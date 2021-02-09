# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Dict, Any
from .core import OCRPredictor
from . import detection, recognition


__all__ = ["ocr_db_sar"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    'ocr_db_sar': {
        'detection': 'db_resnet50',
        'recognition': 'sar_vgg16_bn',
    },
}


def _predictor(arch: str, pretrained: bool, **kwargs: Any) -> OCRPredictor:

    # Detection
    _model = detection.__dict__[default_cfgs[arch]['detection']](pretrained=pretrained)
    det_predictor = detection.DetectionPredictor(
        detection.DetectionPreProcessor(output_size=_model.cfg['input_shape'][:2], batch_size=2),
        _model,
        detection.__dict__[_model.cfg['post_processor']]()
    )

    # Recognition
    _model = recognition.__dict__[default_cfgs[arch]['recognition']](pretrained=pretrained)
    reco_predictor = recognition.RecognitionPredictor(
        recognition.RecognitionPreProcessor(output_size=_model.cfg['input_shape'][:2], batch_size=16),
        _model,
        recognition.__dict__[_model.cfg['post_processor']](_model.cfg['vocab'])
    )

    return OCRPredictor(det_predictor, reco_predictor)


def ocr_db_sar(pretrained: bool, **kwargs: Any) -> OCRPredictor:
    """End-to-end OCR architecture using a DBNet with a ResNet-50 backbone for localization, and SAR with a VGG-16BN
    backbone as text recognition architecture.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        OCR predictor
    """

    return _predictor('ocr_db_sar', pretrained, **kwargs)
