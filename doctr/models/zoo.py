# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Dict, Any
from .core import OCRPredictor
from .detection import zoo as det_zoo
from .recognition import zoo as reco_zoo


__all__ = ["ocr_db_sar_vgg", "ocr_db_crnn_vgg", "ocr_db_sar_resnet"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    'ocr_db_sar_vgg': {'detection': 'db_resnet50_predictor', 'recognition': 'sar_vgg16_bn_predictor'},
    'ocr_db_sar_resnet': {'detection': 'db_resnet50_predictor', 'recognition': 'sar_resnet31_predictor'},
    'ocr_db_crnn_vgg': {'detection': 'db_resnet50_predictor', 'recognition': 'crnn_vgg16_bn_predictor'},
}


def _predictor(arch: str, pretrained: bool, det_bs=2, reco_bs=32, **kwargs: Any) -> OCRPredictor:

    # Detection
    det_predictor = det_zoo.__dict__[default_cfgs[arch]['detection']](pretrained=pretrained, batch_size=det_bs)

    # Recognition
    reco_predictor = reco_zoo.__dict__[default_cfgs[arch]['recognition']](pretrained=pretrained, batch_size=reco_bs)

    return OCRPredictor(det_predictor, reco_predictor)


def ocr_db_sar_vgg(pretrained: bool = False, **kwargs: Any) -> OCRPredictor:
    """End-to-end OCR architecture using a DBNet with a ResNet-50 backbone for localization, and SAR with a VGG-16BN
    backbone as text recognition architecture.

    Example::
        >>> import numpy as np
        >>> from doctr.models import ocr_db_sar_vgg
        >>> model = ocr_db_sar_vgg(pretrained=True)
        >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
        >>> out = model([[input_page]])

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        OCR predictor
    """

    return _predictor('ocr_db_sar_vgg', pretrained, **kwargs)


def ocr_db_sar_resnet(pretrained: bool = False, **kwargs: Any) -> OCRPredictor:
    """End-to-end OCR architecture using a DBNet with a ResNet-50 backbone for localization, and SAR with a VGG-16BN
    backbone as text recognition architecture.

    Example::
        >>> import numpy as np
        >>> from doctr.models import ocr_db_sar_resnet
        >>> model = ocr_db_sar_resnet(pretrained=True)
        >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
        >>> out = model([[input_page]])

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        OCR predictor
    """

    return _predictor('ocr_db_sar_resnet', pretrained, **kwargs)


def ocr_db_crnn_vgg(pretrained: bool = False, **kwargs: Any) -> OCRPredictor:
    """End-to-end OCR architecture using a DBNet with a ResNet-50 backbone for localization, and CRNN with a VGG-16BN
    backbone as text recognition architecture.

    Example::
        >>> import numpy as np
        >>> from doctr.models import ocr_db_crnn_vgg
        >>> model = ocr_db_crnn_vgg(pretrained=True)
        >>> input_page = (255 * np.random.rand(600, 800, 3)).astype(np.uint8)
        >>> out = model([[input_page]])

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        OCR predictor
    """

    return _predictor('ocr_db_crnn_vgg', pretrained, **kwargs)
