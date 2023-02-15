# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Dict

from torchvision.models.detection import FasterRCNN, faster_rcnn

from ...utils import load_pretrained_params

__all__ = ["fasterrcnn_mobilenet_v3_large_fpn"]


default_cfgs: Dict[str, Dict[str, Any]] = {
    "fasterrcnn_mobilenet_v3_large_fpn": {
        "input_shape": (3, 1024, 1024),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "classes": ["background", "qr_code", "bar_code", "logo", "photo"],
        "url": "https://doctr-static.mindee.com/models?id=v0.4.1/fasterrcnn_mobilenet_v3_large_fpn-d5b2490d.pt&src=0",
    },
}


def _fasterrcnn(arch: str, pretrained: bool, **kwargs: Any) -> FasterRCNN:
    _kwargs = {
        "image_mean": default_cfgs[arch]["mean"],
        "image_std": default_cfgs[arch]["std"],
        "box_detections_per_img": 150,
        "box_score_thresh": 0.5,
        "box_positive_fraction": 0.35,
        "box_nms_thresh": 0.2,
        "rpn_nms_thresh": 0.2,
        "num_classes": len(default_cfgs[arch]["classes"]),
    }

    # Build the model
    _kwargs.update(kwargs)
    model = faster_rcnn.__dict__[arch](pretrained=False, pretrained_backbone=False, **_kwargs)
    model.cfg = default_cfgs[arch]

    if pretrained:
        # Load pretrained parameters
        load_pretrained_params(model, default_cfgs[arch]["url"])
    else:
        # Filter keys
        state_dict = {
            k: v
            for k, v in faster_rcnn.__dict__[arch](pretrained=True).state_dict().items()
            if not k.startswith("roi_heads.")
        }

        # Load state dict
        model.load_state_dict(state_dict, strict=False)

    return model


def fasterrcnn_mobilenet_v3_large_fpn(pretrained: bool = False, **kwargs: Any) -> FasterRCNN:
    """Faster-RCNN architecture with a MobileNet V3 backbone as described in `"Faster R-CNN: Towards Real-Time
    Object Detection with Region Proposal Networks" <https://arxiv.org/pdf/1506.01497.pdf>`_.

    >>> import torch
    >>> from doctr.models.obj_detection import fasterrcnn_mobilenet_v3_large_fpn
    >>> model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    >>> input_tensor = torch.rand((1, 3, 1024, 1024), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our object detection dataset

    Returns:
        object detection architecture
    """

    return _fasterrcnn("fasterrcnn_mobilenet_v3_large_fpn", pretrained, **kwargs)
