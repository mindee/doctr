# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any

from torch import nn
from torchvision.models import vgg as tv_vgg

from doctr.datasets import VOCABS

from ...utils import load_pretrained_params

__all__ = ["vgg16_bn_r"]


default_cfgs: dict[str, dict[str, Any]] = {
    "vgg16_bn_r": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.4.1/vgg16_bn_r-d108c19c.pt&src=0",
    },
}


def _vgg(
    arch: str,
    pretrained: bool,
    tv_arch: str,
    num_rect_pools: int = 3,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> tv_vgg.VGG:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = tv_vgg.__dict__[tv_arch](**kwargs, weights=None)
    # list the MaxPool2d
    pool_idcs = [idx for idx, m in enumerate(model.features) if isinstance(m, nn.MaxPool2d)]
    # Replace their kernel with rectangular ones
    for idx in pool_idcs[-num_rect_pools:]:
        model.features[idx] = nn.MaxPool2d((2, 1))
    # Patch average pool & classification head
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Linear(512, kwargs["num_classes"])
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    model.cfg = _cfg

    return model


def vgg16_bn_r(pretrained: bool = False, **kwargs: Any) -> tv_vgg.VGG:
    """VGG-16 architecture as described in `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/pdf/1409.1556.pdf>`_, modified by adding batch normalization, rectangular pooling and a simpler
    classification head.

    >>> import torch
    >>> from doctr.models import vgg16_bn_r
    >>> model = vgg16_bn_r(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        **kwargs: keyword arguments of the VGG architecture

    Returns:
        VGG feature extractor
    """
    return _vgg(
        "vgg16_bn_r",
        pretrained,
        "vgg16_bn",
        3,
        ignore_keys=["classifier.weight", "classifier.bias"],
        **kwargs,
    )
