# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Greatly inspired by https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv3.py

from copy import deepcopy
from typing import Any, Dict, List, Optional

from torchvision.models import mobilenetv3

from doctr.datasets import VOCABS

from ...utils import load_pretrained_params

__all__ = [
    "mobilenet_v3_small",
    "mobilenet_v3_small_r",
    "mobilenet_v3_large",
    "mobilenet_v3_large_r",
    "mobilenet_v3_small_orientation",
]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "mobilenet_v3_large": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.4.1/mobilenet_v3_large-11fc8cb9.pt&src=0",
    },
    "mobilenet_v3_large_r": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.4.1/mobilenet_v3_large_r-74a22066.pt&src=0",
    },
    "mobilenet_v3_small": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.4.1/mobilenet_v3_small-6a4bfa6b.pt&src=0",
    },
    "mobilenet_v3_small_r": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.4.1/mobilenet_v3_small_r-1a8a3530.pt&src=0",
    },
    "mobilenet_v3_small_orientation": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 128, 128),
        "classes": [0, 90, 180, 270],
        "url": "https://doctr-static.mindee.com/models?id=v0.4.1/classif_mobilenet_v3_small-24f8ff57.pt&src=0",
    },
}


def _mobilenet_v3(
    arch: str,
    pretrained: bool,
    rect_strides: Optional[List[str]] = None,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> mobilenetv3.MobileNetV3:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    if arch.startswith("mobilenet_v3_small"):
        model = mobilenetv3.mobilenet_v3_small(**kwargs, weights=None)
    else:
        model = mobilenetv3.mobilenet_v3_large(**kwargs, weights=None)

    # Rectangular strides
    if isinstance(rect_strides, list):
        for layer_name in rect_strides:
            m = model
            for child in layer_name.split("."):
                m = getattr(m, child)
            m.stride = (2, 1)

    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    model.cfg = _cfg

    return model


def mobilenet_v3_small(pretrained: bool = False, **kwargs: Any) -> mobilenetv3.MobileNetV3:
    """MobileNetV3-Small architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_.

    >>> import torch
    >>> from doctr.models import mobilenet_v3_small
    >>> model = mobilenetv3_small(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the MobileNetV3 architecture

    Returns:
    -------
        a torch.nn.Module
    """
    return _mobilenet_v3(
        "mobilenet_v3_small", pretrained, ignore_keys=["classifier.3.weight", "classifier.3.bias"], **kwargs
    )


def mobilenet_v3_small_r(pretrained: bool = False, **kwargs: Any) -> mobilenetv3.MobileNetV3:
    """MobileNetV3-Small architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_, with rectangular pooling.

    >>> import torch
    >>> from doctr.models import mobilenet_v3_small_r
    >>> model = mobilenet_v3_small_r(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the MobileNetV3 architecture

    Returns:
    -------
        a torch.nn.Module
    """
    return _mobilenet_v3(
        "mobilenet_v3_small_r",
        pretrained,
        ["features.2.block.1.0", "features.4.block.1.0", "features.9.block.1.0"],
        ignore_keys=["classifier.3.weight", "classifier.3.bias"],
        **kwargs,
    )


def mobilenet_v3_large(pretrained: bool = False, **kwargs: Any) -> mobilenetv3.MobileNetV3:
    """MobileNetV3-Large architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_.

    >>> import torch
    >>> from doctr.models import mobilenet_v3_large
    >>> model = mobilenet_v3_large(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the MobileNetV3 architecture

    Returns:
    -------
        a torch.nn.Module
    """
    return _mobilenet_v3(
        "mobilenet_v3_large",
        pretrained,
        ignore_keys=["classifier.3.weight", "classifier.3.bias"],
        **kwargs,
    )


def mobilenet_v3_large_r(pretrained: bool = False, **kwargs: Any) -> mobilenetv3.MobileNetV3:
    """MobileNetV3-Large architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_, with rectangular pooling.

    >>> import torch
    >>> from doctr.models import mobilenet_v3_large_r
    >>> model = mobilenet_v3_large_r(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the MobileNetV3 architecture

    Returns:
    -------
        a torch.nn.Module
    """
    return _mobilenet_v3(
        "mobilenet_v3_large_r",
        pretrained,
        ["features.4.block.1.0", "features.7.block.1.0", "features.13.block.1.0"],
        ignore_keys=["classifier.3.weight", "classifier.3.bias"],
        **kwargs,
    )


def mobilenet_v3_small_orientation(pretrained: bool = False, **kwargs: Any) -> mobilenetv3.MobileNetV3:
    """MobileNetV3-Small architecture as described in
    `"Searching for MobileNetV3",
    <https://arxiv.org/pdf/1905.02244.pdf>`_.

    >>> import torch
    >>> from doctr.models import mobilenet_v3_small_orientation
    >>> model = mobilenet_v3_small_orientation(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
    ----
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the MobileNetV3 architecture

    Returns:
    -------
        a torch.nn.Module
    """
    return _mobilenet_v3(
        "mobilenet_v3_small_orientation",
        pretrained,
        ignore_keys=["classifier.3.weight", "classifier.3.bias"],
        **kwargs,
    )
