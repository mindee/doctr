# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from copy import deepcopy
from typing import Any, Dict, List, Optional

import torch.nn as nn

from doctr.datasets import VOCABS
from doctr.models.modules.layers.pytorch import RepConvLayer
from doctr.models.utils.pytorch import conv_sequence_pt as conv_sequence

from ...utils import load_pretrained_params

__all__ = ["textnetfast_tiny", "textnetfast_small", "textnetfast_base"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "textnetfast_tiny": {
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
    "textnetfast_small": {
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
    "textnetfast_base": {
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": None,
    },
}


class TextNetFast(nn.Sequential):
    """Implements TextNet architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    Args:
        num_blocks: number of resnet block in each stage
        output_channels: number of channels in each stage
        stage_conv: whether to add a conv_sequence after each stage
        stage_pooling: pooling to add after each stage (if None, no pooling)
        origin_stem: whether to use the orginal ResNet stem or ResNet-31's
        stem_channels: number of output channels of the stem convolutions
        attn_module: attention module to use in each stage
        include_top: whether the classifier head should be instantiated
        num_classes: number of output classes
    """

    def __init__(
        self,
        stage1: Dict,
        stage2: Dict,
        stage3: Dict,
        stage4: Dict,
        include_top: bool = True,
        num_classes: int = 1000,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        _layers: List[nn.Module]
        self.first_conv = conv_sequence(in_channels=3, out_channels=64, relu=True, bn=True, kernel_size=3, stride=2)

        _layers = [*self.first_conv]

        for stage in [stage1, stage2, stage3, stage4]:
            stage_ = nn.ModuleList([RepConvLayer(**params) for params in stage])
            _layers.extend([*stage_])

        if include_top:
            _layers.extend(
                [
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1),
                    nn.Linear(512, num_classes, bias=True),
                ]
            )

        super().__init__(*_layers)
        self.cfg = cfg

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def _textnetfast(
    arch: str,
    pretrained: bool,
    arch_fn,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> TextNetFast:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = arch_fn(**kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    model.cfg = _cfg

    return model


def textnetfast_tiny(pretrained: bool = False, **kwargs: Any) -> TextNetFast:
    """Implements TextNet architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    >>> import torch
    >>> from doctr.models import textnetfast_tiny
    >>> model = textnetfast_tiny(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A TextNet model
    """

    return _textnetfast(
        "textnetfast_tiny",
        pretrained,
        TextNetFast,
        stage1=[
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
        ],
        stage2=[
            {"in_channels": 64, "out_channels": 128, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
        ],
        stage3=[
            {"in_channels": 128, "out_channels": 256, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [1, 3], "stride": 1},
        ],
        stage4=[
            {"in_channels": 256, "out_channels": 512, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 3], "stride": 1},
        ],
        ignore_keys=["10.weight", "10.bias"],
        **kwargs,
    )


def textnetfast_small(pretrained: bool = False, **kwargs: Any) -> TextNetFast:
    """Implements TextNet architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    >>> import torch
    >>> from doctr.models import textnetfast_small
    >>> model = textnetfast_small(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A TextNetFast model
    """

    return _textnetfast(
        "textnetfast_small",
        pretrained,
        TextNetFast,
        stage1=[
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 2},
        ],
        stage2=[
            {"in_channels": 64, "out_channels": 128, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
        ],
        stage3=[
            {"in_channels": 128, "out_channels": 256, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
        ],
        stage4=[
            {"in_channels": 256, "out_channels": 512, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},
        ],
        ignore_keys=["10.weight", "10.bias"],
        **kwargs,
    )


def textnetfast_base(pretrained: bool = False, **kwargs: Any) -> TextNetFast:
    """Implements TextNet architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    >>> import torch
    >>> from doctr.models import textnetfast_base
    >>> model = textnetfast_base(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained

    Returns:
        A TextNetFast model
    """

    return _textnetfast(
        "textnetfast_base",
        pretrained,
        TextNetFast,
        stage1=[
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": [3, 3], "stride": 1},
        ],
        stage2=[
            {"in_channels": 64, "out_channels": 128, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 128, "out_channels": 128, "kernel_size": [3, 3], "stride": 1},
        ],
        stage3=[
            {"in_channels": 128, "out_channels": 256, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 3], "stride": 1},
            {"in_channels": 256, "out_channels": 256, "kernel_size": [3, 1], "stride": 1},
        ],
        stage4=[
            {"in_channels": 256, "out_channels": 512, "kernel_size": [3, 3], "stride": 2},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [3, 1], "stride": 1},
            {"in_channels": 512, "out_channels": 512, "kernel_size": [1, 3], "stride": 1},
        ],
        ignore_keys=["10.weight", "10.bias"],
        **kwargs,
    )
