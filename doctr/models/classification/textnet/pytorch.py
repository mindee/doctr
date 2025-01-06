# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from copy import deepcopy
from typing import Any

from torch import nn

from doctr.datasets import VOCABS

from ...modules.layers.pytorch import FASTConvLayer
from ...utils import conv_sequence_pt, load_pretrained_params

__all__ = ["textnet_tiny", "textnet_small", "textnet_base"]

default_cfgs: dict[str, dict[str, Any]] = {
    "textnet_tiny": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.8.1/textnet_tiny-27288d12.pt&src=0",
    },
    "textnet_small": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.8.1/textnet_small-43166ee6.pt&src=0",
    },
    "textnet_base": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.8.1/textnet_base-7f68d7e0.pt&src=0",
    },
}


class TextNet(nn.Sequential):
    """Implements TextNet architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    Args:
        stages (list[dict[str, list[int]]]): list of dictionaries containing the parameters of each stage.
        include_top (bool, optional): Whether to include the classifier head. Defaults to True.
        num_classes (int, optional): Number of output classes. Defaults to 1000.
        cfg (dict[str, Any], optional): Additional configuration. Defaults to None.
    """

    def __init__(
        self,
        stages: list[dict[str, list[int]]],
        input_shape: tuple[int, int, int] = (3, 32, 32),
        num_classes: int = 1000,
        include_top: bool = True,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        _layers: list[nn.Module] = [
            *conv_sequence_pt(
                in_channels=3, out_channels=64, relu=True, bn=True, kernel_size=3, stride=2, padding=(1, 1)
            ),
            *[
                nn.Sequential(*[
                    FASTConvLayer(**params)  # type: ignore[arg-type]
                    for params in [{key: stage[key][i] for key in stage} for i in range(len(stage["in_channels"]))]
                ])
                for stage in stages
            ],
        ]

        if include_top:
            _layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1),
                    nn.Linear(stages[-1]["out_channels"][-1], num_classes),
                )
            )

        super().__init__(*_layers)
        self.cfg = cfg

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def _textnet(
    arch: str,
    pretrained: bool,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> TextNet:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = TextNet(**kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    model.cfg = _cfg

    return model


def textnet_tiny(pretrained: bool = False, **kwargs: Any) -> TextNet:
    """Implements TextNet architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    >>> import torch
    >>> from doctr.models import textnet_tiny
    >>> model = textnet_tiny(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the TextNet architecture

    Returns:
        A textnet tiny model
    """
    return _textnet(
        "textnet_tiny",
        pretrained,
        stages=[
            {"in_channels": [64] * 3, "out_channels": [64] * 3, "kernel_size": [(3, 3)] * 3, "stride": [1, 2, 1]},
            {
                "in_channels": [64, 128, 128, 128],
                "out_channels": [128] * 4,
                "kernel_size": [(3, 3), (1, 3), (3, 3), (3, 1)],
                "stride": [2, 1, 1, 1],
            },
            {
                "in_channels": [128, 256, 256, 256],
                "out_channels": [256] * 4,
                "kernel_size": [(3, 3), (3, 3), (3, 1), (1, 3)],
                "stride": [2, 1, 1, 1],
            },
            {
                "in_channels": [256, 512, 512, 512],
                "out_channels": [512] * 4,
                "kernel_size": [(3, 3), (3, 1), (1, 3), (3, 3)],
                "stride": [2, 1, 1, 1],
            },
        ],
        ignore_keys=["7.2.weight", "7.2.bias"],
        **kwargs,
    )


def textnet_small(pretrained: bool = False, **kwargs: Any) -> TextNet:
    """Implements TextNet architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    >>> import torch
    >>> from doctr.models import textnet_small
    >>> model = textnet_small(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the TextNet architecture

    Returns:
        A TextNet small model
    """
    return _textnet(
        "textnet_small",
        pretrained,
        stages=[
            {"in_channels": [64] * 2, "out_channels": [64] * 2, "kernel_size": [(3, 3)] * 2, "stride": [1, 2]},
            {
                "in_channels": [64, 128, 128, 128, 128, 128, 128, 128],
                "out_channels": [128] * 8,
                "kernel_size": [(3, 3), (1, 3), (3, 3), (3, 1), (3, 3), (3, 1), (1, 3), (3, 3)],
                "stride": [2, 1, 1, 1, 1, 1, 1, 1],
            },
            {
                "in_channels": [128, 256, 256, 256, 256, 256, 256, 256],
                "out_channels": [256] * 8,
                "kernel_size": [(3, 3), (3, 3), (1, 3), (3, 1), (3, 3), (1, 3), (3, 1), (3, 3)],
                "stride": [2, 1, 1, 1, 1, 1, 1, 1],
            },
            {
                "in_channels": [256, 512, 512, 512, 512],
                "out_channels": [512] * 5,
                "kernel_size": [(3, 3), (3, 1), (1, 3), (1, 3), (3, 1)],
                "stride": [2, 1, 1, 1, 1],
            },
        ],
        ignore_keys=["7.2.weight", "7.2.bias"],
        **kwargs,
    )


def textnet_base(pretrained: bool = False, **kwargs: Any) -> TextNet:
    """Implements TextNet architecture from `"FAST: Faster Arbitrarily-Shaped Text Detector with
    Minimalist Kernel Representation" <https://arxiv.org/abs/2111.02394>`_.
    Implementation based on the official Pytorch implementation: <https://github.com/czczup/FAST>`_.

    >>> import torch
    >>> from doctr.models import textnet_base
    >>> model = textnet_base(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the TextNet architecture

    Returns:
        A TextNet base model
    """
    return _textnet(
        "textnet_base",
        pretrained,
        stages=[
            {
                "in_channels": [64] * 10,
                "out_channels": [64] * 10,
                "kernel_size": [(3, 3), (3, 3), (3, 1), (3, 3), (3, 1), (3, 3), (3, 3), (1, 3), (3, 3), (3, 3)],
                "stride": [1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            {
                "in_channels": [64, 128, 128, 128, 128, 128, 128, 128, 128, 128],
                "out_channels": [128] * 10,
                "kernel_size": [(3, 3), (1, 3), (3, 3), (3, 1), (3, 3), (3, 3), (3, 1), (3, 1), (3, 3), (3, 3)],
                "stride": [2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            },
            {
                "in_channels": [128, 256, 256, 256, 256, 256, 256, 256],
                "out_channels": [256] * 8,
                "kernel_size": [(3, 3), (3, 3), (3, 3), (1, 3), (3, 3), (3, 1), (3, 3), (3, 1)],
                "stride": [2, 1, 1, 1, 1, 1, 1, 1],
            },
            {
                "in_channels": [256, 512, 512, 512, 512],
                "out_channels": [512] * 5,
                "kernel_size": [(3, 3), (1, 3), (3, 1), (3, 1), (1, 3)],
                "stride": [2, 1, 1, 1, 1],
            },
        ],
        ignore_keys=["7.2.weight", "7.2.bias"],
        **kwargs,
    )
