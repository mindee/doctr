# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from collections.abc import Callable
from copy import deepcopy
from typing import Any

from torch import nn
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import ResNet as TVResNet
from torchvision.models.resnet import resnet18 as tv_resnet18
from torchvision.models.resnet import resnet34 as tv_resnet34
from torchvision.models.resnet import resnet50 as tv_resnet50

from doctr.datasets import VOCABS

from ...utils import conv_sequence_pt, load_pretrained_params

__all__ = ["ResNet", "resnet18", "resnet31", "resnet34", "resnet50", "resnet34_wide", "resnet_stage"]


default_cfgs: dict[str, dict[str, Any]] = {
    "resnet18": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.4.1/resnet18-244bf390.pt&src=0",
    },
    "resnet31": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.4.1/resnet31-1056cc5c.pt&src=0",
    },
    "resnet34": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.5.0/resnet34-bd8725db.pt&src=0",
    },
    "resnet50": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.5.0/resnet50-1a6c155e.pt&src=0",
    },
    "resnet34_wide": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://doctr-static.mindee.com/models?id=v0.6.0/resnet34_wide-b4b3e39e.pt&src=0",
    },
}


def resnet_stage(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> list[nn.Module]:
    """Build a ResNet stage"""
    _layers: list[nn.Module] = []

    in_chan = in_channels
    s = stride
    for _ in range(num_blocks):
        downsample = None
        if in_chan != out_channels:
            downsample = nn.Sequential(*conv_sequence_pt(in_chan, out_channels, False, True, kernel_size=1, stride=s))

        _layers.append(BasicBlock(in_chan, out_channels, stride=s, downsample=downsample))
        in_chan = out_channels
        # Only the first block can have stride != 1
        s = 1

    return _layers


class ResNet(nn.Sequential):
    """Implements a ResNet-31 architecture from `"Show, Attend and Read:A Simple and Strong Baseline for Irregular
    Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

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
        num_blocks: list[int],
        output_channels: list[int],
        stage_stride: list[int],
        stage_conv: list[bool],
        stage_pooling: list[tuple[int, int] | None],
        origin_stem: bool = True,
        stem_channels: int = 64,
        attn_module: Callable[[int], nn.Module] | None = None,
        include_top: bool = True,
        num_classes: int = 1000,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        _layers: list[nn.Module]
        if origin_stem:
            _layers = [
                *conv_sequence_pt(3, stem_channels, True, True, kernel_size=7, padding=3, stride=2),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        else:
            _layers = [
                *conv_sequence_pt(3, stem_channels // 2, True, True, kernel_size=3, padding=1),
                *conv_sequence_pt(stem_channels // 2, stem_channels, True, True, kernel_size=3, padding=1),
                nn.MaxPool2d(2),
            ]
        in_chans = [stem_channels] + output_channels[:-1]
        for n_blocks, in_chan, out_chan, stride, conv, pool in zip(
            num_blocks, in_chans, output_channels, stage_stride, stage_conv, stage_pooling
        ):
            _stage = resnet_stage(in_chan, out_chan, n_blocks, stride)
            if attn_module is not None:
                _stage.append(attn_module(out_chan))
            if conv:
                _stage.extend(conv_sequence_pt(out_chan, out_chan, True, True, kernel_size=3, padding=1))
            if pool is not None:
                _stage.append(nn.MaxPool2d(pool))
            _layers.append(nn.Sequential(*_stage))

        if include_top:
            _layers.extend([
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(output_channels[-1], num_classes, bias=True),
            ])

        super().__init__(*_layers)
        self.cfg = cfg

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def _resnet(
    arch: str,
    pretrained: bool,
    num_blocks: list[int],
    output_channels: list[int],
    stage_stride: list[int],
    stage_conv: list[bool],
    stage_pooling: list[tuple[int, int] | None],
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> ResNet:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = ResNet(num_blocks, output_channels, stage_stride, stage_conv, stage_pooling, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    return model


def _tv_resnet(
    arch: str,
    pretrained: bool,
    arch_fn,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> TVResNet:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = arch_fn(**kwargs, weights=None)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        load_pretrained_params(model, default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    model.cfg = _cfg

    return model


def resnet18(pretrained: bool = False, **kwargs: Any) -> TVResNet:
    """ResNet-18 architecture as described in `"Deep Residual Learning for Image Recognition",
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    >>> import torch
    >>> from doctr.models import resnet18
    >>> model = resnet18(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the ResNet architecture

    Returns:
        A resnet18 model
    """
    return _tv_resnet(
        "resnet18",
        pretrained,
        tv_resnet18,
        ignore_keys=["fc.weight", "fc.bias"],
        **kwargs,
    )


def resnet31(pretrained: bool = False, **kwargs: Any) -> ResNet:
    """Resnet31 architecture with rectangular pooling windows as described in
    `"Show, Attend and Read:A Simple and Strong Baseline for Irregular Text Recognition",
    <https://arxiv.org/pdf/1811.00751.pdf>`_. Downsizing: (H, W) --> (H/8, W/4)

    >>> import torch
    >>> from doctr.models import resnet31
    >>> model = resnet31(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the ResNet architecture

    Returns:
        A resnet31 model
    """
    return _resnet(
        "resnet31",
        pretrained,
        [1, 2, 5, 3],
        [256, 256, 512, 512],
        [1, 1, 1, 1],
        [True] * 4,
        [(2, 2), (2, 1), None, None],
        origin_stem=False,
        stem_channels=128,
        ignore_keys=["13.weight", "13.bias"],
        **kwargs,
    )


def resnet34(pretrained: bool = False, **kwargs: Any) -> TVResNet:
    """ResNet-34 architecture as described in `"Deep Residual Learning for Image Recognition",
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    >>> import torch
    >>> from doctr.models import resnet34
    >>> model = resnet34(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the ResNet architecture

    Returns:
        A resnet34 model
    """
    return _tv_resnet(
        "resnet34",
        pretrained,
        tv_resnet34,
        ignore_keys=["fc.weight", "fc.bias"],
        **kwargs,
    )


def resnet34_wide(pretrained: bool = False, **kwargs: Any) -> ResNet:
    """ResNet-34 architecture as described in `"Deep Residual Learning for Image Recognition",
    <https://arxiv.org/pdf/1512.03385.pdf>`_ with twice as many output channels.

    >>> import torch
    >>> from doctr.models import resnet34_wide
    >>> model = resnet34_wide(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the ResNet architecture

    Returns:
        A resnet34_wide model
    """
    return _resnet(
        "resnet34_wide",
        pretrained,
        [3, 4, 6, 3],
        [128, 256, 512, 1024],
        [1, 2, 2, 2],
        [False] * 4,
        [None] * 4,
        origin_stem=True,
        stem_channels=128,
        ignore_keys=["10.weight", "10.bias"],
        **kwargs,
    )


def resnet50(pretrained: bool = False, **kwargs: Any) -> TVResNet:
    """ResNet-50 architecture as described in `"Deep Residual Learning for Image Recognition",
    <https://arxiv.org/pdf/1512.03385.pdf>`_.

    >>> import torch
    >>> from doctr.models import resnet50
    >>> model = resnet50(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 512, 512), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the ResNet architecture

    Returns:
        A resnet50 model
    """
    return _tv_resnet(
        "resnet50",
        pretrained,
        tv_resnet50,
        ignore_keys=["fc.weight", "fc.bias"],
        **kwargs,
    )
