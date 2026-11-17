# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: architecture of PP-HGNetV2 from PaddleClas / PaddleOCR (https://github.com/PaddlePaddle/PaddleOCR,
# Apache 2.0), as also used by D-FINE (https://github.com/Peterande/D-FINE, Apache 2.0).

from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from doctr.datasets import VOCABS

from ...modules.layers import DropPath
from ...utils import load_pretrained_params

__all__ = ["HGNetV2", "hgnetv2_b4"]


default_cfgs: dict[str, dict[str, Any]] = {
    "hgnetv2_b4": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (3, 32, 32),
        "classes": list(VOCABS["french"]),
        "url": "https://github.com/mindee/doctr/releases/download/v1.1.0/hgnetv2_b4-0f90b8d0.pt",
    },
}


class ConvBNAct(nn.Sequential):
    """A convolution (without bias) followed by batch-norm and an optional ReLU

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: size of the convolution kernel
        stride: stride of the convolution
        groups: number of groups of the convolution
        use_act: whether to add the ReLU activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        use_act: bool = True,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if use_act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class LightConvBNAct(nn.Sequential):
    """A point-wise convolution followed by a depth-wise convolution

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: size of the depth-wise convolution kernel
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(
            ConvBNAct(in_channels, out_channels, kernel_size=1, use_act=False),
            ConvBNAct(out_channels, out_channels, kernel_size=kernel_size, groups=out_channels),
        )


class StemBlock(nn.Module):
    """Stem of HGNetV2 (stride 4): a strided convolution, then a two-branch (convolutions | max-pooling) block

    Args:
        in_channels: number of input channels
        mid_channels: number of intermediate channels
        out_channels: number of output channels
    """

    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.stem1 = ConvBNAct(in_channels, mid_channels, kernel_size=3, stride=2)
        # 2x2 kernels, padded on the bottom-right to keep the resolution
        self.stem2a = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels // 2, 2, bias=False), nn.BatchNorm2d(mid_channels // 2), nn.ReLU(True)
        )
        self.stem2b = nn.Sequential(
            nn.Conv2d(mid_channels // 2, mid_channels, 2, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)
        self.stem3 = ConvBNAct(mid_channels * 2, mid_channels, kernel_size=3, stride=2)
        self.stem4 = ConvBNAct(mid_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(self.stem1(x), (0, 1, 0, 1))
        x2 = self.stem2b(F.pad(self.stem2a(x), (0, 1, 0, 1)))
        x1 = self.pool(x)
        return self.stem4(self.stem3(torch.cat([x1, x2], dim=1)))


class HGBlock(nn.Module):
    """HGNetV2 block: a chain of convolutions whose outputs are all concatenated (with the input) and aggregated by
    point-wise convolutions, with an optional residual connection

    Args:
        in_channels: number of input channels
        mid_channels: number of channels of each convolution of the chain
        out_channels: number of output channels
        layer_num: number of convolutions in the chain
        kernel_size: size of the convolution kernels
        residual: whether to add a residual connection
        light_block: whether to use light (point-wise + depth-wise) convolutions
        drop_path: stochastic depth rate of the residual branch
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        layer_num: int,
        kernel_size: int = 3,
        residual: bool = False,
        light_block: bool = False,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList([
            (LightConvBNAct if light_block else ConvBNAct)(
                in_channels if idx == 0 else mid_channels, mid_channels, kernel_size=kernel_size
            )
            for idx in range(layer_num)
        ])
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation = nn.Sequential(
            ConvBNAct(total_channels, out_channels // 2, kernel_size=1),
            ConvBNAct(out_channels // 2, out_channels, kernel_size=1),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        outputs = [x]
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        x = self.aggregation(torch.cat(outputs, dim=1))
        return self.drop_path(x) + identity if self.residual else x


class HGStage(nn.Sequential):
    """HGNetV2 stage: an optional depth-wise strided convolution, then a sequence of HGBlocks

    Args:
        in_channels: number of input channels
        mid_channels: number of channels of the convolutions of the blocks
        out_channels: number of output channels
        num_blocks: number of blocks
        downsample: whether to downsample the input by 2 first
        light_block: whether the blocks use light convolutions
        kernel_size: size of the convolution kernels of the blocks
        layer_num: number of convolutions per block
        drop_path: stochastic depth rate of each block
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        num_blocks: int,
        downsample: bool,
        light_block: bool,
        kernel_size: int,
        layer_num: int,
        drop_path: list[float],
    ) -> None:
        super().__init__(
            ConvBNAct(in_channels, in_channels, kernel_size=3, stride=2, groups=in_channels, use_act=False)
            if downsample
            else nn.Identity(),
            *[
                HGBlock(
                    in_channels if idx == 0 else out_channels,
                    mid_channels,
                    out_channels,
                    layer_num,
                    kernel_size=kernel_size,
                    residual=idx > 0,
                    light_block=light_block,
                    drop_path=drop_path[idx],
                )
                for idx in range(num_blocks)
            ],
        )


class HGNetV2(nn.Sequential):
    """Implements the HGNetV2 architecture from PaddlePaddle's `PP-HGNetV2
    <https://github.com/PaddlePaddle/PaddleClas/blob/release/2.6/docs/en/models/PP-HGNetV2_en.md>`_.

    The model is a stem (stride 4) followed by 4 stages (strides 4, 8, 16 and 32).

    Args:
        stem_channels: the input, intermediate and output channels of the stem
        stages: the configuration of each stage `(in_channels, mid_channels, out_channels, num_blocks, downsample,
            light_block, kernel_size, layer_num)`
        drop_path_rate: maximum stochastic depth rate (linearly scaled across blocks)
        num_classes: number of output classes of the classification head
        include_top: whether to add the classification head
        cfg: the configuration dict of the model
    """

    def __init__(
        self,
        stem_channels: tuple[int, int, int],
        stages: list[tuple[int, int, int, int, bool, bool, int, int]],
        drop_path_rate: float = 0.0,
        num_classes: int = 1000,
        include_top: bool = True,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(stage[3] for stage in stages))]
        _layers: list[nn.Module] = [StemBlock(*stem_channels)]
        cur = 0
        for in_ch, mid_ch, out_ch, num_blocks, downsample, light_block, kernel_size, layer_num in stages:
            _layers.append(
                HGStage(
                    in_ch,
                    mid_ch,
                    out_ch,
                    num_blocks,
                    downsample,
                    light_block,
                    kernel_size,
                    layer_num,
                    dpr[cur : cur + num_blocks],
                )
            )
            cur += num_blocks

        if include_top:
            _layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1),
                    nn.Linear(stages[-1][2], num_classes),
                )
            )

        super().__init__(*_layers)
        # Output channels of the stem and of each stage
        self.channels = [stem_channels[2], *(stage[2] for stage in stages)]
        self.cfg = cfg

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def from_pretrained(self, path_or_url: str, **kwargs: Any) -> None:
        """Load pretrained parameters onto the model

        Args:
            path_or_url: the path or URL to the model parameters (checkpoint)
            **kwargs: additional arguments to be passed to `doctr.models.utils.load_pretrained_params`
        """
        load_pretrained_params(self, path_or_url, **kwargs)


def _hgnetv2(
    arch: str,
    pretrained: bool,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> HGNetV2:
    kwargs["num_classes"] = kwargs.get("num_classes", len(default_cfgs[arch]["classes"]))
    kwargs["classes"] = kwargs.get("classes", default_cfgs[arch]["classes"])

    _cfg = deepcopy(default_cfgs[arch])
    _cfg["num_classes"] = kwargs["num_classes"]
    _cfg["classes"] = kwargs["classes"]
    kwargs.pop("classes")

    # Build the model
    model = HGNetV2(**kwargs)
    # Load pretrained parameters
    if pretrained:
        # The number of classes is not the same as the number of classes in the pretrained model =>
        # remove the last layer weights
        _ignore_keys = ignore_keys if kwargs["num_classes"] != len(default_cfgs[arch]["classes"]) else None
        model.from_pretrained(default_cfgs[arch]["url"], ignore_keys=_ignore_keys)

    model.cfg = _cfg

    return model


def hgnetv2_b4(pretrained: bool = False, **kwargs: Any) -> HGNetV2:
    """HGNetV2-B4 from PaddlePaddle's `PP-HGNetV2
    <https://github.com/PaddlePaddle/PaddleClas/blob/release/2.6/docs/en/models/PP-HGNetV2_en.md>`_.

    >>> import torch
    >>> from doctr.models import hgnetv2_b4
    >>> model = hgnetv2_b4(pretrained=False)
    >>> input_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained: boolean, True if model is pretrained
        **kwargs: keyword arguments of the HGNetV2 architecture

    Returns:
        A HGNetV2-B4 model
    """
    return _hgnetv2(
        "hgnetv2_b4",
        pretrained,
        stem_channels=(3, 32, 48),
        stages=[
            (48, 48, 128, 1, False, False, 3, 6),
            (128, 96, 512, 1, True, False, 3, 6),
            (512, 192, 1024, 3, True, True, 5, 6),
            (1024, 384, 2048, 1, True, True, 5, 6),
        ],
        ignore_keys=["5.2.weight", "5.2.bias"],
        **kwargs,
    )
