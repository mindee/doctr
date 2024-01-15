# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Tuple, Union

import torch
import torch.nn as nn

__all__ = ["RepConvLayer"]


class RepConvLayer(nn.Module):
    """Reparameterized Convolutional Layer"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        converted_ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        self.hor_conv, self.hor_bn = None, None
        self.ver_conv, self.ver_bn = None, None

        padding = (int(((converted_ks[0] - 1) * dilation) / 2), int(((converted_ks[1] - 1) * dilation) / 2))

        self.activation = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=converted_ks,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.bn = nn.BatchNorm2d(out_channels)

        if converted_ks[1] != 1:
            self.ver_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(converted_ks[0], 1),
                padding=(int(((converted_ks[0] - 1) * dilation) / 2), 0),
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            self.ver_bn = nn.BatchNorm2d(out_channels)

        if converted_ks[0] != 1:
            self.hor_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, converted_ks[1]),
                padding=(0, int(((converted_ks[1] - 1) * dilation) / 2)),
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            self.hor_bn = nn.BatchNorm2d(out_channels)

        self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_outputs = self.bn(self.conv(x))

        if self.ver_conv is not None and self.ver_bn is not None:
            vertical_outputs = self.ver_bn(self.ver_conv(x))
        else:
            vertical_outputs = 0

        if self.hor_bn is not None and self.hor_conv is not None:
            horizontal_outputs = self.hor_bn(self.hor_conv(x))
        else:
            horizontal_outputs = 0

        if self.rbr_identity is not None and self.ver_bn is not None:
            id_out = self.rbr_identity(x)
        else:
            id_out = 0

        return self.activation(main_outputs + vertical_outputs + horizontal_outputs + id_out)
