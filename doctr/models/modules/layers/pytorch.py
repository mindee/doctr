from typing import Any, List, Tuple, Union

import torch
import torch.nn as nn

__all__ = ["RepConvLayer"]


class RepConvLayer(nn.Module):
    """Reparameterized Convolutional Layer"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: Union[List[int], Tuple[int, int], int], **kwargs: Any
    ) -> None:
        super().__init__()

        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        dilation = kwargs.get("dilation", 1)
        stride = kwargs.get("stride", 1)
        kwargs.pop("padding", None)
        kwargs.pop("bias", None)

        self.hor_conv, self.hor_bn = None, None
        self.ver_conv, self.ver_bn = None, None

        padding = (int(((kernel_size[0] - 1) * dilation) / 2), int(((kernel_size[1] - 1) * dilation) / 2))

        self.activation = nn.ReLU(inplace=True)
        self.main_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            **kwargs,
        )

        self.main_bn = nn.BatchNorm2d(out_channels)

        if kernel_size[1] != 1:
            self.ver_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size[0], 1),
                padding=(int(((kernel_size[0] - 1) * dilation) / 2), 0),
                bias=False,
                **kwargs,
            )
            self.ver_bn = nn.BatchNorm2d(out_channels)

        if kernel_size[0] != 1:
            self.hor_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size[1]),
                padding=(0, int(((kernel_size[1] - 1) * dilation) / 2)),
                bias=False,
                **kwargs,
            )
            self.hor_bn = nn.BatchNorm2d(out_channels)

        self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_outputs = self.main_bn(self.main_conv(x))
        vertical_outputs = self.ver_bn(self.ver_conv(x)) if self.ver_conv is not None else 0
        horizontal_outputs = self.hor_bn(self.hor_conv(x)) if self.hor_conv is not None else 0
        id_out = self.rbr_identity(x) if self.rbr_identity is not None else 0

        return self.activation(main_outputs + vertical_outputs + horizontal_outputs + id_out)
