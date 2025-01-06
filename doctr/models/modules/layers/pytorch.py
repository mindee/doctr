# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import numpy as np
import torch
import torch.nn as nn

__all__ = ["FASTConvLayer"]


class FASTConvLayer(nn.Module):
    """Convolutional layer used in the TextNet and FAST architectures"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.groups = groups
        self.in_channels = in_channels
        self.converted_ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        self.hor_conv, self.hor_bn = None, None
        self.ver_conv, self.ver_bn = None, None

        padding = (int(((self.converted_ks[0] - 1) * dilation) / 2), int(((self.converted_ks[1] - 1) * dilation) / 2))

        self.activation = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.converted_ks,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.bn = nn.BatchNorm2d(out_channels)

        if self.converted_ks[1] != 1:
            self.ver_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(self.converted_ks[0], 1),
                padding=(int(((self.converted_ks[0] - 1) * dilation) / 2), 0),
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            self.ver_bn = nn.BatchNorm2d(out_channels)

        if self.converted_ks[0] != 1:
            self.hor_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, self.converted_ks[1]),
                padding=(0, int(((self.converted_ks[1] - 1) * dilation) / 2)),
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
            self.hor_bn = nn.BatchNorm2d(out_channels)

        self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "fused_conv"):
            return self.activation(self.fused_conv(x))

        main_outputs = self.bn(self.conv(x))
        vertical_outputs = self.ver_bn(self.ver_conv(x)) if self.ver_conv is not None and self.ver_bn is not None else 0
        horizontal_outputs = (
            self.hor_bn(self.hor_conv(x)) if self.hor_bn is not None and self.hor_conv is not None else 0
        )
        id_out = self.rbr_identity(x) if self.rbr_identity is not None else 0

        return self.activation(main_outputs + vertical_outputs + horizontal_outputs + id_out)

    # The following logic is used to reparametrize the layer
    # Borrowed from: https://github.com/czczup/FAST/blob/main/models/utils/nas_utils.py
    def _identity_to_conv(self, identity: nn.BatchNorm2d | None) -> tuple[torch.Tensor, torch.Tensor] | tuple[int, int]:
        if identity is None or identity.running_var is None:
            return 0, 0
        if not hasattr(self, "id_tensor"):
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 0, 0] = 1
            id_tensor = torch.from_numpy(kernel_value).to(identity.weight.device)
            self.id_tensor = self._pad_to_mxn_tensor(id_tensor)
        kernel = self.id_tensor
        std = (identity.running_var + identity.eps).sqrt()  # type: ignore
        t = (identity.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, identity.bias - identity.running_mean * identity.weight / std

    def _fuse_bn_tensor(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = conv.weight
        kernel = self._pad_to_mxn_tensor(kernel)
        std = (bn.running_var + bn.eps).sqrt()  # type: ignore
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def _get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        kernel_mxn, bias_mxn = self._fuse_bn_tensor(self.conv, self.bn)
        if self.ver_conv is not None:
            kernel_mx1, bias_mx1 = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)  # type: ignore[arg-type]
        else:
            kernel_mx1, bias_mx1 = 0, 0  # type: ignore[assignment]
        if self.hor_conv is not None:
            kernel_1xn, bias_1xn = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)  # type: ignore[arg-type]
        else:
            kernel_1xn, bias_1xn = 0, 0  # type: ignore[assignment]
        kernel_id, bias_id = self._identity_to_conv(self.rbr_identity)
        kernel_mxn = kernel_mxn + kernel_mx1 + kernel_1xn + kernel_id
        bias_mxn = bias_mxn + bias_mx1 + bias_1xn + bias_id
        return kernel_mxn, bias_mxn

    def _pad_to_mxn_tensor(self, kernel: torch.Tensor) -> torch.Tensor:
        kernel_height, kernel_width = self.converted_ks
        height, width = kernel.shape[2:]
        pad_left_right = (kernel_width - width) // 2
        pad_top_down = (kernel_height - height) // 2
        return torch.nn.functional.pad(kernel, [pad_left_right, pad_left_right, pad_top_down, pad_top_down], value=0)

    def reparameterize_layer(self):
        if hasattr(self, "fused_conv"):
            return
        kernel, bias = self._get_equivalent_kernel_bias()
        self.fused_conv = nn.Conv2d(
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            kernel_size=self.conv.kernel_size,  # type: ignore[arg-type]
            stride=self.conv.stride,  # type: ignore[arg-type]
            padding=self.conv.padding,  # type: ignore[arg-type]
            dilation=self.conv.dilation,  # type: ignore[arg-type]
            groups=self.conv.groups,
            bias=True,
        )
        self.fused_conv.weight.data = kernel
        self.fused_conv.bias.data = bias  # type: ignore[union-attr]
        for para in self.parameters():
            para.detach_()
        for attr in ["conv", "bn", "ver_conv", "ver_bn", "hor_conv", "hor_bn"]:
            if hasattr(self, attr):
                self.__delattr__(attr)

        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
