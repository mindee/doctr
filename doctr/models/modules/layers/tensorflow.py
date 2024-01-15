# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers

from doctr.utils.repr import NestedObject

__all__ = ["RepConvLayer"]


class RepConvLayer(layers.Layer, NestedObject):
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

        padding = ((converted_ks[0] - 1) * dilation // 2, (converted_ks[1] - 1) * dilation // 2)

        self.activation = layers.ReLU()
        self.conv_pad = layers.ZeroPadding2D(padding=padding)
        self.conv = layers.Conv2D(
            filters=out_channels,
            kernel_size=converted_ks,
            strides=stride,
            dilation_rate=dilation,
            groups=groups,
            use_bias=bias,
        )

        self.bn = layers.BatchNormalization()

        if converted_ks[1] != 1:
            self.ver_pad = layers.ZeroPadding2D(
                padding=(int(((converted_ks[0] - 1) * dilation) / 2), 0),
            )
            self.ver_conv = layers.Conv2D(
                filters=out_channels,
                kernel_size=(converted_ks[0], 1),
                strides=stride,
                dilation_rate=dilation,
                groups=groups,
                use_bias=bias,
            )
            self.ver_bn = layers.BatchNormalization()

        if converted_ks[0] != 1:
            self.hor_pad = layers.ZeroPadding2D(
                padding=(0, int(((converted_ks[1] - 1) * dilation) / 2)),
            )
            self.hor_conv = layers.Conv2D(
                filters=out_channels,
                kernel_size=(1, converted_ks[1]),
                padding="valid",
                strides=stride,
                dilation_rate=dilation,
                groups=groups,
                use_bias=bias,
            )
            self.hor_bn = layers.BatchNormalization()

        self.rbr_identity = layers.BatchNormalization() if out_channels == in_channels and stride == 1 else None

    def call(self, x: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        main_outputs = self.bn(self.conv(self.conv_pad(x, **kwargs), **kwargs), **kwargs)

        if self.ver_conv is not None and self.ver_bn is not None:
            vertical_outputs = self.ver_bn(self.ver_conv(self.ver_pad(x, **kwargs), **kwargs), **kwargs)
        else:
            vertical_outputs = 0

        if self.hor_bn is not None and self.hor_conv is not None:
            horizontal_outputs = self.hor_bn(self.hor_conv(self.hor_pad(x, **kwargs), **kwargs), **kwargs)
        else:
            horizontal_outputs = 0

        if self.rbr_identity is not None and self.ver_bn is not None:
            id_out = self.rbr_identity(x, **kwargs)
        else:
            id_out = 0

        return self.activation(main_outputs + vertical_outputs + horizontal_outputs + id_out)
