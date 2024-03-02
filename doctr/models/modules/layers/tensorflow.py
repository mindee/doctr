# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from doctr.utils.repr import NestedObject

__all__ = ["FASTConvLayer"]


class FASTConvLayer(layers.Layer, NestedObject):
    """Convolutional layer used in the TextNet and FAST architectures"""

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

        self.set_rep = False
        self.groups = groups
        self.in_channels = in_channels
        self.converted_ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        self.hor_conv, self.hor_bn = None, None
        self.ver_conv, self.ver_bn = None, None

        padding = ((self.converted_ks[0] - 1) * dilation // 2, (self.converted_ks[1] - 1) * dilation // 2)

        self.activation = layers.ReLU()
        self.conv_pad = layers.ZeroPadding2D(padding=padding)

        if self.set_rep:
            self.fused_conv = layers.Conv2D(
                filters=out_channels,
                kernel_size=self.converted_ks,
                strides=stride,
                dilation_rate=dilation,
                groups=groups,
                use_bias=True,
            )

        else:
            self.conv = layers.Conv2D(
                filters=out_channels,
                kernel_size=self.converted_ks,
                strides=stride,
                dilation_rate=dilation,
                groups=groups,
                use_bias=bias,
            )

            self.bn = layers.BatchNormalization()

            if self.converted_ks[1] != 1:
                self.ver_pad = layers.ZeroPadding2D(
                    padding=(int(((self.converted_ks[0] - 1) * dilation) / 2), 0),
                )
                self.ver_conv = layers.Conv2D(
                    filters=out_channels,
                    kernel_size=(self.converted_ks[0], 1),
                    strides=stride,
                    dilation_rate=dilation,
                    groups=groups,
                    use_bias=bias,
                )
                self.ver_bn = layers.BatchNormalization()

            if self.converted_ks[0] != 1:
                self.hor_pad = layers.ZeroPadding2D(
                    padding=(0, int(((self.converted_ks[1] - 1) * dilation) / 2)),
                )
                self.hor_conv = layers.Conv2D(
                    filters=out_channels,
                    kernel_size=(1, self.converted_ks[1]),
                    strides=stride,
                    dilation_rate=dilation,
                    groups=groups,
                    use_bias=bias,
                )
                self.hor_bn = layers.BatchNormalization()

            self.rbr_identity = layers.BatchNormalization() if out_channels == in_channels and stride == 1 else None

    def call(self, x: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        if hasattr(self, "fused_conv"):
            return self.activation(self.fused_conv(self.conv_pad(x, **kwargs), **kwargs))

        main_outputs = self.bn(self.conv(self.conv_pad(x, **kwargs), **kwargs), **kwargs)
        vertical_outputs = (
            self.ver_bn(self.ver_conv(self.ver_pad(x, **kwargs), **kwargs), **kwargs)
            if self.ver_conv is not None and self.ver_bn is not None
            else 0
        )
        horizontal_outputs = (
            self.hor_bn(self.hor_conv(self.hor_pad(x, **kwargs), **kwargs), **kwargs)
            if self.hor_bn is not None and self.hor_conv is not None
            else 0
        )
        id_out = self.rbr_identity(x, **kwargs) if self.rbr_identity is not None and self.ver_bn is not None else 0

        return self.activation(main_outputs + vertical_outputs + horizontal_outputs + id_out)

    def _identity_to_conv(self, identity):
        if identity is None:
            return 0, 0
        assert isinstance(identity, layers.BatchNormalization)
        if not hasattr(self, "id_tensor"):
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 1, 1), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 0, 0] = 1
            id_tensor = tf.constant(kernel_value, dtype=tf.float32)
            self.id_tensor = self._pad_to_mxn_tensor(id_tensor)
        kernel = self.id_tensor
        running_mean = identity.moving_mean
        running_var = identity.moving_variance
        gamma = identity.gamma
        beta = identity.beta
        eps = identity.epsilon
        std = tf.sqrt(running_var + eps)
        t = tf.reshape(gamma / std, (-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.kernel
        kernel = self._pad_to_mxn_tensor(kernel)
        running_mean = bn.moving_mean
        running_var = bn.moving_variance
        gamma = bn.gamma
        beta = bn.beta
        eps = bn.epsilon
        std = tf.sqrt(running_var + eps)
        t = tf.reshape(gamma / std, (-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel_mxn, bias_mxn = self._fuse_bn_tensor(self.conv, self.bn)
        if self.ver_conv is not None:
            kernel_mx1, bias_mx1 = self._fuse_bn_tensor(self.ver_conv, self.ver_bn)
        else:
            kernel_mx1, bias_mx1 = 0, 0
        if self.hor_conv is not None:
            kernel_1xn, bias_1xn = self._fuse_bn_tensor(self.hor_conv, self.hor_bn)
        else:
            kernel_1xn, bias_1xn = 0, 0
        kernel_id, bias_id = self._identity_to_conv(self.rbr_identity)
        kernel_mxn = kernel_mxn + kernel_mx1 + kernel_1xn + kernel_id
        bias_mxn = bias_mxn + bias_mx1 + bias_1xn + bias_id
        return kernel_mxn, bias_mxn

    def _pad_to_mxn_tensor(self, kernel):
        kernel_height, kernel_width = self.converted_ks
        height, width = kernel.shape[2:]
        pad_left_right = (kernel_width - width) // 2
        pad_top_down = (kernel_height - height) // 2
        return tf.pad(kernel, [[0, 0], [pad_top_down, pad_top_down], [pad_left_right, pad_left_right], [0, 0]])

    def reparameterize(self):
        self.set_rep = True
        if hasattr(self, "fused_conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.fused_conv = layers.Conv2D(
            filters=self.conv.filters,
            kernel_size=self.conv.kernel_size,
            strides=self.conv.strides,
            padding=self.conv.padding,
            dilation_rate=self.conv.dilation_rate,
            groups=self.conv.groups,
            use_bias=True,
        )
        self.fused_conv.build(input_shape=(None, None, None, self.conv.filters))
        self.fused_conv.set_weights([kernel.numpy(), bias.numpy()])
        self.deploy = True
        for para in self.trainable_variables:
            para._trainable = False  # Equivalent to para.detach_()
        for attr in ["conv", "bn", "ver_conv", "ver_bn", "hor_conv", "hor_bn"]:
            if hasattr(self, attr):
                delattr(self, attr)

        if hasattr(self, "rbr_identity"):
            delattr(self, "rbr_identity")
