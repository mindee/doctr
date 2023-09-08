from typing import Any

import tensorflow as tf
from tensorflow.keras import layers

__all__ = ["RepConvLayer"]


class RepConvLayer(layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, groups=1):
        super(RepConvLayer, self).__init__()

        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        padding = (int(((kernel_size[0] - 1) * dilation) / 2), int(((kernel_size[1] - 1) * dilation) / 2))

        self.activation = layers.ReLU()
        self.main_conv = tf.keras.Sequential(
            [
                layers.ZeroPadding2D(padding=padding),
                layers.Conv2D(
                    filters=out_channels,
                    kernel_size=kernel_size,
                    strides=stride,
                    dilation_rate=dilation,
                    groups=groups,
                    use_bias=False,
                    input_shape=(None, None, in_channels),
                ),
            ]
        )

        self.main_bn = layers.BatchNormalization()

        if kernel_size[1] != 1:
            self.ver_conv = tf.keras.Sequential(
                [
                    layers.ZeroPadding2D(padding=(int(((kernel_size[0] - 1) * dilation) / 2), 0)),
                    layers.Conv2D(
                        filters=out_channels,
                        kernel_size=(kernel_size[0], 1),
                        strides=stride,
                        dilation_rate=(dilation, 1),
                        groups=groups,
                        use_bias=False,
                        input_shape=(None, None, in_channels),
                    ),
                ]
            )

            self.ver_bn = layers.BatchNormalization()
        else:
            self.ver_conv, self.ver_bn = None, None

        if kernel_size[0] != 1:
            self.hor_conv = tf.keras.Sequential(
                [
                    layers.ZeroPadding2D(padding=(0, int(((kernel_size[1] - 1) * dilation) / 2))),
                    layers.Conv2D(
                        filters=out_channels,
                        kernel_size=(1, kernel_size[1]),
                        strides=stride,
                        dilation_rate=dilation,
                        groups=groups,
                        use_bias=False,
                        input_shape=(None, None, in_channels),
                    ),
                ]
            )

            self.hor_bn = layers.BatchNormalization()
        else:
            self.hor_conv, self.hor_bn = None, None

        self.rbr_identity = layers.BatchNormalization() if out_channels == in_channels and stride == 1 else None

    def call(
        self,
        x: tf.Tensor,
        **kwargs: Any,
    ) -> tf.Tensor:
        main_outputs = self.main_bn(self.main_conv(x, **kwargs), **kwargs)
        vertical_outputs = self.ver_bn(self.ver_conv(x, **kwargs), **kwargs) if self.ver_conv is not None else 0
        horizontal_outputs = self.hor_bn(self.hor_conv(x, **kwargs), **kwargs) if self.hor_conv is not None else 0
        id_out = self.rbr_identity(x, **kwargs) if self.rbr_identity is not None else 0

        p = main_outputs + vertical_outputs
        q = horizontal_outputs + id_out
        r = p + q

        return self.activation(r)
