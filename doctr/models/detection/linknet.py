# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

from copy import deepcopy
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from typing import Dict, Any, Tuple, Optional

from ..backbones import ResnetStage
from ..utils import conv_sequence, load_pretrained_params
from ...utils.repr import NestedObject

__all__ = ['LinkNet', 'linknet']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'linknet': {
        'mean': (0.798, 0.785, 0.772),
        'std': (0.264, 0.2749, 0.287),
        'out_chan': 1,
        'input_shape': (1024, 1024, 3),
        'post_processor': None,
        'url': None,
    },
}


class DecoderBlock(Sequential):
    """This class implements a Linknet decoder block as described in paper/

    Args:
        in_chan: input channels (must be a multiple of 4)
        out_chan: output channels

    """
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
    ) -> None:
        _layers = []
        _layers.extend(conv_sequence(in_chan // 4, 'relu', True, kernel_size=1))
        _layers.append(
            layers.Conv2DTranspose(
                filters=in_chan // 4,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                kernel_initializer='he_normal'
            )
        )
        _layers.append(layers.BatchNormalization())
        _layers.append(layers.Activation('relu'))
        _layers.extend(conv_sequence(out_chan, 'relu', True, kernel_size=1))
        super().__init__(_layers)


class LinkNetPyramid(layers.Layer, NestedObject):
    """LinkNet Encoder-Decoder module

    """

    def __init__(
        self,
    ) -> None:

        super().__init__()
        self.encoder_1 = ResnetStage(num_blocks=2, output_channels=64, downsample=True)
        self.encoder_2 = ResnetStage(num_blocks=2, output_channels=128, downsample=True)
        self.encoder_3 = ResnetStage(num_blocks=2, output_channels=256, downsample=True)
        self.encoder_4 = ResnetStage(num_blocks=2, output_channels=512, downsample=True)
        self.decoder_1 = DecoderBlock(in_chan=64, out_chan=64)
        self.decoder_2 = DecoderBlock(in_chan=128, out_chan=64)
        self.decoder_3 = DecoderBlock(in_chan=256, out_chan=128)
        self.decoder_4 = DecoderBlock(in_chan=512, out_chan=256)

    def call(
        self,
        x: tf.Tensor
    ) -> tf.Tensor:
        x_1 = self.encoder_1(x)
        x_2 = self.encoder_2(x_1)
        x_3 = self.encoder_3(x_2)
        x_4 = self.encoder_4(x_3)
        y_4 = self.decoder_4(x_4)
        y_3 = self.decoder_3(y_4 + x_3)
        y_2 = self.decoder_2(y_3 + x_2)
        y_1 = self.decoder_1(y_2 + x_1)
        return y_1


class LinkNet(Sequential, NestedObject):
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    Args:
        out_chan: number of channels for the output
    """

    def __init__(
        self,
        out_chan: int = 1,
        input_shape: Tuple[int, int, int] = (512, 512, 3),
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        _layers = []
        _layers.extend(conv_sequence(64, 'relu', True, strides=2, kernel_size=7, input_shape=input_shape))
        _layers.append(layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same'))
        _layers.append(LinkNetPyramid())
        _layers.append(
            layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                kernel_initializer='he_normal'
            )
        )
        _layers.append(layers.BatchNormalization())
        _layers.append(layers.Activation('relu'))
        _layers.extend(conv_sequence(32, 'relu', True, strides=1, kernel_size=3))
        _layers.append(
            layers.Conv2DTranspose(
                filters=out_chan,
                kernel_size=2,
                strides=2,
                padding="same",
                use_bias=False,
                kernel_initializer='he_normal'
            )
        )
        super().__init__(_layers)


def _linknet(arch: str, pretrained: bool, input_shape: Tuple[int, int, int] = None, **kwargs: Any) -> LinkNet:

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']
    _cfg['out_chan'] = kwargs.get('out_chan', _cfg['out_chan'])

    kwargs['out_chan'] = _cfg['out_chan']
    kwargs['input_shape'] = _cfg['input_shape']
    # Build the model
    model = LinkNet(cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, _cfg['url'])

    return model


def linknet(pretrained: bool = False, **kwargs: Any) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import linknet
        >>> model = linknet(pretrained=True)
        >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _linknet('linknet', pretrained, **kwargs)
