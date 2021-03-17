# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from copy import deepcopy
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from typing import Tuple, Dict, Any, Optional

from .. import vgg, resnet
from ..utils import load_pretrained_params
from .core import RecognitionModel

__all__ = ['CRNN', 'crnn_vgg16_bn', 'crnn_resnet31']

default_cfgs: Dict[str, Dict[str, Any]] = {
    'crnn_vgg16_bn': {
        'mean': (.5, .5, .5),
        'std': (1., 1., 1.),
        'backbone': 'vgg16_bn', 'rnn_units': 128,
        'input_shape': (32, 128, 3),
        'post_processor': 'CTCPostProcessor',
        'vocab': ('3K}7eé;5àÎYho]QwV6qU~W"XnbBvcADfËmy.9ÔpÛ*{CôïE%M4#ÈR:g@T$x?0î£|za1ù8,OG€P-'
                  'kçHëÀÂ2É/ûIJ\'j(LNÙFut[)èZs+&°Sd=Ï!<â_Ç>rêi`l'),
        'url': 'https://github.com/mindee/doctr/releases/download/v0.1.0/crnn_vgg16_bn-748c855f.zip',
    },
    'crnn_resnet31': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'backbone': 'resnet31', 'rnn_units': 128,
        'input_shape': (32, 128, 3),
        'post_processor': 'CTCPostProcessor',
        'vocab': ('3K}7eé;5àÎYho]QwV6qU~W"XnbBvcADfËmy.9ÔpÛ*{CôïE%M4#ÈR:g@T$x?0î£|za1ù8,OG€P-'
                  'kçHëÀÂ2É/ûIJ\'j(LNÙFut[)èZs+&°Sd=Ï!<â_Ç>rêi`l'),
        'url': None,
    },
}


class CRNN(RecognitionModel):
    """Implements a CRNN architecture as described in `"Convolutional RNN: an Enhanced Model for Extracting Features
    from Sequential Data" <https://arxiv.org/pdf/1602.05875.pdf>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab_size: number of output classes
        rnn_units: number of units in the LSTM layers
    """
    def __init__(
        self,
        feature_extractor: tf.keras.Model,
        vocab_size: int = 118,
        rnn_units: int = 128,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(cfg=cfg)
        self.feat_extractor = feature_extractor

        # Initialize kernels
        h, w, c = self.feat_extractor.output_shape[1:]

        self.decoder = Sequential(
            [
                layers.Bidirectional(layers.LSTM(units=rnn_units, return_sequences=True)),
                layers.Bidirectional(layers.LSTM(units=rnn_units, return_sequences=True)),
                layers.Dense(units=vocab_size + 1)
            ]
        )
        self.decoder.build(input_shape=(None, w, h * c))

    def call(
        self,
        x: tf.Tensor,
        **kwargs: Any,
    ) -> tf.Tensor:

        features = self.feat_extractor(x, **kwargs)
        # B x H x W x C --> B x W x H x C
        transposed_feat = tf.transpose(features, perm=[0, 2, 1, 3])
        w, h, c = transposed_feat.get_shape().as_list()[1:]
        # B x W x H x C --> B x W x H * C
        features_seq = tf.reshape(transposed_feat, shape=(-1, w, h * c))
        decoded_features = self.decoder(features_seq, **kwargs)
        return decoded_features


def _crnn_vgg(arch: str, pretrained: bool, input_shape: Optional[Tuple[int, int, int]] = None, **kwargs: Any) -> CRNN:

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']
    _cfg['vocab_size'] = kwargs.get('vocab_size', len(_cfg['vocab']))
    _cfg['rnn_units'] = kwargs.get('rnn_units', _cfg['rnn_units'])

    # Feature extractor
    feat_extractor = vgg.__dict__[_cfg['backbone']](
        input_shape=_cfg['input_shape'],
        include_top=False,
    )

    kwargs['vocab_size'] = _cfg['vocab_size']
    kwargs['rnn_units'] = _cfg['rnn_units']

    # Build the model
    model = CRNN(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, _cfg['url'])

    return model


def _crnn_resnet(
    arch: str, pretrained: bool, input_shape: Optional[Tuple[int, int, int]] = None, **kwargs: Any
) -> CRNN:

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']
    _cfg['vocab_size'] = kwargs.get('vocab_size', len(_cfg['vocab']))
    _cfg['rnn_units'] = kwargs.get('rnn_units', _cfg['rnn_units'])

    # Feature extractor
    feat_extractor = resnet.__dict__[_cfg['backbone']](
        input_shape=_cfg['input_shape'],
        include_top=False,
    )

    kwargs['vocab_size'] = _cfg['vocab_size']
    kwargs['rnn_units'] = _cfg['rnn_units']

    # Build the model
    model = CRNN(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, _cfg['url'])

    return model


def crnn_vgg16_bn(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with a VGG-16 backbone as described in `"Convolutional RNN: an Enhanced Model for Extracting Features
    from Sequential Data" <https://arxiv.org/pdf/1602.05875.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import crnn_vgg16_bn
        >>> model = crnn_vgg16_bn(pretrained=True)
        >>> input_tensor = tf.random.uniform(shape=[1, 32, 128, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        text recognition architecture
    """

    return _crnn_vgg('crnn_vgg16_bn', pretrained, **kwargs)


def crnn_resnet31(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with a resnet31 backbone as described in `"Convolutional RNN: an Enhanced Model for Extracting Features
    from Sequential Data" <https://arxiv.org/pdf/1602.05875.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import crnn_resnet31
        >>> model = crnn_resnet31(pretrained=True)
        >>> input_tensor = tf.random.uniform(shape=[1, 32, 128, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        text recognition architecture
    """

    return _crnn_resnet('crnn_resnet31', pretrained, **kwargs)
