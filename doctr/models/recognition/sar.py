# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from copy import deepcopy
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from typing import Tuple, Dict, List, Any, Optional

from .. import vgg, resnet
from ..utils import load_pretrained_params
from .core import RecognitionModel
from .core import RecognitionPostProcessor
from doctr.utils.repr import NestedObject

__all__ = ['SAR', 'SARPostProcessor', 'sar_vgg16_bn', 'sar_resnet31']

default_cfgs: Dict[str, Dict[str, Any]] = {
    'sar_vgg16_bn': {
        'mean': (.5, .5, .5),
        'std': (1., 1., 1.),
        'backbone': 'vgg16_bn', 'rnn_units': 512, 'max_length': 30, 'num_decoders': 2,
        'input_shape': (32, 128, 3),
        'post_processor': 'SARPostProcessor',
        'vocab': ('3K}7eé;5àÎYho]QwV6qU~W"XnbBvcADfËmy.9ÔpÛ*{CôïE%M4#ÈR:g@T$x?0î£|za1ù8,OG€P-'
                  'kçHëÀÂ2É/ûIJ\'j(LNÙFut[)èZs+&°Sd=Ï!<â_Ç>rêi`l'),
        'url': 'https://github.com/mindee/doctr/releases/download/v0.1-models/sar_vgg16bn-0d7e2c26.zip',
    },
    'sar_resnet31': {
        'mean': (.5, .5, .5),
        'std': (1., 1., 1.),
        'backbone': 'resnet31', 'rnn_units': 512, 'max_length': 30, 'num_decoders': 2,
        'input_shape': (32, 128, 3),
        'post_processor': 'SARPostProcessor',
        'vocab': ('3K}7eé;5àÎYho]QwV6qU~W"XnbBvcADfËmy.9ÔpÛ*{CôïE%M4#ÈR:g@T$x?0î£|za1ù8,OG€P-'
                  'kçHëÀÂ2É/ûIJ\'j(LNÙFut[)èZs+&°Sd=Ï!<â_Ç>rêi`l'),
        'url': 'https://github.com/mindee/doctr/releases/download/v0.1.0/sar_resnet31-ea202587.zip',
    },
}


class AttentionModule(layers.Layer, NestedObject):
    """Implements attention module of the SAR model

    Args:
        attention_units: number of hidden attention units

    """
    def __init__(
        self,
        attention_units: int
    ) -> None:

        super().__init__()
        self.hidden_state_projector = layers.Conv2D(
            attention_units, 1, strides=1, use_bias=False, padding='same', kernel_initializer='he_normal',
        )
        self.features_projector = layers.Conv2D(
            attention_units, 3, strides=1, use_bias=True, padding='same', kernel_initializer='he_normal',
        )
        self.attention_projector = layers.Conv2D(
            1, 1, strides=1, use_bias=False, padding="same", kernel_initializer='he_normal',
        )
        self.flatten = layers.Flatten()

    def call(
        self,
        features: tf.Tensor,
        hidden_state: tf.Tensor,
        **kwargs: Any,
    ) -> tf.Tensor:

        [H, W] = features.get_shape().as_list()[1:3]
        # shape (N, 1, 1, rnn_units) -> (N, 1, 1, attention_units)
        hidden_state_projection = self.hidden_state_projector(hidden_state, **kwargs)
        # shape (N, H, W, vgg_units) -> (N, H, W, attention_units)
        features_projection = self.features_projector(features, **kwargs)
        projection = tf.math.tanh(hidden_state_projection + features_projection)
        # shape (N, H, W, attention_units) -> (N, H, W, 1)
        attention = self.attention_projector(projection, **kwargs)
        # shape (N, H, W, 1) -> (N, H * W)
        attention = self.flatten(attention)
        attention = tf.nn.softmax(attention)
        # shape (N, H * W) -> (N, H, W, 1)
        attention_map = tf.reshape(attention, [-1, H, W, 1])
        glimpse = tf.math.multiply(features, attention_map)
        # shape (N, H * W) -> (N, 1)
        glimpse = tf.reduce_sum(glimpse, axis=[1, 2])
        return glimpse


class SARDecoder(layers.Layer, NestedObject):
    """Implements decoder module of the SAR model

    Args:
        rnn_units: number of hidden units in recurrent cells
        max_length: maximum length of a sequence
        vocab_size: number of classes in the model alphabet
        embedding_units: number of hidden embedding units
        attention_units: number of hidden attention units
        num_decoder_layers: number of LSTM layers to stack

    """
    def __init__(
        self,
        rnn_units: int,
        max_length: int,
        vocab_size: int,
        embedding_units: int,
        attention_units: int,
        num_decoder_layers: int = 2,
        input_shape: Optional[List[Tuple[Optional[int]]]] = None,
    ) -> None:

        super().__init__()
        self.vocab_size = vocab_size
        self.lstm_decoder = layers.StackedRNNCells(
            [layers.LSTMCell(rnn_units, dtype=tf.float32, implementation=1) for _ in range(num_decoder_layers)]
        )
        self.embed = layers.Dense(embedding_units, use_bias=False, input_shape=(None, self.vocab_size + 1))
        self.attention_module = AttentionModule(attention_units)
        self.output_dense = layers.Dense(vocab_size + 1, use_bias=True, input_shape=(None, 2 * rnn_units))
        self.max_length = max_length

        # Initialize kernels
        if input_shape is not None:
            self.attention_module.call(layers.Input(input_shape[0][1:]), layers.Input((1, 1, rnn_units)))

    def call(
        self,
        features: tf.Tensor,
        holistic: tf.Tensor,
        labels: Optional[tf.sparse.SparseTensor] = None,
        **kwargs: Any,
    ) -> tf.Tensor:

        # initialize states (each of shape (N, rnn_units))
        states = self.lstm_decoder.get_initial_state(
            inputs=None, batch_size=features.shape[0], dtype=tf.float32
        )
        # run first step of lstm
        # holistic: shape (N, rnn_units)
        _, states = self.lstm_decoder(holistic, states, **kwargs)
        # Initialize with the index of virtual START symbol (placed after <eos>)
        symbol = tf.fill(features.shape[0], self.vocab_size + 1)
        logits_list = []
        for t in range(self.max_length + 1):  # keep 1 step for <eos>
            # one-hot symbol with depth vocab_size + 1
            # embeded_symbol: shape (N, embedding_units)
            embeded_symbol = self.embed(tf.one_hot(symbol, depth=self.vocab_size + 1), **kwargs)
            logits, states = self.lstm_decoder(embeded_symbol, states, **kwargs)
            glimpse = self.attention_module(
                features, tf.expand_dims(tf.expand_dims(logits, axis=1), axis=1), **kwargs,
            )
            # logits: shape (N, rnn_units), glimpse: shape (N, 1)
            logits = tf.concat([logits, glimpse], axis=-1)
            # shape (N, rnn_units + 1) -> (N, vocab_size + 1)
            logits = self.output_dense(logits, **kwargs)
            # update symbol with predicted logits for t+1 step
            if kwargs.get('training'):
                dense_labels = tf.sparse.to_dense(
                    labels, default_value=self.vocab_size
                )
                # padding dense_labels: shape (N, sequence_length) -> (N, max_length + 1)
                # with constant values: eos symbol = vocab_size
                batch_size = dense_labels.shape[0]
                s = tf.shape(dense_labels)
                paddings = [[0, m - s[i]] for (i, m) in enumerate([batch_size, self.max_length + 1])]
                dense_labels = tf.pad(dense_labels, paddings, 'CONSTANT', constant_values=self.vocab_size)
                symbol = dense_labels[:, t]
            else:
                symbol = tf.argmax(logits, axis=-1)
            logits_list.append(logits)
        outputs = tf.stack(logits_list, axis=1)  # shape (N, max_length + 1, vocab_size + 1)

        return outputs


class SAR(RecognitionModel):
    """Implements a SAR architecture as described in `"Show, Attend and Read:A Simple and Strong Baseline for
    Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab_size: size of the alphabet
        rnn_units: number of hidden units in both encoder and decoder LSTM
        embedding_units: number of embedding units
        attention_units: number of hidden units in attention module
        max_length: maximum word length handled by the model
        num_decoders: number of LSTM to stack in decoder layer

    """

    _children_names: List[str] = ['feat_extractor', 'encoder', 'decoder']

    def __init__(
        self,
        feature_extractor,
        vocab_size: int = 110,
        rnn_units: int = 512,
        embedding_units: int = 512,
        attention_units: int = 512,
        max_length: int = 30,
        num_decoders: int = 2,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(cfg=cfg)

        self.feat_extractor = feature_extractor

        self.encoder = Sequential(
            [
                layers.LSTM(units=rnn_units, return_sequences=True),
                layers.LSTM(units=rnn_units, return_sequences=False)
            ]
        )
        # Initialize the kernels (watch out for reduce_max)
        self.encoder.build(input_shape=(None,) + self.feat_extractor.output_shape[2:])

        self.decoder = SARDecoder(
            rnn_units, max_length, vocab_size, embedding_units, attention_units, num_decoders,
            input_shape=[self.feat_extractor.output_shape, self.encoder.output_shape]
        )

    def call(
        self,
        x: tf.Tensor,
        labels: Optional[tf.sparse.SparseTensor] = None,
        **kwargs: Any,
    ) -> tf.Tensor:

        features = self.feat_extractor(x, **kwargs)
        pooled_features = tf.reduce_max(features, axis=1)  # vertical max pooling
        encoded = self.encoder(pooled_features, **kwargs)
        if kwargs.get('training'):
            if labels is None:
                raise ValueError('Need to provide labels during training for teacher forcing')
            decoded = self.decoder(features, encoded, labels, **kwargs)
        else:
            decoded = self.decoder(features, encoded, **kwargs)

        return decoded


class SARPostProcessor(RecognitionPostProcessor):
    """Post processor for SAR architectures

    Args:
        vocab: string containing the ordered sequence of supported characters
        ignore_case: if True, ignore case of letters
        ignore_accents: if True, ignore accents of letters
    """

    def __call__(
        self,
        logits: tf.Tensor,
    ) -> List[str]:
        # compute pred with argmax for attention models
        pred = tf.math.argmax(logits, axis=2)

        # decode raw output of the model with tf_label_to_idx
        pred = tf.cast(pred, dtype='int32')
        decoded_strings_pred = tf.strings.reduce_join(inputs=tf.nn.embedding_lookup(self._embedding, pred), axis=-1)
        decoded_strings_pred = tf.strings.split(decoded_strings_pred, "<eos>")
        decoded_strings_pred = tf.sparse.to_dense(decoded_strings_pred.to_sparse(), default_value='not valid')[:, 0]
        words_list = [word.decode() for word in list(decoded_strings_pred.numpy())]

        if self.ignore_case:
            words_list = [word.lower() for word in words_list]

        if self.ignore_accents:
            raise NotImplementedError

        return words_list


def _sar_vgg(arch: str, pretrained: bool, input_shape: Tuple[int, int, int] = None, **kwargs: Any) -> SAR:

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']
    _cfg['vocab_size'] = kwargs.get('vocab_size', len(_cfg['vocab']))
    _cfg['rnn_units'] = kwargs.get('rnn_units', _cfg['rnn_units'])
    _cfg['embedding_units'] = kwargs.get('embedding_units', _cfg['rnn_units'])
    _cfg['attention_units'] = kwargs.get('attention_units', _cfg['rnn_units'])
    _cfg['max_length'] = kwargs.get('max_length', _cfg['max_length'])
    _cfg['num_decoders'] = kwargs.get('num_decoders', _cfg['num_decoders'])

    # Feature extractor
    feat_extractor = vgg.__dict__[default_cfgs[arch]['backbone']](
        input_shape=_cfg['input_shape'],
        include_top=False,
    )

    kwargs['vocab_size'] = _cfg['vocab_size']
    kwargs['rnn_units'] = _cfg['rnn_units']
    kwargs['embedding_units'] = _cfg['embedding_units']
    kwargs['attention_units'] = _cfg['attention_units']
    kwargs['max_length'] = _cfg['max_length']
    kwargs['num_decoders'] = _cfg['num_decoders']

    # Build the model
    model = SAR(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def sar_vgg16_bn(pretrained: bool = False, **kwargs: Any) -> SAR:
    """SAR with a VGG16 feature extractor as described in `"Show, Attend and Read:A Simple and Strong
    Baseline for Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import sar_vgg16_bn
        >>> model = sar_vgg16_bn(pretrained=False)
        >>> input_tensor = tf.random.uniform(shape=[1, 64, 256, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        text recognition architecture
    """

    return _sar_vgg('sar_vgg16_bn', pretrained, **kwargs)


def _sar_resnet(arch: str, pretrained: bool, input_shape: Tuple[int, int, int] = None, **kwargs: Any) -> SAR:

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']
    _cfg['vocab_size'] = kwargs.get('vocab_size', len(_cfg['vocab']))
    _cfg['rnn_units'] = kwargs.get('rnn_units', _cfg['rnn_units'])
    _cfg['embedding_units'] = kwargs.get('embedding_units', _cfg['rnn_units'])
    _cfg['attention_units'] = kwargs.get('attention_units', _cfg['rnn_units'])
    _cfg['max_length'] = kwargs.get('max_length', _cfg['max_length'])
    _cfg['num_decoders'] = kwargs.get('num_decoders', _cfg['num_decoders'])

    # Feature extractor
    feat_extractor = resnet.__dict__[default_cfgs[arch]['backbone']](
        input_shape=_cfg['input_shape'],
        include_top=False,
    )

    kwargs['vocab_size'] = _cfg['vocab_size']
    kwargs['rnn_units'] = _cfg['rnn_units']
    kwargs['embedding_units'] = _cfg['embedding_units']
    kwargs['attention_units'] = _cfg['attention_units']
    kwargs['max_length'] = _cfg['max_length']
    kwargs['num_decoders'] = _cfg['num_decoders']

    # Build the model
    model = SAR(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def sar_resnet31(pretrained: bool = False, **kwargs: Any) -> SAR:
    """SAR with a resnet-31 feature extractor as described in `"Show, Attend and Read:A Simple and Strong
    Baseline for Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Example:
        >>> import tensorflow as tf
        >>> from doctr.models import sar_resnet31
        >>> model = sar_resnet31(pretrained=False)
        >>> input_tensor = tf.random.uniform(shape=[1, 64, 256, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        text recognition architecture
    """

    return _sar_resnet('sar_resnet31', pretrained, **kwargs)
