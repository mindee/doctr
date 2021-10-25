# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from copy import deepcopy
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from typing import Tuple, Dict, Any, Optional, List

from ...backbones import vgg16_bn, resnet31, mobilenet_v3_small_r, mobilenet_v3_large_r
from ...utils import load_pretrained_params
from ..core import RecognitionModel, RecognitionPostProcessor
from ....datasets import VOCABS

__all__ = ['CRNN', 'crnn_vgg16_bn', 'CTCPostProcessor', 'crnn_mobilenet_v3_small',
           'crnn_mobilenet_v3_large']

default_cfgs: Dict[str, Dict[str, Any]] = {
    'crnn_vgg16_bn': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'backbone': vgg16_bn, 'rnn_units': 128,
        'input_shape': (32, 128, 3),
        'vocab': VOCABS['legacy_french'],
        'url': 'https://github.com/mindee/doctr/releases/download/v0.3.0/crnn_vgg16_bn-76b7f2c6.zip',
    },
    'crnn_mobilenet_v3_small': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'backbone': mobilenet_v3_small_r, 'rnn_units': 128,
        'input_shape': (32, 128, 3),
        'vocab': VOCABS['french'],
        'url': 'https://github.com/mindee/doctr/releases/download/v0.3.1/crnn_mobilenet_v3_small-7f36edec.zip',
    },
    'crnn_mobilenet_v3_large': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'backbone': mobilenet_v3_large_r, 'rnn_units': 128,
        'input_shape': (32, 128, 3),
        'vocab': VOCABS['french'],
        'url': None,
    },
}


class CTCPostProcessor(RecognitionPostProcessor):
    """
    Postprocess raw prediction of the model (logits) to a list of words using CTC decoding

    Args:
        vocab: string containing the ordered sequence of supported characters
        ignore_case: if True, ignore case of letters
        ignore_accents: if True, ignore accents of letters
    """

    def __call__(
        self,
        logits: tf.Tensor
    ) -> List[Tuple[str, float]]:
        """
        Performs decoding of raw output with CTC and decoding of CTC predictions
        with label_to_idx mapping dictionnary

        Args:
            logits: raw output of the model, shape BATCH_SIZE X SEQ_LEN X NUM_CLASSES + 1

        Returns:
            A list of decoded words of length BATCH_SIZE

        """
        # Decode CTC
        _decoded, _log_prob = tf.nn.ctc_beam_search_decoder(
            tf.transpose(logits, perm=[1, 0, 2]),
            tf.fill(logits.shape[0], logits.shape[1]),
            beam_width=1, top_paths=1,
        )
        out_idxs = tf.sparse.to_dense(_decoded[0], default_value=len(self.vocab))
        probs = tf.math.exp(tf.squeeze(_log_prob, axis=1))

        # Map it to characters
        _decoded_strings_pred = tf.strings.reduce_join(
            inputs=tf.nn.embedding_lookup(tf.constant(self._embedding, dtype=tf.string), out_idxs),
            axis=-1
        )
        _decoded_strings_pred = tf.strings.split(_decoded_strings_pred, "<eos>")
        decoded_strings_pred = tf.sparse.to_dense(_decoded_strings_pred.to_sparse(), default_value='not valid')[:, 0]
        word_values = [word.decode() for word in decoded_strings_pred.numpy().tolist()]

        return list(zip(word_values, probs.numpy().tolist()))


class CRNN(RecognitionModel, Model):
    """Implements a CRNN architecture as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of units in the LSTM layers
        cfg: configuration dictionary
    """

    _children_names: List[str] = ['feat_extractor', 'decoder', 'postprocessor']

    def __init__(
        self,
        feature_extractor: tf.keras.Model,
        vocab: str,
        rnn_units: int = 128,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Initialize kernels
        h, w, c = feature_extractor.output_shape[1:]

        super().__init__()
        self.vocab = vocab
        self.max_length = w
        self.cfg = cfg
        self.feat_extractor = feature_extractor

        self.decoder = Sequential(
            [
                layers.Bidirectional(layers.LSTM(units=rnn_units, return_sequences=True)),
                layers.Bidirectional(layers.LSTM(units=rnn_units, return_sequences=True)),
                layers.Dense(units=len(vocab) + 1)
            ]
        )
        self.decoder.build(input_shape=(None, w, h * c))

        self.postprocessor = CTCPostProcessor(vocab=vocab)

    def compute_loss(
        self,
        model_output: tf.Tensor,
        target: List[str],
    ) -> tf.Tensor:
        """Compute CTC loss for the model.

        Args:
            gt: the encoded tensor with gt labels
            model_output: predicted logits of the model
            seq_len: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        gt, seq_len = self.compute_target(target)
        batch_len = model_output.shape[0]
        input_length = tf.fill((batch_len,), model_output.shape[1])
        ctc_loss = tf.nn.ctc_loss(
            gt, model_output, seq_len, input_length, logits_time_major=False, blank_index=len(self.vocab)
        )
        return ctc_loss

    def call(
        self,
        x: tf.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        features = self.feat_extractor(x, **kwargs)
        # B x H x W x C --> B x W x H x C
        transposed_feat = tf.transpose(features, perm=[0, 2, 1, 3])
        w, h, c = transposed_feat.get_shape().as_list()[1:]
        # B x W x H x C --> B x W x H * C
        features_seq = tf.reshape(transposed_feat, shape=(-1, w, h * c))
        logits = self.decoder(features_seq, **kwargs)

        out: Dict[str, tf.Tensor] = {}
        if return_model_output:
            out["out_map"] = logits

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(logits)

        if target is not None:
            out['loss'] = self.compute_loss(logits, target)

        return out


def _crnn(
    arch: str,
    pretrained: bool,
    pretrained_backbone: bool = True,
    input_shape: Optional[Tuple[int, int, int]] = None,
    **kwargs: Any
) -> CRNN:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']
    _cfg['vocab'] = kwargs.get('vocab', _cfg['vocab'])
    _cfg['rnn_units'] = kwargs.get('rnn_units', _cfg['rnn_units'])

    # Feature extractor
    feat_extractor = _cfg['backbone'](
        input_shape=_cfg['input_shape'],
        include_top=False,
        pretrained=pretrained_backbone,
    )

    kwargs['vocab'] = _cfg['vocab']
    kwargs['rnn_units'] = _cfg['rnn_units']

    # Build the model
    model = CRNN(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, _cfg['url'])

    return model


def crnn_vgg16_bn(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with a VGG-16 backbone as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import crnn_vgg16_bn
        >>> model = crnn_vgg16_bn(pretrained=True)
        >>> input_tensor = tf.random.uniform(shape=[1, 32, 128, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _crnn('crnn_vgg16_bn', pretrained, **kwargs)


def crnn_mobilenet_v3_small(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with a MobileNet V3 Small backbone as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import crnn_mobilenet_v3_small
        >>> model = crnn_mobilenet_v3_small(pretrained=True)
        >>> input_tensor = tf.random.uniform(shape=[1, 32, 128, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _crnn('crnn_mobilenet_v3_small', pretrained, **kwargs)


def crnn_mobilenet_v3_large(pretrained: bool = False, **kwargs: Any) -> CRNN:
    """CRNN with a MobileNet V3 Large backbone as described in `"An End-to-End Trainable Neural Network for Image-based
    Sequence Recognition and Its Application to Scene Text Recognition" <https://arxiv.org/pdf/1507.05717.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import crnn_mobilenet_v3_large
        >>> model = crnn_mobilenet_v3_large(pretrained=True)
        >>> input_tensor = tf.random.uniform(shape=[1, 32, 128, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _crnn('crnn_mobilenet_v3_large', pretrained, **kwargs)
