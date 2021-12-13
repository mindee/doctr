# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers

from doctr.datasets import VOCABS
from doctr.models import backbones

from ...utils import load_pretrained_params
from ..transformer import Decoder, create_look_ahead_mask, create_padding_mask, positional_encoding
from .base import _MASTER, _MASTERPostProcessor

__all__ = ['MASTER', 'master']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'master': {
        'backbone': 'magc_resnet31',
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (32, 128, 3),
        'vocab': VOCABS['legacy_french'],
        'url': 'https://github.com/mindee/doctr/releases/download/v0.3.0/master-bade6eae.zip',
    },
}


class MASTER(_MASTER, Model):

    """Implements MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.
    Implementation based on the official TF implementation: <https://github.com/jiangxiluning/MASTER-TF>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary, (without EOS, SOS, PAD)
        d_model: d parameter for the transformer decoder
        dff: depth of the pointwise feed-forward layer
        num_heads: number of heads for the mutli-head attention module
        num_layers: number of decoder layers to stack
        max_length: maximum length of character sequence handled by the model
        dropout: dropout probability of the decoder
        input_shape: size of the image inputs
        cfg: dictionary containing information about the model
    """

    def __init__(
        self,
        feature_extractor: tf.keras.Model,
        vocab: str,
        d_model: int = 512,
        dff: int = 2048,
        num_heads: int = 8,  # number of heads in the transformer decoder
        num_layers: int = 3,
        max_length: int = 50,
        dropout: float = 0.2,
        input_shape: Tuple[int, int, int] = (32, 128, 3),
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.vocab = vocab
        self.max_length = max_length
        self.cfg = cfg
        self.vocab_size = len(vocab)

        self.feat_extractor = feature_extractor
        self.seq_embedding = layers.Embedding(self.vocab_size + 3, d_model)  # 3 more classes: EOS/PAD/SOS

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=self.vocab_size,
            maximum_position_encoding=max_length,
            dropout=dropout,
        )
        self.feature_pe = positional_encoding(input_shape[0] * input_shape[1], d_model)
        self.linear = layers.Dense(self.vocab_size + 3, kernel_initializer=tf.initializers.he_uniform())

        self.postprocessor = MASTERPostProcessor(vocab=self.vocab)

    def make_mask(self, target: tf.Tensor) -> tf.Tensor:
        look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
        target_padding_mask = create_padding_mask(target, self.vocab_size + 2)  # Pad symbol
        combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)
        return combined_mask

    @staticmethod
    def compute_loss(
        model_output: tf.Tensor,
        gt: tf.Tensor,
        seq_len: List[int],
    ) -> tf.Tensor:
        """Compute categorical cross-entropy loss for the model.
        Sequences are masked after the EOS character.

        Args:
            gt: the encoded tensor with gt labels
            model_output: predicted logits of the model
            seq_len: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        # Input length : number of timesteps
        input_len = tf.shape(model_output)[1]
        # Add one for additional <eos> token (sos disappear in shift!)
        seq_len = tf.cast(seq_len, tf.int32) + 1
        # One-hot gt labels
        oh_gt = tf.one_hot(gt, depth=model_output.shape[2])
        # Compute loss: don't forget to shift gt! Otherwise the model learns to output the gt[t-1]!
        # The "masked" first gt char is <sos>. Delete last logit of the model output.
        cce = tf.nn.softmax_cross_entropy_with_logits(oh_gt[:, 1:, :], model_output[:, :-1, :])
        # Compute mask
        mask_values = tf.zeros_like(cce)
        mask_2d = tf.sequence_mask(seq_len, input_len - 1)  # delete the last mask timestep as well
        masked_loss = tf.where(mask_2d, cce, mask_values)
        ce_loss = tf.math.divide(tf.reduce_sum(masked_loss, axis=1), tf.cast(seq_len, model_output.dtype))

        return tf.expand_dims(ce_loss, axis=1)

    def call(
        self,
        x: tf.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Call function for training

        Args:
            x: images
            target: list of str labels
            return_model_output: if True, return logits
            return_preds: if True, decode logits

        Return:
            A dictionnary containing eventually loss, logits and predictions.
        """

        # Encode
        feature = self.feat_extractor(x, **kwargs)
        b, h, w, c = (tf.shape(feature)[i] for i in range(4))
        feature = tf.reshape(feature, shape=(b, h * w, c))
        encoded = feature + tf.cast(self.feature_pe[:, :h * w, :], dtype=feature.dtype)

        out: Dict[str, tf.Tensor] = {}

        if target is not None:
            # Compute target: tensor of gts and sequence lengths
            gt, seq_len = self.build_target(target)

        if kwargs.get('training', False):
            if target is None:
                raise AssertionError("In training mode, you need to pass a value to 'target'")
            tgt_mask = self.make_mask(gt)
            # Compute logits
            output = self.decoder(gt, encoded, tgt_mask, None, **kwargs)
            logits = self.linear(output, **kwargs)

        else:
            # When not training, we want to compute logits in with the decoder, although
            # we have access to gts (we need gts to compute the loss, but not in the decoder)
            logits = self.decode(encoded, **kwargs)

        if target is not None:
            out['loss'] = self.compute_loss(logits, gt, seq_len)

        if return_model_output:
            out['out_map'] = logits

        if return_preds:
            predictions = self.postprocessor(logits)
            out['preds'] = predictions

        return out

    def decode(self, encoded: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Decode function for prediction

        Args:
            encoded: encoded features

        Return:
            A Tuple of tf.Tensor: predictions, logits
        """
        b = tf.shape(encoded)[0]
        max_len = tf.constant(self.max_length, dtype=tf.int32)
        start_symbol = tf.constant(self.vocab_size + 1, dtype=tf.int32)  # SOS
        padding_symbol = tf.constant(self.vocab_size + 2, dtype=tf.int32)  # PAD

        ys = tf.fill(dims=(b, max_len - 1), value=padding_symbol)
        start_vector = tf.fill(dims=(b, 1), value=start_symbol)
        ys = tf.concat([start_vector, ys], axis=-1)

        logits = tf.zeros(shape=(b, max_len - 1, self.vocab_size + 3), dtype=encoded.dtype)  # 3 symbols
        # max_len = len + 2 (sos + eos)
        for i in range(self.max_length - 1):
            ys_mask = self.make_mask(ys)
            output = self.decoder(ys, encoded, ys_mask, None, **kwargs)
            logits = self.linear(output, **kwargs)
            prob = tf.nn.softmax(logits, axis=-1)
            next_word = tf.argmax(prob, axis=-1, output_type=ys.dtype)
            # ys.shape = B, T
            i_mesh, j_mesh = tf.meshgrid(tf.range(b), tf.range(max_len), indexing='ij')
            indices = tf.stack([i_mesh[:, i + 1], j_mesh[:, i + 1]], axis=1)

            ys = tf.tensor_scatter_nd_update(ys, indices, next_word[:, i + 1])

        # final_logits of shape (N, max_length - 1, vocab_size + 1) (whithout sos)
        return logits


class MASTERPostProcessor(_MASTERPostProcessor):
    """Post processor for MASTER architectures

    Args:
        vocab: string containing the ordered sequence of supported characters
        ignore_case: if True, ignore case of letters
        ignore_accents: if True, ignore accents of letters
    """

    def __call__(
        self,
        logits: tf.Tensor,
    ) -> List[Tuple[str, float]]:
        # compute pred with argmax for attention models
        out_idxs = tf.math.argmax(logits, axis=2)
        # N x L
        probs = tf.gather(tf.nn.softmax(logits, axis=-1), out_idxs, axis=-1, batch_dims=2)
        # Take the minimum confidence of the sequence
        probs = tf.math.reduce_min(probs, axis=1)

        # decode raw output of the model with tf_label_to_idx
        out_idxs = tf.cast(out_idxs, dtype='int32')
        embedding = tf.constant(self._embedding, dtype=tf.string)
        decoded_strings_pred = tf.strings.reduce_join(inputs=tf.nn.embedding_lookup(embedding, out_idxs), axis=-1)
        decoded_strings_pred = tf.strings.split(decoded_strings_pred, "<eos>")
        decoded_strings_pred = tf.sparse.to_dense(decoded_strings_pred.to_sparse(), default_value='not valid')[:, 0]
        word_values = [word.decode() for word in decoded_strings_pred.numpy().tolist()]

        return list(zip(word_values, probs.numpy().tolist()))


def _master(
    arch: str,
    pretrained: bool,
    pretrained_backbone: bool = True,
    input_shape: Tuple[int, int, int] = None,
    **kwargs: Any
) -> MASTER:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']
    _cfg['vocab'] = kwargs.get('vocab', _cfg['vocab'])

    kwargs['vocab'] = _cfg['vocab']

    # Build the model
    model = MASTER(
        backbones.__dict__[_cfg['backbone']](pretrained=pretrained_backbone, input_shape=_cfg['input_shape']),
        cfg=_cfg,
        **kwargs,
    )
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]['url'])

    return model


def master(pretrained: bool = False, **kwargs: Any) -> MASTER:
    """MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import master
        >>> model = master(pretrained=False)
        >>> input_tensor = tf.random.uniform(shape=[1, 48, 160, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _master('master', pretrained, **kwargs)
