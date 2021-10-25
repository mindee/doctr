# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from typing import Tuple, List, Dict, Any, Optional
from copy import deepcopy

from ...backbones.resnet import ResnetStage
from ...utils import conv_sequence, load_pretrained_params
from ..transformer import Decoder, positional_encoding, create_look_ahead_mask, create_padding_mask
from ....datasets import VOCABS
from .base import _MASTER, _MASTERPostProcessor


__all__ = ['MASTER', 'master', 'MASTERPostProcessor']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'master': {
        'mean': (0.694, 0.695, 0.693),
        'std': (0.299, 0.296, 0.301),
        'input_shape': (32, 128, 3),
        'vocab': VOCABS['legacy_french'],
        'url': 'https://github.com/mindee/doctr/releases/download/v0.3.0/master-bade6eae.zip',
    },
}


class MAGC(layers.Layer):

    """Implements the Multi-Aspect Global Context Attention, as described in
    <https://arxiv.org/pdf/1910.02562.pdf>`_.

    Args:
        inplanes: input channels
        headers: number of headers to split channels
        att_scale: if True, re-scale attention to counteract the variance distibutions
        **kwargs
    """

    def __init__(
        self,
        inplanes: int,
        headers: int = 8,
        att_scale: bool = False,
        ratio: float = 0.0625,  # bottleneck ratio of 1/16 as described in paper
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.headers = headers  # h
        self.inplanes = inplanes  # C
        self.att_scale = att_scale
        self.planes = int(inplanes * ratio)

        self.single_header_inplanes = int(inplanes / headers)  # C / h

        self.conv_mask = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            kernel_initializer=tf.initializers.he_normal()
        )

        self.transform = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.planes,
                    kernel_size=1,
                    kernel_initializer=tf.initializers.he_normal()
                ),
                tf.keras.layers.LayerNormalization([1, 2, 3]),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(
                    filters=self.inplanes,
                    kernel_size=1,
                    kernel_initializer=tf.initializers.he_normal()
                ),
            ],
            name='transform'
        )

    def context_modeling(self, inputs: tf.Tensor) -> tf.Tensor:
        b, h, w, c = (tf.shape(inputs)[i] for i in range(4))

        # B, H, W, C -->> B*h, H, W, C/h
        x = tf.reshape(inputs, shape=(b, h, w, self.headers, self.single_header_inplanes))
        x = tf.transpose(x, perm=(0, 3, 1, 2, 4))
        x = tf.reshape(x, shape=(b * self.headers, h, w, self.single_header_inplanes))

        # Compute shorcut
        shortcut = x
        # B*h, 1, H*W, C/h
        shortcut = tf.reshape(shortcut, shape=(b * self.headers, 1, h * w, self.single_header_inplanes))
        # B*h, 1, C/h, H*W
        shortcut = tf.transpose(shortcut, perm=[0, 1, 3, 2])

        # Compute context mask
        # B*h, H, W, 1,
        context_mask = self.conv_mask(x)
        # B*h, 1, H*W, 1
        context_mask = tf.reshape(context_mask, shape=(b * self.headers, 1, h * w, 1))
        # scale variance
        if self.att_scale and self.headers > 1:
            context_mask = context_mask / math.sqrt(self.single_header_inplanes)
        # B*h, 1, H*W, 1
        context_mask = tf.keras.activations.softmax(context_mask, axis=2)

        # Compute context
        # B*h, 1, C/h, 1
        context = tf.matmul(shortcut, context_mask)
        context = tf.reshape(context, shape=(b, 1, c, 1))
        # B, 1, 1, C
        context = tf.transpose(context, perm=(0, 1, 3, 2))
        # Set shape to resolve shape when calling this module in the Sequential MAGCResnet
        batch, chan = inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[-1]
        context.set_shape([batch, 1, 1, chan])
        return context

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # Context modeling: B, H, W, C  ->  B, 1, 1, C
        context = self.context_modeling(inputs)
        # Transform: B, 1, 1, C  ->  B, 1, 1, C
        transformed = self.transform(context)
        return inputs + transformed


class MAGCResnet(Sequential):

    """Implements the modified resnet with MAGC layers, as described in paper.

    Args:
        headers: number of header to split channels in MAGC layers
        input_shape: shape of the model input (without batch dim)
    """

    def __init__(
        self,
        headers: int = 8,
        input_shape: Tuple[int, int, int] = (32, 128, 3),
    ) -> None:
        _layers = [
            # conv_1x
            *conv_sequence(out_channels=64, activation='relu', bn=True, kernel_size=3, input_shape=input_shape),
            *conv_sequence(out_channels=128, activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 2), (2, 2)),
            # conv_2x
            ResnetStage(num_blocks=1, output_channels=256),
            MAGC(inplanes=256, headers=headers, att_scale=True),
            *conv_sequence(out_channels=256, activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 2), (2, 2)),
            # conv_3x
            ResnetStage(num_blocks=2, output_channels=512),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            *conv_sequence(out_channels=512, activation='relu', bn=True, kernel_size=3),
            layers.MaxPooling2D((2, 1), (2, 1)),
            # conv_4x
            ResnetStage(num_blocks=5, output_channels=512),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            *conv_sequence(out_channels=512, activation='relu', bn=True, kernel_size=3),
            # conv_5x
            ResnetStage(num_blocks=3, output_channels=512),
            MAGC(inplanes=512, headers=headers, att_scale=True),
            *conv_sequence(out_channels=512, activation='relu', bn=True, kernel_size=3),
        ]
        super().__init__(_layers)


class MASTER(_MASTER, Model):

    """Implements MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.
    Implementation based on the official TF implementation: <https://github.com/jiangxiluning/MASTER-TF>`_.

    Args:
        vocab: vocabulary, (without EOS, SOS, PAD)
        d_model: d parameter for the transformer decoder
        headers: headers for the MAGC module
        dff: depth of the pointwise feed-forward layer
        num_heads: number of heads for the mutli-head attention module
        num_layers: number of decoder layers to stack
        max_length: maximum length of character sequence handled by the model
        input_size: size of the image inputs
    """

    def __init__(
        self,
        vocab: str,
        d_model: int = 512,
        headers: int = 8,  # number of multi-aspect context
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

        self.feat_extractor = MAGCResnet(headers=headers, input_shape=input_shape)
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

    def compute_loss(
        self,
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
        encoded = feature + self.feature_pe[:, :h * w, :]

        out: Dict[str, tf.Tensor] = {}

        if target is not None:
            # Compute target: tensor of gts and sequence lengths
            gt, seq_len = self.compute_target(target)

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


def _master(arch: str, pretrained: bool, input_shape: Tuple[int, int, int] = None, **kwargs: Any) -> MASTER:

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']
    _cfg['vocab'] = kwargs.get('vocab', _cfg['vocab'])

    kwargs['vocab'] = _cfg['vocab']

    # Build the model
    model = MASTER(cfg=_cfg, **kwargs)
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
