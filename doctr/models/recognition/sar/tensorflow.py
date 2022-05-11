# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, Sequential, layers

from doctr.utils.repr import NestedObject

from ....datasets import VOCABS
from ...backbones import resnet31
from ...utils import load_pretrained_params
from ..core import RecognitionModel, RecognitionPostProcessor

__all__ = ["SAR", "SARPostProcessor", "sar_resnet31"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "sar_resnet31": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "backbone": resnet31,
        "rnn_units": 512,
        "max_length": 30,
        "num_decoders": 2,
        "input_shape": (32, 128, 3),
        "vocab": VOCABS["legacy_french"],
        "url": "https://github.com/mindee/doctr/releases/download/v0.3.0/sar_resnet31-9ee49970.zip",
    },
}


class AttentionModule(layers.Layer, NestedObject):
    """Implements attention module of the SAR model

    Args:
        attention_units: number of hidden attention units

    """

    def __init__(self, attention_units: int) -> None:

        super().__init__()
        self.hidden_state_projector = layers.Conv2D(
            attention_units,
            1,
            strides=1,
            use_bias=False,
            padding="same",
            kernel_initializer="he_normal",
        )
        self.features_projector = layers.Conv2D(
            attention_units,
            3,
            strides=1,
            use_bias=True,
            padding="same",
            kernel_initializer="he_normal",
        )
        self.attention_projector = layers.Conv2D(
            1,
            1,
            strides=1,
            use_bias=False,
            padding="same",
            kernel_initializer="he_normal",
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
        # shape (N, H, W, D) -> (N, H, W, attention_units)
        features_projection = self.features_projector(features, **kwargs)
        # shape (N, 1, 1, attention_units) + (N, H, W, attention_units) -> (N, H, W, attention_units)
        projection = tf.math.tanh(hidden_state_projection + features_projection)
        # shape (N, H, W, attention_units) -> (N, H, W, 1)
        attention = self.attention_projector(projection, **kwargs)
        # shape (N, H, W, 1) -> (N, H * W)
        attention_shape = [tf.shape(attention)[k] for k in range(len(attention.shape))]
        attention = tf.reshape(attention, attention_shape[:-3] + [H * W])
        attention = tf.nn.softmax(attention)
        # shape (N, H * W) -> (N, H, W, 1)
        attention_map = tf.reshape(attention, attention_shape)
        # shape (N, H, W, D) x (N, H, W, 1) -> (N, H, W, D)
        glimpse = tf.math.multiply(features, attention_map)
        # shape (N, H, W, D) -> (N, D)
        glimpse = tf.reduce_sum(glimpse, axis=[-3, -2])
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
        attention_units: int,
        num_decoder_layers: int = 2,
        input_shape: Optional[List[Tuple[Optional[int]]]] = None,
    ) -> None:

        super().__init__()
        self.vocab_size = vocab_size
        self.lstm_decoder = layers.StackedRNNCells(
            [layers.LSTMCell(rnn_units, implementation=1) for _ in range(num_decoder_layers)]
        )
        self.embed = layers.Dense(rnn_units, use_bias=False, input_shape=(None, self.vocab_size + 1))
        self.attention_module = AttentionModule(attention_units)
        self.output_dense = layers.Dense(vocab_size + 1, use_bias=True)
        self.max_length = max_length

        # Initialize kernels
        if input_shape is not None:
            self.attention_module.call(layers.Input(input_shape[0][1:]), layers.Input((1, 1, rnn_units)))

    def embed_token(self, token: tf.Tensor, **kwargs):
        return self.embed(tf.one_hot(token, depth=self.vocab_size + 1), **kwargs)

    def greedy_decode(self, symbol, states, features, gt, **kwargs):
        # if gt is None:
        #     raise ValueError("Need to provide labels during training for teacher forcing")

        logits_list = []

        for t in range(self.max_length + 1):  # keep 1 step for <eos>
            # one-hot symbol with depth vocab_size + 1
            # embeded_symbol: shape (N, embedding_units)
            embeded_symbol = self.embed(tf.one_hot(symbol, depth=self.vocab_size + 1), **kwargs)
            logits, states = self.lstm_decoder(embeded_symbol, states, **kwargs)
            glimpse = self.attention_module(
                features,
                tf.expand_dims(tf.expand_dims(logits, axis=1), axis=1),
                **kwargs,
            )
            # logits: shape (N, rnn_units), glimpse: shape (N, 1)
            logits = tf.concat([logits, glimpse], axis=-1)
            # shape (N, rnn_units + 1) -> (N, vocab_size + 1)
            logits = self.output_dense(logits, **kwargs)
            # update symbol with predicted logits for t+1 step
            if gt is not None:
                # Teacher forcing
                symbol = gt[:, t]  # type: ignore[index]
            else:
                symbol = tf.argmax(logits, axis=-1)
            logits_list.append(logits)
        scores = tf.stack(logits_list, axis=1)

        return scores

    def beam_decode(self, symbol, states, features, beam_width, **kwargs):
        states = tfa.seq2seq.tile_batch(states, multiplier=beam_width)

        cell_output_layer = SARCellOutputLayer(
            attention_module=self.attention_module,
            output_dense=self.output_dense,
            feature_map=features,
        )

        beam_search_decoder = tfa.seq2seq.BeamSearchDecoder(
            cell=self.lstm_decoder,
            beam_width=beam_width,
            embedding_fn=self.embed_token,
            output_layer=cell_output_layer,
            output_all_scores=False,
            maximum_iterations=tf.constant(self.max_length + 1, dtype=tf.int32),
        )

        outputs, _, _ = beam_search_decoder(
            embedding=None, start_tokens=symbol, end_token=self.vocab_size, initial_state=states, **kwargs
        )

        # shape (N, seq_length, beam_width)
        log_probs = outputs.beam_search_decoder_output.scores
        # shape (N, beam_width, seq_length)
        log_probs = tf.transpose(log_probs, (0, 2, 1))

        # shape (N, seq_length, beam_width)
        ids = outputs.predicted_ids
        # shape (N, beam_width, seq_length)
        ids = tf.transpose(ids, (0, 2, 1))

        # Build beam sequences scores as in greedy decoding: shape (N, beam_width, seq_length, vocab_size + 1)
        oh_ids = tf.one_hot(ids, axis=-1, depth=self.vocab_size + 1)
        oh_ids = tf.where(oh_ids == 0.0, np.nan, oh_ids)
        scores = oh_ids * log_probs[..., None]

        return scores

    def call(
        self,
        features: tf.Tensor,
        holistic: tf.Tensor,
        gt: Optional[tf.Tensor] = None,
        beam_width: Optional[int] = None,
        **kwargs: Any,
    ) -> tf.Tensor:

        # initialize states (each of shape (N, rnn_units))
        states = self.lstm_decoder.get_initial_state(
            inputs=None, batch_size=tf.shape(features)[0], dtype=features.dtype
        )
        # run first step of lstm
        # holistic: shape (N, rnn_units)
        _, states = self.lstm_decoder(holistic, states, **kwargs)
        # Initialize with the index of virtual START symbol (placed after <eos> so that the one-hot is only zeros)
        symbol = tf.fill((tf.shape(features)[0],), self.vocab_size + 1)

        if kwargs.get("training") or beam_width is None:
            # Scores here are logits: shape (N, max_length + 1, vocab_size + 1)
            scores = self.greedy_decode(symbol=symbol, states=states, gt=gt, features=features, **kwargs)
        else:
            # Scores here are probabilities: shape (N, beam_width, seq_length, vocab_size + 1)
            scores = self.beam_decode(symbol=symbol, states=states, features=features, beam_width=beam_width, **kwargs)

        return scores


class SARCellOutputLayer(layers.Layer):
    def __init__(
        self,
        attention_module: AttentionModule,
        output_dense: layers.Dense,
        feature_map: tf.Tensor,
        **kwargs,
    ):
        self.attention_module = attention_module
        self.output_dense = output_dense
        self.feature_map = feature_map
        super(SARCellOutputLayer, self).__init__(**kwargs)

    def call(self, inputs: tf.Tensor, **kwargs):
        # inputs: shape (N, B, rnn_units) -> (B, N, rnn_units)
        inputs = tf.transpose(inputs, (1, 0, 2))
        # glimpse: shape (N, H, W, D) | (B, N, 1, 1, rnn_units) -> (B, N, D)
        glimpse = self.attention_module(
            self.feature_map,
            tf.expand_dims(tf.expand_dims(inputs, axis=-2), axis=-2),
            **kwargs,
        )
        # logits: shape [inputs (B, N, rnn_units), glimpse (B, N, D)] -> (B, N, rnn_units + D)
        logits = tf.concat([inputs, glimpse], axis=-1)
        # shape (B, N, rnn_units + D) -> (B, N, vocab_size + 1)
        logits = self.output_dense(logits, **kwargs)
        # shape (B, N, vocab_size + 1) -> (N, B, vocab_size + 1)
        logits = tf.transpose(logits, (1, 0, 2))

        return logits

    def compute_output_shape(self, input_shape: tf.Tensor):
        return self.output_dense.compute_output_shape(input_shape)


class SAR(Model, RecognitionModel):
    """Implements a SAR architecture as described in `"Show, Attend and Read:A Simple and Strong Baseline for
    Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        rnn_units: number of hidden units in both encoder and decoder LSTM
        embedding_units: number of embedding units
        attention_units: number of hidden units in attention module
        max_length: maximum word length handled by the model
        num_decoders: number of LSTM to stack in decoder layer

    """

    _children_names: List[str] = ["feat_extractor", "encoder", "decoder", "postprocessor"]

    def __init__(
        self,
        feature_extractor,
        vocab: str,
        rnn_units: int = 512,
        attention_units: int = 512,
        max_length: int = 30,
        num_decoders: int = 2,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__()
        self.vocab = vocab
        self.cfg = cfg

        self.max_length = max_length + 1  # Add 1 timestep for EOS after the longest word

        self.feat_extractor = feature_extractor

        self.encoder = Sequential(
            [
                layers.LSTM(units=rnn_units, return_sequences=True),
                layers.LSTM(units=rnn_units, return_sequences=False),
            ]
        )
        # Initialize the kernels (watch out for reduce_max)
        self.encoder.build(input_shape=(None,) + self.feat_extractor.output_shape[2:])

        self.decoder = SARDecoder(
            rnn_units=rnn_units,
            max_length=max_length,
            vocab_size=len(vocab),
            attention_units=attention_units,
            num_decoder_layers=num_decoders,
        )

        self.postprocessor = SARPostProcessor(vocab=vocab)

    @staticmethod
    def compute_loss(
        model_output: tf.Tensor,
        gt: tf.Tensor,
        seq_len: tf.Tensor,
        from_logits: bool = True,
    ) -> tf.Tensor:
        """Compute categorical cross-entropy loss for the model.
        Sequences are masked after the EOS character.

        Args:
            gt: the encoded tensor with gt labels
            model_output: predicted logits of the model
            seq_len: lengths of each gt word inside the batch
            from_logits: Whether model_output is expected to be a logits tensor. By default, we assume that
            model_output encodes logits.

        Returns:
            The loss of the model on the batch
        """
        # Input length : number of timesteps
        input_len = tf.shape(model_output)[-2]
        # Add one for additional <eos> token
        seq_len = seq_len + 1
        # One-hot gt labels
        oh_gt = tf.one_hot(gt, depth=model_output.shape[-1])

        if from_logits:
            # Compute loss
            cce = tf.nn.softmax_cross_entropy_with_logits(oh_gt, model_output)

        else:
            # Take the top sequence
            model_output = model_output[:, 0, ...]
            # Convert scores from log probabilities to probabilities
            model_output = tf.math.exp(model_output)
            # Compute loss
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction="none")
            cce = loss(oh_gt[:, :input_len], model_output[:, :input_len])

        # Compute mask
        mask_values = tf.zeros_like(cce)
        mask_2d = tf.sequence_mask(seq_len, input_len)
        masked_loss = tf.where(mask_2d, cce, mask_values)
        ce_loss = tf.math.divide(tf.reduce_sum(masked_loss, axis=1), tf.cast(seq_len, model_output.dtype))
        return tf.expand_dims(ce_loss, axis=1)

    def call(
        self,
        x: tf.Tensor,
        target: Optional[List[str]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        beam_width: Optional[int] = None,
        top_sequences: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Feature extraction: shape (N, H, W, D)
        features = self.feat_extractor(x, **kwargs)
        # Vertical max pooling: shape (N, H, W, D) -> (N, W, D)
        pooled_features = tf.reduce_max(features, axis=1)
        # Bilayer LSTM Encoder: shape (N, W, D) -> (N, rnn_units)
        encoded = self.encoder(pooled_features, **kwargs)
        # Encode target
        if target is not None:
            gt, seq_len = self.compute_target(target)
            seq_len = tf.cast(seq_len, tf.int32)
        # Bilayer LSTM with attention decoder
        decoded_features = self.decoder(
            features=features, holistic=encoded, gt=None if target is None else gt, beam_width=beam_width, **kwargs
        )

        out: Dict[str, tf.Tensor] = {}
        if return_model_output:
            out["out_map"] = decoded_features

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(decoded_features, top_sequences=top_sequences)

        if target is not None:
            out["loss"] = self.compute_loss(
                decoded_features, gt, seq_len, from_logits=kwargs.get("training") or beam_width is None
            )

        return out


class SARPostProcessor(RecognitionPostProcessor):
    """Post processor for SAR architectures

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __call__(self, scores: tf.Tensor, top_sequences: Optional[int] = None) -> List[Tuple[str, float]]:
        # compute pred with argmax for attention models
        out_idxs = tf.math.argmax(scores, axis=-1)

        # N x L
        if top_sequences:
            # Compute sequence log-probabilities (scores are log probabilities)
            log_probs = tf.gather(scores, out_idxs, axis=-1, batch_dims=3)
            # Take the geometric mean probability of the sequence
            probs = tf.math.exp(tf.math.reduce_mean(log_probs, axis=-1))[:, :top_sequences]
        else:
            # Compute sequence probabilities
            probs = tf.nn.softmax(scores, axis=-1)
            probs = tf.gather(probs, out_idxs, axis=-1, batch_dims=2)
            # Take the geometric mean probability of the sequence
            probs = tf.math.exp(tf.math.reduce_mean(tf.math.log(probs), axis=-1))

        # decode raw output of the model with tf_label_to_idx
        out_idxs = tf.cast(out_idxs, dtype="int32")
        embedding = tf.constant(self._embedding, dtype=tf.string)
        decoded_strings_pred = tf.strings.reduce_join(inputs=tf.nn.embedding_lookup(embedding, out_idxs), axis=-1)
        decoded_strings_pred = tf.strings.split(decoded_strings_pred, "<eos>")
        decoded_strings_pred = tf.sparse.to_dense(decoded_strings_pred.to_sparse(), default_value="not valid")[..., 0]

        if top_sequences:
            decoded_strings_pred = decoded_strings_pred[:, :top_sequences]

        return {"predictions": decoded_strings_pred, "probabilities": probs}


def _sar(
    arch: str,
    pretrained: bool,
    pretrained_backbone: bool = True,
    input_shape: Tuple[int, int, int] = None,
    **kwargs: Any,
) -> SAR:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["input_shape"] = input_shape or _cfg["input_shape"]
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])
    _cfg["rnn_units"] = kwargs.get("rnn_units", _cfg["rnn_units"])
    _cfg["attention_units"] = kwargs.get("attention_units", _cfg["rnn_units"])
    _cfg["max_length"] = kwargs.get("max_length", _cfg["max_length"])
    _cfg["num_decoders"] = kwargs.get("num_decoders", _cfg["num_decoders"])

    # Feature extractor
    feat_extractor = default_cfgs[arch]["backbone"](
        input_shape=_cfg["input_shape"],
        pretrained=pretrained_backbone,
        include_top=False,
    )

    kwargs["vocab"] = _cfg["vocab"]
    kwargs["rnn_units"] = _cfg["rnn_units"]
    kwargs["attention_units"] = _cfg["attention_units"]
    kwargs["max_length"] = _cfg["max_length"]
    kwargs["num_decoders"] = _cfg["num_decoders"]

    # Build the model
    model = SAR(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"])

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
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _sar("sar_resnet31", pretrained, **kwargs)
