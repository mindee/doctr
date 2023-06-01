# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers

from doctr.datasets import VOCABS
from doctr.utils.repr import NestedObject

from ...classification import resnet31
from ...utils.tensorflow import load_pretrained_params
from ..core import RecognitionModel, RecognitionPostProcessor

__all__ = ["SAR", "sar_resnet31"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "sar_resnet31": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (32, 128, 3),
        "vocab": VOCABS["french"],
        "url": None,
    },
}


class SAREncoder(layers.Layer, NestedObject):
    """Implements encoder module of the SAR model

    Args:
        rnn_units: number of hidden rnn units
        dropout_prob: dropout probability
    """

    def __init__(self, rnn_units: int, dropout_prob: float = 0.0) -> None:
        super().__init__()
        self.rnn = Sequential(
            [
                layers.LSTM(units=rnn_units, return_sequences=True, recurrent_dropout=dropout_prob),
                layers.LSTM(units=rnn_units, return_sequences=False, recurrent_dropout=dropout_prob),
            ]
        )

    def call(
        self,
        x: tf.Tensor,
        **kwargs: Any,
    ) -> tf.Tensor:
        # (N, C)
        return self.rnn(x, **kwargs)


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
        # shape (N, H, W, vgg_units) -> (N, H, W, attention_units)
        features_projection = self.features_projector(features, **kwargs)
        # shape (N, 1, 1, rnn_units) -> (N, 1, 1, attention_units)
        hidden_state = tf.expand_dims(tf.expand_dims(hidden_state, axis=1), axis=1)
        hidden_state_projection = self.hidden_state_projector(hidden_state, **kwargs)
        projection = tf.math.tanh(hidden_state_projection + features_projection)
        # shape (N, H, W, attention_units) -> (N, H, W, 1)
        attention = self.attention_projector(projection, **kwargs)
        # shape (N, H, W, 1) -> (N, H * W)
        attention = self.flatten(attention)
        attention = tf.nn.softmax(attention)
        # shape (N, H * W) -> (N, H, W, 1)
        attention_map = tf.reshape(attention, [-1, H, W, 1])
        glimpse = tf.math.multiply(features, attention_map)
        # shape (N, H * W) -> (N, C)
        return tf.reduce_sum(glimpse, axis=[1, 2])


class SARDecoder(layers.Layer, NestedObject):
    """Implements decoder module of the SAR model

    Args:
        rnn_units: number of hidden units in recurrent cells
        max_length: maximum length of a sequence
        vocab_size: number of classes in the model alphabet
        embedding_units: number of hidden embedding units
        attention_units: number of hidden attention units
        num_decoder_cells: number of LSTMCell layers to stack
        dropout_prob: dropout probability

    """

    def __init__(
        self,
        rnn_units: int,
        max_length: int,
        vocab_size: int,
        embedding_units: int,
        attention_units: int,
        num_decoder_cells: int = 2,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.embed = layers.Dense(embedding_units, use_bias=False)
        self.embed_tgt = layers.Embedding(embedding_units, self.vocab_size + 1)

        self.lstm_cells = layers.StackedRNNCells(
            [layers.LSTMCell(rnn_units, implementation=1) for _ in range(num_decoder_cells)]
        )
        self.attention_module = AttentionModule(attention_units)
        self.output_dense = layers.Dense(self.vocab_size + 1, use_bias=True)
        self.dropout = layers.Dropout(dropout_prob)

    def call(
        self,
        features: tf.Tensor,
        holistic: tf.Tensor,
        gt: Optional[tf.Tensor] = None,
        **kwargs: Any,
    ) -> tf.Tensor:
        if gt is not None:
            gt_embedding = self.embed_tgt(gt, **kwargs)

        logits_list: List[tf.Tensor] = []

        for t in range(self.max_length + 1):  # 32
            if t == 0:
                # step to init the first states of the LSTMCell
                states = self.lstm_cells.get_initial_state(
                    inputs=None, batch_size=features.shape[0], dtype=features.dtype
                )
                prev_symbol = holistic
            elif t == 1:
                # step to init a 'blank' sequence of length vocab_size + 1 filled with zeros
                # (N, vocab_size + 1) --> (N, embedding_units)
                prev_symbol = tf.zeros([features.shape[0], self.vocab_size + 1])
                prev_symbol = self.embed(prev_symbol, **kwargs)
            else:
                if gt is not None:
                    # (N, embedding_units) -2 because of <bos> and <eos> (same)
                    prev_symbol = self.embed(gt_embedding[:, t - 2], **kwargs)
                else:
                    # -1 to start at timestep where prev_symbol was initialized
                    index = tf.argmax(logits_list[t - 1], axis=-1)
                    # update prev_symbol with ones at the index of the previous logit vector
                    # (N, embedding_units)
                    index = tf.ones_like(index)
                    prev_symbol = tf.scatter_nd(
                        tf.expand_dims(index, axis=1),
                        prev_symbol,
                        tf.constant([features.shape[0], features.shape[-1]], dtype=tf.int64),
                    )

            # (N, C), (N, C)  take the last hidden state and cell state from current timestep
            _, states = self.lstm_cells(prev_symbol, states, **kwargs)
            # states = (hidden_state, cell_state)
            hidden_state = states[0][0]
            # (N, H, W, C), (N, C) --> (N, C)
            glimpse = self.attention_module(features, hidden_state, **kwargs)
            # (N, C), (N, C) --> (N, 2 * C)
            logits = tf.concat([hidden_state, glimpse], axis=1)
            logits = self.dropout(logits, **kwargs)
            # (N, vocab_size + 1)
            logits_list.append(self.output_dense(logits, **kwargs))

        # (max_length + 1, N, vocab_size + 1) --> (N, max_length + 1, vocab_size + 1)
        return tf.transpose(tf.stack(logits_list[1:]), (1, 0, 2))


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
        num_decoder_cells: number of LSTMCell layers to stack
        dropout_prob: dropout probability for the encoder and decoder
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    _children_names: List[str] = ["feat_extractor", "encoder", "decoder", "postprocessor"]

    def __init__(
        self,
        feature_extractor,
        vocab: str,
        rnn_units: int = 512,
        embedding_units: int = 512,
        attention_units: int = 512,
        max_length: int = 30,
        num_decoder_cells: int = 2,
        dropout_prob: float = 0.0,
        exportable: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length + 1  # Add 1 timestep for EOS after the longest word

        self.feat_extractor = feature_extractor

        self.encoder = SAREncoder(rnn_units, dropout_prob)
        self.decoder = SARDecoder(
            rnn_units,
            self.max_length,
            len(vocab),
            embedding_units,
            attention_units,
            num_decoder_cells,
            dropout_prob,
        )

        self.postprocessor = SARPostProcessor(vocab=vocab)

    @staticmethod
    def compute_loss(
        model_output: tf.Tensor,
        gt: tf.Tensor,
        seq_len: tf.Tensor,
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
        # Add one for additional <eos> token
        seq_len = seq_len + 1
        # One-hot gt labels
        oh_gt = tf.one_hot(gt, depth=model_output.shape[2])
        # Compute loss
        cce = tf.nn.softmax_cross_entropy_with_logits(oh_gt, model_output)
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        features = self.feat_extractor(x, **kwargs)
        # vertical max pooling --> (N, C, W)
        pooled_features = tf.reduce_max(features, axis=1)
        # holistic (N, C)
        encoded = self.encoder(pooled_features, **kwargs)

        if target is not None:
            gt, seq_len = self.build_target(target)
            seq_len = tf.cast(seq_len, tf.int32)

        if kwargs.get("training", False) and target is None:
            raise ValueError("Need to provide labels during training for teacher forcing")

        decoded_features = self.decoder(features, encoded, gt=None if target is None else gt, **kwargs)

        out: Dict[str, tf.Tensor] = {}
        if self.exportable:
            out["logits"] = decoded_features
            return out

        if return_model_output:
            out["out_map"] = decoded_features

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(decoded_features)

        if target is not None:
            out["loss"] = self.compute_loss(decoded_features, gt, seq_len)

        return out


class SARPostProcessor(RecognitionPostProcessor):
    """Post processor for SAR architectures

    Args:
        vocab: string containing the ordered sequence of supported characters
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
        out_idxs = tf.cast(out_idxs, dtype="int32")
        embedding = tf.constant(self._embedding, dtype=tf.string)
        decoded_strings_pred = tf.strings.reduce_join(inputs=tf.nn.embedding_lookup(embedding, out_idxs), axis=-1)
        decoded_strings_pred = tf.strings.split(decoded_strings_pred, "<eos>")
        decoded_strings_pred = tf.sparse.to_dense(decoded_strings_pred.to_sparse(), default_value="not valid")[:, 0]
        word_values = [word.decode() for word in decoded_strings_pred.numpy().tolist()]

        return list(zip(word_values, probs.numpy().tolist()))


def _sar(
    arch: str,
    pretrained: bool,
    backbone_fn,
    pretrained_backbone: bool = True,
    input_shape: Optional[Tuple[int, int, int]] = None,
    **kwargs: Any,
) -> SAR:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["input_shape"] = input_shape or _cfg["input_shape"]
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])

    # Feature extractor
    feat_extractor = backbone_fn(
        pretrained=pretrained_backbone,
        input_shape=_cfg["input_shape"],
        include_top=False,
    )

    kwargs["vocab"] = _cfg["vocab"]

    # Build the model
    model = SAR(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"])

    return model


def sar_resnet31(pretrained: bool = False, **kwargs: Any) -> SAR:
    """SAR with a resnet-31 feature extractor as described in `"Show, Attend and Read:A Simple and Strong
    Baseline for Irregular Text Recognition" <https://arxiv.org/pdf/1811.00751.pdf>`_.

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

    return _sar("sar_resnet31", pretrained, resnet31, **kwargs)
