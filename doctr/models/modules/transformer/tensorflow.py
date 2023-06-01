# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from typing import Any, Callable, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers

from doctr.utils.repr import NestedObject

__all__ = ["Decoder", "PositionalEncoding", "EncoderBlock"]

tf.config.run_functions_eagerly(True)


class PositionalEncoding(layers.Layer, NestedObject):
    """Compute positional encoding"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = layers.Dropout(rate=dropout)

        # Compute the positional encodings once in log space.
        pe = tf.Variable(tf.zeros((max_len, d_model)))
        position = tf.cast(
            tf.expand_dims(tf.experimental.numpy.arange(start=0, stop=max_len), axis=1), dtype=tf.float32
        )
        div_term = tf.math.exp(
            tf.cast(tf.experimental.numpy.arange(start=0, stop=d_model, step=2), dtype=tf.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe = pe.numpy()
        pe[:, 0::2] = tf.math.sin(position * div_term)
        pe[:, 1::2] = tf.math.cos(position * div_term)
        self.pe = tf.expand_dims(tf.convert_to_tensor(pe), axis=0)

    def call(
        self,
        x: tf.Tensor,
        **kwargs: Any,
    ) -> tf.Tensor:
        """
        Args:
            x: embeddings (batch, max_len, d_model)
            **kwargs: additional arguments

        Returns:
            positional embeddings (batch, max_len, d_model)
        """
        if x.dtype == tf.float16:  # amp fix: cast to half
            x = x + tf.cast(self.pe[:, : x.shape[1]], dtype=tf.half)
        else:
            x = x + self.pe[:, : x.shape[1]]
        return self.dropout(x, **kwargs)


@tf.function
def scaled_dot_product_attention(
    query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, mask: Optional[tf.Tensor] = None
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Scaled Dot-Product Attention"""

    scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) / math.sqrt(query.shape[-1])
    if mask is not None:
        # NOTE: to ensure the ONNX compatibility, tf.where works only with bool type condition
        scores = tf.where(mask == False, float("-inf"), scores)  # noqa: E712
    p_attn = tf.nn.softmax(scores, axis=-1)
    return tf.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(layers.Layer, NestedObject):
    """Position-wise Feed-Forward Network"""

    def __init__(
        self, d_model: int, ffd: int, dropout=0.1, activation_fct: Callable[[Any], Any] = layers.ReLU()
    ) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.activation_fct = activation_fct

        self.first_linear = layers.Dense(ffd, kernel_initializer=tf.initializers.he_uniform())
        self.sec_linear = layers.Dense(d_model, kernel_initializer=tf.initializers.he_uniform())
        self.dropout = layers.Dropout(rate=dropout)

    def call(self, x: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        x = self.first_linear(x, **kwargs)
        x = self.activation_fct(x)
        x = self.dropout(x, **kwargs)
        x = self.sec_linear(x, **kwargs)
        x = self.dropout(x, **kwargs)
        return x


class MultiHeadAttention(layers.Layer, NestedObject):
    """Multi-Head Attention"""

    def __init__(self, num_heads: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.linear_layers = [layers.Dense(d_model, kernel_initializer=tf.initializers.he_uniform()) for _ in range(3)]
        self.output_linear = layers.Dense(d_model, kernel_initializer=tf.initializers.he_uniform())

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: tf.Tensor = None,
        **kwargs: Any,
    ) -> tf.Tensor:
        batch_size = query.shape[0]

        # linear projections of Q, K, V
        query, key, value = [
            tf.transpose(
                tf.reshape(linear(x, **kwargs), shape=[batch_size, -1, self.num_heads, self.d_k]), perm=[0, 2, 1, 3]
            )
            for linear, x in zip(self.linear_layers, (query, key, value))
        ]

        # apply attention on all the projected vectors in batch
        x, attn = scaled_dot_product_attention(query, key, value, mask=mask)

        # Concat attention heads
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[batch_size, -1, self.num_heads * self.d_k])

        return self.output_linear(x, **kwargs)


class EncoderBlock(layers.Layer, NestedObject):
    """Transformer Encoder Block"""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        dff: int,  # hidden dimension of the feedforward network
        dropout: float,
        activation_fct: Callable[[Any], Any] = layers.ReLU(),
    ) -> None:
        super().__init__()

        self.num_layers = num_layers

        self.layer_norm_input = layers.LayerNormalization(epsilon=1e-5)
        self.layer_norm_attention = layers.LayerNormalization(epsilon=1e-5)
        self.layer_norm_output = layers.LayerNormalization(epsilon=1e-5)
        self.dropout = layers.Dropout(rate=dropout)

        self.attention = [MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)]
        self.position_feed_forward = [
            PositionwiseFeedForward(d_model, dff, dropout, activation_fct) for _ in range(self.num_layers)
        ]

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None, **kwargs: Any) -> tf.Tensor:
        output = x

        for i in range(self.num_layers):
            normed_output = self.layer_norm_input(output, **kwargs)
            output = output + self.dropout(
                self.attention[i](normed_output, normed_output, normed_output, mask, **kwargs),
                **kwargs,
            )
            normed_output = self.layer_norm_attention(output, **kwargs)
            output = output + self.dropout(self.position_feed_forward[i](normed_output, **kwargs), **kwargs)

        # (batch_size, seq_len, d_model)
        return self.layer_norm_output(output, **kwargs)


class Decoder(layers.Layer, NestedObject):
    """Transformer Decoder"""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        vocab_size: int,
        dropout: float = 0.2,
        dff: int = 2048,  # hidden dimension of the feedforward network
        maximum_position_encoding: int = 50,
    ) -> None:
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.layer_norm_input = layers.LayerNormalization(epsilon=1e-5)
        self.layer_norm_masked_attention = layers.LayerNormalization(epsilon=1e-5)
        self.layer_norm_attention = layers.LayerNormalization(epsilon=1e-5)
        self.layer_norm_output = layers.LayerNormalization(epsilon=1e-5)

        self.dropout = layers.Dropout(rate=dropout)
        self.embed = layers.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, maximum_position_encoding)

        self.attention = [MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)]
        self.source_attention = [MultiHeadAttention(num_heads, d_model, dropout) for _ in range(self.num_layers)]
        self.position_feed_forward = [PositionwiseFeedForward(d_model, dff, dropout) for _ in range(self.num_layers)]

    def call(
        self,
        tgt: tf.Tensor,
        memory: tf.Tensor,
        source_mask: Optional[tf.Tensor] = None,
        target_mask: Optional[tf.Tensor] = None,
        **kwargs: Any,
    ) -> tf.Tensor:
        tgt = self.embed(tgt, **kwargs) * math.sqrt(self.d_model)
        pos_enc_tgt = self.positional_encoding(tgt, **kwargs)
        output = pos_enc_tgt

        for i in range(self.num_layers):
            normed_output = self.layer_norm_input(output, **kwargs)
            output = output + self.dropout(
                self.attention[i](normed_output, normed_output, normed_output, target_mask, **kwargs),
                **kwargs,
            )
            normed_output = self.layer_norm_masked_attention(output, **kwargs)
            output = output + self.dropout(
                self.source_attention[i](normed_output, memory, memory, source_mask, **kwargs),
                **kwargs,
            )
            normed_output = self.layer_norm_attention(output, **kwargs)
            output = output + self.dropout(self.position_feed_forward[i](normed_output, **kwargs), **kwargs)

        # (batch_size, seq_len, d_model)
        return self.layer_norm_output(output, **kwargs)
