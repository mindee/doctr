# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# This module 'transformer.py' is 100% inspired from this Tensorflow tutorial:
# https://www.tensorflow.org/text/tutorials/transformer


from typing import Tuple, Any

import tensorflow as tf
import numpy as np


__all__ = ['Decoder', 'positional_encoding', 'create_look_ahead_mask', 'create_padding_mask']


def get_angles(pos: np.array, i: np.array, d_model: int = 512) -> np.array:
    """This function compute the 2D array of angles for sinusoidal positional encoding.

    Args:
        pos: range of positions to encode
        i: range of depth to encode positions
        d_model: depth parameter of the model

    Returns:
        2D array of angles, len(pos) x len(i)
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position: int, d_model: int = 512, dtype=tf.float32) -> tf.Tensor:
    """This function computes the 2D positional encoding of the position, on a depth d_model

    Args:
        position: Number of positions to encode
        d_model: depth of the encoding

    Returns:
        2D positional encoding as described in Transformer paper.
    """
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model,
    )
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=dtype)


@tf.function
def create_padding_mask(seq: tf.Tensor, padding: int = 0, dtype=tf.float32) -> tf.Tensor:
    seq = tf.cast(tf.math.equal(seq, padding), dtype)
    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


@tf.function
def create_look_ahead_mask(size: int) -> tf.Tensor:
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


@tf.function
def scaled_dot_product_attention(
    q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:

    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], q.dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model: int = 512, num_heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, kernel_initializer=tf.initializers.he_uniform())
        self.wk = tf.keras.layers.Dense(d_model, kernel_initializer=tf.initializers.he_uniform())
        self.wv = tf.keras.layers.Dense(d_model, kernel_initializer=tf.initializers.he_uniform())

        self.dense = tf.keras.layers.Dense(d_model, kernel_initializer=tf.initializers.he_uniform())

    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self,
        v: tf.Tensor,
        k: tf.Tensor,
        q: tf.Tensor,
        mask: tf.Tensor,
        **kwargs: Any,
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        batch_size = tf.shape(q)[0]

        q = self.wq(q, **kwargs)  # (batch_size, seq_len, d_model)
        k = self.wk(k, **kwargs)  # (batch_size, seq_len, d_model)
        v = self.wv(v, **kwargs)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention, **kwargs)  # (batch_size, seq_len_q, d_model)

        return output


def point_wise_feed_forward_network(d_model: int = 512, dff: int = 2048) -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            dff, activation='relu', kernel_initializer=tf.initializers.he_uniform()
        ),  # (batch, seq_len, dff)
        tf.keras.layers.Dense(d_model, kernel_initializer=tf.initializers.he_uniform())  # (batch, seq_len, d_model)
    ])


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dff: int = 2048,
        dropout: float = 0.2,
    ) -> None:
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(
        self,
        x: tf.Tensor,
        enc_output: tf.Tensor,
        look_ahead_mask: tf.Tensor,
        padding_mask: tf.Tensor,
        **kwargs: Any,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1 = self.mha1(x, x, x, look_ahead_mask, **kwargs)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, **kwargs)
        out1 = self.layernorm1(attn1 + x, **kwargs)

        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask, **kwargs)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, **kwargs)
        out2 = self.layernorm2(attn2 + out1, **kwargs)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2, **kwargs)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, **kwargs)
        out3 = self.layernorm3(ffn_output + out2, **kwargs)  # (batch_size, target_seq_len, d_model)

        return out3


class Decoder(tf.keras.layers.Layer):

    def __init__(
        self,
        num_layers: int = 3,
        d_model: int = 512,
        num_heads: int = 8,
        dff: int = 2048,
        vocab_size: int = 120,
        maximum_position_encoding: int = 50,
        dropout: float = 0.2,
    ) -> None:
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(vocab_size + 3, d_model)  # 3 more classes EOS/SOS/PAD
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(
        self,
        x: tf.Tensor,
        enc_output: tf.Tensor,
        look_ahead_mask: tf.Tensor,
        padding_mask: tf.Tensor,
        **kwargs: Any,
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        seq_len = tf.shape(x)[1]

        x = self.embedding(x, **kwargs)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, x.dtype))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, **kwargs)

        for i in range(self.num_layers):
            x = self.dec_layers[i](
                x, enc_output, look_ahead_mask, padding_mask, **kwargs
            )

        # x.shape == (batch_size, target_seq_len, d_model)
        return x
