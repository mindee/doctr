# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math
from copy import deepcopy
from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

from doctr.datasets import VOCABS
from doctr.models.modules.transformer import MultiHeadAttention, PositionwiseFeedForward

from ...classification import vit_s
from ...utils.tensorflow import load_pretrained_params
from .base import _PARSeq, _PARSeqPostProcessor

__all__ = ["PARSeq", "parseq"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "parseq": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (32, 128, 3),
        "vocab": VOCABS["french"],
        "url": None,
    },
}


class CharEmbedding(layers.Layer):
    """Implements the character embedding module

    Args:
        vocab_size: size of the vocabulary
        d_model: dimension of the model
    """

    def __init__(self, vocab_size: int, d_model: int):
        super(CharEmbedding, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return math.sqrt(self.d_model) * self.embedding(x)


class PARSeqDecoder(layers.Layer):
    """Implements decoder module of the PARSeq model

    Args:
        d_model: dimension of the model
        num_heads: number of attention heads
        ffd: dimension of the feed forward layer
        ffd_ratio: depth multiplier for the feed forward layer
        dropout: dropout rate
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 12,
        ffd: int = 2048,
        ffd_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super(PARSeqDecoder, self).__init__()
        self.attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.cross_attention = MultiHeadAttention(num_heads, d_model, dropout=dropout)
        self.position_feed_forward = PositionwiseFeedForward(
            d_model, ffd * ffd_ratio, dropout, layers.Activation(tf.nn.gelu)
        )

        self.attention_norm = layers.LayerNormalization(epsilon=1e-5)
        self.cross_attention_norm = layers.LayerNormalization(epsilon=1e-5)
        self.query_norm = layers.LayerNormalization(epsilon=1e-5)
        self.content_norm = layers.LayerNormalization(epsilon=1e-5)
        self.feed_forward_norm = layers.LayerNormalization(epsilon=1e-5)
        self.output_norm = layers.LayerNormalization(epsilon=1e-5)
        self.attention_dropout = layers.Dropout(dropout)
        self.cross_attention_dropout = layers.Dropout(dropout)
        self.feed_forward_dropout = layers.Dropout(dropout)

    def call(
        self,
        target,
        content,
        memory,
        target_mask=None,
        **kwargs: Any,
    ):
        query_norm = self.query_norm(target, **kwargs)
        content_norm = self.content_norm(content, **kwargs)
        target = target + self.attention_dropout(
            self.attention(query_norm, content_norm, content_norm, mask=target_mask, **kwargs), **kwargs
        )
        target = target + self.cross_attention_dropout(
            self.cross_attention(self.query_norm(target, **kwargs), memory, memory, **kwargs), **kwargs
        )
        target = target + self.feed_forward_dropout(
            self.position_feed_forward(self.feed_forward_norm(target, **kwargs), **kwargs), **kwargs
        )
        return self.output_norm(target, **kwargs)


class PARSeq(_PARSeq, Model):
    """Implements a PARSeq architecture as described in `"Scene Text Recognition
    with Permuted Autoregressive Sequence Models" <https://arxiv.org/pdf/2207.06966>`_.
    Modified implementation based on the official Pytorch implementation: <https://github.com/baudm/parseq/tree/main`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        embedding_units: number of embedding units
        max_length: maximum word length handled by the model
        dropout_prob: dropout probability for the decoder
        dec_num_heads: number of attention heads in the decoder
        dec_ff_dim: dimension of the feed forward layer in the decoder
        dec_ffd_ratio: depth multiplier for the feed forward layer in the decoder
        input_shape: input shape of the image
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    _children_names: List[str] = ["feat_extractor", "postprocessor"]

    def __init__(
        self,
        feature_extractor,
        vocab: str,
        embedding_units: int,
        max_length: int = 32,  # different from paper
        dropout_prob: float = 0.1,
        dec_num_heads: int = 12,
        dec_ff_dim: int = 2048,
        dec_ffd_ratio: int = 4,
        input_shape: Tuple[int, int, int] = (32, 128, 3),
        exportable: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length
        self.vocab_size = len(vocab)
        self.rng = np.random.default_rng()

        self.feat_extractor = feature_extractor
        self.decoder = PARSeqDecoder(embedding_units, dec_num_heads, dec_ff_dim, dec_ffd_ratio, dropout_prob)
        self.embed = CharEmbedding(self.vocab_size + 3, embedding_units)  # +3 for SOS, EOS, PAD
        self.head = layers.Dense(self.vocab_size + 1, name="head")  # +1 for EOS
        self.pos_queries = self.add_weight(
            shape=(1, self.max_length + 1, embedding_units),
            initializer="zeros",
            trainable=True,
            name="positions",
        )
        self.dropout = layers.Dropout(dropout_prob)

        self.postprocessor = PARSeqPostProcessor(vocab=self.vocab)

    @tf.function
    def generate_permutations(self, seqlen: tf.Tensor) -> tf.Tensor:
        # Generates permutations of the target sequence.
        # Translated from https://github.com/baudm/parseq/blob/main/strhub/models/parseq/system.py
        # with small modifications

        max_num_chars = int(tf.reduce_max(seqlen))  # get longest sequence length in batch
        perms = [tf.range(max_num_chars, dtype=tf.int32)]

        max_perms = math.factorial(max_num_chars) // 2
        num_gen_perms = min(3, max_perms)
        if max_num_chars < 5:
            # Pool of permutations to sample from. We only need the first half (if complementary option is selected)
            # Special handling for max_num_chars == 4 which correctly divides the pool into the flipped halves
            if max_num_chars == 4:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool_candidates = list(permutations(range(max_num_chars), max_num_chars))
            perm_pool = tf.convert_to_tensor([perm_pool_candidates[i] for i in selector])
            # If the forward permutation is always selected, no need to add it to the pool for sampling
            perm_pool = perm_pool[1:]
            final_perms = tf.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(final_perms), replace=False)
                final_perms = tf.concat([final_perms, perm_pool[i[0] : i[1]]], axis=0)
        else:
            perms.extend(
                [tf.random.shuffle(tf.range(max_num_chars, dtype=tf.int32)) for _ in range(num_gen_perms - len(perms))]
            )
            final_perms = tf.stack(perms)

        comp = tf.reverse(final_perms, axis=[-1])
        final_perms = tf.stack([final_perms, comp])
        final_perms = tf.transpose(final_perms, perm=[1, 0, 2])
        final_perms = tf.reshape(final_perms, shape=(-1, max_num_chars))

        sos_idx = tf.zeros([tf.shape(final_perms)[0], 1], dtype=tf.int32)
        eos_idx = tf.fill([tf.shape(final_perms)[0], 1], max_num_chars + 1)
        combined = tf.concat([sos_idx, final_perms + 1, eos_idx], axis=1)
        combined = tf.cast(combined, dtype=tf.int32)
        if tf.shape(combined)[0] > 1:
            combined = tf.tensor_scatter_nd_update(
                combined, [[1, i] for i in range(1, max_num_chars + 2)], max_num_chars + 1 - tf.range(max_num_chars + 1)
            )
        # we pad to max length with eos idx to fit the mask generation
        return tf.pad(
            combined, [[0, 0], [0, self.max_length + 1 - tf.shape(combined)[1]]], constant_values=max_num_chars + 2
        )  # (num_perms, self.max_length + 1)

    @tf.function
    def generate_permutations_attention_masks(self, permutation: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Generate source and target mask for the decoder attention.
        sz = permutation.shape[0]
        mask = tf.ones((sz, sz), dtype=tf.float32)

        for i in range(sz - 1):
            query_idx = int(permutation[i])
            masked_keys = permutation[i + 1 :].numpy().tolist()
            indices = tf.constant([[query_idx, j] for j in masked_keys], dtype=tf.int32)
            mask = tf.tensor_scatter_nd_update(mask, indices, tf.zeros(len(masked_keys), dtype=tf.float32))

        source_mask = tf.identity(mask[:-1, :-1])
        eye_indices = tf.eye(sz, dtype=tf.bool)
        mask = tf.tensor_scatter_nd_update(
            mask, tf.where(eye_indices), tf.zeros_like(tf.boolean_mask(mask, eye_indices))
        )
        target_mask = mask[1:, :-1]

        return tf.cast(source_mask, dtype=tf.bool), tf.cast(target_mask, dtype=tf.bool)

    @tf.function
    def decode(
        self,
        target: tf.Tensor,
        memory: tf,
        target_mask: Optional[tf.Tensor] = None,
        target_query: Optional[tf.Tensor] = None,
        **kwargs: Any,
    ) -> tf.Tensor:
        batch_size, sequence_length = target.shape
        # apply positional information to the target sequence excluding the SOS token
        null_ctx = self.embed(target[:, :1])
        content = self.pos_queries[:, : sequence_length - 1] + self.embed(target[:, 1:])
        content = self.dropout(tf.concat([null_ctx, content], axis=1), **kwargs)
        if target_query is None:
            target_query = tf.tile(self.pos_queries[:, :sequence_length], [batch_size, 1, 1])
        target_query = self.dropout(target_query, **kwargs)
        return self.decoder(target_query, content, memory, target_mask, **kwargs)

    @tf.function
    def decode_autoregressive(self, features: tf.Tensor) -> tf.Tensor:
        """Generate predictions for the given features."""
        # Padding symbol + SOS at the beginning
        b = tf.shape(features)[0]
        ys = tf.fill(dims=(b, self.max_length), value=self.vocab_size + 2)
        start_vector = tf.fill(dims=(b, 1), value=self.vocab_size + 1)
        ys = tf.concat([start_vector, ys], axis=-1)
        pos_queries = tf.tile(self.pos_queries[:, : self.max_length + 1], [b, 1, 1])
        query_mask = tf.cast(
            tf.linalg.band_part(tf.ones((self.max_length + 1, self.max_length + 1)), -1, 0), dtype=tf.bool
        )

        pos_logits = []
        for i in range(self.max_length):
            # Decode one token at a time without providing information about the future tokens
            tgt_out = self.decode(
                ys[:, : i + 1],
                features,
                query_mask[i : i + 1, : i + 1],
                target_query=pos_queries[:, i : i + 1],
            )
            pos_prob = self.head(tgt_out)
            pos_logits.append(pos_prob)

            if i + 1 < self.max_length:
                # update ys with the next token
                i_mesh, j_mesh = tf.meshgrid(tf.range(b), tf.range(self.max_length), indexing="ij")
                indices = tf.stack([i_mesh[:, i + 1], j_mesh[:, i + 1]], axis=1)
                ys = tf.tensor_scatter_nd_update(
                    ys, indices, tf.cast(tf.argmax(pos_prob[:, -1, :], axis=-1), dtype=tf.int32)
                )

                # Stop decoding if all sequences have reached the EOS token
                # We need to check it on True to be compatible with ONNX
                if tf.reduce_any(tf.reduce_all(tf.equal(ys, tf.constant(self.vocab_size)), axis=-1)) is True:
                    break

        logits = tf.concat(pos_logits, axis=1)  # (N, max_length, vocab_size + 1)

        # One refine iteration
        # Update query mask
        query_mask = tf.cast(1 - tf.linalg.diag(tf.ones(self.max_length, dtype=tf.int32), k=-1), dtype=tf.bool)

        sos = tf.fill((tf.shape(features)[0], 1), self.vocab_size + 1)
        ys = tf.concat([sos, tf.cast(tf.argmax(logits[:, :-1], axis=-1), dtype=tf.int32)], axis=1)
        # Create padding mask for refined target input maskes all behind EOS token as False
        # (N, 1, 1, max_length)
        target_pad_mask = tf.cumsum(tf.cast(tf.equal(ys, self.vocab_size), dtype=tf.int32), axis=1, reverse=False)
        target_pad_mask = tf.logical_not(tf.cast(target_pad_mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.bool))
        mask = tf.math.logical_and(target_pad_mask, query_mask[:, : ys.shape[1]])
        logits = self.head(self.decode(ys, features, mask, target_query=pos_queries))

        return logits  # (N, max_length, vocab_size + 1)

    @tf.function
    def decode_non_autoregressive(self, features: tf.Tensor) -> tf.Tensor:
        """Decode the given features at once"""
        pos_queries = tf.tile(self.pos_queries[:, : self.max_length + 1], [tf.shape(features)[0], 1, 1])
        ys = tf.fill((tf.shape(features)[0], 1), self.vocab_size + 1)
        return self.head(self.decode(ys, features, target_query=pos_queries))[:, : self.max_length]

    @staticmethod
    def compute_loss(
        model_output: tf.Tensor,
        gt: tf.Tensor,
        seq_len: List[int],
    ) -> tf.Tensor:
        """Compute categorical cross-entropy loss for the model.
        Sequences are masked after the EOS character.

        Args:
            model_output: predicted logits of the model
            gt: the encoded tensor with gt labels
            seq_len: lengths of each gt word inside the batch

        Returns:
            The loss of the model on the batch
        """
        # Input length : number of steps
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
        features = self.feat_extractor(x, **kwargs)  # (batch_size, patches_seqlen, d_model)
        # remove cls token
        features = features[:, 1:, :]

        if target is not None:
            gt, seq_len = self.build_target(target)
            seq_len = tf.cast(seq_len, tf.int32)

        if kwargs.get("training", False) and target is None:
            raise ValueError("Need to provide labels during training")

        if target is not None:
            gt, seq_len = self.build_target(target)
            seq_len = tf.cast(seq_len, tf.int32)

            if kwargs.get("training", False):
                # Generate permutations of the target sequences
                tgt_perms = self.generate_permutations(seq_len)

                # Create padding mask for target input
                # [True, True, True, ..., False, False, False] -> False is masked
                padding_mask = ((gt != self.vocab_size + 2) | (gt != self.vocab_size))[:, tf.newaxis, tf.newaxis, :]

                for perm in tgt_perms:
                    # Generate attention masks for the permutations
                    _, target_mask = self.generate_permutations_attention_masks(perm)
                    # combine target padding mask and query mask
                    mask = tf.math.logical_and(target_mask, padding_mask)
                    logits = self.head(self.decode(gt, features, mask))
            else:
                # eval step - use non-autoregressive decoding while training evaluation
                logits = self.decode_non_autoregressive(features)
        else:
            logits = self.decode_autoregressive(features)

        out: Dict[str, tf.Tensor] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output:
            out["out_map"] = logits

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = self.postprocessor(logits)

        if target is not None:
            out["loss"] = self.compute_loss(logits, gt, seq_len)

        return out


class PARSeqPostProcessor(_PARSeqPostProcessor):
    """Post processor for PARSeq architecture

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


def _parseq(
    arch: str,
    pretrained: bool,
    backbone_fn,
    input_shape: Optional[Tuple[int, int, int]] = None,
    **kwargs: Any,
) -> PARSeq:
    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["input_shape"] = input_shape or _cfg["input_shape"]
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])
    patch_size = kwargs.get("patch_size", (4, 8))

    kwargs["vocab"] = _cfg["vocab"]

    # Feature extractor
    feat_extractor = backbone_fn(
        # NOTE: we don't use a pretrained backbone for non-rectangular patches to avoid the pos embed mismatch
        pretrained=False,
        input_shape=_cfg["input_shape"],
        patch_size=patch_size,
        include_top=False,
    )

    kwargs.pop("patch_size", None)
    kwargs.pop("pretrained_backbone", None)

    # Build the model
    model = PARSeq(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"])

    return model


def parseq(pretrained: bool = False, **kwargs: Any) -> PARSeq:
    """PARSeq architecture from
    `"Scene Text Recognition with Permuted Autoregressive Sequence Models" <https://arxiv.org/pdf/2207.06966>`_.

    >>> import tensorflow as tf
    >>> from doctr.models import parseq
    >>> model = parseq(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 32, 128, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _parseq(
        "parseq",
        pretrained,
        vit_s,
        embedding_units=384,
        patch_size=(4, 8),
        **kwargs,
    )
