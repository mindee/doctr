# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any

import tensorflow as tf
from tensorflow.keras import Model, layers

from doctr.datasets import VOCABS
from doctr.models.classification import magc_resnet31
from doctr.models.modules.transformer import Decoder, PositionalEncoding

from ...utils.tensorflow import _bf16_to_float32, _build_model, load_pretrained_params
from .base import _MASTER, _MASTERPostProcessor

__all__ = ["MASTER", "master"]


default_cfgs: dict[str, dict[str, Any]] = {
    "master": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (32, 128, 3),
        "vocab": VOCABS["french"],
        "url": "https://doctr-static.mindee.com/models?id=v0.9.0/master-d7fdaeff.weights.h5&src=0",
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
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    def __init__(
        self,
        feature_extractor: Model,
        vocab: str,
        d_model: int = 512,
        dff: int = 2048,
        num_heads: int = 8,  # number of heads in the transformer decoder
        num_layers: int = 3,
        max_length: int = 50,
        dropout: float = 0.2,
        input_shape: tuple[int, int, int] = (32, 128, 3),  # different from the paper
        exportable: bool = False,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.exportable = exportable
        self.max_length = max_length
        self.d_model = d_model
        self.vocab = vocab
        self.cfg = cfg
        self.vocab_size = len(vocab)

        self.feat_extractor = feature_extractor
        self.positional_encoding = PositionalEncoding(self.d_model, dropout, max_len=input_shape[0] * input_shape[1])

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=self.d_model,
            num_heads=num_heads,
            vocab_size=self.vocab_size + 3,  # EOS, SOS, PAD
            dff=dff,
            dropout=dropout,
            maximum_position_encoding=self.max_length,
        )

        self.linear = layers.Dense(self.vocab_size + 3, kernel_initializer=tf.initializers.he_uniform())
        self.postprocessor = MASTERPostProcessor(vocab=self.vocab)

    @tf.function
    def make_source_and_target_mask(self, source: tf.Tensor, target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # [1, 1, 1, ..., 0, 0, 0] -> 0 is masked
        # (N, 1, 1, max_length)
        target_pad_mask = tf.cast(tf.math.not_equal(target, self.vocab_size + 2), dtype=tf.uint8)
        target_pad_mask = target_pad_mask[:, tf.newaxis, tf.newaxis, :]
        target_length = target.shape[1]
        # sub mask filled diagonal with 1 = see 0 = masked (max_length, max_length)
        target_sub_mask = tf.linalg.band_part(tf.ones((target_length, target_length)), -1, 0)
        # source mask filled with ones (max_length, positional_encoded_seq_len)
        source_mask = tf.ones((target_length, source.shape[1]))
        # combine the two masks into one boolean mask where False is masked (N, 1, max_length, max_length)
        target_mask = tf.math.logical_and(
            tf.cast(target_sub_mask, dtype=tf.bool), tf.cast(target_pad_mask, dtype=tf.bool)
        )
        return source_mask, target_mask

    @staticmethod
    def compute_loss(
        model_output: tf.Tensor,
        gt: tf.Tensor,
        seq_len: list[int],
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
        target: list[str] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Call function for training

        Args:
            x: images
            target: list of str labels
            return_model_output: if True, return logits
            return_preds: if True, decode logits
            **kwargs: keyword arguments passed to the decoder

        Returns:
            A dictionnary containing eventually loss, logits and predictions.
        """
        # Encode
        feature = self.feat_extractor(x, **kwargs)
        b, h, w, c = feature.get_shape()
        # (N, H, W, C) --> (N, H * W, C)
        feature = tf.reshape(feature, shape=(b, h * w, c))
        # add positional encoding to features
        encoded = self.positional_encoding(feature, **kwargs)

        out: dict[str, tf.Tensor] = {}

        if kwargs.get("training", False) and target is None:
            raise ValueError("Need to provide labels during training")

        if target is not None:
            # Compute target: tensor of gts and sequence lengths
            gt, seq_len = self.build_target(target)
            # Compute decoder masks
            source_mask, target_mask = self.make_source_and_target_mask(encoded, gt)
            # Compute logits
            output = self.decoder(gt, encoded, source_mask, target_mask, **kwargs)
            logits = self.linear(output, **kwargs)
        else:
            logits = self.decode(encoded, **kwargs)

        logits = _bf16_to_float32(logits)

        if self.exportable:
            out["logits"] = logits
            return out

        if target is not None:
            out["loss"] = self.compute_loss(logits, gt, seq_len)

        if return_model_output:
            out["out_map"] = logits

        if return_preds:
            out["preds"] = self.postprocessor(logits)

        return out

    @tf.function
    def decode(self, encoded: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        """Decode function for prediction

        Args:
            encoded: encoded features
            **kwargs: keyword arguments passed to the decoder

        Returns:
            A tuple of tf.Tensor: predictions, logits
        """
        b = encoded.shape[0]

        start_symbol = tf.constant(self.vocab_size + 1, dtype=tf.int32)  # SOS
        padding_symbol = tf.constant(self.vocab_size + 2, dtype=tf.int32)  # PAD

        ys = tf.fill(dims=(b, self.max_length - 1), value=padding_symbol)
        start_vector = tf.fill(dims=(b, 1), value=start_symbol)
        ys = tf.concat([start_vector, ys], axis=-1)

        # Final dimension include EOS/SOS/PAD
        for i in range(self.max_length - 1):
            source_mask, target_mask = self.make_source_and_target_mask(encoded, ys)
            output = self.decoder(ys, encoded, source_mask, target_mask, **kwargs)
            logits = self.linear(output, **kwargs)
            prob = tf.nn.softmax(logits, axis=-1)
            next_token = tf.argmax(prob, axis=-1, output_type=ys.dtype)
            # update ys with the next token and ignore the first token (SOS)
            i_mesh, j_mesh = tf.meshgrid(tf.range(b), tf.range(self.max_length), indexing="ij")
            indices = tf.stack([i_mesh[:, i + 1], j_mesh[:, i + 1]], axis=1)

            ys = tf.tensor_scatter_nd_update(ys, indices, next_token[:, i])

        # Shape (N, max_length, vocab_size + 1)
        return logits


class MASTERPostProcessor(_MASTERPostProcessor):
    """Post processor for MASTER architectures

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __call__(
        self,
        logits: tf.Tensor,
    ) -> list[tuple[str, float]]:
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

        return list(zip(word_values, probs.numpy().clip(0, 1).tolist()))


def _master(arch: str, pretrained: bool, backbone_fn, pretrained_backbone: bool = True, **kwargs: Any) -> MASTER:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["input_shape"] = kwargs.get("input_shape", _cfg["input_shape"])
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])

    kwargs["vocab"] = _cfg["vocab"]
    kwargs["input_shape"] = _cfg["input_shape"]

    # Build the model
    model = MASTER(
        backbone_fn(pretrained=pretrained_backbone, input_shape=_cfg["input_shape"], include_top=False),
        cfg=_cfg,
        **kwargs,
    )
    _build_model(model)

    # Load pretrained parameters
    if pretrained:
        # The given vocab differs from the pretrained model => skip the mismatching layers for fine tuning
        load_pretrained_params(
            model, default_cfgs[arch]["url"], skip_mismatch=kwargs["vocab"] != default_cfgs[arch]["vocab"]
        )

    return model


def master(pretrained: bool = False, **kwargs: Any) -> MASTER:
    """MASTER as described in paper: <https://arxiv.org/pdf/1910.02562.pdf>`_.

    >>> import tensorflow as tf
    >>> from doctr.models import master
    >>> model = master(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 32, 128, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
        **kwargs: keywoard arguments passed to the MASTER architecture

    Returns:
        text recognition architecture
    """
    return _master("master", pretrained, magc_resnet31, **kwargs)
