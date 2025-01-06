# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any

import tensorflow as tf
from tensorflow.keras import Model, layers

from doctr.datasets import VOCABS

from ...classification import vit_b, vit_s
from ...utils.tensorflow import _bf16_to_float32, _build_model, load_pretrained_params
from .base import _ViTSTR, _ViTSTRPostProcessor

__all__ = ["ViTSTR", "vitstr_small", "vitstr_base"]

default_cfgs: dict[str, dict[str, Any]] = {
    "vitstr_small": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (32, 128, 3),
        "vocab": VOCABS["french"],
        "url": "https://doctr-static.mindee.com/models?id=v0.9.0/vitstr_small-d28b8d92.weights.h5&src=0",
    },
    "vitstr_base": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (32, 128, 3),
        "vocab": VOCABS["french"],
        "url": "https://doctr-static.mindee.com/models?id=v0.9.0/vitstr_base-9ad6eb84.weights.h5&src=0",
    },
}


class ViTSTR(_ViTSTR, Model):
    """Implements a ViTSTR architecture as described in `"Vision Transformer for Fast and
    Efficient Scene Text Recognition" <https://arxiv.org/pdf/2105.08582.pdf>`_.

    Args:
        feature_extractor: the backbone serving as feature extractor
        vocab: vocabulary used for encoding
        embedding_units: number of embedding units
        max_length: maximum word length handled by the model
        dropout_prob: dropout probability for the encoder and decoder
        input_shape: input shape of the image
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    _children_names: list[str] = ["feat_extractor", "postprocessor"]

    def __init__(
        self,
        feature_extractor,
        vocab: str,
        embedding_units: int,
        max_length: int = 32,
        dropout_prob: float = 0.0,
        input_shape: tuple[int, int, int] = (32, 128, 3),  # different from paper
        exportable: bool = False,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length + 2  # +2 for SOS and EOS

        self.feat_extractor = feature_extractor
        self.head = layers.Dense(len(self.vocab) + 1, name="head")  # +1 for EOS

        self.postprocessor = ViTSTRPostProcessor(vocab=self.vocab)

    @staticmethod
    def compute_loss(
        model_output: tf.Tensor,
        gt: tf.Tensor,
        seq_len: list[int],
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
        # The "masked" first gt char is <sos>.
        cce = tf.nn.softmax_cross_entropy_with_logits(oh_gt[:, 1:, :], model_output)
        # Compute mask
        mask_values = tf.zeros_like(cce)
        mask_2d = tf.sequence_mask(seq_len, input_len)
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
        features = self.feat_extractor(x, **kwargs)  # (batch_size, patches_seqlen, d_model)

        if target is not None:
            gt, seq_len = self.build_target(target)
            seq_len = tf.cast(seq_len, tf.int32)

        if kwargs.get("training", False) and target is None:
            raise ValueError("Need to provide labels during training")

        features = features[:, : self.max_length]  # (batch_size, max_length, d_model)
        B, N, E = features.shape
        features = tf.reshape(features, (B * N, E))
        logits = tf.reshape(
            self.head(features, **kwargs), (B, N, len(self.vocab) + 1)
        )  # (batch_size, max_length, vocab + 1)
        decoded_features = _bf16_to_float32(logits[:, 1:])  # remove cls_token

        out: dict[str, tf.Tensor] = {}
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


class ViTSTRPostProcessor(_ViTSTRPostProcessor):
    """Post processor for ViTSTR architecture

    Args:
        vocab: string containing the ordered sequence of supported characters
    """

    def __call__(
        self,
        logits: tf.Tensor,
    ) -> list[tuple[str, float]]:
        # compute pred with argmax for attention models
        out_idxs = tf.math.argmax(logits, axis=2)
        preds_prob = tf.math.reduce_max(tf.nn.softmax(logits, axis=-1), axis=-1)

        # decode raw output of the model with tf_label_to_idx
        out_idxs = tf.cast(out_idxs, dtype="int32")
        embedding = tf.constant(self._embedding, dtype=tf.string)
        decoded_strings_pred = tf.strings.reduce_join(inputs=tf.nn.embedding_lookup(embedding, out_idxs), axis=-1)
        decoded_strings_pred = tf.strings.split(decoded_strings_pred, "<eos>")
        decoded_strings_pred = tf.sparse.to_dense(decoded_strings_pred.to_sparse(), default_value="not valid")[:, 0]
        word_values = [word.decode() for word in decoded_strings_pred.numpy().tolist()]

        # compute probabilties for each word up to the EOS token
        probs = [
            preds_prob[i, : len(word)].numpy().clip(0, 1).mean().item() if word else 0.0
            for i, word in enumerate(word_values)
        ]

        return list(zip(word_values, probs))


def _vitstr(
    arch: str,
    pretrained: bool,
    backbone_fn,
    input_shape: tuple[int, int, int] | None = None,
    **kwargs: Any,
) -> ViTSTR:
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
    model = ViTSTR(feat_extractor, cfg=_cfg, **kwargs)
    _build_model(model)

    # Load pretrained parameters
    if pretrained:
        # The given vocab differs from the pretrained model => skip the mismatching layers for fine tuning
        load_pretrained_params(
            model, default_cfgs[arch]["url"], skip_mismatch=kwargs["vocab"] != default_cfgs[arch]["vocab"]
        )

    return model


def vitstr_small(pretrained: bool = False, **kwargs: Any) -> ViTSTR:
    """ViTSTR-Small as described in `"Vision Transformer for Fast and Efficient Scene Text Recognition"
    <https://arxiv.org/pdf/2105.08582.pdf>`_.

    >>> import tensorflow as tf
    >>> from doctr.models import vitstr_small
    >>> model = vitstr_small(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 32, 128, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
        **kwargs: keyword arguments of the ViTSTR architecture

    Returns:
        text recognition architecture
    """
    return _vitstr(
        "vitstr_small",
        pretrained,
        vit_s,
        embedding_units=384,
        patch_size=(4, 8),
        **kwargs,
    )


def vitstr_base(pretrained: bool = False, **kwargs: Any) -> ViTSTR:
    """ViTSTR-Base as described in `"Vision Transformer for Fast and Efficient Scene Text Recognition"
    <https://arxiv.org/pdf/2105.08582.pdf>`_.

    >>> import tensorflow as tf
    >>> from doctr.models import vitstr_base
    >>> model = vitstr_base(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 32, 128, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset
        **kwargs: keyword arguments of the ViTSTR architecture

    Returns:
        text recognition architecture
    """
    return _vitstr(
        "vitstr_base",
        pretrained,
        vit_b,
        embedding_units=768,
        patch_size=(4, 8),
        **kwargs,
    )
