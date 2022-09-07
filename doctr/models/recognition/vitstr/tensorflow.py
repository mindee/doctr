# Copyright (C) 2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers

from doctr.datasets import VOCABS

from ...modules import VisionTransformer
from ...utils.tensorflow import load_pretrained_params
from ..core import RecognitionModel, RecognitionPostProcessor

__all__ = ["ViTSTR", "vitstr"]

default_cfgs: Dict[str, Dict[str, Any]] = {
    "vitstr": {
        "mean": (0.694, 0.695, 0.693),
        "std": (0.299, 0.296, 0.301),
        "input_shape": (32, 128, 3),
        "vocab": VOCABS["french"],
        "url": None,
    },
}


class ViTSTR(Model, RecognitionModel):
    """Implements a ViTSTR architecture as described in `"Vision Transformer for Fast and
    Efficient Scene Text Recognition" <https://arxiv.org/pdf/2105.08582.pdf>`_.

    Args:
        vocab: vocabulary used for encoding
        embedding_units: number of embedding units
        max_length: maximum word length handled by the model
        dropout_prob: dropout probability for the encoder and decoder
        input_shape: input shape of the image
        patch_size: size of the patches
        exportable: onnx exportable returns only logits
        cfg: dictionary containing information about the model
    """

    _children_names: List[str] = ["feat_extractor", "postprocessor"]

    def __init__(
        self,
        vocab: str,
        embedding_units: int = 384,
        max_length: int = 25,
        dropout_prob: float = 0.0,
        input_shape: Tuple[int, int, int] = (32, 128, 3),  # different from paper
        patch_size: Tuple[int, int] = (4, 8),  # different from paper to match our size
        exportable: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__()
        self.vocab = vocab
        self.exportable = exportable
        self.cfg = cfg
        self.max_length = max_length + 2  # Add 1 timestep for EOS after the longest word and 1 for ViT cls token

        self.feat_extractor = VisionTransformer(
            img_size=input_shape[:-1],
            patch_size=patch_size,
            d_model=embedding_units,
            num_layers=12,
            num_heads=6,
            dropout=dropout_prob,
        )
        self.head = layers.Dense(len(self.vocab))

        self.postprocessor = ViTSTRPostProcessor(vocab=self.vocab)

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

        if target is not None:
            gt, seq_len = self.build_target(target)
            seq_len = tf.cast(seq_len, tf.int32)

        if kwargs.get("training", False) and target is None:
            raise ValueError("Need to provide labels during training")

        features = features[:, : self.max_length]
        # batch, seqlen, embedding_size
        B, N, E = features.shape
        features = tf.reshape(features, (B * N, E))
        logits = tf.reshape(self.head(features), (B, N, len(self.vocab)))  # (batch, seqlen, vocab)
        decoded_features = logits[:, 1:]  # remove cls_token

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


class ViTSTRPostProcessor(RecognitionPostProcessor):
    """Post processor for ViTSTR architecture

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


def _vitstr(
    arch: str,
    pretrained: bool,
    input_shape: Optional[Tuple[int, int, int]] = None,
    **kwargs: Any,
) -> ViTSTR:

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["input_shape"] = input_shape or _cfg["input_shape"]
    _cfg["vocab"] = kwargs.get("vocab", _cfg["vocab"])

    kwargs["vocab"] = _cfg["vocab"]

    # Build the model
    model = ViTSTR(cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, default_cfgs[arch]["url"])

    return model


def vitstr(pretrained: bool = False, **kwargs: Any) -> ViTSTR:
    """ViTSTR as described in `"Vision Transformer for Fast and Efficient Scene Text Recognition"
    <https://arxiv.org/pdf/2105.08582.pdf>`_.

    >>> import tensorflow as tf
    >>> from doctr.models import vitstr
    >>> model = vitstr(pretrained=False)
    >>> input_tensor = tf.random.uniform(shape=[1, 32, 128, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text recognition dataset

    Returns:
        text recognition architecture
    """

    return _vitstr("vitstr", pretrained, **kwargs)
