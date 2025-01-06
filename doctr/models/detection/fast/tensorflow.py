# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

from copy import deepcopy
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers

from doctr.file_utils import CLASS_NAME
from doctr.models.utils import IntermediateLayerGetter, _bf16_to_float32, _build_model, load_pretrained_params
from doctr.utils.repr import NestedObject

from ...classification import textnet_base, textnet_small, textnet_tiny
from ...modules.layers import FASTConvLayer
from .base import _FAST, FASTPostProcessor

__all__ = ["FAST", "fast_tiny", "fast_small", "fast_base", "reparameterize"]


default_cfgs: dict[str, dict[str, Any]] = {
    "fast_tiny": {
        "input_shape": (1024, 1024, 3),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.9.0/fast_tiny-d7379d7b.weights.h5&src=0",
    },
    "fast_small": {
        "input_shape": (1024, 1024, 3),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.9.0/fast_small-44b27eb6.weights.h5&src=0",
    },
    "fast_base": {
        "input_shape": (1024, 1024, 3),
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "url": "https://doctr-static.mindee.com/models?id=v0.9.0/fast_base-f2c6c736.weights.h5&src=0",
    },
}


class FastNeck(layers.Layer, NestedObject):
    """Neck of the FAST architecture, composed of a series of 3x3 convolutions and upsampling layer.

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 128,
    ) -> None:
        super().__init__()
        self.reduction = [FASTConvLayer(in_channels * scale, out_channels, kernel_size=3) for scale in [1, 2, 4, 8]]

    def _upsample(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        return tf.image.resize(x, size=y.shape[1:3], method="bilinear")

    def call(self, x: tf.Tensor, **kwargs: Any) -> tf.Tensor:
        f1, f2, f3, f4 = x
        f1, f2, f3, f4 = [reduction(f, **kwargs) for reduction, f in zip(self.reduction, (f1, f2, f3, f4))]
        f2, f3, f4 = [self._upsample(f, f1) for f in (f2, f3, f4)]
        f = tf.concat((f1, f2, f3, f4), axis=-1)
        return f


class FastHead(Sequential):
    """Head of the FAST architecture

    Args:
        in_channels: number of input channels
        num_classes: number of output classes
        out_channels: number of output channels
        dropout: dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        out_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        _layers = [
            FASTConvLayer(in_channels, out_channels, kernel_size=3),
            layers.Dropout(dropout),
            layers.Conv2D(num_classes, kernel_size=1, use_bias=False),
        ]
        super().__init__(_layers)


class FAST(_FAST, Model, NestedObject):
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_.

    Args:
        feature extractor: the backbone serving as feature extractor
        bin_thresh: threshold for binarization
        box_thresh: minimal objectness score to consider a box
        dropout_prob: dropout probability
        pooling_size: size of the pooling layer
        assume_straight_pages: if True, fit straight bounding boxes only
        exportable: onnx exportable returns only logits
        cfg: the configuration dict of the model
        class_names: list of class names
    """

    _children_names: list[str] = ["feat_extractor", "neck", "head", "postprocessor"]

    def __init__(
        self,
        feature_extractor: IntermediateLayerGetter,
        bin_thresh: float = 0.1,
        box_thresh: float = 0.1,
        dropout_prob: float = 0.1,
        pooling_size: int = 4,  # different from paper performs better on close text-rich images
        assume_straight_pages: bool = True,
        exportable: bool = False,
        cfg: dict[str, Any] = {},
        class_names: list[str] = [CLASS_NAME],
    ) -> None:
        super().__init__()
        self.class_names = class_names
        num_classes: int = len(self.class_names)
        self.cfg = cfg

        self.feat_extractor = feature_extractor
        self.exportable = exportable
        self.assume_straight_pages = assume_straight_pages

        # Identify the number of channels for the neck & head initialization
        feat_out_channels = [
            layers.Input(shape=in_shape[1:]).shape[-1] for in_shape in self.feat_extractor.output_shape
        ]
        # Initialize neck & head
        self.neck = FastNeck(feat_out_channels[0], feat_out_channels[1])
        self.head = FastHead(feat_out_channels[-1], num_classes, feat_out_channels[1], dropout_prob)

        # NOTE: The post processing from the paper works not well for text-rich images
        # so we use a modified version from DBNet
        self.postprocessor = FASTPostProcessor(
            assume_straight_pages=assume_straight_pages, bin_thresh=bin_thresh, box_thresh=box_thresh
        )

        # Pooling layer as erosion reversal as described in the paper
        self.pooling = layers.MaxPooling2D(pool_size=pooling_size // 2 + 1, strides=1, padding="same")

    def compute_loss(
        self,
        out_map: tf.Tensor,
        target: list[dict[str, np.ndarray]],
        eps: float = 1e-6,
    ) -> tf.Tensor:
        """Compute fast loss, 2 x Dice loss where the text kernel loss is scaled by 0.5.

        Args:
            out_map: output feature map of the model of shape (N, num_classes, H, W)
            target: list of dictionary where each dict has a `boxes` and a `flags` entry
            eps: epsilon factor in dice loss

        Returns:
            A loss tensor
        """
        targets = self.build_target(target, out_map.shape[1:], True)

        seg_target = tf.convert_to_tensor(targets[0], dtype=out_map.dtype)
        seg_mask = tf.convert_to_tensor(targets[1], dtype=out_map.dtype)
        shrunken_kernel = tf.convert_to_tensor(targets[2], dtype=out_map.dtype)

        def ohem(score: tf.Tensor, gt: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
            pos_num = tf.reduce_sum(tf.cast(gt > 0.5, dtype=tf.int32)) - tf.reduce_sum(
                tf.cast((gt > 0.5) & (mask <= 0.5), dtype=tf.int32)
            )
            neg_num = tf.reduce_sum(tf.cast(gt <= 0.5, dtype=tf.int32))
            neg_num = tf.minimum(pos_num * 3, neg_num)

            if neg_num == 0 or pos_num == 0:
                return mask

            neg_score_sorted, _ = tf.nn.top_k(-tf.boolean_mask(score, gt <= 0.5), k=neg_num)
            threshold = -neg_score_sorted[-1]

            selected_mask = tf.math.logical_and((score >= threshold) | (gt > 0.5), (mask > 0.5))
            return tf.cast(selected_mask, dtype=tf.float32)

        if len(self.class_names) > 1:
            kernels = tf.nn.softmax(out_map, axis=-1)
            prob_map = tf.nn.softmax(self.pooling(out_map), axis=-1)
        else:
            kernels = tf.sigmoid(out_map)
            prob_map = tf.sigmoid(self.pooling(out_map))

        # As described in the paper, we use the Dice loss for the text segmentation map and the Dice loss scaled by 0.5.
        selected_masks = tf.stack(
            [ohem(score, gt, mask) for score, gt, mask in zip(prob_map, seg_target, seg_mask)], axis=0
        )
        inter = tf.reduce_sum(selected_masks * prob_map * seg_target, axis=(0, 1, 2))
        cardinality = tf.reduce_sum(selected_masks * (prob_map + seg_target), axis=(0, 1, 2))
        text_loss = tf.reduce_mean((1 - 2 * inter / (cardinality + eps))) * 0.5

        # As described in the paper, we use the Dice loss for the text kernel map.
        selected_masks = seg_target * seg_mask
        inter = tf.reduce_sum(selected_masks * kernels * shrunken_kernel, axis=(0, 1, 2))
        cardinality = tf.reduce_sum(selected_masks * (kernels + shrunken_kernel), axis=(0, 1, 2))
        kernel_loss = tf.reduce_mean((1 - 2 * inter / (cardinality + eps)))

        return text_loss + kernel_loss

    def call(
        self,
        x: tf.Tensor,
        target: list[dict[str, np.ndarray]] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        feat_maps = self.feat_extractor(x, **kwargs)
        # Pass through the Neck & Head & Upsample
        feat_concat = self.neck(feat_maps, **kwargs)
        logits: tf.Tensor = self.head(feat_concat, **kwargs)
        logits = layers.UpSampling2D(size=x.shape[-2] // logits.shape[-2], interpolation="bilinear")(logits, **kwargs)

        out: dict[str, tf.Tensor] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output or target is None or return_preds:
            prob_map = _bf16_to_float32(tf.math.sigmoid(self.pooling(logits, **kwargs)))

        if return_model_output:
            out["out_map"] = prob_map

        if target is None or return_preds:
            # Post-process boxes (keep only text predictions)
            out["preds"] = [dict(zip(self.class_names, preds)) for preds in self.postprocessor(prob_map.numpy())]

        if target is not None:
            loss = self.compute_loss(logits, target)
            out["loss"] = loss

        return out


def reparameterize(model: FAST | layers.Layer) -> FAST:
    """Fuse batchnorm and conv layers and reparameterize the model

    args:

        model: the FAST model to reparameterize

    Returns:
        the reparameterized model
    """
    last_conv = None
    last_conv_idx = None

    for idx, layer in enumerate(model.layers):
        if hasattr(layer, "layers") or isinstance(
            layer, (FASTConvLayer, FastNeck, FastHead, layers.BatchNormalization, layers.Conv2D)
        ):
            if isinstance(layer, layers.BatchNormalization):
                # fuse batchnorm only if it is followed by a conv layer
                if last_conv is None:
                    continue
                conv_w = last_conv.kernel
                conv_b = last_conv.bias if last_conv.use_bias else tf.zeros_like(layer.moving_mean)

                factor = layer.gamma / tf.sqrt(layer.moving_variance + layer.epsilon)
                last_conv.kernel = conv_w * factor.numpy().reshape([1, 1, 1, -1])
                if last_conv.use_bias:
                    last_conv.bias.assign((conv_b - layer.moving_mean) * factor + layer.beta)
                model.layers[last_conv_idx] = last_conv  # Replace the last conv layer with the fused version
                model.layers[idx] = layers.Lambda(lambda x: x)
                last_conv = None
            elif isinstance(layer, layers.Conv2D):
                last_conv = layer
                last_conv_idx = idx
            elif isinstance(layer, FASTConvLayer):
                layer.reparameterize_layer()
            elif isinstance(layer, FastNeck):
                for reduction in layer.reduction:
                    reduction.reparameterize_layer()
            elif isinstance(layer, FastHead):
                reparameterize(layer)
            else:
                reparameterize(layer)
    return model


def _fast(
    arch: str,
    pretrained: bool,
    backbone_fn,
    feat_layers: list[str],
    pretrained_backbone: bool = True,
    input_shape: tuple[int, int, int] | None = None,
    **kwargs: Any,
) -> FAST:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["input_shape"] = input_shape or _cfg["input_shape"]
    if not kwargs.get("class_names", None):
        kwargs["class_names"] = _cfg.get("class_names", [CLASS_NAME])
    else:
        kwargs["class_names"] = sorted(kwargs["class_names"])

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(
            input_shape=_cfg["input_shape"],
            include_top=False,
            pretrained=pretrained_backbone,
        ),
        feat_layers,
    )

    # Build the model
    model = FAST(feat_extractor, cfg=_cfg, **kwargs)
    _build_model(model)

    # Load pretrained parameters
    if pretrained:
        # The given class_names differs from the pretrained model => skip the mismatching layers for fine tuning
        load_pretrained_params(
            model,
            _cfg["url"],
            skip_mismatch=kwargs["class_names"] != default_cfgs[arch].get("class_names", [CLASS_NAME]),
        )

    return model


def fast_tiny(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a tiny TextNet backbone.

    >>> import tensorflow as tf
    >>> from doctr.models import fast_tiny
    >>> model = fast_tiny(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _fast(
        "fast_tiny",
        pretrained,
        textnet_tiny,
        ["stage_0", "stage_1", "stage_2", "stage_3"],
        **kwargs,
    )


def fast_small(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a small TextNet backbone.

    >>> import tensorflow as tf
    >>> from doctr.models import fast_small
    >>> model = fast_small(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _fast(
        "fast_small",
        pretrained,
        textnet_small,
        ["stage_0", "stage_1", "stage_2", "stage_3"],
        **kwargs,
    )


def fast_base(pretrained: bool = False, **kwargs: Any) -> FAST:
    """FAST as described in `"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation"
    <https://arxiv.org/pdf/2111.02394.pdf>`_, using a base TextNet backbone.

    >>> import tensorflow as tf
    >>> from doctr.models import fast_base
    >>> model = fast_base(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _fast(
        "fast_base",
        pretrained,
        textnet_base,
        ["stage_0", "stage_1", "stage_2", "stage_3"],
        **kwargs,
    )
