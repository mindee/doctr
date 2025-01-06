# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

from copy import deepcopy
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers, losses
from tensorflow.keras.applications import ResNet50

from doctr.file_utils import CLASS_NAME
from doctr.models.utils import (
    IntermediateLayerGetter,
    _bf16_to_float32,
    _build_model,
    conv_sequence,
    load_pretrained_params,
)
from doctr.utils.repr import NestedObject

from ...classification import mobilenet_v3_large
from .base import DBPostProcessor, _DBNet

__all__ = ["DBNet", "db_resnet50", "db_mobilenet_v3_large"]


default_cfgs: dict[str, dict[str, Any]] = {
    "db_resnet50": {
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "input_shape": (1024, 1024, 3),
        "url": "https://doctr-static.mindee.com/models?id=v0.9.0/db_resnet50-649fa22b.weights.h5&src=0",
    },
    "db_mobilenet_v3_large": {
        "mean": (0.798, 0.785, 0.772),
        "std": (0.264, 0.2749, 0.287),
        "input_shape": (1024, 1024, 3),
        "url": "https://doctr-static.mindee.com/models?id=v0.9.0/db_mobilenet_v3_large-ee2e1dbe.weights.h5&src=0",
    },
}


class FeaturePyramidNetwork(layers.Layer, NestedObject):
    """Feature Pyramid Network as described in `"Feature Pyramid Networks for Object Detection"
    <https://arxiv.org/pdf/1612.03144.pdf>`_.

    Args:
        channels: number of channel to output
    """

    def __init__(
        self,
        channels: int,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.upsample = layers.UpSampling2D(size=(2, 2), interpolation="nearest")
        self.inner_blocks = [layers.Conv2D(channels, 1, strides=1, kernel_initializer="he_normal") for _ in range(4)]
        self.layer_blocks = [self.build_upsampling(channels, dilation_factor=2**idx) for idx in range(4)]

    @staticmethod
    def build_upsampling(
        channels: int,
        dilation_factor: int = 1,
    ) -> layers.Layer:
        """Module which performs a 3x3 convolution followed by up-sampling

        Args:
            channels: number of output channels
            dilation_factor (int): dilation factor to scale the convolution output before concatenation

        Returns:
            a keras.layers.Layer object, wrapping these operations in a sequential module

        """
        _layers = conv_sequence(channels, "relu", True, kernel_size=3)

        if dilation_factor > 1:
            _layers.append(layers.UpSampling2D(size=(dilation_factor, dilation_factor), interpolation="nearest"))

        module = Sequential(_layers)

        return module

    def extra_repr(self) -> str:
        return f"channels={self.channels}"

    def call(
        self,
        x: list[tf.Tensor],
        **kwargs: Any,
    ) -> tf.Tensor:
        # Channel mapping
        results = [block(fmap, **kwargs) for block, fmap in zip(self.inner_blocks, x)]
        # Upsample & sum
        for idx in range(len(results) - 1, -1):
            results[idx] += self.upsample(results[idx + 1])
        # Conv & upsample
        results = [block(fmap, **kwargs) for block, fmap in zip(self.layer_blocks, results)]

        return layers.concatenate(results)


class DBNet(_DBNet, Model, NestedObject):
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_.

    Args:
        feature extractor: the backbone serving as feature extractor
        fpn_channels: number of channels each extracted feature maps is mapped to
        bin_thresh: threshold for binarization
        box_thresh: minimal objectness score to consider a box
        assume_straight_pages: if True, fit straight bounding boxes only
        exportable: onnx exportable returns only logits
        cfg: the configuration dict of the model
        class_names: list of class names
    """

    _children_names: list[str] = ["feat_extractor", "fpn", "probability_head", "threshold_head", "postprocessor"]

    def __init__(
        self,
        feature_extractor: IntermediateLayerGetter,
        fpn_channels: int = 128,  # to be set to 256 to represent the author's initial idea
        bin_thresh: float = 0.3,
        box_thresh: float = 0.1,
        assume_straight_pages: bool = True,
        exportable: bool = False,
        cfg: dict[str, Any] | None = None,
        class_names: list[str] = [CLASS_NAME],
    ) -> None:
        super().__init__()
        self.class_names = class_names
        num_classes: int = len(self.class_names)
        self.cfg = cfg

        self.feat_extractor = feature_extractor
        self.exportable = exportable
        self.assume_straight_pages = assume_straight_pages

        self.fpn = FeaturePyramidNetwork(channels=fpn_channels)
        # Initialize kernels
        _inputs = [layers.Input(shape=in_shape[1:]) for in_shape in self.feat_extractor.output_shape]
        output_shape = tuple(self.fpn(_inputs).shape)

        self.probability_head = Sequential([
            *conv_sequence(64, "relu", True, kernel_size=3, input_shape=output_shape[1:]),
            layers.Conv2DTranspose(64, 2, strides=2, use_bias=False, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2DTranspose(num_classes, 2, strides=2, kernel_initializer="he_normal"),
        ])
        self.threshold_head = Sequential([
            *conv_sequence(64, "relu", True, kernel_size=3, input_shape=output_shape[1:]),
            layers.Conv2DTranspose(64, 2, strides=2, use_bias=False, kernel_initializer="he_normal"),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Conv2DTranspose(num_classes, 2, strides=2, kernel_initializer="he_normal"),
        ])

        self.postprocessor = DBPostProcessor(
            assume_straight_pages=assume_straight_pages, bin_thresh=bin_thresh, box_thresh=box_thresh
        )

    def compute_loss(
        self,
        out_map: tf.Tensor,
        thresh_map: tf.Tensor,
        target: list[dict[str, np.ndarray]],
        gamma: float = 2.0,
        alpha: float = 0.5,
        eps: float = 1e-8,
    ) -> tf.Tensor:
        """Compute a batch of gts, masks, thresh_gts, thresh_masks from a list of boxes
        and a list of masks for each image. From there it computes the loss with the model output

        Args:
            out_map: output feature map of the model of shape (N, H, W, C)
            thresh_map: threshold map of shape (N, H, W, C)
            target: list of dictionary where each dict has a `boxes` and a `flags` entry
            gamma: modulating factor in the focal loss formula
            alpha: balancing factor in the focal loss formula
            eps: epsilon factor in dice loss

        Returns:
            A loss tensor
        """
        if gamma < 0:
            raise ValueError("Value of gamma should be greater than or equal to zero.")

        prob_map = tf.math.sigmoid(out_map)
        thresh_map = tf.math.sigmoid(thresh_map)

        seg_target, seg_mask, thresh_target, thresh_mask = self.build_target(target, out_map.shape[1:], True)
        seg_target = tf.convert_to_tensor(seg_target, dtype=out_map.dtype)
        seg_mask = tf.convert_to_tensor(seg_mask, dtype=tf.bool)
        seg_mask = tf.cast(seg_mask, tf.float32)
        thresh_target = tf.convert_to_tensor(thresh_target, dtype=out_map.dtype)
        thresh_mask = tf.convert_to_tensor(thresh_mask, dtype=tf.bool)

        # Focal loss
        focal_scale = 10.0
        bce_loss = losses.binary_crossentropy(seg_target[..., None], out_map[..., None], from_logits=True)

        # Convert logits to prob, compute gamma factor
        p_t = (seg_target * prob_map) + ((1 - seg_target) * (1 - prob_map))
        alpha_t = seg_target * alpha + (1 - seg_target) * (1 - alpha)
        # Unreduced loss
        focal_loss = alpha_t * (1 - p_t) ** gamma * bce_loss
        # Class reduced
        focal_loss = tf.reduce_sum(seg_mask * focal_loss, (0, 1, 2, 3)) / tf.reduce_sum(seg_mask, (0, 1, 2, 3))

        # Compute dice loss for each class or for approx binary_map
        if len(self.class_names) > 1:
            dice_map = tf.nn.softmax(out_map, axis=-1)
        else:
            # compute binary map instead
            dice_map = 1.0 / (1.0 + tf.exp(-50 * (prob_map - thresh_map)))
        # Class-reduced dice loss
        inter = tf.reduce_sum(seg_mask * dice_map * seg_target, axis=[0, 1, 2])
        cardinality = tf.reduce_sum(seg_mask * (dice_map + seg_target), axis=[0, 1, 2])
        dice_loss = tf.reduce_mean(1 - 2 * inter / (cardinality + eps))

        # Compute l1 loss for thresh_map
        if tf.reduce_any(thresh_mask):
            thresh_mask = tf.cast(thresh_mask, tf.float32)
            l1_loss = tf.reduce_sum(tf.abs(thresh_map - thresh_target) * thresh_mask) / (
                tf.reduce_sum(thresh_mask) + eps
            )
        else:
            l1_loss = tf.constant(0.0)

        return l1_loss + focal_scale * focal_loss + dice_loss

    def call(
        self,
        x: tf.Tensor,
        target: list[dict[str, np.ndarray]] | None = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        feat_maps = self.feat_extractor(x, **kwargs)
        feat_concat = self.fpn(feat_maps, **kwargs)
        logits = self.probability_head(feat_concat, **kwargs)

        out: dict[str, tf.Tensor] = {}
        if self.exportable:
            out["logits"] = logits
            return out

        if return_model_output or target is None or return_preds:
            prob_map = _bf16_to_float32(tf.math.sigmoid(logits))

        if return_model_output:
            out["out_map"] = prob_map

        if target is None or return_preds:
            # Post-process boxes (keep only text predictions)
            out["preds"] = [dict(zip(self.class_names, preds)) for preds in self.postprocessor(prob_map.numpy())]

        if target is not None:
            thresh_map = self.threshold_head(feat_concat, **kwargs)
            loss = self.compute_loss(logits, thresh_map, target)
            out["loss"] = loss

        return out


def _db_resnet(
    arch: str,
    pretrained: bool,
    backbone_fn,
    fpn_layers: list[str],
    pretrained_backbone: bool = True,
    input_shape: tuple[int, int, int] | None = None,
    **kwargs: Any,
) -> DBNet:
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
            weights="imagenet" if pretrained_backbone else None,
            include_top=False,
            pooling=None,
            input_shape=_cfg["input_shape"],
        ),
        fpn_layers,
    )

    # Build the model
    model = DBNet(feat_extractor, cfg=_cfg, **kwargs)
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


def _db_mobilenet(
    arch: str,
    pretrained: bool,
    backbone_fn,
    fpn_layers: list[str],
    pretrained_backbone: bool = True,
    input_shape: tuple[int, int, int] | None = None,
    **kwargs: Any,
) -> DBNet:
    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg["input_shape"] = input_shape or _cfg["input_shape"]
    if not kwargs.get("class_names", None):
        kwargs["class_names"] = default_cfgs[arch].get("class_names", [CLASS_NAME])
    else:
        kwargs["class_names"] = sorted(kwargs["class_names"])

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(
            input_shape=_cfg["input_shape"],
            include_top=False,
            pretrained=pretrained_backbone,
        ),
        fpn_layers,
    )

    # Build the model
    model = DBNet(feat_extractor, cfg=_cfg, **kwargs)
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


def db_resnet50(pretrained: bool = False, **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a ResNet-50 backbone.

    >>> import tensorflow as tf
    >>> from doctr.models import db_resnet50
    >>> model = db_resnet50(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _db_resnet(
        "db_resnet50",
        pretrained,
        ResNet50,
        ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"],
        **kwargs,
    )


def db_mobilenet_v3_large(pretrained: bool = False, **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a mobilenet v3 large backbone.

    >>> import tensorflow as tf
    >>> from doctr.models import db_mobilenet_v3_large
    >>> model = db_mobilenet_v3_large(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset
        **kwargs: keyword arguments of the DBNet architecture

    Returns:
        text detection architecture
    """
    return _db_mobilenet(
        "db_mobilenet_v3_large",
        pretrained,
        mobilenet_v3_large,
        ["inverted_2", "inverted_5", "inverted_11", "final_block"],
        **kwargs,
    )
