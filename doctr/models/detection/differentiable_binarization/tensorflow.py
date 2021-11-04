# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50

from doctr.models.utils import IntermediateLayerGetter, conv_sequence, load_pretrained_params
from doctr.utils.repr import NestedObject

from ...backbones import mobilenet_v3_large
from .base import DBPostProcessor, _DBNet

__all__ = ['DBNet', 'db_resnet50', 'db_mobilenet_v3_large']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'db_resnet50': {
        'mean': (0.798, 0.785, 0.772),
        'std': (0.264, 0.2749, 0.287),
        'backbone': ResNet50,
        'fpn_layers': ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"],
        'input_shape': (1024, 1024, 3),
        'url': 'https://github.com/mindee/doctr/releases/download/v0.2.0/db_resnet50-adcafc63.zip',
    },
    'db_mobilenet_v3_large': {
        'mean': (0.798, 0.785, 0.772),
        'std': (0.264, 0.2749, 0.287),
        'backbone': mobilenet_v3_large,
        'fpn_layers': ["inverted_2", "inverted_5", "inverted_11", "final_block"],
        'input_shape': (1024, 1024, 3),
        'url': 'https://github.com/mindee/doctr/releases/download/v0.3.1/db_mobilenet_v3_large-8c16d5bf.zip',
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
        self.upsample = layers.UpSampling2D(size=(2, 2), interpolation='nearest')
        self.inner_blocks = [layers.Conv2D(channels, 1, strides=1, kernel_initializer='he_normal') for _ in range(4)]
        self.layer_blocks = [self.build_upsampling(channels, dilation_factor=2 ** idx) for idx in range(4)]

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

        _layers = conv_sequence(channels, 'relu', True, kernel_size=3)

        if dilation_factor > 1:
            _layers.append(layers.UpSampling2D(size=(dilation_factor, dilation_factor), interpolation='nearest'))

        module = keras.Sequential(_layers)

        return module

    def extra_repr(self) -> str:
        return f"channels={self.channels}"

    def call(
        self,
        x: List[tf.Tensor],
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


class DBNet(_DBNet, keras.Model, NestedObject):
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_.

    Args:
        feature extractor: the backbone serving as feature extractor
        fpn_channels: number of channels each extracted feature maps is mapped to
        rotated_bbox: whether the segmentation map can include rotated bounding boxes
        cfg: the configuration dict of the model
    """

    _children_names: List[str] = ['feat_extractor', 'fpn', 'probability_head', 'threshold_head', 'postprocessor']

    def __init__(
        self,
        feature_extractor: IntermediateLayerGetter,
        fpn_channels: int = 128,  # to be set to 256 to represent the author's initial idea
        rotated_bbox: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__()
        self.cfg = cfg

        self.feat_extractor = feature_extractor
        self.rotated_bbox = rotated_bbox

        self.fpn = FeaturePyramidNetwork(channels=fpn_channels)
        # Initialize kernels
        _inputs = [layers.Input(shape=in_shape[1:]) for in_shape in self.feat_extractor.output_shape]
        output_shape = tuple(self.fpn(_inputs).shape)

        self.probability_head = keras.Sequential(
            [
                *conv_sequence(64, 'relu', True, kernel_size=3, input_shape=output_shape[1:]),
                layers.Conv2DTranspose(64, 2, strides=2, use_bias=False, kernel_initializer='he_normal'),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Conv2DTranspose(1, 2, strides=2, kernel_initializer='he_normal'),
            ]
        )
        self.threshold_head = keras.Sequential(
            [
                *conv_sequence(64, 'relu', True, kernel_size=3, input_shape=output_shape[1:]),
                layers.Conv2DTranspose(64, 2, strides=2, use_bias=False, kernel_initializer='he_normal'),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Conv2DTranspose(1, 2, strides=2, kernel_initializer='he_normal'),
            ]
        )

        self.postprocessor = DBPostProcessor(rotated_bbox=rotated_bbox)

    def compute_loss(
        self,
        out_map: tf.Tensor,
        thresh_map: tf.Tensor,
        target: List[np.ndarray]
    ) -> tf.Tensor:
        """Compute a batch of gts, masks, thresh_gts, thresh_masks from a list of boxes
        and a list of masks for each image. From there it computes the loss with the model output

        Args:
            out_map: output feature map of the model of shape (N, H, W, C)
            thresh_map: threshold map of shape (N, H, W, C)
            target: list of dictionary where each dict has a `boxes` and a `flags` entry

        Returns:
            A loss tensor
        """

        prob_map = tf.math.sigmoid(tf.squeeze(out_map, axis=[-1]))
        thresh_map = tf.math.sigmoid(tf.squeeze(thresh_map, axis=[-1]))

        seg_target, seg_mask, thresh_target, thresh_mask = self.compute_target(target, out_map.shape[:3])
        seg_target = tf.convert_to_tensor(seg_target, dtype=out_map.dtype)
        seg_mask = tf.convert_to_tensor(seg_mask, dtype=tf.bool)
        thresh_target = tf.convert_to_tensor(thresh_target, dtype=out_map.dtype)
        thresh_mask = tf.convert_to_tensor(thresh_mask, dtype=tf.bool)

        # Compute balanced BCE loss for proba_map
        bce_scale = 5.
        bce_loss = tf.keras.losses.binary_crossentropy(seg_target[..., None], out_map, from_logits=True)[seg_mask]

        neg_target = 1 - seg_target[seg_mask]
        positive_count = tf.math.reduce_sum(seg_target[seg_mask])
        negative_count = tf.math.reduce_min([tf.math.reduce_sum(neg_target), 3. * positive_count])
        negative_loss = bce_loss * neg_target
        negative_loss, _ = tf.nn.top_k(negative_loss, tf.cast(negative_count, tf.int32))
        sum_losses = tf.math.reduce_sum(bce_loss * seg_target[seg_mask]) + tf.math.reduce_sum(negative_loss)
        balanced_bce_loss = sum_losses / (positive_count + negative_count + 1e-6)

        # Compute dice loss for approxbin_map
        bin_map = 1 / (1 + tf.exp(-50. * (prob_map[seg_mask] - thresh_map[seg_mask])))

        bce_min = tf.math.reduce_min(bce_loss)
        weights = (bce_loss - bce_min) / (tf.math.reduce_max(bce_loss) - bce_min) + 1.
        inter = tf.math.reduce_sum(bin_map * seg_target[seg_mask] * weights)
        union = tf.math.reduce_sum(bin_map) + tf.math.reduce_sum(seg_target[seg_mask]) + 1e-8
        dice_loss = 1 - 2.0 * inter / union

        # Compute l1 loss for thresh_map
        l1_scale = 10.
        if tf.reduce_any(thresh_mask):
            l1_loss = tf.math.reduce_mean(tf.math.abs(thresh_map[thresh_mask] - thresh_target[thresh_mask]))
        else:
            l1_loss = tf.constant(0.)

        return l1_scale * l1_loss + bce_scale * balanced_bce_loss + dice_loss

    def call(
        self,
        x: tf.Tensor,
        target: Optional[List[np.ndarray]] = None,
        return_model_output: bool = False,
        return_boxes: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        feat_maps = self.feat_extractor(x, **kwargs)
        feat_concat = self.fpn(feat_maps, **kwargs)
        logits = self.probability_head(feat_concat, **kwargs)

        out: Dict[str, tf.Tensor] = {}
        if return_model_output or target is None or return_boxes:
            prob_map = tf.math.sigmoid(logits)

        if return_model_output:
            out["out_map"] = prob_map

        if target is None or return_boxes:
            # Post-process boxes
            out["preds"] = self.postprocessor(tf.squeeze(prob_map, axis=-1).numpy())

        if target is not None:
            thresh_map = self.threshold_head(feat_concat, **kwargs)
            loss = self.compute_loss(logits, thresh_map, target)
            out['loss'] = loss

        return out


def _db_resnet(
    arch: str,
    pretrained: bool,
    pretrained_backbone: bool = False,
    input_shape: Tuple[int, int, int] = None,
    **kwargs: Any
) -> DBNet:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        _cfg['backbone'](
            include_top=False,
            weights='imagenet' if pretrained_backbone else None,
            input_shape=_cfg['input_shape'],
            pooling=None,
        ),
        _cfg['fpn_layers'],
    )

    # Build the model
    model = DBNet(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, _cfg['url'])

    return model


def _db_mobilenet(
    arch: str,
    pretrained: bool,
    pretrained_backbone: bool = True,
    input_shape: Tuple[int, int, int] = None,
    **kwargs: Any
) -> DBNet:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        _cfg['backbone'](
            input_shape=_cfg['input_shape'],
            include_top=False,
            pretrained=pretrained_backbone,
        ),
        _cfg['fpn_layers'],
    )

    # Build the model
    model = DBNet(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, _cfg['url'])

    return model


def db_resnet50(pretrained: bool = False, **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a ResNet-50 backbone.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import db_resnet50
        >>> model = db_resnet50(pretrained=True)
        >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _db_resnet('db_resnet50', pretrained, **kwargs)


def db_mobilenet_v3_large(pretrained: bool = False, **kwargs: Any) -> DBNet:
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_, using a mobilenet v3 large backbone.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import db_mobilenet_v3_large
        >>> model = db_mobilenet_v3_large(pretrained=True)
        >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _db_mobilenet('db_mobilenet_v3_large', pretrained, **kwargs)
