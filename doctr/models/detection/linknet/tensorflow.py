# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential, layers

from doctr.models.classification import resnet18, resnet34, resnet50
from doctr.models.utils import IntermediateLayerGetter, conv_sequence, load_pretrained_params
from doctr.utils.repr import NestedObject

from .base import LinkNetPostProcessor, _LinkNet

__all__ = ['LinkNet', 'linknet_resnet18', 'linknet_resnet34', 'linknet_resnet50', 'linknet_resnet18_rotation']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'linknet_resnet18': {
        'mean': (0.798, 0.785, 0.772),
        'std': (0.264, 0.2749, 0.287),
        'input_shape': (1024, 1024, 3),
        'url': None,
    },
    'linknet_resnet18_rotation': {
        'mean': (0.798, 0.785, 0.772),
        'std': (0.264, 0.2749, 0.287),
        'input_shape': (1024, 1024, 3),
        'url': 'https://github.com/mindee/doctr/releases/download/v0.5.0/linknet_resnet18-a48e6ed3.zip',
    },
    'linknet_resnet34': {
        'mean': (0.798, 0.785, 0.772),
        'std': (0.264, 0.2749, 0.287),
        'input_shape': (1024, 1024, 3),
        'url': None,
    },
    'linknet_resnet50': {
        'mean': (0.798, 0.785, 0.772),
        'std': (0.264, 0.2749, 0.287),
        'input_shape': (1024, 1024, 3),
        'url': None,
    },
}


def decoder_block(in_chan: int, out_chan: int, stride: int, **kwargs: Any) -> Sequential:
    """Creates a LinkNet decoder block"""

    return Sequential([
        *conv_sequence(in_chan // 4, 'relu', True, kernel_size=1, **kwargs),
        layers.Conv2DTranspose(
            filters=in_chan // 4,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer='he_normal'
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        *conv_sequence(out_chan, 'relu', True, kernel_size=1),
    ])


class LinkNetFPN(Model, NestedObject):
    """LinkNet Decoder module"""

    def __init__(
        self,
        out_chans: int,
        in_shapes: List[Tuple[int, ...]],
    ) -> None:

        super().__init__()
        self.out_chans = out_chans
        strides = [2] * (len(in_shapes) - 1) + [1]
        i_chans = [s[-1] for s in in_shapes[::-1]]
        o_chans = i_chans[1:] + [out_chans]
        self.decoders = [
            decoder_block(in_chan, out_chan, s, input_shape=in_shape)
            for in_chan, out_chan, s, in_shape in zip(i_chans, o_chans, strides, in_shapes[::-1])
        ]

    def call(
        self,
        x: List[tf.Tensor]
    ) -> tf.Tensor:
        out = 0
        for decoder, fmap in zip(self.decoders, x[::-1]):
            out = decoder(out + fmap)
        return out

    def extra_repr(self) -> str:
        return f"out_chans={self.out_chans}"


class LinkNet(_LinkNet, keras.Model):
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    Args:
        num_classes: number of channels for the output
    """

    _children_names: List[str] = ['feat_extractor', 'fpn', 'classifier', 'postprocessor']

    def __init__(
        self,
        feat_extractor: IntermediateLayerGetter,
        fpn_channels: int = 64,
        num_classes: int = 1,
        assume_straight_pages: bool = True,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(cfg=cfg)

        self.assume_straight_pages = assume_straight_pages

        self.feat_extractor = feat_extractor

        self.fpn = LinkNetFPN(fpn_channels, [_shape[1:] for _shape in self.feat_extractor.output_shape])
        self.fpn.build(self.feat_extractor.output_shape)

        self.classifier = Sequential([
            layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                kernel_initializer='he_normal',
                input_shape=self.fpn.decoders[-1].output_shape[1:],
            ),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            *conv_sequence(32, 'relu', True, kernel_size=3, strides=1),
            layers.Conv2DTranspose(
                filters=num_classes,
                kernel_size=2,
                strides=2,
                padding="same",
                use_bias=True,
                kernel_initializer='he_normal'
            ),
        ])

        self.postprocessor = LinkNetPostProcessor(assume_straight_pages=assume_straight_pages)

    def compute_loss(
        self,
        out_map: tf.Tensor,
        target: List[np.ndarray],
        gamma: float = 2.,
        alpha: float = .5,
        eps: float = 1e-8,
    ) -> tf.Tensor:
        """Compute linknet loss, BCE with boosted box edges or focal loss. Focal loss implementation based on
        <https://github.com/tensorflow/addons/>`_.

        Args:
            out_map: output feature map of the model of shape N x H x W x 1
            target: list of dictionary where each dict has a `boxes` and a `flags` entry
            gamma: modulating factor in the focal loss formula
            alpha: balancing factor in the focal loss formula

        Returns:
            A loss tensor
        """
        seg_target, seg_mask = self.build_target(target, out_map.shape[1:3])

        seg_target = tf.convert_to_tensor(seg_target, dtype=out_map.dtype)
        seg_mask = tf.convert_to_tensor(seg_mask, dtype=tf.bool)
        seg_mask = tf.cast(seg_mask, tf.float32)

        bce_loss = tf.keras.losses.binary_crossentropy(seg_target, out_map, from_logits=True)[..., None]
        proba_map = tf.sigmoid(out_map)

        # Focal loss
        if gamma < 0:
            raise ValueError("Value of gamma should be greater than or equal to zero.")
        # Convert logits to prob, compute gamma factor
        p_t = (seg_target * proba_map) + ((1 - seg_target) * (1 - proba_map))
        alpha_t = seg_target * alpha + (1 - seg_target) * (1 - alpha)
        # Unreduced loss
        focal_loss = alpha_t * (1 - p_t) ** gamma * bce_loss
        # Class reduced
        focal_loss = tf.reduce_sum(seg_mask * focal_loss, (0, 1, 2)) / tf.reduce_sum(seg_mask, (0, 1, 2))

        # Dice loss
        inter = tf.math.reduce_sum(seg_mask * proba_map * seg_target, (0, 1, 2))
        cardinality = tf.math.reduce_sum(seg_mask * (proba_map + seg_target), (0, 1, 2))
        dice_loss = 1 - 2 * (inter + eps) / (cardinality + eps)

        return tf.reduce_mean(focal_loss) + tf.reduce_mean(dice_loss)

    def call(
        self,
        x: tf.Tensor,
        target: Optional[List[np.ndarray]] = None,
        return_model_output: bool = False,
        return_preds: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        feat_maps = self.feat_extractor(x, **kwargs)
        logits = self.fpn(feat_maps, **kwargs)
        logits = self.classifier(logits, **kwargs)

        out: Dict[str, tf.Tensor] = {}
        if return_model_output or target is None or return_preds:
            prob_map = tf.math.sigmoid(logits)
        if return_model_output:
            out["out_map"] = prob_map

        if target is None or return_preds:
            # Post-process boxes
            out["preds"] = [preds[0] for preds in self.postprocessor(prob_map.numpy())]

        if target is not None:
            loss = self.compute_loss(logits, target)
            out['loss'] = loss

        return out


def _linknet(
    arch: str,
    pretrained: bool,
    backbone_fn,
    fpn_layers: List[str],
    pretrained_backbone: bool = True,
    input_shape: Optional[Tuple[int, int, int]] = None,
    **kwargs: Any
) -> LinkNet:

    pretrained_backbone = pretrained_backbone and not pretrained

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or default_cfgs[arch]['input_shape']

    # Feature extractor
    feat_extractor = IntermediateLayerGetter(
        backbone_fn(
            pretrained=pretrained_backbone,
            include_top=False,
            input_shape=_cfg['input_shape'],
        ),
        fpn_layers,
    )

    # Build the model
    model = LinkNet(feat_extractor, cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, _cfg['url'])

    return model


def linknet_resnet18(pretrained: bool = False, **kwargs: Any) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    >>> import tensorflow as tf
    >>> from doctr.models import linknet_resnet18
    >>> model = linknet_resnet18(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _linknet(
        'linknet_resnet18',
        pretrained,
        resnet18,
        ['resnet_block_1', 'resnet_block_3', 'resnet_block_5', 'resnet_block_7'],
        **kwargs,
    )


def linknet_resnet18_rotation(pretrained: bool = False, **kwargs: Any) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    >>> import tensorflow as tf
    >>> from doctr.models import linknet_resnet18_rotation
    >>> model = linknet_resnet18_rotation(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _linknet(
        'linknet_resnet18_rotation',
        pretrained,
        resnet18,
        ['resnet_block_1', 'resnet_block_3', 'resnet_block_5', 'resnet_block_7'],
        **kwargs,
    )


def linknet_resnet34(pretrained: bool = False, **kwargs: Any) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    >>> import tensorflow as tf
    >>> from doctr.models import linknet_resnet34
    >>> model = linknet_resnet34(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _linknet(
        'linknet_resnet34',
        pretrained,
        resnet34,
        ['resnet_block_2', 'resnet_block_6', 'resnet_block_12', 'resnet_block_15'],
        **kwargs,
    )


def linknet_resnet50(pretrained: bool = False, **kwargs: Any) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    >>> import tensorflow as tf
    >>> from doctr.models import linknet_resnet50
    >>> model = linknet_resnet50(pretrained=True)
    >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
    >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _linknet(
        'linknet_resnet50',
        pretrained,
        resnet50,
        ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"],
        **kwargs,
    )
