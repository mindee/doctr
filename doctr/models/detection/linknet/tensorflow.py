# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

from copy import deepcopy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from typing import Dict, Any, Tuple, Optional, List

from doctr.utils.repr import NestedObject
from doctr.models.backbones import ResnetStage
from doctr.models.utils import conv_sequence, load_pretrained_params
from .base import LinkNetPostProcessor, _LinkNet

__all__ = ['LinkNet', 'linknet16']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'linknet16': {
        'mean': (0.798, 0.785, 0.772),
        'std': (0.264, 0.2749, 0.287),
        'input_shape': (1024, 1024, 3),
        'url': None,
    },
}


def decoder_block(in_chan: int, out_chan: int) -> Sequential:
    """Creates a LinkNet decoder block"""

    return Sequential([
        *conv_sequence(in_chan // 4, 'relu', True, kernel_size=1),
        layers.Conv2DTranspose(
            filters=in_chan // 4,
            kernel_size=3,
            strides=2,
            padding="same",
            use_bias=False,
            kernel_initializer='he_normal'
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        *conv_sequence(out_chan, 'relu', True, kernel_size=1),
    ])


class LinkNetFPN(layers.Layer, NestedObject):
    """LinkNet Encoder-Decoder module"""

    def __init__(
        self,
    ) -> None:

        super().__init__()
        self.encoder_1 = ResnetStage(num_blocks=2, output_channels=64, downsample=True)
        self.encoder_2 = ResnetStage(num_blocks=2, output_channels=128, downsample=True)
        self.encoder_3 = ResnetStage(num_blocks=2, output_channels=256, downsample=True)
        self.encoder_4 = ResnetStage(num_blocks=2, output_channels=512, downsample=True)
        self.decoder_1 = decoder_block(in_chan=64, out_chan=64)
        self.decoder_2 = decoder_block(in_chan=128, out_chan=64)
        self.decoder_3 = decoder_block(in_chan=256, out_chan=128)
        self.decoder_4 = decoder_block(in_chan=512, out_chan=256)

    def call(
        self,
        x: tf.Tensor
    ) -> tf.Tensor:
        x_1 = self.encoder_1(x)
        x_2 = self.encoder_2(x_1)
        x_3 = self.encoder_3(x_2)
        x_4 = self.encoder_4(x_3)
        y_4 = self.decoder_4(x_4)
        y_3 = self.decoder_3(y_4 + x_3)
        y_2 = self.decoder_2(y_3 + x_2)
        y_1 = self.decoder_1(y_2 + x_1)
        return y_1


class LinkNet(_LinkNet, keras.Model):
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    Args:
        num_classes: number of channels for the output
    """

    _children_names: List[str] = ['stem', 'fpn', 'classifier', 'postprocessor']

    def __init__(
        self,
        num_classes: int = 1,
        input_shape: Tuple[int, int, int] = (512, 512, 3),
        rotated_bbox: bool = False,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(cfg=cfg)

        self.rotated_bbox = rotated_bbox

        self.stem = Sequential([
            *conv_sequence(64, 'relu', True, strides=2, kernel_size=7, input_shape=input_shape),
            layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same'),
        ])

        self.fpn = LinkNetFPN()

        self.classifier = Sequential([
            layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                kernel_initializer='he_normal'
            ),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            *conv_sequence(32, 'relu', True, strides=1, kernel_size=3),
            layers.Conv2DTranspose(
                filters=num_classes,
                kernel_size=2,
                strides=2,
                padding="same",
                use_bias=False,
                kernel_initializer='he_normal'
            ),
        ])

        self.postprocessor = LinkNetPostProcessor(rotated_bbox=rotated_bbox)

    def compute_loss(
        self,
        out_map: tf.Tensor,
        target: List[np.ndarray],
        focal_loss: bool = False,
        alpha: float = .5,
        gamma: float = 2.,
        edge_factor: float = 2.,
    ) -> tf.Tensor:
        """Compute linknet loss, BCE with boosted box edges or focal loss. Focal loss implementation based on
        <https://github.com/tensorflow/addons/>`_.

        Args:
            out_map: output feature map of the model of shape N x H x W x 1
            target: list of dictionary where each dict has a `boxes` and a `flags` entry
            focal_loss: if True, use focal loss instead of BCE
            edge_factor: boost factor for box edges (in case of BCE)
            alpha: balancing factor in the focal loss formula
            gammma: modulating factor in the focal loss formula

        Returns:
            A loss tensor
        """
        seg_target, seg_mask, edge_mask = self.compute_target(target, out_map.shape[:3])
        seg_target = tf.convert_to_tensor(seg_target, dtype=out_map.dtype)
        edge_mask = tf.convert_to_tensor(seg_mask, dtype=tf.bool)
        seg_mask = tf.convert_to_tensor(seg_mask, dtype=tf.bool)

        # Get the cross_entropy for each entry
        bce = tf.keras.losses.binary_crossentropy(
            seg_target[seg_mask],
            tf.squeeze(out_map, axis=[-1])[seg_mask],
            from_logits=True)

        if focal_loss:
            if gamma and gamma < 0:
                raise ValueError("Value of gamma should be greater than or equal to zero.")

            # Convert logits to prob, compute gamma factor
            pred_prob = tf.sigmoid(tf.squeeze(out_map, axis=[-1])[seg_mask])
            p_t = (seg_target[seg_mask] * pred_prob) + ((1 - seg_target[seg_mask]) * (1 - pred_prob))
            modulating_factor = tf.pow((1.0 - p_t), gamma)

            # Compute alpha factor
            alpha_factor = seg_target[seg_mask] * alpha + (1 - seg_target[seg_mask]) * (1 - alpha)

            # compute the final loss
            loss = tf.reduce_mean(alpha_factor * modulating_factor * bce)

        else:
            # Compute BCE loss with highlighted edges
            loss = tf.math.multiply(
                1 + (edge_factor - 1) * tf.cast(edge_mask, out_map.dtype),
                bce
            )
            loss = tf.reduce_mean(loss)

        return loss

    def call(
        self,
        x: tf.Tensor,
        target: Optional[List[np.ndarray]] = None,
        return_model_output: bool = False,
        return_boxes: bool = False,
        focal_loss: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        logits = self.stem(x)
        logits = self.fpn(logits)
        logits = self.classifier(logits)

        out: Dict[str, tf.Tensor] = {}
        if return_model_output or target is None or return_boxes:
            prob_map = tf.math.sigmoid(logits)
        if return_model_output:
            out["out_map"] = prob_map

        if target is None or return_boxes:
            # Post-process boxes
            out["preds"] = self.postprocessor(tf.squeeze(prob_map, axis=-1).numpy())

        if target is not None:
            loss = self.compute_loss(logits, target, focal_loss)
            out['loss'] = loss

        return out


def _linknet(arch: str, pretrained: bool, input_shape: Tuple[int, int, int] = None, **kwargs: Any) -> LinkNet:

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']

    kwargs['input_shape'] = _cfg['input_shape']
    # Build the model
    model = LinkNet(cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, _cfg['url'])

    return model


def linknet16(pretrained: bool = False, **kwargs: Any) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import linknet16
        >>> model = linknet16(pretrained=True)
        >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _linknet('linknet16', pretrained, **kwargs)
