# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

from copy import deepcopy
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import layers, Sequential
from typing import Dict, Any, Tuple, Optional, List

from .core import DetectionModel, DetectionPostProcessor
from ..backbones import ResnetStage
from ..utils import conv_sequence, load_pretrained_params
from ...utils.repr import NestedObject

__all__ = ['LinkNet', 'linknet', 'LinkNetPostProcessor']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'linknet': {
        'mean': (0.798, 0.785, 0.772),
        'std': (0.264, 0.2749, 0.287),
        'out_chan': 1,
        'input_shape': (1024, 1024, 3),
        'post_processor': 'LinkNetPostProcessor',
        'url': None,
    },
}


class LinkNetPostProcessor(DetectionPostProcessor):
    """Implements a post processor for LinkNet model.

    Args:
        min_size_box: minimal length (pix) to keep a box
        box_thresh: minimal objectness score to consider a box
        bin_thresh: threshold used to binzarized p_map at inference time

    """
    def __init__(
        self,
        min_size_box: int = 3,
        bin_thresh: float = 0.15,
        box_thresh: float = 0.1,
    ) -> None:
        super().__init__(
            box_thresh,
            bin_thresh
        )

    def bitmap_to_boxes(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
    ) -> np.ndarray:
        """Compute boxes from a bitmap/pred_map: find connected components then filter boxes

        Args:
            pred: Pred map from differentiable linknet output
            bitmap: Bitmap map computed from pred (binarized)

        Returns:
            np tensor boxes for the bitmap, each box is a 5-element list
                containing x, y, w, h, score for the box
        """
        label_num, labelimage = cv2.connectedComponents(bitmap.astype(np.uint8), connectivity=4)
        height, width = bitmap.shape[:2]
        min_size_box = 1 + int(height / 512)
        boxes = []
        for label in range(1, label_num + 1):
            points = np.array(np.where(labelimage == label)[::-1]).T
            if points.shape[0] < 4:  # remove polygons with 3 points or less
                continue
            score = self.box_score(pred, points.reshape(-1, 2))
            if self.box_thresh > score:   # remove polygons with a weak objectness
                continue
            x, y, w, h = cv2.boundingRect(points)
            if min(w, h) < min_size_box:  # filter too small boxes
                continue
            # compute relative polygon to get rid of img shape
            xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
            boxes.append([xmin, ymin, xmax, ymax, score])
        return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5), dtype=np.float32)


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
    """LinkNet Encoder-Decoder module

    """

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


class LinkNet(DetectionModel, NestedObject):
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    Args:
        out_chan: number of channels for the output
    """

    _children_names: List[str] = ['stem', 'fpn', 'classifier', 'postprocessor']

    def __init__(
        self,
        out_chan: int = 1,
        input_shape: Tuple[int, int, int] = (512, 512, 3),
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(cfg=cfg)

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
                filters=out_chan,
                kernel_size=2,
                strides=2,
                padding="same",
                use_bias=False,
                kernel_initializer='he_normal'
            ),
        ])

        self.min_size_box = 3

        self.postprocessor = LinkNetPostProcessor()

    def compute_target(
        self,
        target: List[Dict[str, Any]],
        output_shape: Tuple[int, int, int],
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        seg_target = np.zeros(output_shape, dtype=np.bool)
        seg_mask = np.ones(output_shape, dtype=np.bool)

        for idx, _target in enumerate(target):
            # Draw each polygon on gt
            if _target['boxes'].shape[0] == 0:
                # Empty image, full masked
                seg_mask[idx] = False

            # Absolute bounding boxes
            abs_boxes = _target['boxes'].copy()
            abs_boxes[:, [0, 2]] *= output_shape[-1]
            abs_boxes[:, [1, 3]] *= output_shape[-2]
            abs_boxes = abs_boxes.round().astype(np.int32)

            boxes_size = np.minimum(abs_boxes[:, 2] - abs_boxes[:, 0], abs_boxes[:, 3] - abs_boxes[:, 1])

            for box, box_size, is_ambiguous in zip(abs_boxes, boxes_size, _target['flags']):
                # Mask ambiguous boxes
                if is_ambiguous:
                    seg_mask[idx, box[1]: box[3] + 1, box[0]: box[2] + 1] = False
                    continue
                # Mask boxes that are too small
                if box_size < self.min_size_box:
                    seg_mask[idx, box[1]: box[3] + 1, box[0]: box[2] + 1] = False
                    continue
                # Fill polygon with 1
                seg_target[idx, box[1]: box[3] + 1, box[0]: box[2] + 1] = True

        seg_target = tf.convert_to_tensor(seg_target, dtype=tf.float32)
        seg_mask = tf.convert_to_tensor(seg_mask, dtype=tf.bool)

        return seg_target, seg_mask

    def compute_loss(
        self,
        out_map: tf.Tensor,
        target: List[Dict[str, Any]]
    ) -> tf.Tensor:
        """Compute a batch of gts and masks from a list of boxes and a list of masks for each image
        Then, it computes the loss function with proba_map, gts and masks

        Args:
            out_map: output feature map of the model of shape N x H x W x 1
            target: list of dictionary where each dict has a `boxes` and a `flags` entry

        Returns:
            A loss tensor
        """
        seg_target, seg_mask = self.compute_target(target, out_map.shape[:3])

        # Compute BCE loss
        return tf.math.reduce_mean(tf.keras.losses.binary_crossentropy(
            seg_target[seg_mask],
            tf.squeeze(out_map, axis=[-1])[seg_mask],
            from_logits=True
        ))

    def call(
        self,
        x: tf.Tensor,
        target: Optional[List[Dict[str, Any]]] = None,
        return_model_output: bool = False,
        return_boxes: bool = False,
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
            out["boxes"] = self.postprocessor(prob_map)

        if target is not None:
            loss = self.compute_loss(logits, target)
            out['loss'] = loss

        return out


def _linknet(arch: str, pretrained: bool, input_shape: Tuple[int, int, int] = None, **kwargs: Any) -> LinkNet:

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']
    _cfg['out_chan'] = kwargs.get('out_chan', _cfg['out_chan'])

    kwargs['out_chan'] = _cfg['out_chan']
    kwargs['input_shape'] = _cfg['input_shape']
    # Build the model
    model = LinkNet(cfg=_cfg, **kwargs)
    # Load pretrained parameters
    if pretrained:
        load_pretrained_params(model, _cfg['url'])

    return model


def linknet(pretrained: bool = False, **kwargs: Any) -> LinkNet:
    """LinkNet as described in `"LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation"
    <https://arxiv.org/pdf/1707.03718.pdf>`_.

    Example::
        >>> import tensorflow as tf
        >>> from doctr.models import linknet
        >>> model = linknet(pretrained=True)
        >>> input_tensor = tf.random.uniform(shape=[1, 1024, 1024, 3], maxval=1, dtype=tf.float32)
        >>> out = model(input_tensor)

    Args:
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _linknet('linknet', pretrained, **kwargs)
