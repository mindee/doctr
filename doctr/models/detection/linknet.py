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


class DecoderBlock(Sequential):
    """This class implements a Linknet decoder block as described in paper/

    Args:
        in_chan: input channels (must be a multiple of 4)
        out_chan: output channels

    """
    def __init__(
        self,
        in_chan: int,
        out_chan: int,
    ) -> None:
        _layers = []
        _layers.extend(conv_sequence(in_chan // 4, 'relu', True, kernel_size=1))
        _layers.append(
            layers.Conv2DTranspose(
                filters=in_chan // 4,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                kernel_initializer='he_normal'
            )
        )
        _layers.append(layers.BatchNormalization())
        _layers.append(layers.Activation('relu'))
        _layers.extend(conv_sequence(out_chan, 'relu', True, kernel_size=1))
        super().__init__(_layers)


class LinkNetPyramid(layers.Layer, NestedObject):
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
        self.decoder_1 = DecoderBlock(in_chan=64, out_chan=64)
        self.decoder_2 = DecoderBlock(in_chan=128, out_chan=64)
        self.decoder_3 = DecoderBlock(in_chan=256, out_chan=128)
        self.decoder_4 = DecoderBlock(in_chan=512, out_chan=256)

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

    def __init__(
        self,
        out_chan: int = 1,
        input_shape: Tuple[int, int, int] = (512, 512, 3),
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(cfg=cfg)
        _layers = []
        _layers.extend(conv_sequence(64, 'relu', True, strides=2, kernel_size=7, input_shape=input_shape))
        _layers.append(layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same'))
        _layers.append(LinkNetPyramid())
        _layers.append(
            layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                kernel_initializer='he_normal'
            )
        )
        _layers.append(layers.BatchNormalization())
        _layers.append(layers.Activation('relu'))
        _layers.extend(conv_sequence(32, 'relu', True, strides=1, kernel_size=3))
        _layers.append(
            layers.Conv2DTranspose(
                filters=out_chan,
                kernel_size=2,
                strides=2,
                padding="same",
                use_bias=False,
                kernel_initializer='he_normal'
            )
        )
        _layers.append(layers.Activation('sigmoid'))
        self._layers = _layers
        self.min_size_box = 3

        self.postprocessor = LinkNetPostProcessor()

    def compute_loss(
        self,
        model_output: Dict[str, tf.Tensor],
        batch_boxes: List[np.array],
        batch_flags: List[List[bool]]
    ) -> tf.Tensor:
        """Compute a batch of gts and masks from a list of boxes and a list of masks for each image
        Then, it computes the loss function with proba_map, gts and masks

        Args:
            model_output: dictionary containing the output of the model, proba_map, shape: N x H x W x C
            batch_boxes: list of boxes for each image of the batch
            batch_flags: list of boxes to mask for each image of the batch

        Returns:
            A loss tensor
        """
        # Get model output
        proba_map = model_output["proba_map"]
        batch_size, h, w, _ = proba_map.shape

        # Compute masks and gts
        batch_gts, batch_masks = [], []
        for batch_idx in range(batch_size):
            # Initialize mask and gt
            gt = np.zeros((h, w), dtype=np.float32)
            mask = np.ones((h, w), dtype=np.float32)

            # Draw each polygon on gt
            if batch_boxes[batch_idx].shape[0] == 0:
                # Empty image, full masked
                mask = np.zeros((h, w), dtype=np.float32)

            # Absolute bounding boxes
            abs_boxes = batch_boxes[batch_idx].copy()
            abs_boxes[:, [0, 2]] *= w
            abs_boxes[:, [1, 3]] *= h
            abs_boxes = abs_boxes.astype(np.int32)

            boxes_size = np.minimum(abs_boxes[:, 2] - abs_boxes[:, 0], abs_boxes[:, 3] - abs_boxes[:, 1])

            for box, box_size, flag in zip(abs_boxes, boxes_size, batch_flags[batch_idx]):
                if flag is True:
                    mask[box[1]: box[3] + 1, box[0]: box[2] + 1] = 0
                    continue
                if box_size < self.min_size_box:
                    mask[box[1]: box[3] + 1, box[0]: box[2] + 1] = 0
                    continue
                # Fill polygon with 1
                gt[box[1]: box[3] + 1, box[0]: box[2] + 1] = 1

            # Cast
            gts = tf.convert_to_tensor(gt, tf.float32)
            masks = tf.convert_to_tensor(mask, tf.float32)

            # Batch
            batch_gts.append(gts)
            batch_masks.append(masks)

        # Stack
        gts = tf.stack(batch_gts, axis=0)
        masks = tf.stack(batch_masks, axis=0)

        proba_map = tf.squeeze(proba_map, axis=[-1])

        # Compute BCE loss
        bce_loss = tf.keras.losses.binary_crossentropy(gts * mask, proba_map * masks)

        return bce_loss

    def call(
        self,
        x: tf.Tensor,
        **kwargs: Any,
    ) -> Dict[str, tf.Tensor]:
        return dict(proba_map=Sequential(self._layers)(x))


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
