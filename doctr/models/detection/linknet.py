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
        self.min_size_box = min_size_box
        self.bin_thresh = bin_thresh
        self.box_thresh = box_thresh

    @staticmethod
    def box_score(
        pred: np.ndarray,
        points: np.ndarray
    ) -> float:
        """Compute the confidence score for a polygon : mean of the p values on the polygon

        Args:
            pred (np.ndarray): p map returned by the model
            points: array of x, y coordinates

        Returns:
            polygon objectness
        """
        h, w = pred.shape[:2]
        xmin = np.clip(np.floor(points[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(points[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(points[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(points[:, 1].max()).astype(np.int), 0, h - 1)

        return pred[ymin:ymax + 1, xmin:xmax + 1].mean()

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
        boxes = []
        for label in range(1, label_num + 1):
            points = np.array(np.where(labelimage == label)[::-1]).T
            if points.shape[0] < 4:  # remove polygons with 3 points or less
                continue
            score = self.box_score(pred, points.reshape(-1, 2))
            if self.box_thresh > score:   # remove polygons with a weak objectness
                continue
            x, y, w, h = cv2.boundingRect(points)
            if min(w, h) < self.min_size_box:  # filter too small boxes
                continue
            # compute relative polygon to get rid of img shape
            xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
            boxes.append([xmin, ymin, xmax, ymax, score])
        return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5), dtype=np.float32)

    def __call__(
        self,
        x: Dict[str, tf.Tensor],
    ) -> List[np.ndarray]:
        """Performs postprocessing for a dict of model outputs

        Args:
            x: dictionary of the model output

        returns:
            list of N tensors (for each input sample), with each tensor of shape (*, 5).
        """
        p = x["proba_map"]
        p = tf.squeeze(p, axis=-1)  # remove last dim
        bitmap = tf.cast(p > self.bin_thresh, tf.float32)

        p = tf.unstack(p, axis=0)
        bitmap = tf.unstack(bitmap, axis=0)

        boxes_batch = []

        for p_, bitmap_ in zip(p, bitmap):
            p_ = p_.numpy()
            bitmap_ = bitmap_.numpy()
            # perform opening (erosion + dilatation)
            kernel = np.ones((3, 3), np.uint8)
            bitmap_ = cv2.morphologyEx(bitmap_, cv2.MORPH_OPEN, kernel)
            boxes = self.bitmap_to_boxes(pred=p_, bitmap=bitmap_)
            boxes_batch.append(boxes)

        return boxes_batch


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

    def compute_loss(
        self,
        model_output: Dict[str, tf.Tensor],
        batch_polys: List[List[List[List[float]]]],
        batch_flags: List[List[bool]]
    ) -> tf.Tensor:
        """Compute a batch of gts and masks from a list of boxes and a list of masks for each image
        Then, it computes the loss function with proba_map, gts and masks

        Args:
            model_output: dictionary containing the output of the model, proba_map, shape: N x H x W x C
            batch_polys: list of boxes for each image of the batch
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
            if batch_polys[batch_idx] == batch_flags[batch_idx] == []:
                # Empty image, full masked
                mask = np.zeros((h, w), dtype=np.float32)
            for poly, flag in zip(batch_polys[batch_idx], batch_flags[batch_idx]):
                # Convert polygon to absolute polygon and to np array
                poly = [[int(w * x), int(h * y)] for [x, y] in poly]
                poly = np.array(poly)
                if flag is True:
                    cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                height = max(poly[:, 1]) - min(poly[:, 1])
                width = max(poly[:, 0]) - min(poly[:, 0])
                if min(height, width) < self.min_size_box:
                    cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                # Fill polygon with 1
                cv2.fillPoly(gt, [poly.astype(np.int32)], 1)

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
