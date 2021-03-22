# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

import cv2
from copy import deepcopy
import numpy as np
from shapely.geometry import Polygon
import pyclipper
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Union, List, Tuple, Optional, Any, Dict

from .core import DetectionModel, DetectionPostProcessor
from ..utils import IntermediateLayerGetter, load_pretrained_params, conv_sequence
from doctr.utils.repr import NestedObject

__all__ = ['DBPostProcessor', 'DBNet', 'db_resnet50']


default_cfgs: Dict[str, Dict[str, Any]] = {
    'db_resnet50': {
        'mean': (0.798, 0.785, 0.772),
        'std': (0.264, 0.2749, 0.287),
        'backbone': 'ResNet50',
        'fpn_layers': ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"],
        'fpn_channels': 128,
        'input_shape': (1024, 1024, 3),
        'post_processor': 'DBPostProcessor',
        'url': 'https://github.com/mindee/doctr/releases/download/v0.1.0/db_resnet50-091c08a5.zip',
    },
}


class DBPostProcessor(DetectionPostProcessor):
    """Implements a post processor for DBNet adapted from the implementation of `xuannianz
    <https://github.com/xuannianz/DifferentiableBinarization>`_.

    Args:
        unclip ratio: ratio used to unshrink polygons
        min_size_box: minimal length (pix) to keep a box
        max_candidates: maximum boxes to consider in a single page
        box_thresh: minimal objectness score to consider a box
        bin_thresh: threshold used to binzarized p_map at inference time

    """
    def __init__(
        self,
        unclip_ratio: Union[float, int] = 1.5,
        min_size_box: int = 3,
        max_candidates: int = 1000,
        box_thresh: float = 0.1,
        bin_thresh: float = 0.15,
    ) -> None:

        super().__init__(
            min_size_box,
            max_candidates,
            box_thresh
        )
        self.unclip_ratio = unclip_ratio
        self.bin_thresh = bin_thresh

    @staticmethod
    def box_score(
        pred: np.ndarray,
        points: np.ndarray
    ) -> float:
        """Compute the confidence score for a polygon : mean of the p values on the polygon

        Args:
            pred (np.ndarray): p map returned by the model

        Returns:
            polygon objectness
        """
        h, w = pred.shape[:2]
        xmin = np.clip(np.floor(points[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(points[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(points[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(points[:, 1].max()).astype(np.int), 0, h - 1)

        return pred[ymin:ymax + 1, xmin:xmax + 1].mean()

    def polygon_to_box(
        self,
        points: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Expand a polygon (points) by a factor unclip_ratio, and returns a 4-points box

        Args:
            points: The first parameter.

        Returns:
            a box in absolute coordinates (x, y, w, h)
        """
        poly = Polygon(points)
        distance = poly.area * self.unclip_ratio / poly.length  # compute distance to expand polygon
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        _points = offset.Execute(distance)
        # Take biggest stack of points
        idx = 0
        if len(_points) > 1:
            max_size = 0
            for _idx, p in enumerate(_points):
                if len(p) > max_size:
                    idx = _idx
                    max_size = len(p)
            # We ensure that _points can be correctly casted to a ndarray
            _points = [_points[idx]]
        expanded_points = np.asarray(_points)  # expand polygon
        if len(expanded_points) < 1:
            return None
        x, y, w, h = cv2.boundingRect(expanded_points)  # compute a 4-points box from expanded polygon
        return x, y, w, h

    def bitmap_to_boxes(
        self,
        pred: np.ndarray,
        bitmap: np.ndarray,
    ) -> List[List[float]]:
        """Compute boxes from a bitmap/pred_map

        Args:
            pred: Pred map from differentiable binarization output
            bitmap: Bitmap map computed from pred (binarized)

        Returns:
            list of boxes for the bitmap, each box is a 5-element list
                containing x, y, w, h, score for the box
        """
        height, width = bitmap.shape[:2]
        boxes = []
        # get contours from connected components on the bitmap
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours[:self.max_candidates]:
            # Check whether smallest enclosing bounding box is not too small
            if np.any(contour[:, 0].max(axis=0) - contour[:, 0].min(axis=0) <= self.min_size_box):
                continue
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)  # approximate contour by a polygon
            points = approx.reshape((-1, 2))  # get polygon points
            if points.shape[0] < 4:  # remove polygons with 3 points or less
                continue
            score = self.box_score(pred, points.reshape(-1, 2))
            if self.box_thresh > score:   # remove polygons with a weak objectness
                continue
            _box = self.polygon_to_box(points)

            if _box is None or _box[2] < self.min_size_box or _box[3] < self.min_size_box:  # remove to small boxes
                continue
            x, y, w, h = _box
            # compute relative polygon to get rid of img shape
            xmin, ymin, xmax, ymax = x / width, y / height, (x + w) / width, (y + h) / height
            boxes.append([xmin, ymin, xmax, ymax, score])
        return np.clip(np.asarray(boxes), 0, 1) if len(boxes) > 0 else np.zeros((0, 5), dtype=np.float32)

    def __call__(
        self,
        x: tf.Tensor,
    ) -> List[np.ndarray]:
        """Performs postprocessing for a list of model outputs

        Args:
            x: raw output of the model, of shape (N, H, W, 1)

        returns:
            list of N tensors (for each input sample), with each tensor of shape (*, 5).
        """
        p = tf.squeeze(x, axis=-1)  # remove last dim
        bitmap = tf.cast(p > self.bin_thresh, tf.float32)

        p = tf.unstack(p, axis=0)
        bitmap = tf.unstack(bitmap, axis=0)

        boxes_batch = []

        for p_, bitmap_ in zip(p, bitmap):
            p_ = p_.numpy()
            bitmap_ = bitmap_.numpy()
            boxes = self.bitmap_to_boxes(pred=p_, bitmap=bitmap_)
            boxes_batch.append(boxes)

        return boxes_batch


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


class DBNet(DetectionModel, NestedObject):
    """DBNet as described in `"Real-time Scene Text Detection with Differentiable Binarization"
    <https://arxiv.org/pdf/1911.08947.pdf>`_.

    Args:
        feature extractor: the backbone serving as feature extractor
        fpn_channels: number of channels each extracted feature maps is mapped to
    """

    _children_names = ['feat_extractor', 'fpn', 'probability_head', 'threshold_head']

    def __init__(
        self,
        feature_extractor: IntermediateLayerGetter,
        fpn_channels: int = 128,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(cfg=cfg)

        self.feat_extractor = feature_extractor

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
                layers.Activation('sigmoid'),
            ]
        )
        self.threshold_head = keras.Sequential(
            [
                *conv_sequence(64, 'relu', True, kernel_size=3, input_shape=output_shape[1:]),
                layers.Conv2DTranspose(64, 2, strides=2, use_bias=False, kernel_initializer='he_normal'),
                layers.BatchNormalization(),
                layers.Activation('relu'),
                layers.Conv2DTranspose(1, 2, strides=2, kernel_initializer='he_normal'),
                layers.Activation('sigmoid'),
            ]
        )

    @staticmethod
    def compute_approx_binmap(
        p: tf.Tensor,
        t: tf.Tensor
    ) -> tf.Tensor:
        """Compute approximate binary map as described in paper,
        from threshold map t and probability map p

        Args:
            p (tf.Tensor): probability map
            t (tf.Tensor): threshold map

        Returns:
            a tf.Tensor
        """
        return 1 / (1 + tf.exp(-50. * (p - t)))

    def call(
        self,
        x: tf.Tensor,
        **kwargs: Any,
    ) -> Union[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor]:

        feat_maps = self.feat_extractor(x, **kwargs)
        feat_concat = self.fpn(feat_maps, **kwargs)
        prob_map = self.probability_head(feat_concat, **kwargs)

        if kwargs.get('training', False):
            thresh_map = self.threshold_head(feat_concat, **kwargs)
            approx_binmap = self.compute_approx_binmap(prob_map, thresh_map)
            return prob_map, thresh_map, approx_binmap

        else:
            return prob_map


def _db_resnet(arch: str, pretrained: bool, input_shape: Tuple[int, int, int] = None, **kwargs: Any) -> DBNet:

    # Patch the config
    _cfg = deepcopy(default_cfgs[arch])
    _cfg['input_shape'] = input_shape or _cfg['input_shape']
    _cfg['fpn_channels'] = kwargs.get('fpn_channels', _cfg['fpn_channels'])

    # Feature extractor
    resnet = tf.keras.applications.__dict__[_cfg['backbone']](
        include_top=False,
        weights=None,
        input_shape=_cfg['input_shape'],
        pooling=None,
    )

    feat_extractor = IntermediateLayerGetter(
        resnet,
        _cfg['fpn_layers'],
    )

    kwargs['fpn_channels'] = _cfg['fpn_channels']

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
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        text detection architecture
    """

    return _db_resnet('db_resnet50', pretrained, **kwargs)
