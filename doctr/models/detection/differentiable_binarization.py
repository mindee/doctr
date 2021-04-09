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
        'url': 'https://github.com/mindee/doctr/releases/download/v0.1.1/db_resnet50-98ba765d.zip',
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
            box_thresh,
            bin_thresh
        )
        self.unclip_ratio = unclip_ratio
        self.max_candidates = max_candidates

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

        self.shrink_ratio = 0.4
        self.thresh_min = 0.3
        self.thresh_max = 0.7
        self.min_size_box = 3

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

        self.postprocessor = DBPostProcessor()

    @staticmethod
    def compute_distance(
        xs: np.array,
        ys: np.array,
        a: np.array,
        b: np.array,
    ) -> float:
        """Compute the distance for each point of the map (xs, ys) to the (a, b) segment

        Args:
            xs : map of x coordinates (height, width)
            ys : map of y coordinates (height, width)
            a: first point defining the [ab] segment
            b: second point defining the [ab] segment

        Returns:
            The computed distance

        """
        square_dist_1 = np.square(xs - a[0]) + np.square(ys - a[1])
        square_dist_2 = np.square(xs - b[0]) + np.square(ys - b[1])
        square_dist = np.square(a[0] - b[0]) + np.square(a[1] - b[1])
        cosin = (square_dist - square_dist_1 - square_dist_2) / (2 * np.sqrt(square_dist_1 * square_dist_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_dist_1 * square_dist_2 * square_sin / square_dist)
        result[cosin < 0] = np.sqrt(np.fmin(square_dist_1, square_dist_2))[cosin < 0]
        return result

    def draw_thresh_map(
        self,
        polygon: np.array,
        canvas: np.array,
        mask: np.array,
    ) -> None:
        """Draw a polygon treshold map on a canvas, as described in the DB paper

        Args:
            polygon : array of coord., to draw the boundary of the polygon
            canvas : threshold map to fill with polygons
            mask : mask for training on threshold polygons
            shrink_ratio : 0.4, as described in the DB paper

        """
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise AttributeError("polygon should be a 2 dimensional array of coords")

        # Augment polygon by shrink_ratio
        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(coor) for coor in polygon]  # Get coord as list of tuples
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])

        # Fill the mask with 1 on the new padded polygon
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        # Get min/max to recover polygon after distance computation
        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1
        # Get absolute polygon for distance computation
        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin
        # Get absolute padded polygon
        xs = np.broadcast_to(np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        # Compute distance map to fill the padded polygon
        distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.compute_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = np.min(distance_map, axis=0)

        # Clip the padded polygon inside the canvas
        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)

        # Fill the canvas with the distances computed inside the valid padded polygon
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin:ymax_valid - ymin + 1,
                xmin_valid - xmin:xmax_valid - xmin + 1
            ],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1]
        )

    def compute_loss(
        self,
        model_output: Dict[str, tf.Tensor],
        batch_polys: List[List[List[List[float]]]],
        batch_flags: List[List[bool]]
    ) -> tf.Tensor:
        """Compute a batch of gts, masks, thresh_gts, thresh_masks from a list of boxes
        and a list of masks for each image. From there it computes the loss with the model output

        Args:
            model_output: dictionnary containing the maps outputed by the model:
                proba_map, binary_map and thresh_map, shapes: Nx H x W x C
            batch_polys: list of boxes for each image of the batch
            batch_flags: list of boxes to mask for each image of the batch

        Returns:
            A loss tensor
        """
        # Load model outputs
        proba_map = model_output["proba_map"]
        binary_map = model_output["binary_map"]
        thresh_map = model_output["thresh_map"]

        # Compute gts and masks
        batch_size, h, w, _ = proba_map.shape
        batch_gts, batch_masks, batch_thresh_gts, batch_thresh_masks = [], [], [], []
        for batch_idx in range(batch_size):
            # Initialize mask and gt
            gt = np.zeros((h, w), dtype=np.float32)
            mask = np.ones((h, w), dtype=np.float32)
            thresh_gt = np.zeros((h, w), dtype=np.float32)
            thresh_mask = np.zeros((h, w), dtype=np.float32)

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

                # Negative shrink for gt, as described in paper
                polygon = Polygon(poly)
                distance = polygon.area * (1 - np.power(self.shrink_ratio, 2)) / polygon.length
                subject = [tuple(coor) for coor in poly]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)

                # Draw polygon on gt if it is valid
                if len(shrinked) == 0:
                    cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                if shrinked.shape[0] <= 2 or not Polygon(shrinked).is_valid:
                    cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)

                # Draw on both thresh map and thresh mask
                self.draw_thresh_map(poly, thresh_gt, thresh_mask)
            thresh_gt = thresh_gt * (self.thresh_max - self.thresh_min) + self.thresh_min

            # Cast
            gts = tf.convert_to_tensor(gt, tf.float32)
            masks = tf.convert_to_tensor(mask, tf.float32)
            thresh_gts = tf.convert_to_tensor(thresh_gt, tf.float32)
            thresh_masks = tf.convert_to_tensor(thresh_mask, tf.float32)

            # Batch
            batch_gts.append(gts)
            batch_masks.append(masks)
            batch_thresh_gts.append(thresh_gts)
            batch_thresh_masks.append(thresh_masks)

        # Stack
        gt = tf.stack(batch_gts, axis=0)
        mask = tf.stack(batch_masks, axis=0)
        thresh_gt = tf.stack(batch_thresh_gts, axis=0)
        thresh_mask = tf.stack(batch_thresh_masks, axis=0)

        # Compute balanced BCE loss for proba_map
        negative_ratio = 3.
        bce_scale = 5.
        positive_mask = (gt * mask)
        negative_mask = ((1 - gt) * mask)
        positive_count = tf.math.reduce_sum(positive_mask)
        negative_count = tf.math.reduce_min([tf.math.reduce_sum(negative_mask), positive_count * negative_ratio])
        gt_exp = tf.expand_dims(gt, axis=-1)
        bce_loss = tf.keras.losses.binary_crossentropy(gt_exp, proba_map)
        positive_loss = bce_loss * positive_mask
        negative_loss = bce_loss * negative_mask
        negative_loss, _ = tf.nn.top_k(tf.reshape(negative_loss, (-1,)), tf.cast(negative_count, tf.int32))
        sum_losses = tf.math.reduce_sum(positive_loss) + tf.math.reduce_sum(negative_loss)
        balanced_bce_loss = sum_losses / (positive_count + negative_count + 1e-6)
        balanced_bce_loss = balanced_bce_loss

        # Compute dice loss for approxbin_map
        binary_map = tf.squeeze(binary_map, axis=[-1])
        wght = bce_loss
        weights = (wght - tf.math.reduce_min(wght)) / (tf.math.reduce_max(wght) - tf.math.reduce_min(wght)) + 1.
        mask = mask * weights
        intersection = tf.math.reduce_sum(binary_map * gt * mask)
        union = tf.math.reduce_sum(binary_map * mask) + tf.math.reduce_sum(gt * mask) + 1e-8
        dice_loss = 1 - 2.0 * intersection / union

        # Compute l1 loss for thresh_map
        l1_scale = 10.
        thresh_map = tf.squeeze(thresh_map, axis=[-1])
        mask_sum = tf.math.reduce_sum(thresh_mask)
        if mask_sum == 0:
            l1_loss = tf.constant(0.)
        else:
            l1_loss = tf.math.reduce_sum(tf.math.abs(thresh_map - thresh_gt) * thresh_mask) / mask_sum

        return l1_scale * l1_loss + bce_scale * balanced_bce_loss + dice_loss

    @staticmethod
    def compute_binary_map(
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
    ) -> Dict[str, tf.Tensor]:

        feat_maps = self.feat_extractor(x, **kwargs)
        feat_concat = self.fpn(feat_maps, **kwargs)
        proba_map = self.probability_head(feat_concat, **kwargs)

        if kwargs.get('training', False):
            thresh_map = self.threshold_head(feat_concat, **kwargs)
            binary_map = self.compute_binary_map(proba_map, thresh_map)
            return dict(proba_map=proba_map, thresh_map=thresh_map, binary_map=binary_map)
        return dict(proba_map=proba_map)


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
        pretrained (bool): If True, returns a model pre-trained on our text detection dataset

    Returns:
        text detection architecture
    """

    return _db_resnet('db_resnet50', pretrained, **kwargs)
