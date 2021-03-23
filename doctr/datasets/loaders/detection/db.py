# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import pyclipper
import cv2
import os
import tensorflow as tf
import numpy as np
from shapely.geometry import Polygon
from typing import List, Tuple

from .core import DataGenerator, load_annotation

__all__ = ["DBGenerator"]


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
    polygon: np.array,
    canvas: np.array,
    mask: np.array,
    shrink_ratio: float = 0.4,
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
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
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
        absolute_distance = compute_distance(xs, ys, polygon[i], polygon[j])
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


class DBGenerator(DataGenerator):
    """Data loader for Differentiable Binarization detection model

    Args:
        input_size: size (h, w) for the images
        images_path: path to the images folder
        labels_path: pathe to the folder containing json label for each image
        batch_size: batch size to train on
        suffle: if True, dataset is shuffled between each epoch
        std_rgb: to normalize dataset
        std_mean: to normalize dataset

    """
    def __init__(
        self,
        input_size: Tuple[int, int],
        images_path: str,
        labels_path: str,
        batch_size: int = 1,
        shuffle: bool = True,
        std_rgb: Tuple[float, float, float] = (0.264, 0.274, 0.287),
        std_mean: Tuple[float, float, float] = (0.798, 0.785, 0.772),
    ) -> None:
        super().__init__(
            input_size,
            images_path,
            labels_path,
            batch_size,
            shuffle,
            std_rgb,
            std_mean,
        )
        self.shrink_ratio = 0.4
        self.thresh_min = 0.3
        self.thresh_max = 0.7

    def __getitem__(
        self,
        index: int
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # Get one batch of data
        indexes = self.indexes[index * self.batch_size:min(self.__len__(), (index + 1) * self.batch_size)]
        # Find list of paths
        list_paths = [os.listdir(self.images_path)[k] for k in indexes]
        # Generate data
        return self.__data_generation(list_paths)

    def __data_generation(
        self,
        list_paths: List[str],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # Init batch arrays
        batch_images, batch_gts, batch_masks, batch_thresh_gts, batch_thresh_masks = [], [], [], [], []
        for image_name in list_paths:
            try:
                polys, to_masks = load_annotation(self.labels_path, image_name)
            except ValueError:
                mask = np.zeros(self.input_size, dtype=np.float32)
                polys, to_masks = [], []

            image = cv2.imread(os.path.join(self.images_path, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            # Initialize mask and gt
            gt = np.zeros((h, w), dtype=np.float32)
            mask = np.ones((h, w), dtype=np.float32)
            thresh_gt = np.zeros((h, w), dtype=np.float32)
            thresh_mask = np.zeros((h, w), dtype=np.float32)

            # Draw each polygon on gt
            for poly, to_mask in zip(polys, to_masks):
                poly = np.array(poly)
                if to_mask is True:
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
                if len(shrinked) == 0:
                    cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                if shrinked.shape[0] <= 2 or not Polygon(shrinked).is_valid:
                    cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
                # Draw on both thresh map and thresh mask
                draw_thresh_map(poly, thresh_gt, thresh_mask, shrink_ratio=self.shrink_ratio)
            thresh_gt = thresh_gt * (self.thresh_max - self.thresh_min) + self.thresh_min

            # Cast
            image = tf.cast(image, tf.float32)
            gt = tf.cast(gt, tf.float32)
            mask = tf.cast(mask, tf.float32)
            thresh_gt = tf.cast(thresh_gt, tf.float32)
            thresh_mask = tf.cast(thresh_mask, tf.float32)
            # Resize
            image = tf.image.resize(image, [*self.input_size], method='bilinear')
            gt = tf.image.resize(tf.expand_dims(gt, -1), [*self.input_size], method='bilinear')
            mask = tf.image.resize(tf.expand_dims(mask, -1), [*self.input_size], method='bilinear')
            thresh_gt = tf.image.resize(tf.expand_dims(thresh_gt, -1), [*self.input_size], method='bilinear')
            thresh_mask = tf.image.resize(tf.expand_dims(thresh_mask, -1), [*self.input_size], method='bilinear')
            # Batch
            batch_images.append(image)
            batch_gts.append(gt)
            batch_masks.append(mask)
            batch_thresh_gts.append(thresh_gt)
            batch_thresh_masks.append(thresh_mask)

        # Stack batches
        batch_images = tf.stack(batch_images, axis=0)
        batch_gts = tf.stack(batch_gts, axis=0)
        batch_masks = tf.stack(batch_masks, axis=0)
        batch_thresh_gts = tf.stack(batch_thresh_gts, axis=0)
        batch_thresh_masks = tf.stack(batch_thresh_masks, axis=0)
        # Normalize
        batch_images = tf.cast(batch_images, tf.float32) * (self.std / 255) - (self.mean / self.std)

        return batch_images, batch_gts, batch_masks, batch_thresh_gts, batch_thresh_masks
