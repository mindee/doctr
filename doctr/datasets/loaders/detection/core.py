# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import cv2
import tensorflow as tf
import numpy as np
from typing import List, Tuple


def load_annotation(
    labels_path: str,
    img_name: str,
) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
    """Loads detection annotations (boxes) for an image, from a folder containing
    annotations in json files for each image

    Agrs:
        labels_path: path to the folder containing all json annotations
        img_name: name of the image to find the corresponding json annotation

    Returns:
        A tuple of 2 lists: a list of polygons and a boolean vector to mask suspicious polygons
    """
    with open(os.path.join(labels_path, img_name + '.json')) as f:
        labels = json.load(f)

    polys = [
        [[int(x), int(y)] for [x, y] in poly] for poly in labels["boxes_1"] + labels["boxes_2"] + labels["boxes_3"]
    ]
    to_masks = [False] * (len(labels["boxes_1"]) + len(labels["boxes_2"])) + [True] * len(labels["boxes_3"])

    return polys, to_masks


class DataGenerator(tf.keras.utils.Sequence):
    """Data loader for Differentiable Binarization detection model

    Args:
        input_size: size (h, w) for the images
        images_path: path to the images folder
        labels_path: pathe to the folder containing json label for each image
        batch_size: batch size to train on
        suffle: if True, dataset is shuffled between each epoch

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
        self.input_size = input_size
        self.batch_size = batch_size
        self.images_path = images_path
        self.labels_path = labels_path
        self.shuffle = shuffle
        self.on_epoch_end()
        self.min_size_box = 3
        self.std = tf.cast(std_rgb, tf.float32)
        self.mean = tf.cast(std_mean, tf.float32)

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.ceil(len(os.listdir(self.images_path)) / self.batch_size))

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(os.listdir(self.images_path)))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __getitem__(
        self,
        index: int
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Get one batch of data
        indexes = self.indexes[index * self.batch_size:min(self.__len__(), (index + 1) * self.batch_size)]
        # Find list of paths
        list_paths = [os.listdir(self.images_path)[k] for k in indexes]
        # Generate data
        return self.__data_generation(list_paths)

    def __data_generation(
        self,
        list_paths: List[str],
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Init batch arrays
        batch_images, batch_gts, batch_masks = [], [], []
        for index, path in enumerate(list_paths):
            image_name = list_paths[index]
            # Load annotation for image
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
                cv2.fillPoly(gt, [poly.astype(np.int32)], 1)

            # Cast
            image = tf.cast(image, tf.float32)
            gt = tf.cast(gt, tf.float32)
            mask = tf.cast(mask, tf.float32)
            # Resize
            image = tf.image.resize(image, [*self.input_size], method='bilinear')
            gt = tf.image.resize(tf.expand_dims(gt, -1), [*self.input_size], method='bilinear')
            mask = tf.image.resize(tf.expand_dims(mask, -1), [*self.input_size], method='bilinear')
            # Batch
            batch_images.append(image)
            batch_gts.append(gt)
            batch_masks.append(mask)

        # Stack batches
        batch_images = tf.stack(batch_images, axis=0)
        batch_gts = tf.stack(batch_gts, axis=0)
        batch_masks = tf.stack(batch_masks, axis=0)
        # Normalize
        batch_images = tf.cast(batch_images, tf.float32) * (self.std / 255) - (self.mean / self.std)

        return batch_images, batch_gts, batch_masks
