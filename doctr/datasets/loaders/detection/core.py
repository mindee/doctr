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
        A tuple of 2 lists: a list of polygons to draw on the segmentation gt map,
        and a list of ambiguous polygons to mask during the training

    """
    with open(os.path.join(labels_path, img_name + '.json')) as f:
        labels = json.load(f)

    polys = [[[int(x), int(y)] for [x, y] in polygon] for polygon in labels["boxes_1"] + labels["boxes_2"]]
    polys_mask = [[[int(x), int(y)] for [x, y] in polygon] for polygon in labels["boxes_3"]]

    return polys, polys_mask


def resize(image, input_size):
    return cv2.resize(image, input_size, interpolation=cv2.INTER_AREA)


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
    ) -> None:
        self.input_size = input_size
        self.batch_size = batch_size
        self.images_path = images_path
        self.labels_path = labels_path
        self.shuffle = shuffle
        self.on_epoch_end()
        self.min_size_box = 3

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(os.listdir(self.images_path)) / self.batch_size))

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(os.listdir(self.images_path)))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __getitem__(
        self,
        index: int
    ) -> Tuple[tf.Tensor, np.array, np.array]:
        # Get one batch of data
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of paths
        list_paths = [os.listdir(self.images_path)[k] for k in indexes]
        # Generate data
        return self.__data_generation(list_paths)

    def __data_generation(
        self,
        list_paths: List[str],
    ) -> Tuple[tf.Tensor, np.array, np.array]:
        # Init batch arrays
        batch_images = np.empty((self.batch_size, *self.input_size, 3))
        batch_gts = np.empty((self.batch_size, *self.input_size))
        batch_masks = np.empty((self.batch_size, *self.input_size))
        for index, path in enumerate(list_paths):
            image_name = list_paths[index]
            # Load annotation for image
            try:
                polys, polys_mask = load_annotation(self.labels_path, image_name)
            except ValueError:
                mask = np.zeros(self.input_size, dtype=np.float32)
                polys, polys_mask = [], []

            image = cv2.imread(os.path.join(self.images_path, image_name))
            h, w = image.shape[:2]

            # Initialize mask and gt
            gt = np.zeros((h, w), dtype=np.float32)
            mask = np.ones((h, w), dtype=np.float32)

            # Draw each polygon on gt
            for poly in polys:
                poly = np.array(poly)
                height = max(poly[:, 1]) - min(poly[:, 1])
                width = max(poly[:, 0]) - min(poly[:, 0])
                # generate true and mask
                if min(height, width) < self.min_size_box:
                    cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                    continue
                cv2.fillPoly(gt, [poly.astype(np.int32)], 1)

            # Fill mask
            for poly in polys_mask:
                poly = np.array(poly)
                cv2.fillPoly(mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

            # Resize and batch
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, [*self.input_size], method='bilinear')
            gt = resize(gt, self.input_size)
            mask = resize(mask, self.input_size)

            batch_images[index, ] = image
            batch_gts[index, ] = gt
            batch_masks[index, ] = mask

        # Normalize
        batch_images = tf.stack(batch_images, axis=0)
        std = tf.cast((0.287, 0.274, 0.264), tf.float32)
        mean = tf.cast((0.772, 0.785, 0.798), tf.float32)
        batch_images = tf.cast(batch_images, tf.float32) * (std / 255) - (mean / std)

        return batch_images, batch_gts, batch_masks
