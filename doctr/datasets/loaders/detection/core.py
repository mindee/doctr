# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import cv2
import tensorflow as tf
import numpy as np
from typing import List, Tuple

__all__ = ["DetectionDataGenerator"]


class DataGenerator(tf.keras.utils.Sequence):
    """Data loader for detection model

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
        indexes = self.indexes[
            index * self.batch_size:min(len(os.listdir(self.images_path)), (index + 1) * self.batch_size)
        ]
        # Find list of paths
        list_paths = [os.listdir(self.images_path)[k] for k in indexes]
        # Generate data
        return self.__data_generation(list_paths)

    @staticmethod
    def load_annotation(
        labels_path: str,
        img_name: str,
    ) -> Tuple[List[List[List[int]]], List[bool]]:
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

    def __data_generation(
        self,
        list_paths: List[str],
    ) -> Tuple[tf.Tensor, List[List[List[List[float]]]], List[List[bool]]]:
        """Generate a batch of images and corresponding relative boxes (as a list of list of boxes),
        and corresponding boxes to mask

        Args:
            list_paths: list of paths to images to batch

        Returns:
            Images, boxes, boxes to mask
        """
        # Init batch lists
        batch_images, batch_polys, batch_masks = [], [], []
        for image_name in list_paths:
            image = cv2.imread(os.path.join(self.images_path, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            # Resize and batch images
            image = tf.image.resize(image, self.input_size)
            image = tf.cast(image, tf.float32)
            batch_images.append(image)

            try:
                polys, to_masks = self.load_annotation(self.labels_path, image_name)
            except ValueError:
                polys, to_masks = [], []
            # Normalize polys
            polys = [[[x / w, y / h] for [x, y] in poly] for poly in polys]
            batch_polys.append(polys)
            batch_masks.append(to_masks)
        batch_images = tf.stack(batch_images, axis=0)

        return batch_images, batch_polys, batch_masks
