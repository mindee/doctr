# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import cv2
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict, Any

__all__ = ["DetectionDataGenerator"]


class DetectionDataGenerator(tf.keras.utils.Sequence):
    """Data loader for detection model

    Args:
        input_size: size (h, w) for the images
        images_path: path to the images folder
        labels_path: pathe to the folder containing json label for each image
        batch_size: batch size to train on
        shuffle: if True, dataset is shuffled between each epoch

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

        self.data: List[Tuple[str, Dict[str, Any]]] = []
        for img_path in os.listdir(self.images_path):
            bboxes, flags = self.load_target(img_path)
            self.data.append((img_path, dict(boxes=bboxes, flags=flags)))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.ceil(len(self.data) / self.batch_size))

    def on_epoch_end(self):
        # Updates indices after each epoch
        self.indices = np.arange(len(self.data))
        if self.shuffle is True:
            np.random.shuffle(self.indices)

    def __getitem__(
        self,
        index: int
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Get one batch of data
        indices = self.indices[
            index * self.batch_size: min(len(self.data), (index + 1) * self.batch_size)
        ]
        # Find list of paths
        samples = [self.data[k] for k in indices]
        # Generate data
        return self.__data_generation(samples)

    def load_target(
        self,
        img_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads detection annotations (boxes) for an image, from a folder containing
        annotations in json files for each image

        Args:
            img_name: name of the image to find the corresponding json annotation

        Returns:
            A tuple of 2 lists: a list of polygons and a boolean vector to mask suspicious polygons
        """
        with open(os.path.join(self.labels_path, img_name + '.json')) as f:
            boxes = json.load(f)

        bboxes = np.asarray(boxes["boxes_1"] + boxes["boxes_2"] + boxes["boxes_3"], dtype=np.float32)
        # Switch to xmin, ymin, xmax, ymax
        bboxes = np.concatenate((bboxes.min(axis=1), bboxes.max(axis=1)), axis=1)

        is_ambiguous = [False] * (len(boxes["boxes_1"]) + len(boxes["boxes_2"])) + [True] * len(boxes["boxes_3"])

        return bboxes, np.asarray(is_ambiguous)

    def __data_generation(
        self,
        samples: List[Tuple[str, Dict[str, Any]]],
    ) -> Tuple[tf.Tensor, List[List[List[List[float]]]], List[List[bool]]]:
        """Generate a batch of images and corresponding relative boxes (as a list of list of boxes),
        and corresponding boxes to mask

        Args:
            list_paths: list of paths to images to batch

        Returns:
            Images, boxes, boxes to mask
        """

        # # Init batch lists
        batch_images, batch_boxes, batch_flags = [], [], []
        for img_name, target in samples:
            image = cv2.imread(os.path.join(self.images_path, img_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            # Resize and batch images
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, self.input_size)
            batch_images.append(image)

            # Switch to relative coords
            boxes = target['boxes']
            boxes[..., [0, 2]] /= w
            boxes[..., [1, 3]] /= h
            batch_boxes.append(boxes)
            batch_flags.append(target['flags'])
        batch_images = tf.stack(batch_images, axis=0)

        return batch_images, batch_boxes, batch_flags
