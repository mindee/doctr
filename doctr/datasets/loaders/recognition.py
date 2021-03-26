# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import tensorflow as tf
import numpy as np
import cv2
from typing import Tuple, List

__all__ = ["RecognitionDataGenerator"]


class RecognitionDataGenerator(tf.keras.utils.Sequence):
    """Data loader for recognition model

    Args:
        input_size: size (h, w) for the images
        images_path: path to the images folder
        labels_path: pathe to the json file containing all labels (character sequences)
        batch_size: batch size to train on
        suffle: if True, dataset is shuffled between each epoch

    """
    def __init__(
        self,
        input_size: Tuple[int, int],
        images_path: str,
        labels_path: str,
        batch_size: int = 64,
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

    def load_annotation(
        self,
        img_name: str,
    ) -> str:
        """Loads recognition annotation (character sequence) for an image from a json file

        Args:
            labels_path: path to the json file containing annotations
            img_name: name of the image to find the corresponding label

        Returns:
            A string (character sequence)
        """
        with open(self.labels_path) as f:
            labels = json.load(f)

        if img_name not in labels.keys():
            raise AttributeError("Image is not in referenced in label file")

        return labels[img_name]

    def __data_generation(
        self,
        list_paths: List[str],
    ) -> Tuple[tf.Tensor, List[str]]:
        # Init batch lists
        batch_images, batch_gts, = [], []
        for image_name in list_paths:
            gt = self.load_annotation(image_name)
            image = cv2.imread(os.path.join(self.images_path, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Cast, resize
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, [*self.input_size], method='bilinear')
            # Batch
            batch_images.append(image)
            batch_gts.append(gt)
        batch_images = tf.stack(batch_images, axis=0)

        return batch_images, batch_gts
