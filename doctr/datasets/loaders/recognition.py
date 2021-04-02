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
        img_folder: path to the images folder
        labels_path: pathe to the json file containing all labels (character sequences)
        batch_size: batch size to train on
        suffle: if True, dataset is shuffled between each epoch

    """
    def __init__(
        self,
        input_size: Tuple[int, int],
        img_folder: str,
        labels_path: str,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> None:
        self.input_size = input_size
        self.batch_size = batch_size
        self.root = img_folder
        self.shuffle = shuffle

        self.data: List[Tuple[str, Dict[str, Any]]] = []
        with open(self.labels_path) as f:
            labels = json.load(f)
        for img_path in os.listdir(self.root):
            label = labels.get(img_path)
            if not isinstance(label, str):
                raise KeyError("Image is not in referenced in label file")
            self.data.append((img_path, label))
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
    ) -> Tuple[tf.Tensor, List[str]]:
        # Get one batch of data
        indices = self.indices[
            index * self.batch_size:min(len(self.data), (index + 1) * self.batch_size)
        ]
        # Find list of paths
        samples = [self.data[k] for k in indices]
        # Generate data
        return self.__data_generation(samples)

    def __data_generation(
        self,
        samples: List[Tuple[str, str]],
    ) -> Tuple[tf.Tensor, List[str]]:
        # Init batch lists
        batch_images, batch_labels = [], []
        for img_name, label in samples:
            image = cv2.imread(os.path.join(self.root, img_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Cast, resize
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, [*self.input_size], method='bilinear')
            # Batch
            batch_images.append(image)
            batch_gts.append(label)
        batch_images = tf.stack(batch_images, axis=0)

        return batch_images, batch_gts
