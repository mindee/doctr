# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import math
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Dict, Any

__all__ = ["DetectionDataset"]


class DetectionDataset:
    def __init__(
        self,
        input_size: Tuple[int, int],
        img_folder: str,
        labels_path: str,
    ) -> None:
        self.input_size = input_size
        self.root = img_folder

        self.data: List[Tuple[str, Dict[str, Any]]] = []
        for img_path in os.listdir(self.root):
            with open(os.path.join(labels_path, img_path + '.json'), 'rb') as f:
                boxes = json.load(f)

            bboxes = np.asarray(boxes["boxes_1"] + boxes["boxes_2"] + boxes["boxes_3"], dtype=np.float32)
            # Switch to xmin, ymin, xmax, ymax
            bboxes = np.concatenate((bboxes.min(axis=1), bboxes.max(axis=1)), axis=1)

            is_ambiguous = [False] * (len(boxes["boxes_1"]) + len(boxes["boxes_2"])) + [True] * len(boxes["boxes_3"])

            self.data.append((img_path, dict(boxes=bboxes, flags=np.asarray(is_ambiguous))))

    def __len__(self):
        return len(self.data)

    def __getitem__(
        self,
        index: int
    ) -> Tuple[tf.Tensor, List[np.ndarray], List[np.ndarray]]:

        img_name, target = self.data[index]
        img = tf.io.read_file(os.path.join(self.root, img_name))
        img = tf.image.decode_jpeg(img, channels=3)
        h, w = img.shape[:2]
        img = tf.image.resize(img, self.input_size, method='bilinear')

        # Boxes
        boxes = target['boxes']
        boxes[..., [0, 2]] /= w
        boxes[..., [1, 3]] /= h

        return img, boxes, target['flags']

    @staticmethod
    def collate_fn(samples):

        images, boxes, flags = zip(*samples)
        images = tf.stack(images, axis=0)

        return images, list(boxes), list(flags)
