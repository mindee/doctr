# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict


def plot_samples(images, targets: List[Dict[str, np.ndarray]]) -> None:
    # Unnormalize image
    nb_samples = 4
    _, axes = plt.subplots(2, nb_samples, figsize=(20, 5))
    for idx in range(nb_samples):
        img = (255 * images[idx].numpy()).round().clip(0, 255).astype(np.uint8)
        if img.shape[0] == 3 and img.shape[2] != 3:
            img = img.transpose(1, 2, 0)

        target = np.zeros(img.shape[:2], np.uint8)
        boxes = targets[idx]['boxes'][np.logical_not(targets[idx]['flags'])]
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * img.shape[1]
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * img.shape[0]
        boxes[:, :4] = boxes[:, :4].round().astype(int)

        for box in boxes:
            if boxes.shape[1] == 5:
                box = cv2.boxPoints(((int(box[0]), int(box[1])), (int(box[2]), int(box[3])), -box[4]))
                box = np.int0(box)
                cv2.fillPoly(target, [box], 1)
            else:
                target[int(box[1]): int(box[3]) + 1, int(box[0]): int(box[2]) + 1] = 1

        axes[0][idx].imshow(img)
        axes[0][idx].axis('off')
        axes[1][idx].imshow(target.astype(bool))
        axes[1][idx].axis('off')
    plt.show()
