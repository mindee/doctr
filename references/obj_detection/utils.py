# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap


def plot_samples(images, targets: List[Dict[str, np.ndarray]], num_classes, alpha=0.5) -> None:
    cmap = get_cmap('gist_rainbow', num_classes)
    # Unnormalize image
    nb_samples = min(len(images), 4)
    _, axes = plt.subplots(1, nb_samples, figsize=(20, 5))
    for idx in range(nb_samples):
        img = (255 * images[idx].numpy()).round().clip(0, 255).astype(np.uint8)
        target = img.copy()
        if img.shape[0] == 3 and img.shape[2] != 3:
            img = img.transpose(1, 2, 0)
        boxes = targets[idx]['boxes'].numpy().copy()
        for box, class_idx in zip(boxes, targets[idx]['labels']):
            r, g, b, _ = cmap(class_idx.numpy())
            color = int(round(255 * r)), int(round(255 * g)), int(round(255 * b))
            ke = {'1': 'qr code', '2': 'bar code', '3': 'logo', '4': 'photo'}
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)
            text_size, _ = cv2.getTextSize(ke[str(class_idx)], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_w, text_h = text_size
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[0]) + text_w, int(box[1]) - text_h), color, -1)
            cv2.putText(img, ke[str(class_idx)], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, )

        axes[idx].imshow(img)
        axes[idx].imshow(((1 - alpha) * img + alpha * target).round().astype(np.uint8))
    # Disable axis
    for ax in axes.ravel():
        ax.axis('off')
    plt.show()
