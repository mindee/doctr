# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import math

import matplotlib.pyplot as plt
import numpy as np


def plot_samples(images, targets):
    # Unnormalize image
    num_samples = min(len(images), 12)
    num_rows = min(len(images), 3)
    num_cols = int(math.ceil(num_samples / num_rows))
    _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5))
    for idx in range(num_samples):
        img = (255 * images[idx].numpy()).round().clip(0, 255).astype(np.uint8)
        if img.shape[0] == 3 and img.shape[2] != 3:
            img = img.transpose(1, 2, 0)

        row_idx = idx // num_cols
        col_idx = idx % num_cols

        axes[row_idx][col_idx].imshow(img)
        axes[row_idx][col_idx].axis('off')
        axes[row_idx][col_idx].set_title(targets[idx])
    plt.show()
