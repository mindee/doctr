# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap


def plot_samples(images, targets: List[Dict[str, np.ndarray]], classes: List[str]) -> None:
    cmap = get_cmap("gist_rainbow", len(classes))
    # Unnormalize image
    nb_samples = min(len(images), 4)
    _, axes = plt.subplots(1, nb_samples, figsize=(20, 5))
    for idx in range(nb_samples):
        img = (255 * images[idx].numpy()).round().clip(0, 255).astype(np.uint8)
        if img.shape[0] == 3 and img.shape[2] != 3:
            img = img.transpose(1, 2, 0)
        target = img.copy()
        for box, class_idx in zip(targets[idx]["boxes"].numpy(), targets[idx]["labels"]):
            r, g, b, _ = cmap(class_idx.numpy())
            color = int(round(255 * r)), int(round(255 * g)), int(round(255 * b))
            cv2.rectangle(target, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            text_size, _ = cv2.getTextSize(classes[class_idx], cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_w, text_h = text_size
            cv2.rectangle(target, (int(box[0]), int(box[1])), (int(box[0]) + text_w, int(box[1]) - text_h), color, -1)
            cv2.putText(
                target, classes[class_idx], (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )

        axes[idx].imshow(target)
    # Disable axis
    for ax in axes.ravel():
        ax.axis("off")
    plt.show()


def plot_recorder(lr_recorder, loss_recorder, beta: float = 0.95, **kwargs) -> None:
    """Display the results of the LR grid search.
    Adapted from https://github.com/frgfm/Holocron/blob/master/holocron/trainer/core.py

    Args:
        lr_recorder: list of LR values
        loss_recorder: list of loss values
        beta (float, optional): smoothing factor
    """

    if len(lr_recorder) != len(loss_recorder) or len(lr_recorder) == 0:
        raise AssertionError("Both `lr_recorder` and `loss_recorder` should have the same length")

    # Exp moving average of loss
    smoothed_losses = []
    avg_loss = 0.0
    for idx, loss in enumerate(loss_recorder):
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_losses.append(avg_loss / (1 - beta ** (idx + 1)))

    # Properly rescale Y-axis
    data_slice = slice(
        min(len(loss_recorder) // 10, 10),
        -min(len(loss_recorder) // 20, 5) if len(loss_recorder) >= 20 else len(loss_recorder),
    )
    vals = np.array(smoothed_losses[data_slice])
    min_idx = vals.argmin()
    max_val = vals.max() if min_idx is None else vals[: min_idx + 1].max()  # type: ignore[misc]
    delta = max_val - vals[min_idx]

    plt.plot(lr_recorder[data_slice], smoothed_losses[data_slice])
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Training loss")
    plt.ylim(vals[min_idx] - 0.1 * delta, max_val + 0.2 * delta)
    plt.grid(True, linestyle="--", axis="x")
    plt.show(**kwargs)
