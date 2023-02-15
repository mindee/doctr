# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import pickle
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_samples(images, targets: List[Dict[str, np.ndarray]]) -> None:
    # Unnormalize image
    nb_samples = min(len(images), 4)
    _, axes = plt.subplots(2, nb_samples, figsize=(20, 5))
    for idx in range(nb_samples):
        img = (255 * images[idx].numpy()).round().clip(0, 255).astype(np.uint8)
        if img.shape[0] == 3 and img.shape[2] != 3:
            img = img.transpose(1, 2, 0)

        target = np.zeros(img.shape[:2], np.uint8)
        tgts = targets[idx].copy()
        for key, boxes in tgts.items():
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * img.shape[1]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * img.shape[0]
            boxes[:, :4] = boxes[:, :4].round().astype(int)

            for box in boxes:
                if boxes.ndim == 3:
                    cv2.fillPoly(target, [np.int0(box)], 1)
                else:
                    target[int(box[1]) : int(box[3]) + 1, int(box[0]) : int(box[2]) + 1] = 1
        if nb_samples > 1:
            axes[0][idx].imshow(img)
            axes[1][idx].imshow(target.astype(bool))
        else:
            axes[0].imshow(img)
            axes[1].imshow(target.astype(bool))

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
        # -min(len(loss_recorder) // 20, 5) if len(loss_recorder) >= 20 else len(loss_recorder)
        len(loss_recorder),
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


def load_backbone(model, weights_path):
    pretrained_backbone_weights = pickle.load(open(weights_path, "rb"))
    model.feat_extractor.set_weights(pretrained_backbone_weights[0])
    model.fpn.set_weights(pretrained_backbone_weights[1])
    return model
