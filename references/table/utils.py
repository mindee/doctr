# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_samples(images: list[Any], targets: list[dict[str, np.ndarray]], max_samples: int = 4) -> None:
    """Display a few training samples with their ground-truth cells overlaid."""
    nb_samples = min(len(images), max_samples)
    _, axes = plt.subplots(2, nb_samples, figsize=(20, 6))
    if nb_samples == 1:
        axes = np.expand_dims(axes, axis=1)

    for idx in range(nb_samples):
        img = (255 * images[idx].detach().cpu().numpy()).round().clip(0, 255).astype(np.uint8)
        if img.shape[0] == 3 and img.shape[2] != 3:
            img = img.transpose(1, 2, 0)

        axes[0][idx].imshow(img)
        axes[0][idx].set_title("Image")

        overlay = img.copy()
        cells = targets[idx]["cells"].copy()
        cells[..., 0] *= img.shape[1]
        cells[..., 1] *= img.shape[0]
        for quad in cells.round().astype(np.intp):
            cv2.polylines(overlay, [quad], True, (255, 0, 0), 1)
        axes[1][idx].imshow(overlay)
        axes[1][idx].set_title("GT cells")

    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def build_param_groups(model: Any, lr: float, backbone_lr: float, weight_decay: float):
    """Build optimizer parameter groups, separating backbone from head parameters and applying weight decay
    only to non-bias / non-norm tensors."""
    no_decay_keys = ("bias", "norm", ".bn", "embed")

    def is_backbone(name: str) -> bool:
        return name.removeprefix("module.").startswith("feat_extractor.")

    groups: dict[tuple[bool, bool], list[Any]] = {
        (False, True): [],
        (False, False): [],
        (True, True): [],
        (True, False): [],
    }
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        decay = not (p.ndim <= 1 or any(k in n.lower() for k in no_decay_keys))
        groups[(is_backbone(n), decay)].append(p)

    return [
        {"params": groups[(False, True)], "lr": lr, "weight_decay": weight_decay},
        {"params": groups[(False, False)], "lr": lr, "weight_decay": 0.0},
        {"params": groups[(True, True)], "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": groups[(True, False)], "lr": backbone_lr, "weight_decay": 0.0},
    ]


def plot_recorder(lr_recorder, loss_recorder, beta: float = 0.95, **kwargs) -> None:
    """Display the results of the LR grid search.
    Adapted from https://github.com/frgfm/Holocron/blob/master/holocron/trainer/core.py
    """
    if len(lr_recorder) != len(loss_recorder) or len(lr_recorder) == 0:
        raise AssertionError("Both `lr_recorder` and `loss_recorder` should have the same length")

    smoothed_losses = []
    avg_loss = 0.0
    for idx, loss in enumerate(loss_recorder):
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_losses.append(avg_loss / (1 - beta ** (idx + 1)))

    data_slice = slice(min(len(loss_recorder) // 10, 10), len(loss_recorder))
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


class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
