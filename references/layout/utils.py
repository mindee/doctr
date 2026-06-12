# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


def convert_target(target: dict[str, list], class_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Convert the target from the dataset format to the format expected by the metric

    Args:
        target: dictionary containing the target boxes and labels for a single sample
        class_names: list of class names

    Returns:
        tuple of (boxes, labels) where boxes is an array of shape (N, 4) or (N, 4, 2) depending on the use of polygons,
            and labels is an array of shape (N,) containing the class indices.
    """
    boxes = []
    labels = []

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name, class_boxes in target.items():
        if len(class_boxes) == 0:
            continue

        boxes.extend(class_boxes)
        labels.extend([class_to_idx[class_name]] * len(class_boxes))

    return np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def plot_samples(
    images: list[Any],
    targets: list[dict[str, np.ndarray]],
    padding_masks: list[Any] | None = None,
    max_samples: int = 4,
) -> None:
    nb_samples = min(len(images), max_samples)
    _, axes = plt.subplots(3, nb_samples, figsize=(20, 8))

    if nb_samples == 1:
        axes = np.expand_dims(axes, axis=1)

    for idx in range(nb_samples):
        img = (255 * images[idx].detach().cpu().numpy()).round().clip(0, 255).astype(np.uint8)
        if img.shape[0] == 3 and img.shape[2] != 3:
            img = img.transpose(1, 2, 0)

        axes[0][idx].imshow(img)
        axes[0][idx].set_title("Image")

        target = np.zeros(img.shape[:2], np.uint8)
        tgts = targets[idx].copy()
        for boxes in tgts.values():
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * img.shape[1]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * img.shape[0]
            boxes[:, :4] = boxes[:, :4].round().astype(int)
            for box in boxes:
                if boxes.ndim == 3:
                    cv2.fillPoly(target, [np.intp(box)], 1)
                else:
                    target[int(box[1]) : int(box[3]) + 1, int(box[0]) : int(box[2]) + 1] = 1

        axes[1][idx].imshow(target.astype(bool), cmap="gray")
        axes[1][idx].set_title("GT Boxes")

        if padding_masks is not None and padding_masks[idx] is not None:
            pm = padding_masks[idx].detach().cpu().numpy()
            pm = pm.squeeze().astype(bool)
            axes[2][idx].imshow(pm, cmap="gray")
            axes[2][idx].set_title("Padding Mask")
        else:
            axes[2][idx].text(0.5, 0.5, "No mask", ha="center", va="center")
            axes[2][idx].set_title("Padding Mask")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def build_param_groups(model: Any, lr: float, backbone_lr: float, weight_decay: float):
    """Build parameter groups for the optimizer, separating backbone and non-backbone parameters,
    and applying weight decay only to non-bias and non-norm parameters.

    Args:
        model: the model containing the parameters
        lr: learning rate for non-backbone parameters
        backbone_lr: learning rate for backbone parameters
        weight_decay: weight decay to apply to non-bias and non-norm parameters

    Returns:
        a list of parameter groups to be passed to the optimizer
    """
    no_decay_keys = ("bias", "norm", ".bn", "embed")  # Embedding, LayerNorm, BN

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

    Args:
        lr_recorder: list of LR values
        loss_recorder: list of loss values
        beta (float, optional): smoothing factor
        **kwargs: keyword arguments from `matplotlib.pyplot.show`
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
