# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import doctr


def plot_samples(images: list[Any], targets: list[dict[str, np.ndarray]], max_samples: int = 2) -> None:
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
        logic = targets[idx]["logic"]

        cells[..., 0] *= img.shape[1]
        cells[..., 1] *= img.shape[0]

        for quad, (start_col, end_col, start_row, end_row) in zip(
            cells.round().astype(np.intp),
            logic,
        ):
            cv2.polylines(overlay, [quad], True, (255, 0, 0), 1)

            center = quad.mean(axis=0)

            # Corner order: 0=TL, 1=TR, 2=BR, 3=BL
            for corner_idx, corner in enumerate(quad):
                # Move the label from the corner toward the polygon center.
                label_position = corner + 0.18 * (center - corner)
                x, y = label_position.astype(np.intp)

                cv2.putText(
                    overlay,
                    str(corner_idx),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 255),  # Pink
                    1,
                    cv2.LINE_AA,
                )

            center_x, center_y = center.astype(np.intp)
            cv2.putText(
                overlay,
                f"C:{start_col}-{end_col} R:{start_row}-{end_row}",
                (center_x, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 100, 0),  # Dark green
                1,
                cv2.LINE_AA,
            )

        axes[1][idx].imshow(overlay)
        axes[1][idx].set_title("GT cells | corners: TL, TR, BR, BL")

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


def resolve_device(device: str | int | None) -> torch.device:
    """Turn a CLI device spec (None, "cpu", "mps", "cuda", "cuda:1", 0) into a torch device.

    With `None`, pick CUDA if available, then MPS, then CPU.

    Args:
        device: the device specification

    Returns:
        the resolved torch device
    """
    if device is None or device == "":
        if torch.cuda.is_available():
            return torch.device("cuda", 0)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
        index = int(device)
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if index >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
        return torch.device("cuda", index)
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if dev.index is None:
            # "cuda" without an index: pin it to the current device so that `torch.cuda.set_device` works
            dev = torch.device("cuda", torch.cuda.current_device())
        elif dev.index >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise AssertionError("MPS backend is not available on this machine.")
    return dev


def model_device(model: torch.nn.Module) -> torch.device:
    """Device holding the parameters of a model (works through DDP wrappers)."""
    return next(model.parameters()).device


def amp_dtype(name: str) -> torch.dtype:
    """Autocast dtype for `--amp-dtype`."""
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def save_checkpoint(model: torch.nn.Module, output_dir: str, name: str) -> Path:
    """Save the model weights as `<output_dir>/<name>.pt`.

    Args:
        model: the model to save (unwrapped from DDP if needed)
        output_dir: destination folder, created if missing
        name: file stem

    Returns:
        the path of the weights file
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = out_dir / f"{name}.pt"
    torch.save(model.state_dict(), weights)
    return weights


def save_run_metadata(output_dir: str, name: str, metadata: dict) -> Path:
    """Write the run metadata as `<output_dir>/<name>.json`.

    The metadata (architecture, task settings such as class names or vocab, dataset hashes, versions, arguments)
    is what is needed to rebuild the model for inference without remembering how it was trained. It describes the
    run, not one of its checkpoints, so it is written once per run and shared by every checkpoint the run saves.

    Args:
        output_dir: destination folder, created if missing
        name: file stem, the experiment name
        metadata: JSON-serializable run description

    Returns:
        the path of the metadata file
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=1, default=str)
    return path


def run_metadata(args, **task_specific) -> dict:
    """Common run description written next to every checkpoint: versions, git revision and the full arguments."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, cwd=Path(__file__).parent
        ).strip()
    except Exception:
        revision = None
    return {
        "framework": "pytorch",
        "doctr_version": doctr.__version__,
        "torch_version": torch.__version__,
        "git_revision": revision,
        "args": dict(vars(args)),
        **task_specific,
    }
