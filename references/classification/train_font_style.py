# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import datetime
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR, OneCycleLR, PolynomialLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms.v2 import ColorJitter, Compose, Normalize

if os.getenv("TQDM_SLACK_TOKEN") and os.getenv("TQDM_SLACK_CHANNEL"):
    from tqdm.contrib.slack import tqdm
else:
    from tqdm.auto import tqdm

from doctr import transforms as T
from doctr.datasets import FONT_STYLE_ATTRIBUTES, FontStyleDataset
from doctr.models import classification, login_to_hub, push_to_hf_hub
from doctr.models.utils import export_model_to_onnx
from doctr.utils import Sample
from utils import (
    EarlyStopper,
    amp_dtype,
    model_device,
    plot_recorder,
    plot_samples,
    resolve_device,
    run_metadata,
    save_checkpoint,
    save_run_metadata,
)

# The classes of each style attribute, in the order of the model outputs
ATTRIBUTES = FONT_STYLE_ATTRIBUTES
GROUP_SIZES = [len(classes) for classes in ATTRIBUTES.values()]
INPUT_SIZE = (64, 256)
# Margin around the word boxes of the OCR predictor, relative to their height: frames them like the training crops
CROP_MARGIN = 0.15

AMP_DTYPE = torch.float16  # set from --amp-dtype in main()


def _autocast():
    return torch.amp.autocast("cuda", dtype=AMP_DTYPE)


def _scaler():
    # bfloat16 has the range of float32: no loss scaling needed (GradScaler only makes sense for float16)
    return torch.amp.GradScaler("cuda", enabled=AMP_DTYPE == torch.float16)


def load_split(path: str, **kwargs) -> FontStyleDataset:
    """Load a split: a folder holding `images/` and `labels.json`, or a folder of such folders (e.g. shards)"""
    root = Path(path)
    folders = (
        [root]
        if (root / "labels.json").is_file()
        else sorted(folder for folder in root.iterdir() if (folder / "labels.json").is_file())
    )
    if len(folders) == 0:
        raise FileNotFoundError(f"no labels.json found in {path} nor in its subfolders")
    datasets = [
        FontStyleDataset(img_folder=str(folder / "images"), labels_path=str(folder / "labels.json"), **kwargs)
        for folder in folders
    ]
    dataset = datasets[0]
    for other in datasets[1:]:
        dataset.merge_dataset(other)
    return dataset


def random_trim(sample: Sample, max_x: float = 0.04, max_y: float = 0.04) -> Sample:
    """Trim the margins of a crop at random, as a text detection box may be tighter than the synthetic crops. The
    vertical trim is kept small: the underline lies under the text and must stay in the crop."""
    _, h, w = sample.image.shape
    left, right = (int(w * max_x * np.random.rand()) for _ in range(2))
    top, bottom = (int(h * max_y * np.random.rand()) for _ in range(2))
    if w - left - right < 4 or h - top - bottom < 4:
        return sample
    return Sample(image=sample.image[:, top : h - bottom, left : w - right], target=sample.target)


def class_weights(
    dataset: FontStyleDataset, device: torch.device, attributes: list[str] | None = None
) -> list[torch.Tensor]:
    """Inverse square root frequency weights of the classes of each attribute (mean 1), against the imbalance of
    the rare styles (e.g. strikethrough, highlight). The attributes left out keep uniform weights: weighting bold
    up would make the classifier call ambiguous regular words bold, whereas most words of a page are regular."""
    targets = np.stack([target for _, target in dataset.data])
    weights = []
    for idx, (attribute, size) in enumerate(zip(ATTRIBUTES, GROUP_SIZES)):
        if attributes and attribute not in attributes:
            weights.append(torch.ones(size, dtype=torch.float32, device=device))
            continue
        counts = np.bincount(targets[:, idx], minlength=size).astype(np.float64)
        weight = np.where(counts > 0, 1 / np.sqrt(np.maximum(counts, 1)), 0)
        weights.append(torch.tensor(weight / weight[counts > 0].mean(), dtype=torch.float32, device=device))
    return weights


def grouped_loss(
    out: torch.Tensor,
    targets: torch.Tensor,
    weights: list[torch.Tensor] | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Sum of the cross entropies of each style attribute"""
    groups = torch.split(out.float(), GROUP_SIZES, dim=1)
    return sum(  # type: ignore[return-value]
        cross_entropy(
            logits,
            targets[:, idx],
            weight=weights[idx] if weights is not None else None,
            label_smoothing=label_smoothing,
        )
        for idx, logits in enumerate(groups)
    )


def record_lr(
    model: torch.nn.Module,
    train_loader: DataLoader,
    batch_transforms,
    optimizer,
    loss_fn,
    start_lr: float = 1e-7,
    end_lr: float = 1,
    num_it: int = 100,
    amp: bool = False,
):
    """Gridsearch the optimal learning rate for the training.
    Adapted from https://github.com/frgfm/Holocron/blob/master/holocron/trainer/core.py
    """
    if num_it > len(train_loader):
        raise ValueError("the value of `num_it` needs to be lower than the number of available batches")

    model = model.train()
    # Update param groups & LR
    optimizer.defaults["lr"] = start_lr
    for pgroup in optimizer.param_groups:
        pgroup["lr"] = start_lr

    gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
    scheduler = MultiplicativeLR(optimizer, lambda step: gamma)

    lr_recorder = [start_lr * gamma**idx for idx in range(num_it)]
    loss_recorder = []

    if amp:
        scaler = _scaler()

    for batch_idx, (images, targets) in enumerate(train_loader):
        targets = torch.from_numpy(np.stack(targets)).to(model_device(model), non_blocking=True)
        images = batch_transforms(images.to(model_device(model), non_blocking=True))

        # Forward, Backward & update
        optimizer.zero_grad()
        if amp:
            with _autocast():
                train_loss = loss_fn(model(images), targets)
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss = loss_fn(model(images), targets)
            train_loss.backward()
            optimizer.step()
        # Update LR
        scheduler.step()

        # Record
        if not torch.isfinite(train_loss):
            if batch_idx == 0:
                raise ValueError("loss value is NaN or inf.")
            else:
                break
        loss_recorder.append(train_loss.item())
        # Stop after the number of iterations
        if batch_idx + 1 == num_it:
            break

    return lr_recorder[: len(loss_recorder)], loss_recorder


def fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, loss_fn, amp=False, log=None):
    if amp:
        scaler = _scaler()

    model.train()
    # Iterate over the batches of the dataset
    epoch_train_loss, batch_cnt = 0.0, 0.0
    pbar = tqdm(train_loader, dynamic_ncols=True)
    for images, targets in pbar:
        targets = torch.from_numpy(np.stack(targets)).to(model_device(model), non_blocking=True)
        images = batch_transforms(images.to(model_device(model), non_blocking=True))

        optimizer.zero_grad()
        if amp:
            with _autocast():
                train_loss = loss_fn(model(images), targets)
            scaler.scale(train_loss).backward()
            # Update the params
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss = loss_fn(model(images), targets)
            train_loss.backward()
            optimizer.step()

        scheduler.step()
        last_lr = scheduler.get_last_lr()[0]

        pbar.set_description(f"Training loss: {train_loss.item():.6} | LR: {last_lr:.6}")
        log(train_loss=train_loss.item(), lr=last_lr)

        epoch_train_loss += train_loss.item()
        batch_cnt += 1

    epoch_train_loss /= batch_cnt
    return epoch_train_loss, last_lr


@torch.no_grad()
def evaluate(model, val_loader, batch_transforms, amp=False, log=None):
    """Evaluate the model: loss, and the accuracy and macro F1 score (mean F1 of the classes present in the
    validation set) of each style attribute"""
    # Model in eval mode
    model.eval()
    # Confusion matrix of each attribute
    confusions = [np.zeros((size, size), dtype=np.int64) for size in GROUP_SIZES]
    val_loss, batch_cnt = 0.0, 0.0
    pbar = tqdm(val_loader, dynamic_ncols=True)
    for images, targets in pbar:
        targets = torch.from_numpy(np.stack(targets)).to(model_device(model), non_blocking=True)
        images = batch_transforms(images.to(model_device(model), non_blocking=True))

        if amp:
            with _autocast():
                out = model(images)
        else:
            out = model(images)
        loss = grouped_loss(out, targets)
        for idx, logits in enumerate(torch.split(out.float(), GROUP_SIZES, dim=1)):
            np.add.at(confusions[idx], (targets[:, idx].cpu().numpy(), logits.argmax(dim=1).cpu().numpy()), 1)

        pbar.set_description(f"Validation loss: {loss.item():.6}")
        if log is not None:
            log(val_loss=loss.item())

        val_loss += loss.item()
        batch_cnt += 1

    val_loss /= batch_cnt
    metrics = {}
    for attribute, confusion in zip(ATTRIBUTES, confusions):
        tp = np.diag(confusion).astype(np.float64)
        support, predicted = confusion.sum(1), confusion.sum(0)
        f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=(support + predicted) > 0)
        metrics[attribute] = {
            "acc": float(tp.sum() / max(confusion.sum(), 1)),
            "f1": float(f1[support > 0].mean()) if (support > 0).any() else 0.0,
        }
    return val_loss, metrics


def format_metrics(metrics: dict) -> str:
    return " | ".join(f"{attr}: {m['acc']:.2%} (F1 {m['f1']:.2%})" for attr, m in metrics.items())


def mean_f1(metrics: dict) -> float:
    return float(np.mean([m["f1"] for m in metrics.values()]))


def main(args):
    global AMP_DTYPE
    AMP_DTYPE = amp_dtype(args.amp_dtype)
    slack_token = os.getenv("TQDM_SLACK_TOKEN")
    slack_channel = os.getenv("TQDM_SLACK_CHANNEL")

    pbar = tqdm(disable=False if slack_token and slack_channel else True)
    if slack_token and slack_channel:
        # Monkey patch tqdm write method to send messages directly to Slack
        pbar.write = lambda msg: pbar.sio.client.chat_postMessage(channel=slack_channel, text=msg)
    pbar.write(str(args))
    device = resolve_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp (automatic mixed precision) is only supported on CUDA devices")

    if args.push_to_hub:
        login_to_hub()

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Load val data generator
    st = time.time()
    val_set = load_split(
        args.val_path,
        img_transforms=T.Resize(INPUT_SIZE, preserve_aspect_ratio=True, symmetric_pad=True),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=SequentialSampler(val_set),
        pin_memory=device.type == "cuda",
        collate_fn=val_set.collate_fn,
    )
    pbar.write(f"Validation set loaded in {time.time() - st:.4}s ({len(val_set)} samples in {len(val_loader)} batches)")

    batch_transforms = Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))

    # Load doctr model: the logits of every attribute concatenated
    model = classification.__dict__[args.arch](
        pretrained=args.pretrained, num_classes=sum(GROUP_SIZES), classes=ATTRIBUTES
    )
    # The input size of the model, used by `font_style_predictor` to preprocess the crops, and the margin the OCR
    # predictor adds around the word boxes to crop them for the model
    model.cfg["input_shape"] = (3, *INPUT_SIZE)
    model.cfg["crop_margin"] = CROP_MARGIN

    # Resume weights
    if isinstance(args.resume, str):
        pbar.write(f"Resuming {args.resume}")
        model.from_pretrained(args.resume)

    # GPU
    if device.type == "cpu":
        logging.warning("No accessible GPU, target device set to CPU.")  # noqa: LOG015
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = model.to(device)

    if args.test_only:
        pbar.write("Running evaluation")
        val_loss, metrics = evaluate(model, val_loader, batch_transforms, amp=args.amp)
        pbar.write(f"Validation loss: {val_loss:.6} (mean F1: {mean_f1(metrics):.2%})\n{format_metrics(metrics)}")
        return

    st = time.time()
    # Color-safe augmentations: the text and highlight colors are labels, so no grayscale, color inversion,
    # hue or saturation change. No shear or perspective either, which would look like italic. The framing, the
    # resolution and the compression vary, as for the words of real pages cropped by the OCR predictor.
    train_set = load_split(
        args.train_path,
        img_transforms=Compose([
            random_trim,
            T.RandomPad(max_ratio=0.15, p=0.5),
            # small words of pages rendered or scanned at 100-150 dpi
            T.RandomLowResolution(height_range=(14, 34), p=0.5),
            T.JPEGCompression(quality_range=(35, 90), p=0.3),
            # faint enough to keep the color labels
            T.RandomColorCast(max_gain=0.04, p=0.3),
            T.Resize(INPUT_SIZE, preserve_aspect_ratio=True, symmetric_pad=True),
            T.RandomApply(T.GaussianNoise(mean=0.0, std=0.04), 0.1),
            T.RandomApply(T.GaussianBlur(sigma=(0.3, 0.9)), 0.15),
            T.ImageTorchvisionTransform(ColorJitter(brightness=0.15, contrast=0.15)),
        ]),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        sampler=RandomSampler(train_set),
        pin_memory=device.type == "cuda",
        collate_fn=train_set.collate_fn,
    )
    pbar.write(f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in {len(train_loader)} batches)")

    if args.show_samples:
        x, target = next(iter(train_loader))
        labels = [
            ", ".join(
                f"{attr}={classes[t[idx]]}"
                for idx, (attr, classes) in enumerate(ATTRIBUTES.items())
                if classes[t[idx]] not in (False, "none")
            )
            for t in target
        ]
        plot_samples(x, labels)
        return

    if args.class_weights and set(args.class_weights) - set(ATTRIBUTES):
        raise ValueError(f"unknown attributes in --class-weights: {set(args.class_weights) - set(ATTRIBUTES)}")
    weights = class_weights(train_set, device, args.class_weights) if args.class_weights is not None else None

    def loss_fn(out, targets):
        return grouped_loss(out, targets, weights, args.label_smoothing)

    # Optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            args.lr,
            betas=(0.95, 0.999),
            eps=1e-6,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            args.lr,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=args.weight_decay or 1e-4,
        )

    # LR Finder
    if args.find_lr:
        lrs, losses = record_lr(model, train_loader, batch_transforms, optimizer, loss_fn, amp=args.amp)
        plot_recorder(lrs, losses)
        return

    # Scheduler
    if args.sched == "cosine":
        scheduler = CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=args.lr / 25e4)
    elif args.sched == "onecycle":
        scheduler = OneCycleLR(optimizer, args.lr, args.epochs * len(train_loader))
    elif args.sched == "poly":
        scheduler = PolynomialLR(optimizer, args.epochs * len(train_loader))

    # Training monitoring
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.arch}_{current_time}" if args.name is None else args.name

    config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "architecture": args.arch,
        "input_size": INPUT_SIZE,
        "optimizer": args.optim,
        "framework": "pytorch",
        "attributes": list(ATTRIBUTES),
        "scheduler": args.sched,
        "pretrained": args.pretrained,
        "class_weights": args.class_weights,
        "label_smoothing": args.label_smoothing,
    }
    # the run metadata describes the whole run: written once, next to the checkpoint `<exp_name>.pt`
    save_run_metadata(
        args.output_dir,
        exp_name,
        {**config, **run_metadata(args, task="font_style", classes=ATTRIBUTES, input_size=INPUT_SIZE)},
    )

    global global_step
    global_step = 0  # Shared global step counter

    # W&B
    if args.wb:
        import wandb

        run = wandb.init(name=exp_name, project="font-style-classification", config=config)

        def wandb_log_at_step(train_loss=None, val_loss=None, lr=None):
            wandb.log({
                **({"train_loss_step": train_loss} if train_loss is not None else {}),
                **({"val_loss_step": val_loss} if val_loss is not None else {}),
                **({"step_lr": lr} if lr is not None else {}),
            })

    # ClearML
    if args.clearml:
        from clearml import Logger, Task

        task = Task.init(project_name="docTR/font-style-classification", task_name=exp_name, reuse_last_task_id=False)
        task.upload_artifact("config", config)

        def clearml_log_at_step(train_loss=None, val_loss=None, lr=None):
            logger = Logger.current_logger()
            if train_loss is not None:
                logger.report_scalar(
                    title="Training Step Loss", series="train_loss_step", iteration=global_step, value=train_loss
                )
            if val_loss is not None:
                logger.report_scalar(
                    title="Validation Step Loss", series="val_loss_step", iteration=global_step, value=val_loss
                )
            if lr is not None:
                logger.report_scalar(title="Step Learning Rate", series="step_lr", iteration=global_step, value=lr)

    # Unified logger
    def log_at_step(train_loss=None, val_loss=None, lr=None):
        global global_step
        if args.wb:
            wandb_log_at_step(train_loss, val_loss, lr)
        if args.clearml:
            clearml_log_at_step(train_loss, val_loss, lr)
        global_step += 1  # Increment the shared global step counter

    # The checkpoint is selected on the mean macro F1 score of the attributes, not dominated by the frequent classes
    best_f1 = -np.inf
    # Training loop
    if args.early_stop:
        early_stopper = EarlyStopper(patience=args.early_stop_epochs, min_delta=args.early_stop_delta)
    for epoch in range(args.epochs):
        train_loss, actual_lr = fit_one_epoch(
            model, train_loader, batch_transforms, optimizer, scheduler, loss_fn, amp=args.amp, log=log_at_step
        )
        pbar.write(f"Epoch {epoch + 1}/{args.epochs} - Training loss: {train_loss:.6} | LR: {actual_lr:.6}")

        # Validation loop at the end of each epoch
        val_loss, metrics = evaluate(model, val_loader, batch_transforms, amp=args.amp, log=log_at_step)
        score = mean_f1(metrics)
        if score > best_f1:
            pbar.write(f"Mean F1 increased {best_f1:.4} --> {score:.4}: saving state...")
            save_checkpoint(model, args.output_dir, exp_name)
            best_f1 = score
        pbar.write(
            f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} (mean F1: {score:.2%})\n"
            f"{format_metrics(metrics)}"
        )

        # W&B
        if args.wb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": actual_lr,
                "mean_f1": score,
                **{f"{attr}_acc": m["acc"] for attr, m in metrics.items()},
                **{f"{attr}_f1": m["f1"] for attr, m in metrics.items()},
            })

        # ClearML
        if args.clearml:
            from clearml import Logger

            logger = Logger.current_logger()
            logger.report_scalar(title="Training Loss", series="train_loss", value=train_loss, iteration=epoch)
            logger.report_scalar(title="Validation Loss", series="val_loss", value=val_loss, iteration=epoch)
            logger.report_scalar(title="Learning Rate", series="lr", value=actual_lr, iteration=epoch)
            logger.report_scalar(title="Mean F1", series="mean_f1", value=score, iteration=epoch)
            for attr, m in metrics.items():
                logger.report_scalar(title="Accuracy", series=attr, value=m["acc"], iteration=epoch)
                logger.report_scalar(title="F1", series=attr, value=m["f1"], iteration=epoch)

        if args.early_stop and early_stopper.early_stop(val_loss):
            pbar.write("Training halted early due to reaching patience limit.")
            break

    if args.wb:
        run.finish()

    if args.push_to_hub:
        push_to_hf_hub(model, exp_name, task="classification", run_config=args)

    if args.export_onnx:
        pbar.write("Exporting model to ONNX...")
        dummy_batch = next(iter(val_loader))
        dummy_input = dummy_batch[0].to(device)
        model_path = export_model_to_onnx(model, exp_name, dummy_input)
        pbar.write(f"Exported model saved in {model_path}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR training script for font style classification of word crops (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("arch", type=str, help="classification model to train")
    parser.add_argument("--output_dir", type=str, default=".", help="path to save checkpoints and final model")
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="path to the training data: a folder holding images/ and labels.json, or a folder of such folders",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        required=True,
        help="path to the validation data: a folder holding images/ and labels.json, or a folder of such folders",
    )
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device for single-process runs: a CUDA index (e.g. 0), 'cuda:N', 'mps' (Apple Silicon) or 'cpu'. "
        "Default: CUDA if available, else MPS, else CPU.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam or AdamW)")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Run the validation loop")
    parser.add_argument(
        "--show-samples", dest="show_samples", action="store_true", help="Display unormalized training samples"
    )
    parser.add_argument(
        "--class-weights",
        dest="class_weights",
        nargs="*",
        default=None,
        metavar="ATTRIBUTE",
        help="weight the classes by their inverse square root frequency (rare styles): of all the attributes, or "
        "only of the given ones (e.g. underline strikethrough color highlight handwritten category family)",
    )
    parser.add_argument(
        "--label-smoothing",
        dest="label_smoothing",
        type=float,
        default=0.0,
        help="label smoothing of the cross entropies (e.g. 0.1, for the many near-identical font families)",
    )
    parser.add_argument("--wb", dest="wb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--clearml", dest="clearml", action="store_true", help="Log to ClearML")
    parser.add_argument("--push-to-hub", dest="push_to_hub", action="store_true", help="Push to Huggingface Hub")
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Load pretrained parameters before starting the training",
    )
    parser.add_argument("--export-onnx", dest="export_onnx", action="store_true", help="Export the model to ONNX")
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "adamw"], help="optimizer to use")
    parser.add_argument(
        "--sched", type=str, default="cosine", choices=["cosine", "onecycle", "poly"], help="scheduler to use"
    )
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="autocast dtype for --amp; bfloat16 (Ampere+ GPUs) avoids float16 overflows, no loss scaling",
    )
    parser.add_argument("--find-lr", action="store_true", help="Gridsearch the optimal LR")
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping")
    parser.add_argument("--early-stop-epochs", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--early-stop-delta", type=float, default=0.01, help="Minimum Delta for early stopping")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
