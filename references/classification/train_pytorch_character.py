# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os

os.environ["USE_TORCH"] = "1"

import datetime
import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR, OneCycleLR, PolynomialLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    Normalize,
    RandomGrayscale,
    RandomPerspective,
    RandomPhotometricDistort,
    RandomRotation,
)
from tqdm.auto import tqdm

from doctr import transforms as T
from doctr.datasets import VOCABS, CharacterGenerator
from doctr.models import classification, login_to_hub, push_to_hf_hub
from doctr.models.utils import export_model_to_onnx
from utils import EarlyStopper, plot_recorder, plot_samples


def record_lr(
    model: torch.nn.Module,
    train_loader: DataLoader,
    batch_transforms,
    optimizer,
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
        scaler = torch.cuda.amp.GradScaler()

    for batch_idx, (images, targets) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda()
            targets = targets.cuda()

        images = batch_transforms(images)

        # Forward, Backward & update
        optimizer.zero_grad()
        if amp:
            with torch.cuda.amp.autocast():
                out = model(images)
                train_loss = cross_entropy(out, targets)
            scaler.scale(train_loss).backward()
            # Update the params
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            train_loss = cross_entropy(out, targets)
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


def fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, amp=False):
    if amp:
        scaler = torch.cuda.amp.GradScaler()

    model.train()
    # Iterate over the batches of the dataset
    pbar = tqdm(train_loader, position=1)
    for images, targets in pbar:
        if torch.cuda.is_available():
            images = images.cuda()
            targets = targets.cuda()

        images = batch_transforms(images)

        optimizer.zero_grad()
        if amp:
            with torch.cuda.amp.autocast():
                out = model(images)
                train_loss = cross_entropy(out, targets)
            scaler.scale(train_loss).backward()
            # Update the params
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            train_loss = cross_entropy(out, targets)
            train_loss.backward()
            optimizer.step()
        scheduler.step()

        pbar.set_description(f"Training loss: {train_loss.item():.6}")


@torch.no_grad()
def evaluate(model, val_loader, batch_transforms, amp=False):
    # Model in eval mode
    model.eval()
    # Validation loop
    val_loss, correct, samples, batch_cnt = 0, 0, 0, 0
    for images, targets in tqdm(val_loader):
        images = batch_transforms(images)

        if torch.cuda.is_available():
            images = images.cuda()
            targets = targets.cuda()

        if amp:
            with torch.cuda.amp.autocast():
                out = model(images)
                loss = cross_entropy(out, targets)
        else:
            out = model(images)
            loss = cross_entropy(out, targets)
        # Compute metric
        correct += (out.argmax(dim=1) == targets).sum().item()

        val_loss += loss.item()
        batch_cnt += 1
        samples += images.shape[0]

    val_loss /= batch_cnt
    acc = correct / samples
    return val_loss, acc


def main(args):
    print(args)

    if args.push_to_hub:
        login_to_hub()

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    torch.backends.cudnn.benchmark = True

    vocab = VOCABS[args.vocab]

    fonts = args.font.split(",")

    # Load val data generator
    st = time.time()
    val_set = CharacterGenerator(
        vocab=vocab,
        num_samples=args.val_samples * len(vocab),
        cache_samples=True,
        img_transforms=Compose([
            T.Resize((args.input_size, args.input_size)),
            # Ensure we have a 90% split of white-background images
            T.RandomApply(T.ColorInversion(), 0.9),
        ]),
        font_family=fonts,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=SequentialSampler(val_set),
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Validation set loaded in {time.time() - st:.4}s ({len(val_set)} samples in {len(val_loader)} batches)")

    batch_transforms = Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))

    # Load doctr model
    model = classification.__dict__[args.arch](pretrained=args.pretrained, num_classes=len(vocab), classes=list(vocab))

    # Resume weights
    if isinstance(args.resume, str):
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint)

    # GPU
    if isinstance(args.device, int):
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if args.device >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
    # Silent default switch to GPU if available
    elif torch.cuda.is_available():
        args.device = 0
    else:
        logging.warning("No accessible GPU, targe device set to CPU.")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        model = model.cuda()

    if args.test_only:
        print("Running evaluation")
        val_loss, acc = evaluate(model, val_loader, batch_transforms)
        print(f"Validation loss: {val_loss:.6} (Acc: {acc:.2%})")
        return

    st = time.time()

    # Load train data generator
    train_set = CharacterGenerator(
        vocab=vocab,
        num_samples=args.train_samples * len(vocab),
        cache_samples=True,
        img_transforms=Compose([
            T.Resize((args.input_size, args.input_size)),
            # Augmentations
            T.RandomApply(T.ColorInversion(), 0.9),
            RandomGrayscale(p=0.1),
            RandomPhotometricDistort(p=0.1),
            T.RandomApply(T.RandomShadow(), p=0.4),
            T.RandomApply(T.GaussianNoise(mean=0, std=0.1), 0.1),
            T.RandomApply(T.GaussianBlur(sigma=(0.5, 1.5)), 0.3),
            RandomPerspective(distortion_scale=0.2, p=0.3),
            RandomRotation(15, interpolation=InterpolationMode.BILINEAR),
        ]),
        font_family=fonts,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        sampler=RandomSampler(train_set),
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in {len(train_loader)} batches)")

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, list(map(vocab.__getitem__, target)))
        return

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
        lrs, losses = record_lr(model, train_loader, batch_transforms, optimizer, amp=args.amp)
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

    # W&B
    if args.wb:
        import wandb

        run = wandb.init(
            name=exp_name,
            project="character-classification",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "architecture": args.arch,
                "input_size": args.input_size,
                "optimizer": args.optim,
                "framework": "pytorch",
                "vocab": args.vocab,
                "scheduler": args.sched,
                "pretrained": args.pretrained,
            },
        )

    # Create loss queue
    min_loss = np.inf
    # Training loop
    if args.early_stop:
        early_stopper = EarlyStopper(patience=args.early_stop_epochs, min_delta=args.early_stop_delta)
    for epoch in range(args.epochs):
        fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler)

        # Validation loop at the end of each epoch
        val_loss, acc = evaluate(model, val_loader, batch_transforms)
        if val_loss < min_loss:
            print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
            torch.save(model.state_dict(), Path(args.output_dir) / f"{exp_name}.pt")
            min_loss = val_loss
        print(f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} (Acc: {acc:.2%})")
        # W&B
        if args.wb:
            wandb.log({
                "val_loss": val_loss,
                "acc": acc,
            })
        if args.early_stop and early_stopper.early_stop(val_loss):
            print("Training halted early due to reaching patience limit.")
            break
    if args.wb:
        run.finish()

    if args.push_to_hub:
        push_to_hf_hub(model, exp_name, task="classification", run_config=args)

    if args.export_onnx:
        print("Exporting model to ONNX...")
        dummy_batch = next(iter(val_loader))
        dummy_input = dummy_batch[0].cuda() if torch.cuda.is_available() else dummy_batch[0]
        model_path = export_model_to_onnx(model, exp_name, dummy_input)
        print(f"Exported model saved in {model_path}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR training script for character classification (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("arch", type=str, help="text-recognition model to train")
    parser.add_argument("--output_dir", type=str, default=".", help="path to save checkpoints and final model")
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("--input_size", type=int, default=32, help="input size H for the model, W = H")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam or AdamW)")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument(
        "--font", type=str, default="FreeMono.ttf,FreeSans.ttf,FreeSerif.ttf", help="Font family to be used"
    )
    parser.add_argument("--vocab", type=str, default="french", help="Vocab to be used for training")
    parser.add_argument(
        "--train-samples",
        dest="train_samples",
        type=int,
        default=1000,
        help="Multiplied by the vocab length gets you the number of training samples that will be used.",
    )
    parser.add_argument(
        "--val-samples",
        dest="val_samples",
        type=int,
        default=20,
        help="Multiplied by the vocab length gets you the number of validation samples that will be used.",
    )
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Run the validation loop")
    parser.add_argument(
        "--show-samples", dest="show_samples", action="store_true", help="Display unormalized training samples"
    )
    parser.add_argument("--wb", dest="wb", action="store_true", help="Log to Weights & Biases")
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
    parser.add_argument("--find-lr", action="store_true", help="Gridsearch the optimal LR")
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping")
    parser.add_argument("--early-stop-epochs", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--early-stop-delta", type=float, default=0.01, help="Minimum Delta for early stopping")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
