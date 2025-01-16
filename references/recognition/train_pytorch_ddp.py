# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os

os.environ["USE_TORCH"] = "1"

import datetime
import hashlib
import multiprocessing
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, PolynomialLR
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    RandomGrayscale,
    RandomPerspective,
    RandomPhotometricDistort,
)

if os.getenv("TQDM_SLACK_TOKEN") and os.getenv("TQDM_SLACK_CHANNEL"):
    from tqdm.contrib.slack import tqdm
else:
    from tqdm.auto import tqdm

from doctr import transforms as T
from doctr.datasets import VOCABS, RecognitionDataset, WordGenerator
from doctr.models import login_to_hub, push_to_hf_hub, recognition
from doctr.utils.metrics import TextMatch
from utils import EarlyStopper, plot_samples


def fit_one_epoch(model, device, train_loader, batch_transforms, optimizer, scheduler, amp=False, clearml_log=False):
    if amp:
        scaler = torch.cuda.amp.GradScaler()

    model.train()
    if clearml_log:
        from clearml import Logger

        logger = Logger.current_logger()

    # Iterate over the batches of the dataset
    pbar = tqdm(train_loader, position=1)
    for images, targets in pbar:
        images = images.to(device)
        images = batch_transforms(images)

        train_loss = model(images, targets)["loss"]

        optimizer.zero_grad()
        if amp:
            with torch.cuda.amp.autocast():
                train_loss = model(images, targets)["loss"]
            scaler.scale(train_loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # Update the params
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss = model(images, targets)["loss"]
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        scheduler.step()

        pbar.set_description(f"Training loss: {train_loss.item():.6}")
        if clearml_log:
            global iteration
            logger.report_scalar(
                title="Training Loss", series="train_loss", value=train_loss.item(), iteration=iteration
            )
            iteration += 1


@torch.no_grad()
def evaluate(model, device, val_loader, batch_transforms, val_metric, amp=False):
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    for images, targets in tqdm(val_loader):
        images = images.to(device)
        images = batch_transforms(images)
        if amp:
            with torch.cuda.amp.autocast():
                out = model(images, targets, return_preds=True)
        else:
            out = model(images, targets, return_preds=True)
        # Compute metric
        if len(out["preds"]):
            words, _ = zip(*out["preds"])
        else:
            words = []
        val_metric.update(targets, words)

        val_loss += out["loss"].item()
        batch_cnt += 1

    val_loss /= batch_cnt
    result = val_metric.summary()
    return val_loss, result["raw"], result["unicase"]


def main(rank: int, world_size: int, args):
    """
    Args:
        rank (int): device id to put the model on
        world_size (int): number of processes participating in the job
        args: other arguments passed through the CLI
    """
    print(args)

    if rank == 0 and args.push_to_hub:
        login_to_hub()

    if not isinstance(args.workers, int):
        args.workers = min(16, multiprocessing.cpu_count())

    torch.backends.cudnn.benchmark = True

    vocab = VOCABS[args.vocab]
    fonts = args.font.split(",")

    if rank == 0:
        # Load val data generator
        st = time.time()
        if isinstance(args.val_path, str):
            with open(os.path.join(args.val_path, "labels.json"), "rb") as f:
                val_hash = hashlib.sha256(f.read()).hexdigest()

            val_set = RecognitionDataset(
                img_folder=os.path.join(args.val_path, "images"),
                labels_path=os.path.join(args.val_path, "labels.json"),
                img_transforms=T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
            )
        else:
            val_hash = None
            # Load synthetic data generator
            val_set = WordGenerator(
                vocab=vocab,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                num_samples=args.val_samples * len(vocab),
                font_family=fonts,
                img_transforms=Compose([
                    T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                    # Ensure we have a 90% split of white-background images
                    T.RandomApply(T.ColorInversion(), 0.9),
                ]),
            )

        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.workers,
            sampler=SequentialSampler(val_set),
            pin_memory=torch.cuda.is_available(),
            collate_fn=val_set.collate_fn,
        )
        print(f"Validation set loaded in {time.time() - st:.4}s ({len(val_set)} samples in {len(val_loader)} batches)")

    batch_transforms = Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))

    # Load doctr model
    model = recognition.__dict__[args.arch](pretrained=args.pretrained, vocab=vocab)

    # Resume weights
    if isinstance(args.resume, str):
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint)

    # Backbone freezing
    if args.freeze_backbone:
        for p in model.feat_extractor.parameters():
            p.requires_grad = False

    # create default process group
    device = torch.device("cuda", args.devices[rank])
    dist.init_process_group(args.backend, rank=rank, world_size=world_size)
    # create local model
    model = model.to(device)
    # construct DDP model
    model = DDP(model, device_ids=[device])

    if rank == 0:
        # Metrics
        val_metric = TextMatch()

    if rank == 0 and args.test_only:
        print("Running evaluation")
        val_loss, exact_match, partial_match = evaluate(
            model, device, val_loader, batch_transforms, val_metric, amp=args.amp
        )
        print(f"Validation loss: {val_loss:.6} (Exact: {exact_match:.2%} | Partial: {partial_match:.2%})")
        return

    st = time.time()

    if isinstance(args.train_path, str):
        # Load train data generator
        base_path = Path(args.train_path)
        parts = (
            [base_path]
            if base_path.joinpath("labels.json").is_file()
            else [base_path.joinpath(sub) for sub in os.listdir(base_path)]
        )
        with open(parts[0].joinpath("labels.json"), "rb") as f:
            train_hash = hashlib.sha256(f.read()).hexdigest()

        train_set = RecognitionDataset(
            parts[0].joinpath("images"),
            parts[0].joinpath("labels.json"),
            img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                # Augmentations
                T.RandomApply(T.ColorInversion(), 0.1),
                RandomGrayscale(p=0.1),
                RandomPhotometricDistort(p=0.1),
                T.RandomApply(T.RandomShadow(), p=0.4),
                T.RandomApply(T.GaussianNoise(mean=0, std=0.1), 0.1),
                T.RandomApply(T.GaussianBlur(sigma=(0.5, 1.5)), 0.3),
                RandomPerspective(distortion_scale=0.2, p=0.3),
            ]),
        )
        if len(parts) > 1:
            for subfolder in parts[1:]:
                train_set.merge_dataset(
                    RecognitionDataset(subfolder.joinpath("images"), subfolder.joinpath("labels.json"))
                )
    else:
        train_hash = None
        # Load synthetic data generator
        train_set = WordGenerator(
            vocab=vocab,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            num_samples=args.train_samples * len(vocab),
            font_family=fonts,
            img_transforms=Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                # Ensure we have a 90% split of white-background images
                T.RandomApply(T.ColorInversion(), 0.9),
                RandomGrayscale(p=0.1),
                RandomPhotometricDistort(p=0.1),
                T.RandomApply(T.RandomShadow(), p=0.4),
                T.RandomApply(T.GaussianNoise(mean=0, std=0.1), 0.1),
                T.RandomApply(T.GaussianBlur(sigma=(0.5, 1.5)), 0.3),
                RandomPerspective(distortion_scale=0.2, p=0.3),
            ]),
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        sampler=DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True),
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_set.collate_fn,
    )
    print(f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in {len(train_loader)} batches)")

    if rank == 0 and args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
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

    if rank == 0:
        config = {
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "architecture": args.arch,
            "input_size": args.input_size,
            "optimizer": args.optim,
            "framework": "pytorch",
            "scheduler": args.sched,
            "train_hash": train_hash,
            "val_hash": val_hash,
            "pretrained": args.pretrained,
            "rotation": args.rotation,
            "amp": args.amp,
        }

    # W&B
    if rank == 0 and args.wb:
        run = wandb.init(
            name=exp_name,
            project="text-recognition",
            config=config,
        )

    # ClearML
    if rank == 0 and args.clearml:
        from clearml import Task

        task = Task.init(project_name="docTR/text-recognition", task_name=exp_name, reuse_last_task_id=False)
        task.upload_artifact("config", config)
        global iteration
        iteration = 0

    # Create loss queue
    min_loss = np.inf
    if args.early_stop:
        early_stopper = EarlyStopper(patience=args.early_stop_epochs, min_delta=args.early_stop_delta)
    # Training loop
    for epoch in range(args.epochs):
        fit_one_epoch(
            model, device, train_loader, batch_transforms, optimizer, scheduler, amp=args.amp, clearml_log=args.clearml
        )

        if rank == 0:
            # Validation loop at the end of each epoch
            val_loss, exact_match, partial_match = evaluate(
                model, device, val_loader, batch_transforms, val_metric, amp=args.amp
            )
            if val_loss < min_loss:
                # All processes should see same parameters as they all start from same
                # random parameters and gradients are synchronized in backward passes.
                # Therefore, saving it in one process is sufficient.
                print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
                torch.save(model.module.state_dict(), Path(args.output_dir) / f"{exp_name}.pt")
            min_loss = val_loss
            print(
                f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} "
                f"(Exact: {exact_match:.2%} | Partial: {partial_match:.2%})"
            )
            # W&B
            if args.wb:
                wandb.log({
                    "val_loss": val_loss,
                    "exact_match": exact_match,
                    "partial_match": partial_match,
                })

            # ClearML
            if args.clearml:
                from clearml import Logger

                logger = Logger.current_logger()
                logger.report_scalar(title="Validation Loss", series="val_loss", value=val_loss, iteration=epoch)
                logger.report_scalar(title="Exact Match", series="exact_match", value=exact_match, iteration=epoch)
                logger.report_scalar(
                    title="Partial Match", series="partial_match", value=partial_match, iteration=epoch
                )

            if args.early_stop and early_stopper.early_stop(val_loss):
                print("Training halted early due to reaching patience limit.")
                break
    if rank == 0:
        if args.wb:
            run.finish()

        if args.push_to_hub:
            push_to_hf_hub(model, exp_name, task="recognition", run_config=args)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR DDP training script for text recognition (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # DDP related args
    parser.add_argument("--backend", default="nccl", type=str, help="Backend to use for Torch DDP")

    parser.add_argument("arch", type=str, help="text-recognition model to train")
    parser.add_argument("--output_dir", type=str, default=".", help="path to save checkpoints and final model")
    parser.add_argument("--train_path", type=str, default=None, help="path to train data folder(s)")
    parser.add_argument("--val_path", type=str, default=None, help="path to val data folder")
    parser.add_argument(
        "--train-samples",
        type=int,
        default=1000,
        help="Multiplied by the vocab length gets you the number of synthetic training samples that will be used.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=20,
        help="Multiplied by the vocab length gets you the number of synthetic validation samples that will be used.",
    )
    parser.add_argument(
        "--font", type=str, default="FreeMono.ttf,FreeSans.ttf,FreeSerif.ttf", help="Font family to be used"
    )
    parser.add_argument("--min-chars", type=int, default=1, help="Minimum number of characters per synthetic sample")
    parser.add_argument("--max-chars", type=int, default=12, help="Maximum number of characters per synthetic sample")
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--devices", default=None, nargs="+", type=int, help="GPU devices to use for training")
    parser.add_argument("--input_size", type=int, default=32, help="input size H for the model, W = 4*H")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam or AdamW)")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument("--vocab", type=str, default="french", help="Vocab to be used for training")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Run the validation loop")
    parser.add_argument(
        "--freeze-backbone", dest="freeze_backbone", action="store_true", help="freeze model backbone for fine-tuning"
    )
    parser.add_argument(
        "--show-samples", dest="show_samples", action="store_true", help="Display unormalized training samples"
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
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "adamw"], help="optimizer to use")
    parser.add_argument(
        "--sched", type=str, default="cosine", choices=["cosine", "onecycle", "poly"], help="scheduler to use"
    )
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping")
    parser.add_argument("--early-stop-epochs", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--early-stop-delta", type=float, default=0.01, help="Minimum Delta for early stopping")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise AssertionError("PyTorch cannot access your GPUs. Please investigate!")
    if not isinstance(args.devices, list):
        args.devices = list(range(torch.cuda.device_count()))
    nprocs = len(args.devices)
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    mp.spawn(main, args=(nprocs, args), nprocs=nprocs, join=True)
