# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import datetime
import hashlib
import logging
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import torch

# The following import is required for DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    MultiplicativeLR,
    OneCycleLR,
    PolynomialLR,
    SequentialLR,
)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.v2 import Compose, Normalize, RandomGrayscale, RandomPhotometricDistort

if os.getenv("TQDM_SLACK_TOKEN") and os.getenv("TQDM_SLACK_CHANNEL"):
    from tqdm.contrib.slack import tqdm
else:
    from tqdm.auto import tqdm

from doctr import transforms as T
from doctr.datasets import TableStructureDataset
from doctr.models import table_structure
from doctr.utils.metrics import TableCellMetric
from utils import EarlyStopper, build_param_groups, plot_recorder, plot_samples


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
    optimizer.defaults["lr"] = start_lr
    for pgroup in optimizer.param_groups:
        pgroup["lr"] = start_lr

    gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
    scheduler = MultiplicativeLR(optimizer, lambda step: gamma)

    lr_recorder = [start_lr * gamma**idx for idx in range(num_it)]
    loss_recorder = []

    if amp:
        scaler = torch.amp.GradScaler("cuda")

    for batch_idx, (images, targets) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)

        optimizer.zero_grad()
        if amp:
            with torch.amp.autocast("cuda"):
                train_loss = model(images, target=targets)["loss"]
            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss = model(images, target=targets)["loss"]
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
        scheduler.step()

        if not torch.isfinite(train_loss):
            if batch_idx == 0:
                raise ValueError("loss value is NaN or inf.")
            break
        loss_recorder.append(train_loss.item())
        if batch_idx + 1 == num_it:
            break

    return lr_recorder[: len(loss_recorder)], loss_recorder


def fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, amp=False, log=None, rank=0):
    if amp:
        scaler = torch.amp.GradScaler("cuda")

    model.train()
    epoch_train_loss, batch_cnt = 0, 0
    pbar = tqdm(train_loader, dynamic_ncols=True, disable=(rank != 0))
    for images, targets in pbar:
        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)

        optimizer.zero_grad()
        if amp:
            with torch.amp.autocast("cuda"):
                train_loss = model(images, target=targets)["loss"]
            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss = model(images, target=targets)["loss"]
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

        scheduler.step()
        last_lr = scheduler.get_last_lr()[0]

        pbar.set_description(f"Training loss: {train_loss.item():.6f} | LR: {last_lr:.6f}")
        if log:
            log(train_loss=train_loss.item(), lr=last_lr)

        epoch_train_loss += train_loss.item()
        batch_cnt += 1

    epoch_train_loss /= batch_cnt
    return epoch_train_loss, last_lr


@torch.no_grad()
def evaluate(model, val_loader, batch_transforms, val_metric, amp=False, log=None):
    model.eval()
    val_metric.reset()
    val_loss, batch_cnt = 0, 0
    pbar = tqdm(val_loader, dynamic_ncols=True)
    for images, targets in pbar:
        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)
        if amp:
            with torch.amp.autocast("cuda"):
                out = model(images, target=targets, return_preds=True)
        else:
            out = model(images, target=targets, return_preds=True)

        # Cells & logical coords are compared in the (relative) model-input space
        for target, pred in zip(targets, out["preds"]):
            val_metric.update(
                np.asarray(target["cells"], dtype=np.float32).reshape(-1, 4, 2),
                np.asarray(target["logic"], dtype=np.int64).reshape(-1, 4),
                pred["polygons"],
                pred["logical"],
            )

        pbar.set_description(f"Validation loss: {out['loss'].item():.6f}")
        if log:
            log(val_loss=out["loss"].item())
        val_loss += out["loss"].item()
        batch_cnt += 1

    val_loss /= batch_cnt
    metrics = val_metric.summary()
    return val_loss, metrics["recall"], metrics["precision"], metrics["f1"], metrics["structure_acc"]


def main(args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend=args.backend)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        rank = 0
        if isinstance(args.device, int):
            if not torch.cuda.is_available():
                raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
            if args.device >= torch.cuda.device_count():
                raise ValueError("Invalid device index")
            device = torch.device("cuda", args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda", 0)
        else:
            logging.warning("No accessible GPU, target device set to CPU.")
            device = torch.device("cpu")

    slack_token = os.getenv("TQDM_SLACK_TOKEN")
    slack_channel = os.getenv("TQDM_SLACK_CHANNEL")
    pbar = tqdm(disable=False if (slack_token and slack_channel) and (rank == 0) else True)
    if slack_token and slack_channel:
        pbar.write = lambda msg: pbar.sio.client.chat_postMessage(channel=slack_channel, text=msg)
    pbar.write(str(args))

    if not isinstance(args.workers, int):
        args.workers = min(16, multiprocessing.cpu_count())

    if rank == 0 and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True

    # Temporary model to recover the configuration (mean/std)
    tmp_model = table_structure.__dict__[args.arch](pretrained=False)
    mean, std = tmp_model.cfg["mean"], tmp_model.cfg["std"]

    # Validation data
    val_hash = None
    if rank == 0:
        st = time.time()
        val_set = TableStructureDataset(
            img_folder=os.path.join(args.val_path, "images"),
            label_path=os.path.join(args.val_path, "labels.json"),
            sample_transforms=T.SampleCompose([
                T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
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
        pbar.write(
            f"Validation set loaded in {time.time() - st:.4f}s ({len(val_set)} samples in {len(val_loader)} batches)"
        )
        with open(os.path.join(args.val_path, "labels.json"), "rb") as f:
            val_hash = hashlib.sha256(f.read()).hexdigest()

    batch_transforms = Normalize(mean=mean, std=std)

    model = table_structure.__dict__[args.arch](pretrained=args.pretrained)
    if isinstance(args.resume, str):
        pbar.write(f"Resuming {args.resume}")
        model.from_pretrained(args.resume)

    if rank == 0:
        val_metric = TableCellMetric(iou_thresh=args.iou_thresh)

    if rank == 0 and args.test_only:
        pbar.write("Running evaluation")
        model = model.to(device)
        val_loss, recall, precision, f1, struct = evaluate(
            model, val_loader, batch_transforms, val_metric, amp=args.amp
        )
        pbar.write(
            f"Validation loss: {val_loss:.6f} | Recall: {(recall or 0):.2%} | Precision: {(precision or 0):.2%} "
            f"| F1: {(f1 or 0):.2%} | Structure acc: {(struct or 0):.2%}"
        )
        return

    st = time.time()
    # Image-only augmentations
    img_transforms = T.OneOf([
        Compose([
            T.RandomApply(T.ColorInversion(), 0.3),
            T.RandomApply(T.GaussianBlur(sigma=(0.5, 1.5)), 0.2),
        ]),
        T.ImageTorchvisionTransform(RandomPhotometricDistort(p=0.3)),
        T.ImageTorchvisionTransform(RandomGrayscale(p=0.1)),
        lambda x: x,  # identity
    ])
    # Image + geometry augmentations (letterbox to a square; the model renders the dense targets)
    sample_transforms = T.SampleCompose([
        T.RandomHorizontalFlip(0.15),
        T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
    ])

    train_set = TableStructureDataset(
        img_folder=os.path.join(args.train_path, "images"),
        label_path=os.path.join(args.train_path, "labels.json"),
        img_transforms=img_transforms,
        sample_transforms=sample_transforms,
    )
    sampler = (
        DistributedSampler(train_set, rank=rank, shuffle=True, drop_last=True)
        if distributed
        else RandomSampler(train_set)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_set.collate_fn,
    )
    if rank == 0:
        pbar.write(
            f"Train set loaded in {time.time() - st:.4f}s ({len(train_set)} samples in {len(train_loader)} batches)"
        )
    with open(os.path.join(args.train_path, "labels.json"), "rb") as f:
        train_hash = hashlib.sha256(f.read()).hexdigest()

    if rank == 0 and args.show_samples:
        images, targets = next(iter(train_loader))
        plot_samples(images, targets)
        return

    if args.freeze_backbone:
        for p in model.feat_extractor.parameters():
            p.requires_grad = False

    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[rank])

    backbone_lr = args.lr * 0.1 if args.pretrained or args.resume is not None else args.lr
    param_groups = build_param_groups(
        model, lr=args.lr, backbone_lr=backbone_lr, weight_decay=args.weight_decay or 1e-4
    )
    optimizer = (
        torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        if args.optim == "adamw"
        else torch.optim.Adam(param_groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    )

    if rank == 0 and args.find_lr:
        lrs, losses = record_lr(model, train_loader, batch_transforms, optimizer, amp=args.amp)
        plot_recorder(lrs, losses)
        return

    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, min(2000, int(0.05 * total_steps)))
    if args.sched == "cosine":
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=args.lr * 0.01)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    elif args.sched == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[g["lr"] for g in optimizer.param_groups],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            div_factor=100,
            final_div_factor=100,
            anneal_strategy="cos",
        )
    else:  # poly
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        poly = PolynomialLR(optimizer, total_iters=total_steps - warmup_steps, power=1.0)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, poly], milestones=[warmup_steps])

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.arch}_{current_time}" if args.name is None else args.name
    if rank == 0:
        config = {
            "learning_rate": args.lr,
            "backbone_learning_rate": backbone_lr,
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
            "amp": args.amp,
        }

    global global_step
    global_step = 0

    if args.wb:
        import wandb

        run = wandb.init(name=exp_name, project="table-structure-recognition", config=config)

        def wandb_log_at_step(train_loss=None, val_loss=None, lr=None):
            wandb.log({
                **({"train_loss_step": train_loss} if train_loss is not None else {}),
                **({"val_loss_step": val_loss} if val_loss is not None else {}),
                **({"step_lr": lr} if lr is not None else {}),
            })

    if args.clearml:
        from clearml import Logger, Task

        task = Task.init(project_name="docTR/table-structure-recognition", task_name=exp_name, reuse_last_task_id=False)
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

    def log_at_step(train_loss=None, val_loss=None, lr=None):
        global global_step
        if args.wb:
            wandb_log_at_step(train_loss, val_loss, lr)
        if args.clearml:
            clearml_log_at_step(train_loss, val_loss, lr)
        global_step += 1

    min_loss = np.inf
    if args.early_stop:
        early_stopper = EarlyStopper(patience=args.early_stop_epochs, min_delta=args.early_stop_delta)

    for epoch in range(args.epochs):
        if distributed:
            sampler.set_epoch(epoch)
        train_loss, actual_lr = fit_one_epoch(
            model, train_loader, batch_transforms, optimizer, scheduler, amp=args.amp, log=log_at_step, rank=rank
        )

        if rank == 0:
            pbar.write(f"Epoch {epoch + 1}/{args.epochs} - Training loss: {train_loss:.6f} | LR: {actual_lr:.6f}")
            val_loss, recall, precision, f1, struct = evaluate(
                model, val_loader, batch_transforms, val_metric, amp=args.amp, log=log_at_step
            )
            params = model.module if hasattr(model, "module") else model
            if val_loss < min_loss:
                pbar.write(f"Validation loss decreased {min_loss:.6f} --> {val_loss:.6f}: saving state...")
                torch.save(params.state_dict(), Path(args.output_dir) / f"{exp_name}.pt")
                min_loss = val_loss
            if args.save_interval_epoch:
                torch.save(params.state_dict(), Path(args.output_dir) / f"{exp_name}_epoch{epoch + 1}.pt")
            log_msg = f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6f} "
            if any(v is None for v in (recall, precision, f1)):
                log_msg += "(Undefined metric value, caused by empty GTs or predictions)"
            else:
                log_msg += (
                    f"| Recall: {recall:.2%} | Precision: {precision:.2%} "
                    f"| F1: {f1:.2%} | Structure acc: {(struct or 0):.2%}"
                )
            pbar.write(log_msg)
            if args.wb:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": actual_lr,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                    "structure_acc": struct,
                })
            if args.early_stop and early_stopper.early_stop(val_loss):
                pbar.write("Training halted early due to reaching patience limit.")
                break

    if rank == 0 and args.wb:
        run.finish()
    if distributed:
        dist.destroy_process_group()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="docTR training script for table structure recognition (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend", default="nccl", type=str, help="Backend to use for torch.distributed")
    parser.add_argument(
        "--device", default=None, type=int, help="GPU index for single-GPU training (ignored under DDP)"
    )
    parser.add_argument("arch", type=str, help="table model to train")
    parser.add_argument("--output_dir", type=str, default=".", help="path to save checkpoints and final model")
    parser.add_argument(
        "--train_path", type=str, required=True, help="path to the training data folder (images/ + labels.json)"
    )
    parser.add_argument(
        "--val_path", type=str, required=True, help="path to the validation data folder (images/ + labels.json)"
    )
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size for training")
    parser.add_argument(
        "--save-interval-epoch", dest="save_interval_epoch", action="store_true", help="Save model every epoch"
    )
    parser.add_argument("--input_size", type=int, default=1024, help="model input size, H = W")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for the optimizer (Adam or AdamW)")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for cell matching in the metric")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Run the validation loop")
    parser.add_argument(
        "--freeze-backbone", dest="freeze_backbone", action="store_true", help="freeze model backbone for fine-tuning"
    )
    parser.add_argument(
        "--show-samples", dest="show_samples", action="store_true", help="Display unnormalized training samples"
    )
    parser.add_argument("--wb", dest="wb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--clearml", dest="clearml", action="store_true", help="Log to ClearML")
    parser.add_argument(
        "--pretrained", dest="pretrained", action="store_true", help="Load pretrained parameters before training"
    )
    parser.add_argument("--optim", type=str, default="adamw", choices=["adam", "adamw"], help="optimizer to use")
    parser.add_argument(
        "--sched", type=str, default="cosine", choices=["cosine", "onecycle", "poly"], help="scheduler to use"
    )
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    parser.add_argument("--find-lr", action="store_true", help="Gridsearch the optimal LR")
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping")
    parser.add_argument("--early-stop-epochs", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--early-stop-delta", type=float, default=0.01, help="Minimum Delta for early stopping")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
