# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import datetime
import hashlib
import logging
import multiprocessing
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
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
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    RandomPerspective,
    RandomPhotometricDistort,
)

if os.getenv("TQDM_SLACK_TOKEN") and os.getenv("TQDM_SLACK_CHANNEL"):
    from tqdm.contrib.slack import tqdm
else:
    from tqdm.auto import tqdm

from ddp_utils import ShardSampler, barrier_download, is_main_rank, reduce_sum, sync_val_metric

from doctr import transforms as T
from doctr.datasets import RecognitionDataset, tokenize_latex
from doctr.models import formula, login_to_hub, push_to_hf_hub
from doctr.utils.metrics import FormulaMetric
from utils import (
    EarlyStopper,
    amp_dtype,
    filter_samples,
    load_vocab,
    model_device,
    plot_recorder,
    plot_samples,
    resolve_device,
    run_metadata,
    save_checkpoint,
    save_run_metadata,
)

AMP_DTYPE = torch.float16  # set from --amp-dtype in main()


def _autocast():
    return torch.amp.autocast("cuda", dtype=AMP_DTYPE)


def _scaler():
    # bfloat16 has the range of float32: no loss scaling needed (GradScaler only makes sense for float16)
    return torch.amp.GradScaler("cuda", enabled=AMP_DTYPE == torch.float16)


def split_batch(images: torch.Tensor, targets: list[str], max_tokens: int) -> list[tuple]:
    """Split a batch into chunks of at most `max_tokens` padded target tokens, to bound the memory of long formulas

    The samples are sorted by length, so that the long ones do not pad the short ones. A batch which fits is returned as
    a single chunk.

    Returns:
        a list of (images, targets, share of the target tokens of the batch) chunks
    """
    lengths = [len(tokenize_latex(target)) + 1 for target in targets]  # tokens + EOS
    if len(targets) * max(lengths) <= max_tokens:
        return [(images, targets, 1.0)]
    total = sum(lengths)
    chunks: list[list[int]] = [[]]
    for idx in sorted(range(len(targets)), key=lengths.__getitem__):
        # The samples come by increasing length: the current one is the longest of its chunk
        if chunks[-1] and (len(chunks[-1]) + 1) * lengths[idx] > max_tokens:
            chunks.append([])
        chunks[-1].append(idx)
    return [
        (images[chunk], [targets[idx] for idx in chunk], sum(lengths[idx] for idx in chunk) / total) for chunk in chunks
    ]


def train_step(model, images, targets, optimizer, scaler=None, max_tokens: int = 32768) -> torch.Tensor:
    """Forward, backward & update on a batch, accumulating the gradients of its chunks (cf. `split_batch`)

    The loss of each chunk is weighted by its share of the target tokens, so the update is the one of the whole batch.

    Returns:
        the loss of the batch
    """
    optimizer.zero_grad()
    chunks = split_batch(images, targets, max_tokens)
    batch_loss = torch.zeros((), device=images.device)
    for idx, (chunk_images, chunk_targets, weight) in enumerate(chunks):
        # With DDP, synchronize the gradients on the last chunk only
        sync = idx == len(chunks) - 1 or not hasattr(model, "no_sync")
        with nullcontext() if sync else model.no_sync():
            if scaler is not None:
                with _autocast():
                    loss = model(chunk_images, chunk_targets)["loss"] * weight
                scaler.scale(loss).backward()
            else:
                loss = model(chunk_images, chunk_targets)["loss"] * weight
                loss.backward()
        batch_loss += loss.detach()
    if scaler is not None:
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        # Update the params
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    return batch_loss


def record_lr(
    model: torch.nn.Module,
    train_loader: DataLoader,
    batch_transforms,
    optimizer,
    start_lr: float = 1e-7,
    end_lr: float = 1,
    num_it: int = 100,
    amp: bool = False,
    max_tokens: int = 32768,
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

    scaler = _scaler() if amp else None

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(model_device(model), non_blocking=True)

        images = batch_transforms(images)

        # Forward, Backward & update
        train_loss = train_step(model, images, targets, optimizer, scaler, max_tokens)
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


def fit_one_epoch(
    model,
    device,
    train_loader,
    batch_transforms,
    optimizer,
    scheduler,
    amp=False,
    log=None,
    rank=0,
    max_tokens: int = 32768,
):
    scaler = _scaler() if amp else None

    model.train()
    # Iterate over the batches of the dataset
    epoch_train_loss, batch_cnt = 0, 0
    pbar = tqdm(train_loader, dynamic_ncols=True, disable=(rank != 0))
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        images = batch_transforms(images)

        train_loss = train_step(model, images, targets, optimizer, scaler, max_tokens)

        scheduler.step()
        last_lr = scheduler.get_last_lr()[0]

        pbar.set_description(f"Training loss: {train_loss.item():.6} | LR: {last_lr:.6}")
        if log:
            log(train_loss=train_loss.item(), lr=last_lr)

        epoch_train_loss += train_loss.item()
        batch_cnt += 1

    epoch_train_loss /= batch_cnt
    return epoch_train_loss, last_lr


@torch.no_grad()
def evaluate(model, device, val_loader, batch_transforms, val_metric, amp=False, log=None):
    # Evaluate the underlying module: DDP's forward hooks add nothing under no_grad and
    # would otherwise tie every rank to an identical number of batches.
    model = model.module if hasattr(model, "module") else model
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Validation loop
    # Weight by samples, not batches, so the result is independent of the sharding
    val_loss, sample_cnt = 0, 0
    pbar = tqdm(val_loader, dynamic_ncols=True, disable=not is_main_rank())
    for images, targets in pbar:
        images = images.to(device)
        images = batch_transforms(images)
        # The loss is computed with teacher forcing, the predictions are decoded greedily
        if amp:
            with _autocast():
                out = model(images, targets, return_preds=True)
        else:
            out = model(images, targets, return_preds=True)
        # Compute metric
        preds = [latex for latex, _ in out["preds"]]
        val_metric.update(targets, preds)

        pbar.set_description(f"Validation loss: {out['loss'].item():.6}")
        if log:
            log(val_loss=out["loss"].item())

        val_loss += out["loss"].item() * images.shape[0]
        sample_cnt += images.shape[0]

    val_loss, sample_cnt = reduce_sum([val_loss, float(sample_cnt)])
    sync_val_metric(val_metric, counters=val_metric.counters)
    val_loss /= sample_cnt
    return val_loss, val_metric.summary()


def _load_dataset(path: Path, img_transforms) -> RecognitionDataset:
    """Load a formula dataset (`images/` + `labels.json`), or the union of the datasets of the subfolders of `path`"""
    parts = [path] if path.joinpath("labels.json").is_file() else sorted(p for p in path.iterdir() if p.is_dir())
    dataset = RecognitionDataset(
        str(parts[0].joinpath("images")), str(parts[0].joinpath("labels.json")), img_transforms=img_transforms
    )
    for subfolder in parts[1:]:
        dataset.merge_dataset(
            RecognitionDataset(str(subfolder.joinpath("images")), str(subfolder.joinpath("labels.json")))
        )
    return dataset


def _labels_hash(path: Path) -> str:
    parts = [path] if path.joinpath("labels.json").is_file() else sorted(p for p in path.iterdir() if p.is_dir())
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.joinpath("labels.json").read_bytes())
    return digest.hexdigest()


def main(args):
    global AMP_DTYPE
    AMP_DTYPE = amp_dtype(args.amp_dtype)
    # Detect distributed setup
    # variable is set by torchrun
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    # GPU setup
    if distributed:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend=args.backend, device_id=device)
    else:
        # single process
        rank = 0
        device = resolve_device(args.device)
        if device.type == "cpu":
            logging.warning("No accessible GPU, target device set to CPU.")  # noqa: LOG015
    if args.amp and device.type != "cuda":
        raise ValueError("--amp (automatic mixed precision) is only supported on CUDA devices")
    use_cuda = device.type == "cuda"

    slack_token = os.getenv("TQDM_SLACK_TOKEN")
    slack_channel = os.getenv("TQDM_SLACK_CHANNEL")

    pbar = tqdm(disable=False if (slack_token and slack_channel) and (rank == 0) else True)
    if slack_token and slack_channel:
        # Monkey patch tqdm write method to send messages directly to Slack
        pbar.write = lambda msg: pbar.sio.client.chat_postMessage(channel=slack_channel, text=msg)
    pbar.write(str(args))

    if rank == 0 and args.push_to_hub:
        login_to_hub()

    if not isinstance(args.workers, int):
        args.workers = min(16, multiprocessing.cpu_count())

    if use_cuda:
        torch.backends.cudnn.benchmark = True

    vocab = load_vocab(args.vocab)
    input_shape = (3, *args.input_size)

    # Load doctr model
    with barrier_download(rank, distributed):
        model = formula.__dict__[args.arch](
            pretrained=args.pretrained,
            vocab=vocab,
            input_shape=input_shape,
            max_length=args.max_length,
            max_decode_length=args.max_decode_length,
        )
    batch_transforms = Normalize(mean=model.cfg["mean"], std=model.cfg["std"])

    # Load val data
    st = time.time()
    val_path = Path(args.val_path)
    val_hash = _labels_hash(val_path)
    val_set = _load_dataset(
        val_path,
        img_transforms=T.Resize(args.input_size, preserve_aspect_ratio=True),
    )
    unknown, too_long = filter_samples(val_set, vocab, args.max_length)
    if rank == 0 and (unknown or too_long):
        pbar.write(f"Validation set: {unknown} samples with unknown tokens and {too_long} too long samples skipped")

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        # Shard without padding so no sample is validated twice (see ShardSampler)
        sampler=ShardSampler(val_set, rank=rank) if distributed else SequentialSampler(val_set),
        pin_memory=use_cuda,
        collate_fn=val_set.collate_fn,
    )
    if rank == 0:
        batch_info = f"{len(val_loader)} batches/rank" if distributed else f"{len(val_loader)} batches"
        pbar.write(f"Validation set loaded in {time.time() - st:.4}s ({len(val_set)} samples in {batch_info})")

    # Resume weights
    if isinstance(args.resume, str):
        pbar.write(f"Resuming {args.resume}")
        model.from_pretrained(args.resume)

    # Backbone freezing
    if args.freeze_backbone:
        for p in model.feat_extractor.parameters():
            p.requires_grad = False

    if use_cuda:
        torch.cuda.set_device(device)
    model = model.to(device)

    if distributed:
        # construct DDP model
        model = DDP(model, device_ids=[rank])

    # Metrics
    val_metric = FormulaMetric()

    if args.test_only:
        if rank == 0:
            pbar.write("Running evaluation")
        val_loss, result = evaluate(model, device, val_loader, batch_transforms, val_metric, amp=args.amp)
        if rank == 0:
            pbar.write(
                f"Validation loss: {val_loss:.6} (Exact match: {result['exact_match']:.2%} | "
                f"Edit similarity: {result['edit_similarity']:.2%} | BLEU: {result['bleu']:.4})"
            )
        if distributed:
            dist.destroy_process_group()
        return

    # Load train data
    st = time.time()
    train_path = Path(args.train_path)
    train_hash = _labels_hash(train_path)
    train_set = _load_dataset(
        train_path,
        img_transforms=Compose([
            # The crops of layout regions come with margins: learn to read them as they are
            T.RandomPad(max_ratio=0.5, p=0.5),
            T.Resize(args.input_size, preserve_aspect_ratio=True),
            # Augmentations
            T.ImageTorchvisionTransform(RandomPhotometricDistort(p=0.1)),
            T.RandomApply(T.GaussianNoise(mean=0, std=0.05), 0.1),
            T.RandomApply(T.GaussianBlur(sigma=(0.5, 1.0)), 0.2),
            T.ImageTorchvisionTransform(RandomPerspective(distortion_scale=0.1, p=0.2)),
        ]),
    )
    unknown, too_long = filter_samples(train_set, vocab, args.max_length)
    if rank == 0 and (unknown or too_long):
        pbar.write(f"Training set: {unknown} samples with unknown tokens and {too_long} too long samples skipped")

    if distributed:
        sampler = DistributedSampler(train_set, rank=rank, shuffle=True, drop_last=True)
    else:
        sampler = RandomSampler(train_set)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        sampler=sampler,
        pin_memory=use_cuda,
        collate_fn=train_set.collate_fn,
    )
    if rank == 0:
        pbar.write(
            f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in {len(train_loader)} batches)"
        )

    if args.show_samples:
        if rank == 0:
            x, target = next(iter(train_loader))
            plot_samples(x, target)
        # Rank 0 arrives late here, so rendezvous before every rank leaves
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
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

    # LR finder
    if args.find_lr:
        if rank == 0:
            # Unwrap DDP: rank 0 runs alone here, gradient sync would deadlock
            lrs, losses = record_lr(
                model.module if hasattr(model, "module") else model,
                train_loader,
                batch_transforms,
                optimizer,
                amp=args.amp,
                max_tokens=args.max_tokens,
            )
            plot_recorder(lrs, losses)
        # Rank 0 arrives late here, so rendezvous before every rank leaves
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    # Scheduler, after a linear warmup
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.warmup_steps, total_steps // 2)
    main_steps = total_steps - warmup_steps
    if args.sched == "cosine":
        scheduler = CosineAnnealingLR(optimizer, main_steps, eta_min=args.lr / 25e4)
    elif args.sched == "onecycle":
        scheduler = OneCycleLR(optimizer, args.lr, main_steps)
    elif args.sched == "poly":
        scheduler = PolynomialLR(optimizer, main_steps)
    if warmup_steps > 0:
        scheduler = SequentialLR(
            optimizer,
            [LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps), scheduler],
            milestones=[warmup_steps],
        )

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
            "max_length": args.max_length,
            "max_decode_length": args.max_decode_length,
            "max_tokens": args.max_tokens,
            "optimizer": args.optim,
            "framework": "pytorch",
            "scheduler": args.sched,
            "warmup_steps": warmup_steps,
            "vocab": args.vocab,
            "train_hash": train_hash,
            "val_hash": val_hash,
            "pretrained": args.pretrained,
            "amp": args.amp,
        }
        # the run metadata describes the whole run: written once, next to the checkpoint `<exp_name>.pt`
        save_run_metadata(
            args.output_dir,
            exp_name,
            {
                **config,
                **run_metadata(
                    args,
                    task="formula",
                    vocab=vocab,
                    input_shape=input_shape,
                    max_length=args.max_length,
                ),
            },
        )

    global global_step
    global_step = 0  # Shared global step counter
    # W&B
    if rank == 0 and args.wb:
        import wandb

        run = wandb.init(
            name=exp_name,
            project="formula-recognition",
            config=config,
        )

        def wandb_log_at_step(train_loss=None, val_loss=None, lr=None):
            wandb.log({
                **({"train_loss_step": train_loss} if train_loss is not None else {}),
                **({"val_loss_step": val_loss} if val_loss is not None else {}),
                **({"step_lr": lr} if lr is not None else {}),
            })

    # ClearML
    if rank == 0 and args.clearml:
        from clearml import Logger, Task

        task = Task.init(project_name="docTR/formula-recognition", task_name=exp_name, reuse_last_task_id=False)
        task.upload_artifact("config", config)

        def clearml_log_at_step(train_loss=None, val_loss=None, lr=None):
            logger = Logger.current_logger()
            if train_loss is not None:
                logger.report_scalar(
                    title="Training Step Loss",
                    series="train_loss_step",
                    iteration=global_step,
                    value=train_loss,
                )
            if val_loss is not None:
                logger.report_scalar(
                    title="Validation Step Loss",
                    series="val_loss_step",
                    iteration=global_step,
                    value=val_loss,
                )
            if lr is not None:
                logger.report_scalar(
                    title="Step Learning Rate",
                    series="step_lr",
                    iteration=global_step,
                    value=lr,
                )

    def log_at_step(train_loss=None, val_loss=None, lr=None):
        global global_step
        if args.wb and rank == 0:
            wandb_log_at_step(train_loss, val_loss, lr)
        if args.clearml and rank == 0:
            clearml_log_at_step(train_loss, val_loss, lr)
        global_step += 1  # Increment the shared global step counter

    # Create loss queue
    min_loss = np.inf
    if args.early_stop:
        early_stopper = EarlyStopper(patience=args.early_stop_epochs, min_delta=args.early_stop_delta)
    # Training loop
    for epoch in range(args.epochs):
        if distributed:
            sampler.set_epoch(epoch)
        train_loss, actual_lr = fit_one_epoch(
            model,
            device,
            train_loader,
            batch_transforms,
            optimizer,
            scheduler,
            amp=args.amp,
            log=log_at_step,
            rank=rank,
            max_tokens=args.max_tokens,
        )

        # Validation loop at the end of each epoch
        val_loss, result = evaluate(
            model, device, val_loader, batch_transforms, val_metric, amp=args.amp, log=log_at_step
        )

        if rank == 0:
            pbar.write(f"Epoch {epoch + 1}/{args.epochs} - Training loss: {train_loss:.6} | LR: {actual_lr:.6}")
            params = model.module if hasattr(model, "module") else model
            if val_loss < min_loss:
                pbar.write(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
                save_checkpoint(params, args.output_dir, exp_name)
                min_loss = val_loss
            if args.save_interval_epoch:
                save_checkpoint(params, args.output_dir, f"{exp_name}_epoch{epoch + 1}")
            pbar.write(
                f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} "
                f"(Exact match: {result['exact_match']:.2%} | Edit similarity: {result['edit_similarity']:.2%} | "
                f"BLEU: {result['bleu']:.4})"
            )
            # W&B
            if args.wb:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": actual_lr,
                    **result,
                })

            # ClearML
            if args.clearml:
                from clearml import Logger

                logger = Logger.current_logger()
                logger.report_scalar(title="Training Loss", series="train_loss", value=train_loss, iteration=epoch)
                logger.report_scalar(title="Validation Loss", series="val_loss", value=val_loss, iteration=epoch)
                logger.report_scalar(title="Learning Rate", series="lr", value=actual_lr, iteration=epoch)
                for name, value in result.items():
                    logger.report_scalar(title=name, series=name, value=value, iteration=epoch)

        if args.early_stop and early_stopper.early_stop(val_loss):
            if rank == 0:
                pbar.write("Training halted early due to reaching patience limit.")
            break

    if rank == 0:
        if args.wb:
            run.finish()

        if args.push_to_hub:
            push_to_hf_hub(model, exp_name, task="formula", run_config=args)

    if distributed:
        dist.destroy_process_group()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR training script for formula recognition (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # DDP related args
    parser.add_argument("--backend", default="nccl", type=str, help="Backend to use for torch.distributed")

    parser.add_argument("arch", type=str, help="formula recognition model to train")
    parser.add_argument("--output_dir", type=str, default=".", help="path to save checkpoints and final model")
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="path to the train data folder (images/ + labels.json), or to a folder of such folders",
    )
    parser.add_argument("--val_path", type=str, required=True, help="path to the validation data folder")
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="batch size for training")
    parser.add_argument(
        "--save-interval-epoch", dest="save_interval_epoch", action="store_true", help="Save model every epoch"
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=(256, 768),
        metavar=("H", "W"),
        help="input size (H, W) of the model: the formulas are resized to it, keeping their aspect ratio",
    )
    parser.add_argument("--max_length", type=int, default=4096, help="maximum number of tokens of a formula")
    parser.add_argument(
        "--max_decode_length",
        type=int,
        default=2048,
        help="maximum number of tokens generated per formula by the greedy decoding of the predictions",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="maximum number of padded target tokens per forward pass: longer batches are split into chunks whose "
        "gradients are accumulated, to bound the memory of long formulas",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default=None,
        help="path to a JSON list of LaTeX tokens (cf. build_vocab.py), the default LaTeX tokens of docTR if omitted",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device for single-process runs: a CUDA index (e.g. 0), 'cuda:N', 'mps' (Apple Silicon) or 'cpu'. "
        "Default: CUDA if available, else MPS, else CPU.",
    )

    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam or AdamW)")
    parser.add_argument("--wd", "--weight-decay", default=0.01, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="number of linear warmup steps of the LR")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
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
    parser.add_argument("--optim", type=str, default="adamw", choices=["adam", "adamw"], help="optimizer to use")
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
    args.input_size = tuple(args.input_size)

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
