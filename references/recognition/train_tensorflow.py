# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os

from doctr.file_utils import ensure_keras_v2

ensure_keras_v2()

os.environ["USE_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import datetime
import hashlib
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, mixed_precision, optimizers

if os.getenv("TQDM_SLACK_TOKEN") and os.getenv("TQDM_SLACK_CHANNEL"):
    from tqdm.contrib.slack import tqdm
else:
    from tqdm.auto import tqdm

from doctr.models import login_to_hub, push_to_hf_hub

gpu_devices = tf.config.list_physical_devices("GPU")
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr import transforms as T
from doctr.datasets import VOCABS, DataLoader, RecognitionDataset, WordGenerator
from doctr.models import recognition
from doctr.utils.metrics import TextMatch
from utils import EarlyStopper, plot_recorder, plot_samples


def record_lr(
    model: Model,
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

    # Update param groups & LR
    gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
    optimizer.learning_rate = start_lr

    lr_recorder = [start_lr * gamma**idx for idx in range(num_it)]
    loss_recorder = []

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = batch_transforms(images)

        # Forward, Backward & update
        with tf.GradientTape() as tape:
            train_loss = model(images, target=targets, training=True)["loss"]
        grads = tape.gradient(train_loss, model.trainable_weights)

        if amp:
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        optimizer.learning_rate = optimizer.learning_rate * gamma

        # Record
        train_loss = train_loss.numpy()
        if np.any(np.isnan(train_loss)):
            if batch_idx == 0:
                raise ValueError("loss value is NaN or inf.")
            else:
                break
        loss_recorder.append(train_loss.mean())
        # Stop after the number of iterations
        if batch_idx + 1 == num_it:
            break

    return lr_recorder[: len(loss_recorder)], loss_recorder


@tf.function
def apply_grads(optimizer, grads, model):
    optimizer.apply_gradients(zip(grads, model.trainable_weights))


def fit_one_epoch(model, train_loader, batch_transforms, optimizer, amp=False, clearml_log=False):
    train_iter = iter(train_loader)
    if clearml_log:
        from clearml import Logger

        logger = Logger.current_logger()

    # Iterate over the batches of the dataset
    pbar = tqdm(train_iter, position=1)
    for images, targets in pbar:
        images = batch_transforms(images)

        with tf.GradientTape() as tape:
            train_loss = model(images, target=targets, training=True)["loss"]
        grads = tape.gradient(train_loss, model.trainable_weights)
        if amp:
            grads = optimizer.get_unscaled_gradients(grads)
        apply_grads(optimizer, grads, model)

        pbar.set_description(f"Training loss: {train_loss.numpy().mean():.6}")
        if clearml_log:
            global iteration
            logger.report_scalar(
                title="Training Loss", series="train_loss", value=train_loss.numpy().mean(), iteration=iteration
            )
            iteration += 1


def evaluate(model, val_loader, batch_transforms, val_metric):
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    val_iter = iter(val_loader)
    for images, targets in tqdm(val_iter):
        images = batch_transforms(images)
        out = model(images, target=targets, return_preds=True, training=False)
        # Compute metric
        if len(out["preds"]):
            words, _ = zip(*out["preds"])
        else:
            words = []
        val_metric.update(targets, words)

        val_loss += out["loss"].numpy().mean()
        batch_cnt += 1

    val_loss /= batch_cnt
    result = val_metric.summary()
    return val_loss, result["raw"], result["unicase"]


def main(args):
    print(args)

    if args.push_to_hub:
        login_to_hub()

    vocab = VOCABS[args.vocab]
    fonts = args.font.split(",")

    # AMP
    if args.amp:
        mixed_precision.set_global_policy("mixed_float16")

    st = time.time()

    if isinstance(args.val_path, str):
        with open(os.path.join(args.val_path, "labels.json"), "rb") as f:
            val_hash = hashlib.sha256(f.read()).hexdigest()

        # Load val data generator
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
            img_transforms=T.Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                # Ensure we have a 90% split of white-background images
                T.RandomApply(T.ColorInversion(), 0.9),
            ]),
        )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    print(
        f"Validation set loaded in {time.time() - st:.4}s ({len(val_set)} samples in {val_loader.num_batches} batches)"
    )

    # Load doctr model
    model = recognition.__dict__[args.arch](
        pretrained=args.pretrained,
        input_shape=(args.input_size, 4 * args.input_size, 3),
        vocab=vocab,
    )
    # Resume weights
    if isinstance(args.resume, str):
        model.load_weights(args.resume)

    # Metrics
    val_metric = TextMatch()

    batch_transforms = T.Compose([
        T.Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301)),
    ])

    if args.test_only:
        print("Running evaluation")
        val_loss, exact_match, partial_match = evaluate(model, val_loader, batch_transforms, val_metric)
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
            img_transforms=T.Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                # Augmentations
                T.RandomApply(T.ColorInversion(), 0.1),
                T.RandomApply(T.ToGray(num_output_channels=3), 0.1),
                T.RandomJpegQuality(60),
                T.RandomSaturation(0.3),
                T.RandomContrast(0.3),
                T.RandomBrightness(0.3),
                # T.RandomApply(T.RandomShadow(), 0.4), # Broken atm on GPU
                T.RandomApply(T.GaussianNoise(mean=0.1, std=0.1), 0.1),
                T.RandomApply(T.GaussianBlur(kernel_shape=3, std=(0.1, 0.1)), 0.3),
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
            img_transforms=T.Compose([
                T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                # Ensure we have a 90% split of white-background images
                T.RandomApply(T.ColorInversion(), 0.9),
                T.RandomApply(T.ToGray(num_output_channels=3), 0.1),
                T.RandomJpegQuality(60),
                T.RandomSaturation(0.3),
                T.RandomContrast(0.3),
                T.RandomBrightness(0.3),
                # T.RandomApply(T.RandomShadow(), 0.4),  # NOTE: RandomShadow is broken atm
                T.RandomApply(T.GaussianNoise(mean=0.1, std=0.1), 0.1),
                T.RandomApply(T.GaussianBlur(kernel_shape=3, std=(0.1, 0.1)), 0.3),
            ]),
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    print(
        f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in {train_loader.num_batches} batches)"
    )

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return

    # Scheduler
    if args.sched == "exponential":
        scheduler = optimizers.schedules.ExponentialDecay(
            args.lr,
            decay_steps=args.epochs * len(train_loader),
            decay_rate=1 / (25e4),  # final lr as a fraction of initial lr
            staircase=False,
            name="ExponentialDecay",
        )
    elif args.sched == "poly":
        scheduler = optimizers.schedules.PolynomialDecay(
            args.lr,
            decay_steps=args.epochs * len(train_loader),
            end_learning_rate=1e-7,
            power=1.0,
            cycle=False,
            name="PolynomialDecay",
        )

    # Optimizer
    if args.optim == "adam":
        optimizer = optimizers.Adam(
            learning_rate=scheduler,
            beta_1=0.95,
            beta_2=0.999,
            epsilon=1e-6,
            clipnorm=5,
            weight_decay=None if args.weight_decay == 0 else args.weight_decay,
        )
    elif args.optim == "adamw":
        optimizer = optimizers.AdamW(
            learning_rate=scheduler,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            clipnorm=5,
            weight_decay=args.weight_decay or 1e-4,
        )

    if args.amp:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    # LR Finder
    if args.find_lr:
        lrs, losses = record_lr(model, train_loader, batch_transforms, optimizer, amp=args.amp)
        plot_recorder(lrs, losses)
        return

    # Tensorboard to monitor training
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.arch}_{current_time}" if args.name is None else args.name

    config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "architecture": args.arch,
        "input_size": args.input_size,
        "optimizer": optimizer.name,
        "framework": "tensorflow",
        "scheduler": scheduler.name,
        "vocab": args.vocab,
        "train_hash": train_hash,
        "val_hash": val_hash,
        "pretrained": args.pretrained,
    }

    # W&B
    if args.wb:
        import wandb

        run = wandb.init(
            name=exp_name,
            project="text-recognition",
            config=config,
        )

    # ClearML
    if args.clearml:
        from clearml import Task

        task = Task.init(project_name="docTR/text-recognition", task_name=exp_name, reuse_last_task_id=False)
        task.upload_artifact("config", config)
        global iteration
        iteration = 0

    # Backbone freezing
    if args.freeze_backbone:
        for layer in model.feat_extractor.layers:
            layer.trainable = False

    min_loss = np.inf
    if args.early_stop:
        early_stopper = EarlyStopper(patience=args.early_stop_epochs, min_delta=args.early_stop_delta)
    # Training loop
    for epoch in range(args.epochs):
        fit_one_epoch(model, train_loader, batch_transforms, optimizer, args.amp, args.clearml)

        # Validation loop at the end of each epoch
        val_loss, exact_match, partial_match = evaluate(model, val_loader, batch_transforms, val_metric)
        if val_loss < min_loss:
            print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
            model.save_weights(Path(args.output_dir) / f"{exp_name}.weights.h5")
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
            logger.report_scalar(title="Partial Match", series="partial_match", value=partial_match, iteration=epoch)

        if args.early_stop and early_stopper.early_stop(val_loss):
            print("Training halted early due to reaching patience limit.")
            break
    if args.wb:
        run.finish()

    if args.push_to_hub:
        push_to_hf_hub(model, exp_name, task="recognition", run_config=args)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR training script for text recognition (TensorFlow)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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
    parser.add_argument("--input_size", type=int, default=32, help="input size H for the model, W = 4*H")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam or AdamW)")
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
        "--sched", type=str, default="exponential", choices=["exponential", "poly"], help="scheduler to use"
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
