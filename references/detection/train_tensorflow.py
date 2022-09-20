# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os

os.environ["USE_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import datetime
import hashlib
import multiprocessing as mp
import time

import numpy as np
import psutil
import tensorflow as tf
import wandb
from fastprogress.fastprogress import master_bar, progress_bar
from tensorflow.keras import mixed_precision

from doctr.models import login_to_hub, push_to_hf_hub

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr import transforms as T
from doctr.datasets import DataLoader, DetectionDataset
from doctr.models import detection
from doctr.utils.metrics import LocalizationConfusion
from utils import plot_recorder, plot_samples


def record_lr(
    model: tf.keras.Model,
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
            train_loss = model(images, targets, training=True)["loss"]
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


def fit_one_epoch(model, train_loader, batch_transforms, optimizer, mb, amp=False):
    train_iter = iter(train_loader)
    # Iterate over the batches of the dataset
    for images, targets in progress_bar(train_iter, parent=mb):

        images = batch_transforms(images)

        with tf.GradientTape() as tape:
            train_loss = model(images, targets, training=True)["loss"]
        grads = tape.gradient(train_loss, model.trainable_weights)
        if amp:
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        mb.child.comment = f"Training loss: {train_loss.numpy():.6}"


def evaluate(model, val_loader, batch_transforms, val_metric):
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    val_iter = iter(val_loader)
    for images, targets in val_iter:
        images = batch_transforms(images)
        out = model(images, targets, training=False, return_preds=True)
        # Compute metric
        loc_preds = out["preds"]
        for boxes_gt, boxes_pred in zip(targets, loc_preds):
            if args.rotation and args.eval_straight:
                # Convert pred to boxes [xmin, ymin, xmax, ymax]  N, 4, 2 --> N, 4
                boxes_pred = np.concatenate((boxes_pred.min(axis=1), boxes_pred.max(axis=1)), axis=-1)
            val_metric.update(gts=boxes_gt, preds=boxes_pred[:, :4])

        val_loss += out["loss"].numpy()
        batch_cnt += 1

    val_loss /= batch_cnt
    recall, precision, mean_iou = val_metric.summary()
    return val_loss, recall, precision, mean_iou


def main(args):

    print(args)

    if args.push_to_hub:
        login_to_hub()

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    system_available_memory = int(psutil.virtual_memory().available / 1024**3)

    # AMP
    if args.amp:
        mixed_precision.set_global_policy("mixed_float16")

    st = time.time()
    val_set = DetectionDataset(
        img_folder=os.path.join(args.val_path, "images"),
        label_path=os.path.join(args.val_path, "labels.json"),
        sample_transforms=T.SampleCompose(
            (
                [T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True)]
                if not args.rotation or args.eval_straight
                else []
            )
            + (
                [
                    T.Resize(args.input_size, preserve_aspect_ratio=True),  # This does not pad
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation and not args.eval_straight
                else []
            )
        ),
        use_polygons=args.rotation and not args.eval_straight,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
    )
    print(
        f"Validation set loaded in {time.time() - st:.4}s ({len(val_set)} samples in "
        f"{val_loader.num_batches} batches)"
    )
    with open(os.path.join(args.val_path, "labels.json"), "rb") as f:
        val_hash = hashlib.sha256(f.read()).hexdigest()

    batch_transforms = T.Compose(
        [
            T.Normalize(mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287)),
        ]
    )

    # Load doctr model
    model = detection.__dict__[args.arch](
        pretrained=args.pretrained,
        input_shape=(args.input_size, args.input_size, 3),
        assume_straight_pages=not args.rotation,
    )

    # Resume weights
    if isinstance(args.resume, str):
        model.load_weights(args.resume)

    # Metrics
    val_metric = LocalizationConfusion(
        use_polygons=args.rotation and not args.eval_straight,
        mask_shape=(args.input_size, args.input_size),
        use_broadcasting=True if system_available_memory > 62 else False,
    )
    if args.test_only:
        print("Running evaluation")
        val_loss, recall, precision, mean_iou = evaluate(model, val_loader, batch_transforms, val_metric)
        print(
            f"Validation loss: {val_loss:.6} (Recall: {recall:.2%} | Precision: {precision:.2%} | "
            f"Mean IoU: {mean_iou:.2%})"
        )
        return

    st = time.time()
    # Load both train and val data generators
    train_set = DetectionDataset(
        img_folder=os.path.join(args.train_path, "images"),
        label_path=os.path.join(args.train_path, "labels.json"),
        img_transforms=T.Compose(
            [
                # Augmentations
                T.RandomApply(T.ColorInversion(), 0.1),
                T.RandomJpegQuality(60),
                T.RandomSaturation(0.3),
                T.RandomContrast(0.3),
                T.RandomBrightness(0.3),
            ]
        ),
        sample_transforms=T.SampleCompose(
            (
                [T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True)]
                if not args.rotation
                else []
            )
            + (
                [
                    T.Resize(args.input_size, preserve_aspect_ratio=True),  # This does not pad
                    T.RandomRotate(90, expand=True),
                    T.Resize((args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True),
                ]
                if args.rotation
                else []
            )
        ),
        use_polygons=args.rotation,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )
    print(
        f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in "
        f"{train_loader.num_batches} batches)"
    )
    with open(os.path.join(args.train_path, "labels.json"), "rb") as f:
        train_hash = hashlib.sha256(f.read()).hexdigest()

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return

    # Optimizer
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr,
        decay_steps=args.epochs * len(train_loader),
        decay_rate=1 / (25e4),  # final lr as a fraction of initial lr
        staircase=False,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler, beta_1=0.95, beta_2=0.99, epsilon=1e-6, clipnorm=5)
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

    # W&B
    if args.wb:

        run = wandb.init(
            name=exp_name,
            project="text-detection",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "weight_decay": 0.0,
                "batch_size": args.batch_size,
                "architecture": args.arch,
                "input_size": args.input_size,
                "optimizer": "adam",
                "framework": "tensorflow",
                "scheduler": "exp_decay",
                "train_hash": train_hash,
                "val_hash": val_hash,
                "pretrained": args.pretrained,
                "rotation": args.rotation,
            },
        )

    if args.freeze_backbone:
        for layer in model.feat_extractor.layers:
            layer.trainable = False

    min_loss = np.inf

    # Training loop
    mb = master_bar(range(args.epochs))
    for epoch in mb:
        fit_one_epoch(model, train_loader, batch_transforms, optimizer, mb, args.amp)
        # Validation loop at the end of each epoch
        val_loss, recall, precision, mean_iou = evaluate(model, val_loader, batch_transforms, val_metric)
        if val_loss < min_loss:
            print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
            model.save_weights(f"./{exp_name}/weights")
            min_loss = val_loss
        log_msg = f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} "
        if any(val is None for val in (recall, precision, mean_iou)):
            log_msg += "(Undefined metric value, caused by empty GTs or predictions)"
        else:
            log_msg += f"(Recall: {recall:.2%} | Precision: {precision:.2%} | Mean IoU: {mean_iou:.2%})"
        mb.write(log_msg)
        # W&B
        if args.wb:
            wandb.log(
                {
                    "val_loss": val_loss,
                    "recall": recall,
                    "precision": precision,
                    "mean_iou": mean_iou,
                }
            )

    if args.wb:
        run.finish()

    if args.push_to_hub:
        push_to_hf_hub(model, exp_name, task="detection", run_config=args)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR training script for text detection (TensorFlow)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("train_path", type=str, help="path to training data folder")
    parser.add_argument("val_path", type=str, help="path to validation data folder")
    parser.add_argument("arch", type=str, help="text-detection model to train")
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size for training")
    parser.add_argument("--input_size", type=int, default=1024, help="model input size, H = W")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam)")
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
    parser.add_argument("--push-to-hub", dest="push_to_hub", action="store_true", help="Push to Huggingface Hub")
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Load pretrained parameters before starting the training",
    )
    parser.add_argument("--rotation", dest="rotation", action="store_true", help="train with rotated documents")
    parser.add_argument(
        "--eval-straight",
        action="store_true",
        help="metrics evaluation with straight boxes instead of polygons to save time + memory",
    )
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    parser.add_argument("--find-lr", action="store_true", help="Gridsearch the optimal LR")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
