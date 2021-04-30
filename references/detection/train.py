# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import datetime
import numpy as np
import tensorflow as tf
from collections import deque
from pathlib import Path
from fastprogress.fastprogress import master_bar, progress_bar

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr.models import detection, DetectionPreProcessor
from doctr.utils import metrics
from doctr.datasets import DetectionDataset, DataLoader
from doctr import transforms as T


def fit_one_epoch(model, train_loader, batch_transforms, optimizer, loss_q, mb, step):
    train_iter = iter(train_loader)
    # Iterate over the batches of the dataset
    for batch_step in progress_bar(range(train_loader.num_batches), parent=mb):
        images, targets = next(train_iter)

        boxes = [target['boxes'] for target in targets]
        flags = [target['flags'] for target in targets]
        images = batch_transforms(images)

        with tf.GradientTape() as tape:
            model_output = model(images, training=True)
            train_loss = model.compute_loss(model_output, boxes, flags)
        grads = tape.gradient(train_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        mb.child.comment = f'Training loss: {train_loss.numpy():.6}'
        # Update steps
        step.assign_add(args.batch_size)
        # Add loss to queue
        loss_q.append(np.mean(train_loss))
        # Log loss and save weights every 100 batch step
        if batch_step % 100 == 0:
            # Compute loss
            loss = sum(loss_q) / len(loss_q)
            with tb_writer.as_default():
                tf.summary.scalar('train_loss', loss, step=step)


def evaluate(model, val_loader, batch_transforms, val_metric):
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    val_iter = iter(val_loader)
    for images, targets in val_iter:
        boxes = [target['boxes'] for target in targets]
        flags = [target['flags'] for target in targets]
        images = batch_transforms(images)
        # If we want to compute val loss, we need to pass training=True to have a thresh_map
        model_output = model(images, training=True)
        loss = model.compute_loss(model_output, boxes, flags)
        decoded = model.postprocessor(model_output)
        # Compute metric
        for boxes_gt, boxes_pred in zip(boxes, decoded):
            boxes_pred = np.array(boxes_pred)[:, :-1]  # Remove scores
            val_metric.update(gts=boxes_gt, preds=boxes_pred)

        val_loss += loss.numpy()
        batch_cnt += 1

    val_loss /= batch_cnt
    recall, precision, mean_iou = val_metric.summary()
    return val_loss, recall, precision, mean_iou


def main(args):

    print(args)

    st = time.time()
    # Load both train and val data generators
    train_set = DetectionDataset(
        img_folder=os.path.join(args.data_path, 'train'),
        label_folder=os.path.join(args.data_path, 'train_labels'),
        sample_transforms=T.Compose([
            T.LambdaTransformation(lambda x: x / 255),
            T.Resize((args.input_size, args.input_size)),
            # Augmentations
            T.RandomApply(T.ColorInversion(), .2),
            T.RandomJpegQuality(60),
            T.RandomSaturation(.3),
            T.RandomContrast(.3),
            T.RandomBrightness(0.3),
        ]),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, workers=args.workers)
    print(f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in "
          f"{train_loader.num_batches} batches)")

    st = time.time()
    val_set = DetectionDataset(
        img_folder=os.path.join(args.data_path, 'val'),
        label_folder=os.path.join(args.data_path, 'val_labels'),
        sample_transforms=T.Compose([
            T.LambdaTransformation(lambda x: x / 255),
            T.Resize((args.input_size, args.input_size)),
        ])
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, workers=args.workers)
    print(f"Validation set loaded in {time.time() - st:.4s} ({len(val_set)} samples in "
          f"{val_loader.num_batches} batches)")

    batch_transforms = T.Compose([
        T.Normalize(mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287)),
    ])

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=5)

    # Load doctr model
    model = detection.__dict__[args.model](pretrained=False, input_shape=(args.input_size, args.input_size, 3))
    # Resume weights
    if isinstance(args.resume, str):
        model.load_weights(args.resume)

    # Tf variable to log steps
    step = tf.Variable(0, dtype="int64")

    # Metrics
    val_metric = metrics.LocalizationConfusion()

    # Postprocessor to decode output (to feed metric during val step with boxes)
    postprocessor = model.postprocessor

    if args.test_only:
        print("Running evaluation")
        val_loss, recall, precision, mean_iou = evaluate(model, val_loader, batch_transforms, val_metric)
        print(f"Validation loss: {val_loss:.6} (Recall: {recall:.2%} | Precision: {precision:.2%} | "
              f"Mean IoU: {mean_iou:.2%})")
        return

    # Tensorboard to monitor training
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.model}_{current_time}" if args.name is None else args.name
    log_dir = Path('logs', exp_name)
    log_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = tf.summary.create_file_writer(str(log_dir))

    # Create loss queue
    loss_q = deque(maxlen=100)
    min_loss = np.inf

    # Training loop
    mb = master_bar(range(args.epochs))
    for epoch in mb:
        fit_one_epoch(model, train_loader, batch_transforms, optimizer, loss_q, mb, step)
        # Validation loop at the end of each epoch
        val_loss, recall, precision, mean_iou = evaluate(model, val_loader, batch_transforms, val_metric)
        if val_loss < min_loss:
            print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
            model.save_weights(f'./{exp_name}/weights')
            min_loss = val_loss
        mb.write(f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} "
                 f"(Recall: {recall:.2%} | Precision: {precision:.2%} | Mean IoU: {mean_iou:.2%})")
        # Tensorboard
        with tb_writer.as_default():
            tf.summary.scalar('val_loss', val_loss, step=step)
            tf.summary.scalar('recall', recall, step=step)
            tf.summary.scalar('precision', precision, step=step)
            tf.summary.scalar('mean_iou', mean_iou, step=step)
        # Reset val metric
        val_metric.reset()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR train text-detection model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str, help='path to data folder')
    parser.add_argument('model', type=str, help='text-detection model to train')
    parser.add_argument('--name', type=str, default=None, help='Name of your training experiment')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model on')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--input_size', type=int, default=1024, help='model input size, H = W)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the optimizer (Adam)')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of workers used for dataloading')
    parser.add_argument('--resume', type=str, default=None, help='Path to your checkpoint')
    parser.add_argument("--test-only", dest='test_only', action='store_true', help="Run the validation loop")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
