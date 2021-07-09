# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os

os.environ['USE_TF'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import datetime
import numpy as np
import tensorflow as tf
from collections import deque
from pathlib import Path
from fastprogress.fastprogress import master_bar, progress_bar
import wandb

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr.models import recognition
from doctr.utils.metrics import TextMatch
from doctr.datasets import RecognitionDataset, DataLoader, VOCABS
from doctr import transforms as T

from utils import plot_samples


def fit_one_epoch(model, train_loader, batch_transforms, optimizer, loss_q, mb, step, tb_writer=None):
    train_iter = iter(train_loader)
    # Iterate over the batches of the dataset
    for batch_step in progress_bar(range(train_loader.num_batches), parent=mb):
        images, targets = next(train_iter)

        images = batch_transforms(images)

        with tf.GradientTape() as tape:
            train_loss = model(images, targets, training=True)['loss']
        grads = tape.gradient(train_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        mb.child.comment = f'Training loss: {train_loss.numpy().mean():.6}'
        # Update steps
        step.assign_add(args.batch_size)
        # Add loss to queue
        loss_q.append(np.mean(train_loss))
        # Log loss and save weights every 100 batch step
        if batch_step % 100 == 0:
            # Compute loss
            loss = sum(loss_q) / len(loss_q)
            if tb_writer is not None:
                with tb_writer.as_default():
                    tf.summary.scalar('train_loss', loss, step=step)


def evaluate(model, val_loader, batch_transforms, val_metric):
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    val_iter = iter(val_loader)
    for images, targets in val_iter:
        images = batch_transforms(images)
        out = model(images, targets, return_preds=True, training=False)
        # Compute metric
        if len(out['preds']):
            words, _ = zip(*out['preds'])
        else:
            words = []
        val_metric.update(targets, words)

        val_loss += out['loss'].numpy().mean()
        batch_cnt += 1

    val_loss /= batch_cnt
    result = val_metric.summary()
    return val_loss, result['raw'], result['unicase']


def main(args):

    print(args)

    st = time.time()
    val_set = RecognitionDataset(
        img_folder=os.path.join(args.data_path, 'val'),
        labels_path=os.path.join(args.data_path, 'val_labels.json'),
        sample_transforms=T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, workers=args.workers)
    print(f"Validation set loaded in {time.time() - st:.4}s ({len(val_set)} samples in "
          f"{val_loader.num_batches} batches)")

    # Load doctr model
    model = recognition.__dict__[args.model](
        pretrained=args.pretrained,
        input_shape=(args.input_size, 4 * args.input_size, 3),
        vocab=VOCABS['french']
    )
    # Resume weights
    if isinstance(args.resume, str):
        model.load_weights(args.resume)

    # Tf variable to log steps
    step = tf.Variable(0, dtype="int64")

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
    # Load both train and val data generators
    train_set = RecognitionDataset(
        img_folder=os.path.join(args.data_path, 'train'),
        labels_path=os.path.join(args.data_path, 'train_labels.json'),
        sample_transforms=T.Compose([
            T.RandomApply(T.ColorInversion(), .1),
            T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
            # Augmentations
            T.RandomJpegQuality(60),
            T.RandomSaturation(.3),
            T.RandomContrast(.3),
            T.RandomBrightness(.3),
        ]),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, workers=args.workers)
    print(f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in "
          f"{train_loader.num_batches} batches)")

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return

    # Optimizer
    total_steps = int(args.epochs * len(train_set) / args.batch_size)
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr,
        decay_steps=total_steps,
        decay_rate=0.01,  # final lr as a fraction of initial lr
        staircase=False
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=scheduler,
        beta_1=0.95,
        beta_2=0.99,
        epsilon=1e-6,
        clipnorm=5
    )

    # Tensorboard to monitor training
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.model}_{current_time}" if args.name is None else args.name

    # Tensorboard
    tb_writer = None
    if args.tb:
        log_dir = Path('logs', exp_name)
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = tf.summary.create_file_writer(str(log_dir))

    # W&B
    if args.wb:

        run = wandb.init(
            name=exp_name,
            project="text-recognition",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "architecture": args.model,
                "input_size": args.input_size,
                "optimizer": "adam",
                "exp_type": "text-recognition",
            }
        )

    # Create loss queue
    loss_q = deque(maxlen=100)
    min_loss = np.inf

    # Training loop
    mb = master_bar(range(args.epochs))
    for epoch in mb:
        fit_one_epoch(model, train_loader, batch_transforms, optimizer, loss_q, mb, step, tb_writer)

        # Validation loop at the end of each epoch
        val_loss, exact_match, partial_match = evaluate(model, val_loader, batch_transforms, val_metric)
        if val_loss < min_loss:
            print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
            model.save_weights(f'./{exp_name}/weights')
            min_loss = val_loss
        mb.write(f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} "
                 f"(Exact: {exact_match:.2%} | Partial: {partial_match:.2%})")
        # Tensorboard
        if args.tb:
            with tb_writer.as_default():
                tf.summary.scalar('val_loss', val_loss, step=step)
                tf.summary.scalar('exact_match', exact_match, step=step)
                tf.summary.scalar('partial_match', partial_match, step=step)
        # W&B
        if args.wb:
            wandb.log({
                'epochs': epoch + 1,
                'val_loss': val_loss,
                'exact_match': exact_match,
                'partial_match': partial_match,
            })
        #reset val metric
        val_metric.reset()

    if args.wb:
        run.finish()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR train text-recognition model (TensorFlow)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str, help='path to data folder')
    parser.add_argument('model', type=str, help='text-recognition model to train')
    parser.add_argument('--name', type=str, default=None, help='Name of your training experiment')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model on')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--input_size', type=int, default=32, help='input size H for the model, W = 4*H')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the optimizer (Adam)')
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of workers used for dataloading')
    parser.add_argument('--resume', type=str, default=None, help='Path to your checkpoint')
    parser.add_argument("--test-only", dest='test_only', action='store_true', help="Run the validation loop")
    parser.add_argument('--show-samples', dest='show_samples', action='store_true',
                        help='Display unormalized training samples')
    parser.add_argument('--tb', dest='tb', action='store_true',
                        help='Log to Tensorboard')
    parser.add_argument('--wb', dest='wb', action='store_true',
                        help='Log to Weights & Biases')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='Load pretrained parameters before starting the training')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
