# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import datetime
import numpy as np
import tensorflow as tf
from collections import deque
from pathlib import Path
from fastprogress.fastprogress import master_bar, progress_bar

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr.models import recognition, RecognitionPreProcessor
from doctr.utils import metrics
from doctr.datasets import RecognitionDataset, DataLoader, VOCABS
from doctr import transforms as T


def main(args):

    # Load both train and val data generators
    train_set = RecognitionDataset(
        img_folder=os.path.join(args.data_path, 'train'),
        labels_path=os.path.join(args.data_path, 'train_labels.json'),
        sample_transforms=T.Compose([
            T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=False),
            # Augmentations
            T.RandomApply(T.InvertColorize(), .2),
            T.RandomJpegQuality(60),
            T.RandomSaturation(.3),
            T.RandomContrast(.3),
            T.RandomBrightness(0.3),
        ]),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, workers=args.workers)

    val_set = RecognitionDataset(
        img_folder=os.path.join(args.data_path, 'val'),
        labels_path=os.path.join(args.data_path, 'val_labels.json'),
        sample_transforms=T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=False),
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, workers=args.workers)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=5)

    # Load doctr model
    model = recognition.__dict__[args.model](
        pretrained=False,
        input_shape=(args.input_size, 4 * args.input_size, 3),
        vocab=VOCABS['french']
    )
    # Resume weights
    if isinstance(args.resume, str):
        model.load_weights(args.resume)

    # Tf variable to log steps
    step = tf.Variable(0, dtype="int64")

    # Metrics
    val_metric = metrics.ExactMatch()

    # Preprocessor to normalize
    MEAN_RGB = (0.694, 0.695, 0.693)
    STD_RGB = (0.299, 0.296, 0.301)
    preprocessor = RecognitionPreProcessor(
        output_size=(args.input_size, 4 * args.input_size),
        batch_size=args.batch_size,
        mean=MEAN_RGB,
        std=STD_RGB
    )

    # Postprocessor to decode output (to feed metric during val step)
    postprocessor = model.postprocessor

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
        train_iter = iter(train_loader)
        # Iterate over the batches of the dataset
        for batch_step in progress_bar(range(train_loader.num_batches), parent=mb):
            images, targets = next(train_iter)

            images = preprocessor(images)
            encoded_gts, seq_len = model.compute_target(targets)

            with tf.GradientTape() as tape:
                if args.teacher_forcing is True:
                    train_logits = model(images, encoded_gts, training=True)
                else:
                    train_logits = model(images, training=True)
                train_loss = model.compute_loss(encoded_gts, train_logits, seq_len)
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
                with tb_writer.as_default():
                    tf.summary.scalar('train_loss', loss, step=step)

        # Validation loop at the end of each epoch
        val_loss, batch_cnt = 0, 0
        val_iter = iter(val_loader)
        for images, targets in val_iter:
            images = preprocessor(images)
            val_logits = model(images, training=False)
            encoded_gts, seq_len = model.compute_target(targets)
            loss = model.compute_loss(encoded_gts, val_logits, seq_len)
            decoded = postprocessor(val_logits)
            val_metric.update(targets, decoded)

            val_loss += loss.numpy().mean()
            batch_cnt += 1

        val_loss /= batch_cnt
        exact_match = val_metric.summary()
        if val_loss < min_loss:
            print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
            model.save_weights(f'./{exp_name}/weights')
            min_loss = val_loss
        mb.write(f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} (Acc: {exact_match:.2%})")
        # Tensorboard
        with tb_writer.as_default():
            tf.summary.scalar('val_loss', val_loss, step=step)
            tf.summary.scalar('exact_match', exact_match, step=step)
        #reset val metric
        val_metric.reset()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR train text-recognition model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str, help='path to data folder')
    parser.add_argument('model', type=str, help='text-recognition model to train')
    parser.add_argument('--name', type=str, default=None, help='Name of your training experiment')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model on')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--input_size', type=int, default=32, help='input size H for the model, W = 4*H')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the optimizer (Adam)')
    parser.add_argument("--teacher_forcing", dest="teacher_forcing",
                        help="enables teacher forcing during training", action="store_true")
    parser.add_argument('-j', '--workers', type=int, default=4, help='number of workers used for dataloading')
    parser.add_argument('--resume', type=str, default=None, help='Path to your checkpoint')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
