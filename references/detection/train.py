# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import datetime
import numpy as np
import tensorflow as tf
from collections import deque

from doctr.models import detection, DetectionPreProcessor
from doctr.utils import metrics
from doctr.datasets import DetectionDataset, DataLoader


def main(args):

    # Load both train and val data generators
    train_set = DetectionDataset(
        input_size=(args.input_size, args.input_size),
        img_folder=os.path.join(args.data_path, 'train'),
        labels_path=os.path.join(args.data_path, 'train_labels')
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, workers=args.workers)

    val_set = DetectionDataset(
        input_size=(args.input_size, args.input_size),
        img_folder=os.path.join(args.data_path, 'val'),
        labels_path=os.path.join(args.data_path, 'val_labels')
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, workers=args.workers)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=5)

    # Load doctr model
    model = detection.__dict__[args.model](pretrained=False, input_shape=(args.input_size, args.input_size, 3))

    # Tf variable to log steps
    step = tf.Variable(0, dtype="int64")

    # Metrics
    val_metric = metrics.LocalizationConfusion()

    # Preprocessor to normalize
    MEAN_RGB = (0.798, 0.785, 0.772)
    STD_RGB = (0.264, 0.2749, 0.287)
    preprocessor = DetectionPreProcessor(
        output_size=(args.input_size, args.input_size),
        batch_size=args.batch_size,
        mean=MEAN_RGB,
        std=STD_RGB
    )

    # Postprocessor to decode output (to feed metric during val step with boxes)
    postprocessor = detection.DBPostProcessor()

    # Tensorboard to monitor training
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'log/' + current_time + '/train'
    val_log_dir = 'log/' + current_time + '/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    def train_step(x):
        with tf.GradientTape() as tape:
            images, targets = x
            boxes = [target['boxes'] for target in targets]
            flags = [target['flags'] for target in targets]
            images = preprocessor(images)
            model_output = model(images, training=True)
            train_loss = model.compute_loss(model_output, boxes, flags)
        grads = tape.gradient(train_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return train_loss

    def test_step(x):
        images, targets = x
        boxes = [target['boxes'] for target in targets]
        flags = [target['flags'] for target in targets]
        images = preprocessor(images)
        # If we want to compute val loss, we need to pass training=True to have a thresh_map
        model_output = model(images, training=True)
        val_loss = model.compute_loss(model_output, boxes, flags)
        decoded = postprocessor(model_output)
        # Compute metric
        for boxes_gt, boxes_pred in zip(boxes, decoded):
            boxes_pred = np.array(boxes_pred)[:, :-1]  # Remove scores
            boxes_gt = np.array([[
                np.array(box).min(axis=0), np.array(box).min(axis=1),
                np.array(box).max(axis=0), np.array(box).max(axis=1)
            ] for box in boxes_gt])  # From a list of [x, y] coords to xmin, ymin, xmax, ymax format
            val_metric.update(gts=boxes_gt, preds=boxes_pred)
        return val_loss

    # Create loss queue
    loss_q = deque(maxlen=100)

    # Training loop
    for _ in range(args.epochs):
        train_iter = iter(train_loader)
        # Iterate over the batches of the dataset
        for batch_step in range(train_loader.num_batches):
            batch = next(train_iter)
            train_loss = train_step(batch)
            # Update steps
            step.assign_add(args.batch_size)
            # Add loss to queue
            loss_q.append(np.mean(train_loss))
            # Log loss and save weights every 100 batch step
            if batch_step % 100 == 0:
                model.save_weights('./checkpointsdet/weights')
                # Compute loss
                loss = sum(loss_q) / len(loss_q)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=step)

        # Validation loop at the end of each epoch
        loss_val = []
        val_iter = iter(val_loader)
        for batch in val_iter:
            val_loss = test_step(batch)
            loss_val.append(np.mean(val_loss))
        mean_loss = sum(loss_val) / len(loss_val)
        #tensorboard
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', mean_loss, step=step)
            tf.summary.scalar('exact_match', val_metric.result(), step=step)
        #reset val metric
        val_metric.reset()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR train text-detection model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str, help='path to data folder')
    parser.add_argument('model', type=str, help='text-detection model to train')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model on')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--input_size', type=int, default=1024, help='model input size, H = W)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for the optimizer (Adam)')
    parser.add_argument('--workers, -j', type=int, default=4, help='number of workers used for dataloading')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
