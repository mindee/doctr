# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
import json
import datetime
import numpy as np
import tensorflow as tf
from collections import deque

from doctr import recognition
from doctr.utils import metrics
from doctr.datasets import RecognitionDataGenerator, VOCABS


def main(args):

    # Load both train and val data generators
    train_dataset = RecognitionDataGenerator(
        input_size=args.input_size,
        batch_size=args.batch_size,
        images_path=os.path.join(args.data_path, 'train'),
        labels_path=os.path.join(args.data_path, 'labels.json')
    )

    val_dataset = RecognitionDataGenerator(
        input_size=args.input_size,
        batch_size=args.batch_size,
        images_path=os.path.join(args.data_path, 'val'),
        labels_path=os.path.join(args.data_path, 'labels.json')
    )

    h, w = args.input_size

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=5)

    # Load doctr model
    model = recognition.__dict__[args.model](pretrained=False, input_shape=args.input_size)

    # Tf variable to log steps
    step = tf.Variable(0, dtype="int64")

    # Metrics
    val_metric = utils.metrics.ExactMatch()

    # Postprocessor to decode output (to feed metric during val step)
    if args.postprocessor == 'sar':
        postprocessor = recognition.SARPostProcessor(vocab=VOCABS["french"])
    else:
        postprocessor = recognition.CTCPostProcessor(vocab=VOCABS["french"])

    # Tensorboard to monitor training
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'log/' + current_time + '/train'
    val_log_dir = 'log/' + current_time + '/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            if args.teacher_forcing is True:
                train_logits = model(x, y, training=True)
            else:
                train_logits = model(x, training=True)
            train_loss = "!!!LOSS FN!!!"
        grads = tape.gradient(train_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return train_loss

    @tf.function
    def test_step(x, y):
        val_logits = model(x, training=False)
        val_loss = "!!!LOSS FN!!!"
        decoded = postprocessor(val_logits)
        val_metric.update(gts=y, preds=decoded)
        return val_loss

    # Create loss queue
    loss_q = deque(maxlen=100)

    # Training loop
    for epoch in range(args.epochs):
        # Iterate over the batches of the dataset
        for x_batch_train, y_batch_train in train_dataset:
            train_loss = train_step(x_batch_train, y_batch_train)
            # Update steps
            step.assign_add(args.batch_size)
            # Add loss to queue
            loss_q.append(np.mean(train_loss))
            # Log loss and save weights every 100 batch step
            if batch_step % 100 == 0:
                model.save_weights('./checkpointsreco/weights')
                # Compute loss
                loss = sum(loss_q) / len(loss_q)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=step)

        # Validation loop at the end of each epoch
        loss_val = []
        for x_batch_val, y_batch_val in val_dataset:
            val_loss = test_step(x_batch_val, y_batch_val)
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
    parser = argparse.ArgumentParser(description='DocTR train text-recognition model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('model', type=str, help='text-recognition model to train')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model on')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--input_size', type=Tuple[int, int], default=(32, 128), help='input size (H, W) for the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for the optimizer (Adam)')
    parser.add_argument('--postprocessor', type=str, default='crnn', help='postprocessor, either crnn or sar')
    parser.add_argument('--teacher_forcing', type=bool, default=False, help='if True, teacher forcing during training')
    parser.add_argument('--data_path', type=str, help='path to data folder')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
