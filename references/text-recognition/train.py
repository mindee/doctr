# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import datetime
import numpy as np
import tensorflow as tf
from collections import deque

from doctr import recognition, utils

from tfrecords_loader import preprocess_pipeline
from parse_text_recognition_tfrecords import load_text_recognition_tfrecords


def main(args):

    # load tfrecords as tf.data.Dataset object
    dataset_train = load_text_recognition_tfrecords('recognition/DATASET_RECO/train.tfrecords')
    dataset_val = load_text_recognition_tfrecords('recognition/DATASET_RECO/val.tfrecords')

    h, w = args.input_size

    # batch, normalize, resize, pad train dataset
    train_dataset = preprocess_pipeline(dataset=dataset_train, img_h=h, img_w=w, batch_size=args.batch_size)
 
    # batch, normalize, resize, pad val dataset
    val_dataset = preprocess_pipeline(dataset=dataset_val, img_h=h, img_w=w, batch_size=args.batch_size)

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=5)

    # load doctr model
    model = recognition.__dict__[args.model](pretrained=False, input_shape=args.input_size)

    # tf variable to log steps
    step = tf.Variable(0, dtype="int64")

    # Metrics
    val_metric = metrics.ExactMatch()

    # tensorboard to monitor training
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'log/' + current_time + '/train'
    val_log_dir = 'log/' + current_time + '/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            train_logits = model(x, y, training=True)
            train_loss = loss_fn(y, train_logits, MODEL, NUM_CLASSES, MAX_LENGTH)
        grads = tape.gradient(train_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return train_loss

    @tf.function
    def test_step(x, y):
        val_logits = model(x, training=False)
        val_loss = loss_fn(y, val_logits, MODEL, NUM_CLASSES, MAX_LENGTH)
        update_accuracy(y, val_logits, val_metric, MODEL, NUM_CLASSES, training=False, dic_val=os.path.join(path_val, 'meta.json'))
        return val_loss

    # create loss queue
    loss_q = deque(maxlen=100)

    # training loop
    for epoch in range(args.epochs):
        # Iterate over the batches of the dataset
        for batch_step, ((x_batch_train, names), y_batch_train) in enumerate(train_dataset):
            train_loss = train_step(x_batch_train, y_batch_train)
            # update steps
            step.assign_add(args.batch_size)
            # add loss to queue
            loss_q.append(np.mean(train_loss))
            # log loss and save weights every 100 batch step
            if batch_step % 100 == 0:
                model.save_weights('./checkpointsreco/weights')
                # compute loss
                loss = sum(loss_q) / len(loss_q)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=step)
                #terminal infos
                print('images ---------- ', end='')
                tf.print(step, end=' ')
                template = 'Loss: {}'
                print(template.format(loss))
                
        # initialize loss
        loss_val = []
        for batch_step_val, ((x_batch_val, names), y_batch_val) in enumerate(val_dataset):
            # update steps
            val_loss = test_step(x_batch_val, y_batch_val)
            loss_val.append(np.mean(val_loss))
        mean_loss = sum(loss_val) / len(loss_val)
        #tensorboard
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', mean_loss, step=step)
            tf.summary.scalar('exact_match', val_metric.result(), step=step)
        #terminal infos
        print('VALIDATION: images ---------- ', end='')
        tf.print(step, end=' ')
        template = 'Loss: {} ---------- Exact Match: {}'
        print (template.format(mean_loss, val_metric.result()))
        #reset val metric
        val_metric.reset_states()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR train text-recognition model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('model', type=str, help='text-recognition model to train')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model on')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--input_size', type=Tuple[int, int], default=(32, 128), help='input size (H, W) for the model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for the optimizer (Adam)')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
