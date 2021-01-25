# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from typing import Tuple, List, Union

__all__ = ['postprocessor']


def ctc_decoder(
    logits: tf.Tensor,
    num_classes: int
) -> tf.Tensor:
    """
    Decode logits with CTC decoder from keras backend

    Args:
        logits: raw output of the model, shape BATCH_SIZE X SEQ_LEN X NUM_CLASSES + 1
        num_classes: number of classes of the model

    Returns:
        decoded logits, shape BATCH_SIZE X SEQ_LEN

    """
    batch_len = tf.cast(tf.shape(logits)[0], dtype=tf.int64)
    sequence_len = tf.cast(tf.shape(logits)[1], dtype=tf.int64)

    # computing prediction with ctc decoder
    decoded_logits = K.ctc_decode(
        tf.nn.softmax(logits), sequence_len * tf.ones(shape=(batch_len,), dtype=tf.int64), greedy=True
    )
    _predictions = tf.squeeze(decoded_logits[0])

    # masking -1 of CTC with num_classes (embedding <eos>)
    pred_shape = tf.shape(_predictions)
    mask_eos = num_classes * tf.ones(pred_shape, dtype=tf.int64)
    mask_1 = -1 * tf.ones(pred_shape, dtype=tf.int64)
    predictions = tf.where(mask_1 != _predictions, _predictions, mask_eos)

    return predictions


def postprocessor(
    num_classes: int,
    logits: tf.Tensor,
    label_to_idx: dict,
    ignore_case: bool = False,
    ignore_accents: bool = False
) -> tf.Tensor:
    """
    Postprocess raw prediction of the model (logits) to str

    Args:
        num_classes: number of classes of the model
        pred: raw output of the model, shape BATCH_SIZE X SEQ_LEN X NUM_CLASSES + 1
        label_to_idx: dictionnary mapping alphabet labels to idx of the model classes
        ignore_case: if True, ignore case of letters
        ignore_accents: if True, ignore accents of letters

    Returns:
        A list of decoded words, length BATCH_SIZE
        Each word has a length < SEQ_LEN

    """
    # decode ctc for ctc models
    predictions = ctc_decoder(logits, num_classes)

    label_mapping = label_to_idx.copy()
    label_mapping['<eos>'] = num_classes
    label, _ = zip(*sorted(label_mapping.items(), key=lambda x: x[1]))
    tf_label_to_idx = tf.constant(value=label, dtype=tf.string, shape=[num_classes + 1], name='label_mapping')
    _decoded_strings_pred = tf.strings.reduce_join(inputs=tf.nn.embedding_lookup(tf_label_to_idx, predictions), axis=-1)
    _decoded_strings_pred = tf.strings.split(_decoded_strings_pred, "<eos>")
    decoded_strings_pred = tf.sparse.to_dense(_decoded_strings_pred.to_sparse(), default_value='not valid')[:, 0]
    words_list = [word.decode() for word in list(decoded_strings_pred.numpy())]

    if ignore_case:
        words_list = [word.lower() for word in words_list]

    if ignore_accents:
        raise NotImplementedError

    return words_list
