# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
import tensorflow.keras.backend as K
from typing import List, Dict

from .core import RecognitionPostProcessor

__all__ = ['CTCPostProcessor']


class CTCPostProcessor(RecognitionPostProcessor):
    """
    Postprocess raw prediction of the model (logits) to a list of words using CTC decoding

    Args:
        num_classes: number of classes of the model
        label_to_idx: dictionnary mapping alphabet labels to idx of the model classes
        ignore_case: if True, ignore case of letters
        ignore_accents: if True, ignore accents of letters

    """
    def __init__(
        self,
        num_classes: int,
        label_to_idx: Dict[str, int],
        ignore_case: bool = False,
        ignore_accents: bool = False
    ) -> None:

        self.num_classes = num_classes
        self.label_to_idx = label_to_idx
        self.ignore_case = ignore_case
        self.ignore_accents = ignore_accents

    def ctc_decoder(
        self,
        logits: tf.Tensor
    ) -> tf.Tensor:
        """
        Decode logits with CTC decoder from keras backend

        Args:
            logits: raw output of the model, shape BATCH_SIZE X SEQ_LEN X NUM_CLASSES + 1

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
        mask_eos = self.num_classes * tf.ones(pred_shape, dtype=tf.int64)
        mask_1 = -1 * tf.ones(pred_shape, dtype=tf.int64)
        predictions = tf.where(mask_1 != _predictions, _predictions, mask_eos)

        return predictions

    def __call__(
        self,
        logits: tf.Tensor
    ) -> List[str]:
        """
        Performs decoding of raw output with CTC and decoding of CTC predictions
        with label_to_idx mapping dictionnary

        Args:
            logits: raw output of the model, shape BATCH_SIZE X SEQ_LEN X NUM_CLASSES + 1

        Returns:
            A list of decoded words of length BATCH_SIZE

        """
        # decode ctc for ctc models
        predictions = self.ctc_decoder(logits)

        label_mapping = self.label_to_idx.copy()
        label_mapping['<eos>'] = self.num_classes
        label, _ = zip(*sorted(label_mapping.items(), key=lambda x: x[1]))
        tf_label_to_idx = tf.constant(value=label, dtype=tf.string, shape=[self.num_classes + 1], name='label_mapping')
        _decoded_strings_pred = tf.strings.reduce_join(
            inputs=tf.nn.embedding_lookup(tf_label_to_idx, predictions),
            axis=-1
        )
        _decoded_strings_pred = tf.strings.split(_decoded_strings_pred, "<eos>")
        decoded_strings_pred = tf.sparse.to_dense(_decoded_strings_pred.to_sparse(), default_value='not valid')[:, 0]
        words_list = [word.decode() for word in list(decoded_strings_pred.numpy())]

        if self.ignore_case:
            words_list = [word.lower() for word in words_list]

        if self.ignore_accents:
            raise NotImplementedError

        return words_list
