# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from typing import List

from .core import RecognitionPostProcessor

__all__ = ['CTCPostProcessor']


class CTCPostProcessor(RecognitionPostProcessor):
    """
    Postprocess raw prediction of the model (logits) to a list of words using CTC decoding

    Args:
        vocab: string containing the ordered sequence of supported characters
        ignore_case: if True, ignore case of letters
        ignore_accents: if True, ignore accents of letters
    """

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
        # computing prediction with ctc decoder
        _prediction = tf.nn.ctc_greedy_decoder(
            tf.nn.softmax(tf.transpose(logits, perm=[1, 0, 2])),
            tf.fill(logits.shape[0], logits.shape[1]),
            merge_repeated=True
        )[0][0]
        prediction = tf.sparse.to_dense(_prediction, default_value=len(self.vocab))

        return prediction

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

        _decoded_strings_pred = tf.strings.reduce_join(
            inputs=tf.nn.embedding_lookup(self._embedding, predictions),
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
