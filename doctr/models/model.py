# Copyright (C) 2021, Mindee.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf

class Model:
    """Implements an abstract model class for both detection and recognition models
    Args:
        is_pretrained: wether the model is pretrained or not
        savedmodel: path to savedmodel file if pretrained

    """

    def __init__(self, is_pretrained: str = True, savedmodel: str) -> None:
        self.is_pretrained = is_pretrained
        self.savedmodel = savedmodel

    def load_weigths(self):
        if self.is_pretrained:
            self.model = tf.saved_model.load(self.savedmodel)

    def predict(self, preprocessed_documents, signature):
        batched_docs, docs_indexes, pages_indexes = preprocessed_documents
        for batch in batched_docs:
            images, names, shapes = batch
            infer = self.model.signatures["serving_default"]
            pred = infer(images)
            pred = pred['output_0']
            return pred
