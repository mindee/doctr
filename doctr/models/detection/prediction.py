# Copyright (C) 2021, Mindee.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
import os
from typing import Union, List, Tuple, Optional, Any, Dict

from doctr.models.detection.inference_utils import batch_documents


def predict_DB(
    path_to_model: str,
    preprocessed_documents:
) -> Dict[str, Any]:
    """
    perform inference on documents
    :param path_to_model: path to dbnet saved_model
    :param documents: documents to run inference on
    """

    model = tf.saved_model.load(path_to_model)

    infer_dict = dict()
    batched_docs, docs_indexes, pages_indexes = preprocessed_documents

    for batch in batched_docs:
        images, names, shapes = batch
        heights = [h for (h, _) in shapes]
        widths = [w for (_, w) in shapes]

        infer = model.signatures["serving_default"]
        pred = infer(images=tf.constant(raw_images), heights=tf.constant(heights), widths=tf.constant(widths))
        pred = pred['output_0']

        boxes_batch, scores_batch = postprocess_img_dbnet(pred, heights, widths)

        for doc_name, boxes, scores in zip(names, boxes_batch, scores_batch):
            infer_dict[doc_name] = {"infer_boxes": boxes, "infer_scores": scores}

    return infer_dict

