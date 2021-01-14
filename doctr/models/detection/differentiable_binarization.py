# Copyright (C) 2021, Mindee.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
import os
from typing import Union, List, Tuple, Optional, Any, Dict

from doctr.models.detection.inference_utils import batch_documents


def inference(
    path_to_model: str,
    documents: Tuple[List[List[Tuple[int, int]]], List[List[bytes]], List[List[str]]]
) -> Dict[str, Any]:
    """
    perform inference on documents
    :param path_to_model: path to dbnet saved_model
    :param documents: documents to run inference on
    """

    model = tf.saved_model.load(path_to_model)

    infer_dict = dict()
    batched_docs, docs_indexes, pages_indexes = batch_documents(documents, batch_size=2)

    for batch in batched_docs:
        shapes, raw_images, documents_names = batch
        heights = [h for (h, _) in shapes]
        widths = [w for (_, w) in shapes]

        infer = model.signatures["serving_default"]
        pred = infer(raw_images=tf.constant(raw_images), heights=tf.constant(heights), widths=tf.constant(widths))
        pred = pred['output_0']

        boxes_batch, scores_batch = postprocess_img_dbnet(pred, heights, widths)

        for doc_name, boxes, scores in zip(documents_names, boxes_batch, scores_batch):
            infer_dict[doc_name] = {"infer_boxes": boxes, "infer_scores": scores}

    return infer_dict


"""
path_to_model = '/home/datascientist-4/tf2_migration/savedmodeldb_1024'
#path_to_images = '/home/datascientist-4/tf2_migration/inference_seg'
path_to_images = "/home/datascientist-4/samplepdf"

filepaths = [os.path.join(path_to_images, fil) for fil in os.listdir(path_to_images)]
documents = read_documents(filepaths)
infer_dict = inference(path_to_model, documents)
show_infer_pdf(filepaths, infer_dict)
"""
