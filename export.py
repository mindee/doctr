# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import tensorflow as tf
from tensorflow.keras import Model
from doctr.models import detection, recognition
from typing import Tuple


class ExportModel(tf.Module):
    def __init__(self, model: Model, input_shape: Tuple[int, int, int]):
        self.model = model(pretrained=True, input_shape=input_shape)
    
    # Ici je ne sais pas comment avoir le input_shape dans le decorateur
    @tf.function(input_signature=[tf.TensorSpec(shape=(None, *input_shape), dtype=tf.float32)])
    def predict(self, inputs):
        result = self.model(inputs, training=False, return_model_output=True)["out_map"]
        return {"out_map": result}


def export_recognition(arch_name: str, export_path: str, input_shape: Tuple[int, int, int]):
    # Build model
    input_t = tf.random.uniform(shape=[1, *input_shape], maxval=1, dtype=tf.float32)
    module = ExportModel(recognition.__dict__[arch_name])
    _ = module.predict(input_t)
    # Save
    tf.saved_model.save(module, export_path, signatures={"out_map": module.predict})


def export_detection(arch_name: str, export_path: str, input_shape: Tuple[int, int, int]):
    # Build model
    input_t = tf.random.uniform(shape=[1, *input_shape], maxval=1, dtype=tf.float32)
    module = ExportModel(detection.__dict__[arch_name])
    _ = module.predict(input_t)
    # Save
    tf.saved_model.save(module, export_path, signatures={"out_map": module.predict})
