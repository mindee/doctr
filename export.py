# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from doctr.models.preprocessor import PreProcessor
import tensorflow as tf
from tensorflow.keras import Model
from doctr.models import detection, recognition
from typing import Tuple, List


def raw_bytes_to_img(raw_bytes: str, height: int, width: int):
    """Convert a raw byte triplet (raw_bytes, height, width) to a RGB tensor
    """
    #decode raw
    image = tf.io.decode_raw(raw_bytes, out_type=tf.float32)
    #reshape flattened img
    image = tf.reshape(image, (height, width, 3))
    return image


def preprocess_raw_images(raw_images: Tuple[List[str], List[int], List[int]], preprocessor: PreProcessor):
    """Convert all raw bytes triplet to images and preprocess the tensors in batches
    """
    raw_bytes, heights, widths = raw_images
    images = tf.vectorized_map(fn=raw_bytes_to_img, elems=(raw_bytes, heights, widths))
    processed_batches = preprocessor(images)
    return processed_batches


class ExportModel(tf.Module):
    
    def __init__(
        self,
        preprocessor: PreProcessor,
        model: Model,
    ) -> None:
        self.preprocessor = preprocessor
        self.model = model

    @tf.function(input_signature=[
        tf.TensorSpec([None, ], dtype=tf.string),
        tf.TensorSpec([None, ], dtype=tf.int32),
        tf.TensorSpec([None, ], dtype=tf.int32)
    ])
    def predict(self, infer_array_inputs, infer_inputs_heights, infer_inputs_widths):
        
        # Batch processing
        raw_images = infer_array_inputs, infer_inputs_heights, infer_inputs_widths
        processed_batches = preprocess_raw_images(raw_images, self.preprocessor)

        # Model prediction
        output = self.model(processed_batches, training=False, return_model_output=True, return_preds=True)
        raw_output, predictions = output["out_map"], output["preds"]

        return {"infer_raw_output": raw_output, "infer_predictions": predictions}


def export_recognition(arch_name: str, export_path: str, input_shape: Tuple[int, int, int]):
    # Build model
    input_t = tf.random.uniform(shape=[1, *input_shape], maxval=1, dtype=tf.float32)
    model = recognition.__dict__[arch_name](pretrained=True, input_shape=input_shape)
    preprocessor = PreProcessor(
        output_size=input_shape[:-1],
        batch_size=64,
        mean=(0.694, 0.695, 0.693),
        std=(0.299, 0.296, 0.301)
    )
    module = ExportModel(preprocessor, model)
    _ = module.predict(input_t)
    # Save
    tf.saved_model.save(module, export_path, signatures=module.predict)


def export_detection(arch_name: str, export_path: str, input_shape: Tuple[int, int, int]):
    # Build model
    input_t = tf.random.uniform(shape=[1, *input_shape], maxval=1, dtype=tf.float32)
    model = detection.__dict__[arch_name](pretrained=True, input_shape=input_shape)
    preprocessor = PreProcessor(
        output_size=input_shape[:-1],
        batch_size=4,
        mean=(0.798, 0.785, 0.772),
        std=(0.264, 0.2749, 0.287)
    )
    module = ExportModel(preprocessor, model)
    _ = module.predict(input_t)
    # Save
    tf.saved_model.save(module, export_path, signatures=module.predict)
