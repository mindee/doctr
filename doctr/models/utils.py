# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from typing import Tuple

logging.getLogger("tensorflow").setLevel(logging.DEBUG)


__all__ = ['convert_to_tflite', 'convert_to_fp16', 'quantize_model']


def convert_to_tflite(tf_model: tf.keras.Model) -> bytes:
    """Converts a model to TFLite format

    Args:
        tf_model: a keras model

    Returns:
        bytes: the model
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    return converter.convert()


def convert_to_fp16(tf_model: tf.keras.Model) -> bytes:
    """Converts a model to half precision

    Args:
        tf_model: a keras model

    Returns:
        bytes: the serialized FP16 model
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter.convert()


def quantize_model(tf_model: tf.keras.Model, input_shape: Tuple[int, int, int]) -> bytes:
    """Quantize a Tensorflow model

    Args:
        tf_model: a keras model
        input_shape: shape of the expected input tensor (excluding batch dimension) with channel last order

    Returns:
        bytes: the serialized quantized model
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Float fallback for operators that do not have an integer implementation
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, *input_shape)
            yield [data.astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()
