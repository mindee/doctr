# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

# Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py

import os
import logging

__all__ = ['is_tf_available', 'is_torch_available']

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "1").upper()
USE_TORCH = os.environ.get("USE_TORCH", "0").upper()


if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = True
    try:
        import torch
        logging.info(f"PyTorch version {torch.__version__} available.")
    except ModuleNotFoundError:
        _torch_available = False
else:
    logging.info("Disabling PyTorch because USE_TF is set")
    _torch_available = False


if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    _tf_available = True
    try:
        import tensorflow as tf
        if int(tf.__version__.split('.')[0]) < 2:
            logging.info(f"TensorFlow found but with version {tf.__version__}. DocTR requires version 2 minimum.")
            _tf_available = False
        else:
            logging.info(f"TensorFlow version {tf.__version__} available.")
    except ModuleNotFoundError:
        _tf_available = False
else:
    logging.info("Disabling Tensorflow because USE_TORCH is set")
    _tf_available = False


def is_torch_available():
    return _torch_available


def is_tf_available():
    return _tf_available
