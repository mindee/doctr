# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py

import importlib.util
import logging
import os
import sys

CLASS_NAME: str = "words"


if sys.version_info < (3, 8):  # pragma: no cover
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


__all__ = ["is_tf_available", "is_torch_available", "CLASS_NAME"]

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()


if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
            logging.info(f"PyTorch version {_torch_version} available.")
        except importlib_metadata.PackageNotFoundError:  # pragma: no cover
            _torch_available = False
else:  # pragma: no cover
    logging.info("Disabling PyTorch because USE_TF is set")
    _torch_available = False


if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
    _tf_available = importlib.util.find_spec("tensorflow") is not None
    if _tf_available:
        candidates = (
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "tensorflow-rocm",
            "tensorflow-macos",
        )
        _tf_version = None
        # For the metadata, we have to look for both tensorflow and tensorflow-cpu
        for pkg in candidates:
            try:
                _tf_version = importlib_metadata.version(pkg)
                break
            except importlib_metadata.PackageNotFoundError:
                pass
        _tf_available = _tf_version is not None
    if _tf_available:
        if int(_tf_version.split(".")[0]) < 2:  # type: ignore[union-attr]  # pragma: no cover
            logging.info(f"TensorFlow found but with version {_tf_version}. DocTR requires version 2 minimum.")
            _tf_available = False
        else:
            logging.info(f"TensorFlow version {_tf_version} available.")
else:  # pragma: no cover
    logging.info("Disabling Tensorflow because USE_TORCH is set")
    _tf_available = False


if not _torch_available and not _tf_available:  # pragma: no cover
    raise ModuleNotFoundError(
        "DocTR requires either TensorFlow or PyTorch to be installed. Please ensure one of them"
        " is installed and that either USE_TF or USE_TORCH is enabled."
    )


def is_torch_available():
    """Whether PyTorch is installed."""
    return _torch_available


def is_tf_available():
    """Whether TensorFlow is installed."""
    return _tf_available
