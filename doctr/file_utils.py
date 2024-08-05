# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py

import importlib.metadata
import importlib.util
import logging
import os
from typing import Optional

CLASS_NAME: str = "words"


__all__ = ["is_tf_available", "is_torch_available", "does_torch_have_compile_capability", "is_triton_available", "requires_package", "CLASS_NAME"]

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_TRITON = os.environ.get("USE_TRITON", "NO").upper()


if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib.metadata.version("torch")
            logging.info(f"PyTorch version {_torch_version} available.")
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
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
                _tf_version = importlib.metadata.version(pkg)
                break
            except importlib.metadata.PackageNotFoundError:
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

if _torch_available:
    import torch
    _torch_has_compile = hasattr(torch, "compile")
    _torch_has_triton = False

    if _torch_has_compile and hasattr(torch.library, 'custom_op'):
        from torch.utils._triton import has_triton
        if USE_TRITON in ENV_VARS_TRUE_AND_AUTO_VALUES:
            if has_triton():
                logging.info("Triton detected!")
                _torch_has_triton = True
            elif USE_TRITON == 'AUTO':
                logging.info("Triton was not found! Continuing without it!")
            else:
                logging.warn("Triton was not found even tough it was requested by the user!")
else:
    _torch_has_compile = False
    _torch_has_triton = False


def requires_package(name: str, extra_message: Optional[str] = None) -> None:  # pragma: no cover
    """
    package requirement helper

    Args:
    ----
        name: name of the package
        extra_message: additional message to display if the package is not found
    """
    try:
        _pkg_version = importlib.metadata.version(name)
        logging.info(f"{name} version {_pkg_version} available.")
    except importlib.metadata.PackageNotFoundError:
        raise ImportError(
            f"\n\n{extra_message if extra_message is not None else ''} "
            f"\nPlease install it with the following command: pip install {name}\n"
        )


def is_torch_available():
    """Whether PyTorch is installed."""
    return _torch_available


def is_tf_available():
    """Whether TensorFlow is installed."""
    return _tf_available

def does_torch_have_compile_capability():
    """Whether Pytorch has compile support."""
    return _torch_has_compile

def is_triton_available():
    """Whether Triton is installed."""
    return _torch_has_triton
