# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import logging
import re
import os
import hashlib
from pathlib import Path
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import get_file
from typing import Optional, List, Any

logging.getLogger("tensorflow").setLevel(logging.DEBUG)


__all__ = ['load_pretrained_params', 'conv_sequence', 'IntermediateLayerGetter']


# matches bfd8deac from resnet18-bfd8deac.ckpt
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')


def load_pretrained_params(
    model: Model,
    url: Optional[str] = None,
    hash_prefix: Optional[str] = None,
    overwrite: bool = False,
    internal_name = 'weights',
    **kwargs: Any,
) -> None:
    """Load a set of parameters onto a model

    Args:
        model: the keras model to be loaded
        url: URL of the set of parameters
        hash_prefix: first characters of SHA256 expected hash
        internal_name: name of the ckpt files
    """

    if url is None:
        logging.warning("Invalid model URL, using default initialization.")
    else:
        filename = url.rpartition('/')[-1]
        archive_path = get_file(filename, url, cache_subdir="models", **kwargs)

        # Check hash in file name
        if hash_prefix is None:
            r = HASH_REGEX.search(filename)
            hash_prefix = r.group(1) if r else None

        if isinstance(hash_prefix, str):
            # Hash the file
            with open(archive_path, 'rb') as f:
                sha_hash = hashlib.sha256(f.read()).hexdigest()

            # Compare to expected hash
            if sha_hash[:len(hash_prefix)] != hash_prefix:
                # Remove file
                os.remove(archive_path)
                raise ValueError(f"corrupted download, the hash of {url} does not match its expected value")

        # Unzip the archive
        archive_path = Path(archive_path)
        params_path = archive_path.parent.joinpath(archive_path.stem)
        if not params_path.is_dir() or overwrite:
            with ZipFile(archive_path, 'r') as f:  # type: ignore[assignment]
                f.extractall(path=params_path)  # type: ignore[attr-defined]

        # Load weights
        model.load_weights(f"{params_path}{os.sep}{internal_name}")


def conv_sequence(
    out_channels: int,
    activation: str = None,
    bn: bool = False,
    padding: str = 'same',
    **kwargs: Any,
) -> List[layers.Layer]:
    """Builds a convolutional-based layer sequence

    Args:
        out_channels: number of output channels
        activation: activation to be used (default: no activation)
        bn: should a batch normalization layer be added
        padding: padding scheme

    Returns:
        list of layers
    """
    conv_seq = [layers.Conv2D(out_channels, padding=padding, **kwargs)]

    if bn:
        conv_seq.append(layers.BatchNormalization())

    if isinstance(activation, str):
        conv_seq.append(layers.Activation(activation))

    return conv_seq


class IntermediateLayerGetter(Model):
    """Implements an intermediate layer getter

    Args:
        model: the model to extract feature maps from
        layer_names: the list of layers to retrieve the feature map from
    """
    def __init__(
        self,
        model: Model,
        layer_names: List[str]
    ) -> None:
        intermediate_fmaps = [model.get_layer(layer_name).output for layer_name in layer_names]
        super().__init__(model.input, outputs=intermediate_fmaps)
