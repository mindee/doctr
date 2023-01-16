# Copyright (C) 2021-2023, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
from typing import Any, List, Optional

import torch
from torch import nn

from doctr.utils.data import download_from_url

__all__ = ["load_pretrained_params", "conv_sequence_pt", "export_model_to_onnx"]


def load_pretrained_params(
    model: nn.Module,
    url: Optional[str] = None,
    hash_prefix: Optional[str] = None,
    overwrite: bool = False,
    ignore_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> None:
    """Load a set of parameters onto a model

    >>> from doctr.models import load_pretrained_params
    >>> load_pretrained_params(model, "https://yoursource.com/yourcheckpoint-yourhash.zip")

    Args:
        model: the PyTorch model to be loaded
        url: URL of the zipped set of parameters
        hash_prefix: first characters of SHA256 expected hash
        overwrite: should the zip extraction be enforced if the archive has already been extracted
        ignore_keys: list of weights to be ignored from the state_dict
    """

    if url is None:
        logging.warning("Invalid model URL, using default initialization.")
    else:
        archive_path = download_from_url(url, hash_prefix=hash_prefix, cache_subdir="models", **kwargs)

        # Read state_dict
        state_dict = torch.load(archive_path, map_location="cpu")

        # Remove weights from the state_dict
        if ignore_keys is not None and len(ignore_keys) > 0:
            for key in ignore_keys:
                state_dict.pop(key)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if set(missing_keys) != set(ignore_keys) or len(unexpected_keys) > 0:
                raise ValueError("unable to load state_dict, due to non-matching keys.")
        else:
            # Load weights
            model.load_state_dict(state_dict)


def conv_sequence_pt(
    in_channels: int,
    out_channels: int,
    relu: bool = False,
    bn: bool = False,
    **kwargs: Any,
) -> List[nn.Module]:
    """Builds a convolutional-based layer sequence

    >>> from torch.nn import Sequential
    >>> from doctr.models import conv_sequence
    >>> module = Sequential(conv_sequence(3, 32, True, True, kernel_size=3))

    Args:
        out_channels: number of output channels
        relu: whether ReLU should be used
        bn: should a batch normalization layer be added

    Returns:
        list of layers
    """
    # No bias before Batch norm
    kwargs["bias"] = kwargs.get("bias", not bn)
    # Add activation directly to the conv if there is no BN
    conv_seq: List[nn.Module] = [nn.Conv2d(in_channels, out_channels, **kwargs)]

    if bn:
        conv_seq.append(nn.BatchNorm2d(out_channels))

    if relu:
        conv_seq.append(nn.ReLU(inplace=True))

    return conv_seq


def export_model_to_onnx(model: nn.Module, model_name: str, dummy_input: torch.Tensor, **kwargs: Any) -> str:
    """Export model to ONNX format.

    >>> import torch
    >>> from doctr.models.classification import resnet18
    >>> from doctr.models.utils import export_model_to_onnx
    >>> model = resnet18(pretrained=True)
    >>> export_model_to_onnx(model, "my_model", dummy_input=torch.randn(1, 3, 32, 32))

    Args:
        model: the PyTorch model to be exported
        model_name: the name for the exported model
        dummy_input: the dummy input to the model
        kwargs: additional arguments to be passed to torch.onnx.export

    Returns:
        the path to the exported model
    """
    torch.onnx.export(
        model,
        dummy_input,
        f"{model_name}.onnx",
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        export_params=True,
        opset_version=14,  # minimum opset which support all operators we use (v0.5.2)
        verbose=False,
        **kwargs,
    )
    logging.info(f"Model exported to {model_name}.onnx")
    return f"{model_name}.onnx"
