# Copyright (C) 2021-2025, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
from typing import Any

import torch
from torch import nn

from doctr.utils.data import download_from_url

__all__ = [
    "load_pretrained_params",
    "conv_sequence_pt",
    "set_device_and_dtype",
    "export_model_to_onnx",
    "_copy_tensor",
    "_bf16_to_float32",
    "_CompiledModule",
]

# torch compiled model type
_CompiledModule = torch._dynamo.eval_frame.OptimizedModule


def _copy_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.clone().detach()


def _bf16_to_float32(x: torch.Tensor) -> torch.Tensor:
    # bfloat16 is not supported in .numpy(): torch/csrc/utils/tensor_numpy.cpp:aten_to_numpy_dtype
    return x.float() if x.dtype == torch.bfloat16 else x


def load_pretrained_params(
    model: nn.Module,
    url: str | None = None,
    hash_prefix: str | None = None,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """Load a set of parameters onto a model

    >>> from doctr.models import load_pretrained_params
    >>> load_pretrained_params(model, "https://yoursource.com/yourcheckpoint-yourhash.zip")

    Args:
        model: the PyTorch model to be loaded
        url: URL of the zipped set of parameters
        hash_prefix: first characters of SHA256 expected hash
        ignore_keys: list of weights to be ignored from the state_dict
        **kwargs: additional arguments to be passed to `doctr.utils.data.download_from_url`
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
) -> list[nn.Module]:
    """Builds a convolutional-based layer sequence

    >>> from torch.nn import Sequential
    >>> from doctr.models import conv_sequence
    >>> module = Sequential(conv_sequence(3, 32, True, True, kernel_size=3))

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        relu: whether ReLU should be used
        bn: should a batch normalization layer be added
        **kwargs: additional arguments to be passed to the convolutional layer

    Returns:
        list of layers
    """
    # No bias before Batch norm
    kwargs["bias"] = kwargs.get("bias", not bn)
    # Add activation directly to the conv if there is no BN
    conv_seq: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, **kwargs)]

    if bn:
        conv_seq.append(nn.BatchNorm2d(out_channels))

    if relu:
        conv_seq.append(nn.ReLU(inplace=True))

    return conv_seq


def set_device_and_dtype(
    model: Any, batches: list[torch.Tensor], device: str | torch.device, dtype: torch.dtype
) -> tuple[Any, list[torch.Tensor]]:
    """Set the device and dtype of a model and its batches

    >>> import torch
    >>> from torch import nn
    >>> from doctr.models.utils import set_device_and_dtype
    >>> model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    >>> batches = [torch.rand(8) for _ in range(2)]
    >>> model, batches = set_device_and_dtype(model, batches, device="cuda", dtype=torch.float16)

    Args:
        model: the model to be set
        batches: the batches to be set
        device: the device to be used
        dtype: the dtype to be used

    Returns:
        the model and batches set
    """
    return model.to(device=device, dtype=dtype), [batch.to(device=device, dtype=dtype) for batch in batches]


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
        verbose=False,
        **kwargs,
    )
    logging.info(f"Model exported to {model_name}.onnx")
    return f"{model_name}.onnx"
