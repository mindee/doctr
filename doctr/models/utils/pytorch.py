# Copyright (C) 2021-2026, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
from collections.abc import Iterable
from typing import Any

import torch
import validators
from torch import nn

from doctr.utils.data import download_from_url

__all__ = [
    "load_pretrained_params",
    "conv_sequence_pt",
    "set_device_and_dtype",
    "export_model_to_onnx",
    "add_whitelist",
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
    path_or_url: str | None = None,
    hash_prefix: str | None = None,
    ignore_keys: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """Load a set of parameters onto a model

    >>> from doctr.models import load_pretrained_params
    >>> load_pretrained_params(model, "https://yoursource.com/yourcheckpoint-yourhash.pt")

    Args:
        model: the PyTorch model to be loaded
        path_or_url: the path or URL to the model parameters (checkpoint)
        hash_prefix: first characters of SHA256 expected hash
        ignore_keys: list of weights to be ignored from the state_dict
        **kwargs: additional arguments to be passed to `doctr.utils.data.download_from_url`
    """
    if path_or_url is None:
        logging.warning("No model URL or Path provided, using default initialization.")
        return

    archive_path = (
        download_from_url(path_or_url, hash_prefix=hash_prefix, cache_subdir="models", **kwargs)
        if validators.url(path_or_url)
        else path_or_url
    )

    # Read state_dict
    state_dict = torch.load(archive_path, map_location="cpu")

    # Remove weights from the state_dict
    if ignore_keys is not None and len(ignore_keys) > 0:
        for key in ignore_keys:
            if key in state_dict:
                state_dict.pop(key)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if any(k not in ignore_keys for k in missing_keys + unexpected_keys):
            raise ValueError(
                "Unable to load state_dict, due to non-matching keys.\n"
                + f"Unexpected keys: {unexpected_keys}\nMissing keys: {missing_keys}"
            )
    else:
        # Load weights
        model.load_state_dict(state_dict)


def conv_sequence_pt(
    in_channels: int,
    out_channels: int,
    act: bool = False,
    bn: bool = False,
    activation: nn.Module = nn.ReLU(inplace=True),
    **kwargs: Any,
) -> list[nn.Module]:
    """Builds a convolutional-based layer sequence

    >>> from torch.nn import Sequential
    >>> from doctr.models import conv_sequence
    >>> module = Sequential(conv_sequence(3, 32, True, True, kernel_size=3))

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        act: should an activation layer be added
        bn: should a batch normalization layer be added
        activation: the activation layer to be added if act is True
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

    if act:
        conv_seq.append(activation)

    return conv_seq


def set_device_and_dtype(
    model: Any,
    batches: list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]],
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[Any, list[torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]]]:
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
    model = model.to(device=device, dtype=dtype)
    if isinstance(batches, tuple):
        return model, [
            (img.to(device=device, dtype=dtype), mask.to(device=device, dtype=torch.bool))
            for img, mask in zip(*batches)
        ]
    return model, [batch.to(device=device, dtype=dtype) for batch in batches]


def export_model_to_onnx(
    model: nn.Module, model_name: str, dummy_input: torch.Tensor | tuple[torch.Tensor, torch.Tensor], **kwargs: Any
) -> str:
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
        dummy_input,  # type: ignore[arg-type]
        f"{model_name}.onnx",
        input_names=["input", "masks"] if isinstance(dummy_input, tuple) else ["input"],
        output_names=["logits", "pred_boxes"] if isinstance(dummy_input, tuple) else ["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "masks": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "pred_boxes": {0: "batch_size"},
        }
        if isinstance(dummy_input, tuple)
        else {"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        export_params=True,
        dynamo=False,
        verbose=False,
        **kwargs,
    )
    logging.info(f"Model exported to {model_name}.onnx")
    return f"{model_name}.onnx"


# Location of the final vocabulary-projection layer for each recognition architecture.
_RECOGNITION_PROJECTIONS: dict[str, str] = {
    "CRNN": "linear",
    "SAR": "decoder.output_dense",
    "MASTER": "linear",
    "ViTSTR": "head",
    "PARSeq": "head",
    "VIPTR": "head",
}


class WhitelistHandle:
    """Removable registration returned by :func:`add_whitelist`.

    Call :meth:`remove` to restore the model's original, unconstrained decoding. The
    handle can also be used as a context manager, in which case the whitelist is removed
    on exit.
    """

    def __init__(self, handles: list[torch.utils.hooks.RemovableHandle]) -> None:
        self._handles = handles

    def remove(self) -> None:
        """Remove the whitelist and restore the model's unconstrained decoding."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "WhitelistHandle":
        return self

    def __exit__(self, *_: Any) -> None:
        self.remove()


def _recognition_models(model: nn.Module) -> list[nn.Module]:
    # Accept an ocr_predictor / kie_predictor / recognition_predictor or a recognition model
    if hasattr(model, "vocab") and hasattr(model, "postprocessor"):
        return [model]
    reco_predictor = getattr(model, "reco_predictor", model)
    reco_model = getattr(reco_predictor, "model", None)
    if reco_model is None:
        raise TypeError(
            "Expected an ocr_predictor, kie_predictor, recognition_predictor or a recognition "
            f"model, but could not find a recognition model on {type(model).__name__}."
        )
    return [reco_model]


def _vocab_projections(model: nn.Module, vocab_size: int) -> list[nn.Linear]:
    path = _RECOGNITION_PROJECTIONS.get(type(model).__name__)
    if path is not None:
        layer: Any = model
        for part in path.split("."):
            layer = getattr(layer, part)
        if isinstance(layer, nn.Linear):
            return [layer]
    # Fallback for unknown architectures: any Linear projecting to the vocab (+ up to 3 specials)
    candidates = [
        module
        for module in model.modules()
        if isinstance(module, nn.Linear) and module.out_features in {vocab_size + 1, vocab_size + 2, vocab_size + 3}
    ]
    if not candidates:
        raise RuntimeError(f"Could not locate the vocabulary projection layer of {type(model).__name__}.")
    return candidates


def add_whitelist(
    model: nn.Module,
    vocabs: str | Iterable[str],
    *,
    verbose: bool = False,
) -> WhitelistHandle:
    """Restrict a recognition model so it can only predict a subset of its vocabulary.

    The whitelist is enforced by masking, at the model's final projection layer, the logits
    of every vocabulary character that is not whitelisted (setting them to ``-inf``) before
    the decoding ``argmax``. Because the projection is the single point every logit flows
    through, this also constrains the autoregressive decoding loop of SAR, MASTER and PARSeq,
    so a forbidden character can never be produced -- not even fed back mid-word. The
    sequence terminator (CTC ``blank`` / attention ``<eos>``) is always kept so decoding
    still terminates. It works with every recognition architecture and with any predictor
    wrapping one (`ocr_predictor`, `kie_predictor`, `recognition_predictor`).

    A whitelist can only restrict a model to characters it already knows: characters that
    are not part of the model's own vocabulary are silently ignored.

    >>> from doctr.datasets import VOCABS
    >>> from doctr.models import ocr_predictor
    >>> from doctr.models.utils import add_whitelist
    >>> predictor = ocr_predictor(pretrained=True)
    >>> handle = add_whitelist(predictor, [VOCABS["polish"], VOCABS["german"]])
    >>> # ... run the predictor - only Polish/German characters can be predicted ...
    >>> handle.remove()  # restore the original, unconstrained decoding

    Args:
        model: an `ocr_predictor`, `kie_predictor`, `recognition_predictor`, or a recognition model.
        vocabs: a vocabulary string (e.g. ``VOCABS["german"]``) or an iterable of vocabulary
            strings (e.g. ``[VOCABS["polish"], VOCABS["german"]]``) whose characters are allowed.
        verbose: if True, log how many characters were kept and forbidden for each model.

    Returns:
        a :class:`WhitelistHandle`; call its :meth:`~WhitelistHandle.remove` method to restore
        the original, unconstrained decoding.
    """
    allowed = set(vocabs) if isinstance(vocabs, str) else {char for vocab in vocabs for char in vocab}

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for reco_model in _recognition_models(model):
        vocab: str = reco_model.vocab  # type: ignore[assignment]
        vocab_size = len(vocab)
        if not any(char in allowed for char in vocab):
            raise ValueError(
                "The whitelist shares no character with the model's vocabulary; the model would "
                "be unable to predict anything."
            )

        for projection in _vocab_projections(reco_model, vocab_size):
            # Keep whitelisted characters and the sequence terminator (index == vocab_size);
            # forbid every other character and any trailing special token (e.g. <sos> / <pad>).
            keep = torch.zeros(projection.out_features, dtype=torch.bool)
            for idx, char in enumerate(vocab):
                keep[idx] = char in allowed
            keep[vocab_size] = True

            def _mask_logits(_module: nn.Module, _inputs: Any, output: torch.Tensor, keep: torch.Tensor = keep):
                output = output.clone()
                output[..., ~keep.to(output.device)] = float("-inf")
                return output

            handles.append(projection.register_forward_hook(_mask_logits))

        if verbose:
            kept = sum(char in allowed for char in vocab)
            logging.info(
                f"add_whitelist: {type(reco_model).__name__} - kept {kept}/{vocab_size} vocabulary "
                f"characters, forbade {vocab_size - kept}."
            )

    return WhitelistHandle(handles)
